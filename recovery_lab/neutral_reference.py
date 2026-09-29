"""Experimental exact-formula neutral references for odd-electron [M]+ ions.

This isolated experiment preserves the candidate's heavy graph, bond orders,
aromaticity, elemental formula, and ion m/z. It represents ionization with the
existing GLACIER ``[M]+`` adduct instead of a charged molecular graph. The graph
itself is a neutral, closed-shell reference; it is NOT a recovered ionic state.
Predictions from this route have not been established to retain accuracy.

Initialize the repository's GlacierAPI before calling the preparation helper so
its pinned upstream featurizer is available. No production policy is modified.
"""

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from functools import lru_cache
import math
import time

from rdkit import Chem, rdBase

from src.chemistry.formula import ionic_mass
from src.chemistry._state_graph import (
    MAX_LEAVES, MAX_CHEMICAL_STATES, MAX_GRAPH_ATOMS,
    MAX_DP_STATES, MAX_SEARCH_NODES,
    _validate_stripped, _local_domains, _validate_state,
    _chemistry_rank, _feature_hash,
)
from src.models.glacier_api import records_from_mol


ROUTE = "neutral_reference_[M]+"
POLICY = "experimental-fixed-bonds-neutral-reference-min-abs-charge-v1"
NEUTRAL_CHARGE_FAMILIES = (0, 2, 4, 6, 8)


def enumerate_neutral_references(
    stripped, max_states=4, *, max_leaves=MAX_LEAVES,
    max_chemical_states=MAX_CHEMICAL_STATES, max_dp_states=MAX_DP_STATES,
    max_search_nodes=MAX_SEARCH_NODES,
):
    """Bounded exact-formula charge-0 search for a charge-1 odd-electron ion.

    Adapted from the production suffix-DP enumerator, with explicit neutral
    reference charge and even absolute-charge families. All production graph,
    formula, sanitization, ordering, and featurization checks are retained.
    """
    started = time.monotonic()
    caps = (max_states, max_leaves, max_chemical_states, max_dp_states, max_search_nodes)
    if any(isinstance(cap, bool) or not isinstance(cap, int) or cap < 1 for cap in caps):
        raise ValueError("Enumeration caps must be positive integers")
    metadata = {
        "policy": POLICY, "route": ROUTE, "experimental": True,
        "accuracy_validated": False,
        "representation": "neutral_closed_shell_reference_plus_M_radical_cation_adduct",
        "ion_charge": 1, "reference_charge": 0,
        "charge_families": list(NEUTRAL_CHARGE_FAMILIES),
        "local_charge_domain": [-1, 0, 1], "radical_domain": "closed_shell_reference",
        "max_leaves": max_leaves, "max_chemical_states": max_chemical_states,
        "max_states": max_states, "leaves_checked": 0, "chemical_valid_assignments": 0,
        "max_graph_atoms": MAX_GRAPH_ATOMS, "max_dp_states": max_dp_states,
        "max_search_nodes": max_search_nodes, "dp_states": 0, "search_nodes": 0,
        "chemical_unique_states": 0, "feature_unique_checked": 0,
        "minimum_abs_charge": None, "families_exhausted": [],
        "rejections": {}, "truncated": False, "stop_reason": None,
    }

    def finish(status, reason, states=None):
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"status": status, "reason": reason, "states": states or [],
                "metadata": metadata, "route": ROUTE, "experimental": True}

    # Check the bound before constructing an RDKit graph, as well as before DP.
    if isinstance(stripped, Mapping) and len(stripped.get("atoms", [])) > MAX_GRAPH_ATOMS:
        return finish("unsupported", "graph_atom_limit")
    try:
        base, counts = _validate_stripped(stripped)
    except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as exc:
        return finish("failed", "invalid_stripped_record: " + str(exc))
    if isinstance(stripped["charge"], bool) or stripped["charge"] != 1:
        return finish("unsupported", "neutral_reference_route_requires_ion_charge_plus_one")
    predicted_mz = ionic_mass(counts, 1)
    try:
        measured_mz = float(stripped["precursor_mz"])
    except (TypeError, ValueError):
        return finish("failed", "precursor_formula_mass_mismatch")
    if (not math.isfinite(measured_mz) or measured_mz <= 0
            or abs(predicted_mz - measured_mz) > max(.005, measured_mz * 20e-6)):
        return finish("failed", "precursor_formula_mass_mismatch")
    metadata["formula_mz"] = predicted_mz
    neutral_electrons = sum(Chem.GetPeriodicTable().GetAtomicNumber(s) * n
                            for s, n in counts.items())
    metadata["ion_electron_parity"] = "odd" if (neutral_electrons - 1) % 2 else "even"
    if neutral_electrons % 2:
        return finish("unsupported", "not_an_odd_electron_ion_formula")
    graph = {"atoms": stripped["atoms"], "bonds": stripped["bonds"]}
    try:
        domains = _local_domains(graph)
    except ValueError as exc:
        return finish("unsupported", str(exc))
    if any(not domain for domain in domains):
        return finish("failed", "empty_local_valence_domain")
    order = sorted(range(len(domains)), key=lambda i: (len(domains[i]), i))
    ordered = [domains[i] for i in order]
    total_h, n = counts.get("H", 0), len(ordered)
    suffix_min_h, suffix_max_h = [0] * (n + 1), [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_min_h[i] = suffix_min_h[i + 1] + min(h for q, h in ordered[i])
        suffix_max_h[i] = suffix_max_h[i + 1] + max(h for q, h in ordered[i])

    class BudgetExceeded(Exception):
        pass

    @lru_cache(maxsize=None)
    def reachable(i, h, q, abs_q):
        metadata["dp_states"] += 1
        if metadata["dp_states"] > max_dp_states:
            raise BudgetExceeded("dp_state_budget")
        if h < suffix_min_h[i] or h > suffix_max_h[i] or abs_q < abs(q) or abs_q < 0:
            return False
        if (abs_q - q) % 2 or abs_q > n - i:
            return False
        if i == n:
            return h == q == abs_q == 0
        return any(reachable(i + 1, h - ah, q - aq, abs_q - abs(aq))
                   for aq, ah in ordered[i])

    assignments, chemical, reject = [None] * n, {}, Counter()
    stopped = None

    def visit(i, h, q, abs_q):
        nonlocal stopped
        if stopped is not None:
            return
        metadata["search_nodes"] += 1
        if metadata["search_nodes"] > max_search_nodes:
            raise BudgetExceeded("search_node_budget")
        if i == n:
            if metadata["leaves_checked"] >= max_leaves:
                stopped = "leaf_budget"
                return
            metadata["leaves_checked"] += 1
            mol = Chem.Mol(base)
            for j, (aq, ah) in enumerate(assignments):
                atom = mol.GetAtomWithIdx(order[j])
                atom.SetFormalCharge(aq)
                atom.SetNumExplicitHs(ah)
                atom.SetNoImplicit(True)
                atom.SetNumRadicalElectrons(0)
            try:
                with rdBase.BlockLogs():
                    Chem.SanitizeMol(mol)
                failure = _validate_state(mol, graph, counts, 0)
                if failure:
                    reject[failure] += 1
                    return
                actual_abs_q = sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms())
                if actual_abs_q != current_family:
                    reject["charge_family_changed"] += 1
                    return
                smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                metadata["chemical_valid_assignments"] += 1
                if smiles not in chemical:
                    chemical[smiles] = _chemistry_rank(mol, smiles)
                if len(chemical) >= max_chemical_states:
                    stopped = "chemical_cap"
            except (ValueError, RuntimeError):
                reject["sanitization_failed"] += 1
            return
        for aq, ah in ordered[i]:
            if reachable(i + 1, h - ah, q - aq, abs_q - abs(aq)):
                assignments[i] = (aq, ah)
                visit(i + 1, h - ah, q - aq, abs_q - abs(aq))
                if stopped is not None:
                    return

    try:
        for current_family in NEUTRAL_CHARGE_FAMILIES:
            if reachable(0, total_h, 0, current_family):
                visit(0, total_h, 0, current_family)
            if chemical:
                metadata["minimum_abs_charge"] = current_family
                break
            if stopped:
                break
            metadata["families_exhausted"].append(current_family)
    except BudgetExceeded as exc:
        stopped = str(exc)
        if chemical:
            metadata["minimum_abs_charge"] = current_family
    metadata["chemical_unique_states"] = len(chemical)
    metadata["rejections"] = dict(sorted(reject.items()))
    metadata["dp_cache"] = reachable.cache_info()._asdict()
    metadata["truncated"] = stopped is not None
    metadata["stop_reason"] = stopped or "minimum_valid_family_exhausted"
    if not chemical:
        metadata["stop_reason"] = stopped or "all_declared_families_exhausted"
        return finish("failed", "search_incomplete" if stopped
                      else "no_valid_neutral_reference_in_declared_domain")
    states, hashes = [], set()
    for smiles, rank in sorted(chemical.items(), key=lambda item: item[1]):
        mol = Chem.MolFromSmiles(smiles)
        try:
            signature = _feature_hash(mol, "[M]+")
        except (ValueError, RuntimeError):
            reject["native_featurization_failed"] += 1
            continue
        if signature in hashes:
            continue
        hashes.add(signature)
        states.append({
            "reference_smiles": smiles, "reference_charge": 0,
            "reference_formula": dict(counts), "ionic_formula": dict(counts),
            "charge": 1, "formula_mz": predicted_mz, "adduct": "[M]+",
            "feature_hash": signature, "chemistry_rank": list(rank),
            "total_abs_charge": rank[0], "route": ROUTE, "experimental": True,
        })
        if len(states) == max_states:
            if len(chemical) > len(states):
                metadata["truncated"] = True
                metadata["feature_selection_capped"] = True
            break
    metadata["feature_unique_checked"] = len(hashes)
    metadata["rejections"] = dict(sorted(reject.items()))
    metadata["retained_states"] = len(states)
    return finish("ok" if states else "failed",
                  None if states else "native_featurization_failed", states)


def prepare_neutral_reference(candidate, collision_energy, instrument="Orbitrap",
                              max_states=4, **enumeration_caps):
    """Return experimental states, audit metadata, and unchanged API inputs.

    Accept the same candidate/stripped-record forms as FragmentGlacierAdapter.
    No hydrogens are added or removed, and no changes are made to the input.
    Each returned input can be passed to ``glacier.predict_records(**input)``.
    """
    if hasattr(candidate, "stripped"):
        stripped = candidate.stripped
    elif isinstance(candidate, Mapping):
        stripped = candidate.get("stripped", candidate)
    else:
        raise TypeError("Expected FragmentCandidate or stripped record")
    ce = float(collision_energy)
    if not math.isfinite(ce):
        raise ValueError("Collision energy must be finite")
    result = enumerate_neutral_references(deepcopy(stripped), max_states,
                                         **enumeration_caps)
    inputs = [{"records": records_from_mol(Chem.MolFromSmiles(state["reference_smiles"])),
               "collision_energy": ce, "adduct": "[M]+", "instrument": instrument}
              for state in result["states"]]
    return {**result, "inputs": inputs}
