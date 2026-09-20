"""Bounded prototype H/Q/|Q| suffix-DP enumerator; scientific order preserved."""
from collections import Counter
from functools import lru_cache
import math
import time

from rdkit import Chem, rdBase

from src.chemistry.formula import ionic_mass
from src.chemistry._state_graph import (
    POLICY_VERSION, MAX_LEAVES, MAX_CHEMICAL_STATES, MAX_GRAPH_ATOMS,
    MAX_DP_STATES, MAX_SEARCH_NODES, CHARGE_FAMILIES,
    _validate_stripped, _local_domains, _validate_state, _chemistry_rank, _feature_hash,
)


def enumerate_microstates(stripped, max_states=4, *, max_leaves=MAX_LEAVES,
                         max_chemical_states=MAX_CHEMICAL_STATES,
                         max_dp_states=MAX_DP_STATES, max_search_nodes=MAX_SEARCH_NODES):
    """Enumerate a bounded minimum-|Q| family using exact H/Q/|Q| suffix DP."""
    started = time.monotonic()
    if min(max_states, max_leaves, max_chemical_states, max_dp_states, max_search_nodes) < 1:
        raise ValueError("Enumeration caps must be positive")
    metadata = {"policy": POLICY_VERSION, "charge_families": list(CHARGE_FAMILIES),
                "local_charge_domain": [-1, 0, 1], "radical_domain": "closed_shell",
                "max_leaves": max_leaves, "max_chemical_states": max_chemical_states,
                "max_states": max_states, "leaves_checked": 0, "chemical_valid_assignments": 0,
                "max_graph_atoms": MAX_GRAPH_ATOMS, "max_dp_states": max_dp_states,
                "max_search_nodes": max_search_nodes, "dp_states": 0, "search_nodes": 0,
                "chemical_unique_states": 0, "feature_unique_checked": 0,
                "minimum_abs_charge": None, "families_exhausted": [],
                "rejections": {}, "truncated": False, "stop_reason": None}
    def finish(status, reason, states=None):
        metadata["elapsed_seconds"] = round(time.monotonic() - started, 6)
        return {"status": status, "reason": reason, "states": states or [], "metadata": metadata}
    try:
        base, counts = _validate_stripped(stripped)
    except (ValueError, RuntimeError, TypeError) as exc:
        return finish("failed", "invalid_stripped_record: " + str(exc))
    if stripped["charge"] != 1:
        return finish("unsupported", "native_direct_route_requires_charge_plus_one")
    if len(stripped["atoms"]) > MAX_GRAPH_ATOMS:
        return finish("unsupported", "graph_atom_limit")
    predicted_mz = ionic_mass(counts, 1)
    measured_mz = stripped["precursor_mz"]
    if not math.isfinite(measured_mz) or abs(predicted_mz-measured_mz) > max(.005, measured_mz*20e-6):
        return finish("failed", "precursor_formula_mass_mismatch")
    metadata["formula_mz"] = predicted_mz
    electrons = sum(Chem.GetPeriodicTable().GetAtomicNumber(s)*n for s, n in counts.items()) - 1
    if electrons % 2:
        return finish("unsupported", "odd_electron_formula_outside_closed_shell_domain")
    graph = {"atoms": stripped["atoms"], "bonds": stripped["bonds"]}
    try:
        domains = _local_domains(graph)
    except ValueError as exc:
        return finish("unsupported", str(exc))
    if any(not domain for domain in domains):
        return finish("failed", "empty_local_valence_domain")
    # Most constrained atom first, with canonical graph index as the only tie-break.
    order = sorted(range(len(domains)), key=lambda i: (len(domains[i]), i))
    ordered = [domains[i] for i in order]
    total_h = counts.get("H", 0)
    n = len(ordered)
    suffix_min_h = [0]*(n+1)
    suffix_max_h = [0]*(n+1)
    for i in range(n-1, -1, -1):
        suffix_min_h[i] = suffix_min_h[i+1] + min(h for q, h in ordered[i])
        suffix_max_h[i] = suffix_max_h[i+1] + max(h for q, h in ordered[i])

    class BudgetExceeded(Exception):
        pass

    @lru_cache(maxsize=None)
    def reachable(i, h, q, abs_q):
        metadata["dp_states"] += 1
        if metadata["dp_states"] > max_dp_states:
            raise BudgetExceeded("dp_state_budget")
        if h < suffix_min_h[i] or h > suffix_max_h[i] or abs_q < abs(q) or abs_q < 0:
            return False
        if (abs_q-q) % 2 or abs_q > n-i:
            return False
        if i == n:
            return h == q == abs_q == 0
        return any(reachable(i+1, h-ah, q-aq, abs_q-abs(aq)) for aq, ah in ordered[i])

    assignments = [None]*n
    chemical = {}
    reject = Counter()
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
                failure = _validate_state(mol, graph, counts, 1)
                if failure:
                    reject[failure] += 1
                    return
                # Cleanup may redistribute Q; require the searched charge family.
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
            if reachable(i+1, h-ah, q-aq, abs_q-abs(aq)):
                assignments[i] = (aq, ah)
                visit(i+1, h-ah, q-aq, abs_q-abs(aq))
                if stopped is not None:
                    return

    try:
        for current_family in CHARGE_FAMILIES:
            if reachable(0, total_h, 1, current_family):
                visit(0, total_h, 1, current_family)
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
        return finish("failed", "search_incomplete" if stopped else "no_valid_state_in_declared_domain")
    states = []
    hashes = set()
    for smiles, rank in sorted(chemical.items(), key=lambda item: item[1]):
        mol = Chem.MolFromSmiles(smiles)
        try:
            signature = _feature_hash(mol, "[M]+")
        except (ValueError, RuntimeError) as exc:
            reject["native_featurization_failed"] += 1
            continue
        if signature in hashes:
            continue
        hashes.add(signature)
        states.append({"ionic_smiles": smiles, "feature_hash": signature,
                       "chemistry_rank": list(rank), "total_abs_charge": rank[0],
                       "ionic_formula": counts, "charge": 1, "formula_mz": predicted_mz,
                       "adduct": "[M]+"})
        if len(states) == max_states:
            if len(chemical) > len(states):
                metadata["truncated"] = True
                metadata["feature_selection_capped"] = True
            break
    metadata["feature_unique_checked"] = len(hashes)
    metadata["rejections"] = dict(sorted(reject.items()))
    metadata["retained_states"] = len(states)
    return finish("ok" if states else "failed", None if states else "native_featurization_failed", states)
