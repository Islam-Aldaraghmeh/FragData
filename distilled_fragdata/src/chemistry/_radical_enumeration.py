"""Opt-in, exact-graph one-radical cations; production enumeration is untouched."""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import math
import time

from rdkit import Chem, rdBase

from src.chemistry.formula import ionic_mass
from src.chemistry._state_graph import (
    CHARGE_FAMILIES, VALENCES, MAX_GRAPH_ATOMS, MAX_LEAVES,
    MAX_CHEMICAL_STATES, MAX_DP_STATES, MAX_SEARCH_NODES,
    _validate_stripped, _local_domains, _graph_record, _canonical_graph,
    _counts_from_mol, _chemistry_rank, _feature_hash,
)

ROUTE = "direct_radical_cation"
PARITY_FAILURE = "odd_electron_formula_outside_closed_shell_domain"


def _radical_domains(graph):
    """Keep r=0 domains; add bounded r=1 valence hypotheses for sanitization.

    Nonaromatic radicals use one less bonding valence than the existing
    closed-shell template at the same formal charge. Aromatic electronic roles
    need bounded H/Q trials; exact aromatic labels must survive sanitization.
    """
    domains = []
    for i, (atom, ordinary) in enumerate(zip(graph["atoms"], _local_domains(graph))):
        edges = [b for b in graph["bonds"] if i in (b["i"], b["j"])]
        options = {(q, h, 0) for q, h in ordinary}
        if atom["aromatic"]:
            options.update((q, h, 1) for q in (-1, 0, 1)
                           for h in range(max(0, 3 - len(edges)) + 1))
        else:
            valence = sum(b["order"] for b in edges)
            for q, totals in VALENCES[atom["element"]].items():
                for total in totals:
                    h = total - valence - 1
                    if h == int(h) and 0 <= h <= 4:
                        options.add((q, int(h), 1))
        domains.append(tuple(sorted(options, key=lambda x: (x[2], abs(x[0]), x[0], x[1]))))
    return domains


def _validate_radical(mol, graph, counts, measured_mz, *, canonical=False):
    represented = _graph_record(mol)
    if canonical:
        represented, graph = _canonical_graph(represented), _canonical_graph(graph)
    if represented != graph:
        return "heavy_graph_or_aromaticity_changed"
    if _counts_from_mol(mol) != counts:
        return "formula_changed"
    if sum(a.GetTotalNumHs() for a in mol.GetAtoms()) != counts.get("H", 0):
        return "hydrogen_count_changed"
    charges = [a.GetFormalCharge() for a in mol.GetAtoms()]
    if sum(charges) != 1 or any(q not in (-1, 0, 1) for q in charges):
        return "charge_changed"
    radicals = [a.GetNumRadicalElectrons() for a in mol.GetAtoms()]
    if sum(radicals) != 1 or any(r not in (0, 1) for r in radicals):
        return "radical_count_changed"
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return "isotope_unsupported"
    if abs(ionic_mass(_counts_from_mol(mol), 1) - measured_mz) > max(.005, measured_mz * 20e-6):
        return "precursor_formula_mass_mismatch"
    return None


def enumerate_radical_states(stripped, max_states=4, *, max_leaves=MAX_LEAVES,
                             max_chemical_states=MAX_CHEMICAL_STATES,
                             max_dp_states=MAX_DP_STATES, max_search_nodes=MAX_SEARCH_NODES):
    """Small extension of the baseline H/Q/|Q| search with one remaining R."""
    caps = (max_states, max_leaves, max_chemical_states, max_dp_states, max_search_nodes)
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in caps):
        raise ValueError("Enumeration caps must be positive integers")
    started = time.monotonic()
    metadata = dict(policy="experimental-exact-graph-one-radical-v1",
                    charge_families=list(CHARGE_FAMILIES), local_charge_domain=[-1, 0, 1],
                    local_radical_domain=[0, 1], required_total_radicals=1,
                    max_states=max_states, max_leaves=max_leaves,
                    max_chemical_states=max_chemical_states, max_dp_states=max_dp_states,
                    max_search_nodes=max_search_nodes, leaves_checked=0, dp_states=0,
                    search_nodes=0, minimum_abs_charge=None, accuracy_validated=False)
    rejections, chemical = Counter(), {}
    stopped = None

    def finish(reason=None, states=None, status="failed"):
        metadata.update(elapsed_seconds=round(time.monotonic() - started, 6),
                        rejections=dict(rejections), chemical_unique_states=len(chemical),
                        truncated=bool(stopped or metadata.get("feature_selection_capped")),
                        stop_reason=stopped or reason or "minimum_valid_family_exhausted")
        return dict(status=status, reason=reason, states=states or [], metadata=metadata)

    if len(stripped.get("atoms", [])) > MAX_GRAPH_ATOMS:
        return finish("graph_atom_limit", status="unsupported")
    try:
        base, counts = _validate_stripped(stripped)
        mz = float(stripped["precursor_mz"])
    except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as exc:
        return finish("invalid_stripped_record: " + str(exc))
    if isinstance(stripped["charge"], bool) or stripped["charge"] != 1:
        return finish("radical_route_requires_charge_plus_one", status="unsupported")
    if not math.isfinite(mz) or mz <= 0 or abs(ionic_mass(counts, 1) - mz) > max(.005, mz * 20e-6):
        return finish("precursor_formula_mass_mismatch")
    electrons = sum(Chem.GetPeriodicTable().GetAtomicNumber(s) * n for s, n in counts.items()) - 1
    if electrons % 2 != 1:
        return finish("not_an_odd_electron_ion_formula", status="unsupported")
    graph = {k: stripped[k] for k in ("atoms", "bonds")}
    try:
        domains = _radical_domains(graph)
    except ValueError as exc:
        return finish(str(exc), status="unsupported")
    if any(not d for d in domains):
        return finish("empty_local_valence_domain")
    order = sorted(range(len(domains)), key=lambda i: (len(domains[i]), i))
    ordered = [domains[i] for i in order]
    n, assignments = len(order), [None] * len(order)
    min_h, max_h = [0] * (n + 1), [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        min_h[i] = min_h[i + 1] + min(h for q, h, r in ordered[i])
        max_h[i] = max_h[i + 1] + max(h for q, h, r in ordered[i])

    class BudgetExceeded(Exception):
        pass

    @lru_cache(maxsize=None)
    def reachable(i, h, q, abs_q, r):
        metadata["dp_states"] += 1
        if metadata["dp_states"] > max_dp_states:
            raise BudgetExceeded("dp_state_budget")
        if (not min_h[i] <= h <= max_h[i] or not 0 <= r <= min(1, n - i)
                or abs_q < abs(q) or abs_q > n - i or (abs_q - q) % 2):
            return False
        if i == n:
            return h == q == abs_q == r == 0
        return any(reachable(i + 1, h - ah, q - aq, abs_q - abs(aq), r - ar)
                   for aq, ah, ar in ordered[i])

    def visit(i, h, q, abs_q, r):
        metadata["search_nodes"] += 1
        if metadata["search_nodes"] > max_search_nodes:
            raise BudgetExceeded("search_node_budget")
        if i < n:
            for aq, ah, ar in ordered[i]:
                if reachable(i + 1, h - ah, q - aq, abs_q - abs(aq), r - ar):
                    assignments[order[i]] = (aq, ah, ar)
                    visit(i + 1, h - ah, q - aq, abs_q - abs(aq), r - ar)
            return
        if metadata["leaves_checked"] >= max_leaves:
            raise BudgetExceeded("leaf_budget")
        metadata["leaves_checked"] += 1
        mol = Chem.Mol(base)
        for atom, (aq, ah, ar) in zip(mol.GetAtoms(), assignments):
            atom.SetFormalCharge(aq)
            atom.SetNumExplicitHs(ah)
            atom.SetNoImplicit(True)
            atom.SetNumRadicalElectrons(ar)
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(mol)
            failure = _validate_radical(mol, graph, counts, mz)
            if failure:
                rejections[failure] += 1
                return
            if [a.GetNumRadicalElectrons() for a in mol.GetAtoms()] != [v[2] for v in assignments]:
                rejections["radical_site_changed"] += 1
                return
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
            rank = _chemistry_rank(mol, smiles)
            if rank[0] != family:
                rejections["charge_family_changed"] += 1
                return
            local = [[a.GetFormalCharge(), a.GetTotalNumHs(), a.GetNumRadicalElectrons()]
                     for a in mol.GetAtoms()]
            chemical.setdefault(smiles, (rank, local))
        except (ValueError, RuntimeError):
            rejections["sanitization_failed"] += 1
            return
        if len(chemical) >= max_chemical_states:
            raise BudgetExceeded("chemical_cap")

    try:
        for family in CHARGE_FAMILIES:
            if reachable(0, counts.get("H", 0), 1, family, 1):
                visit(0, counts.get("H", 0), 1, family, 1)
            if chemical:
                break
    except BudgetExceeded as exc:
        stopped = str(exc)
    if not chemical:
        return finish("search_incomplete" if stopped else "no_valid_radical_state_in_declared_domain")
    metadata["minimum_abs_charge"] = family
    states, seen = [], set()
    for smiles, (rank, local) in sorted(chemical.items(), key=lambda item: item[1][0]):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or _validate_radical(mol, graph, counts, mz, canonical=True):
                rejections["smiles_roundtrip_changed_state"] += 1
                continue
            signature = _feature_hash(mol, "[M]+")
        except (ValueError, RuntimeError):
            rejections["native_featurization_failed"] += 1
            continue
        if signature in seen:
            continue
        seen.add(signature)
        states.append(dict(ionic_smiles=smiles, feature_hash=signature,
                           chemistry_rank=list(rank), total_abs_charge=rank[0],
                           ionic_formula=counts, charge=1, formula_mz=ionic_mass(counts, 1),
                           radical_electrons=1,
                           # Indices refer to parsed ionic_smiles, not original graph.
                           radical_atom_indices=[a.GetIdx() for a in mol.GetAtoms()
                                                 if a.GetNumRadicalElectrons()],
                           # Triples are in the original candidate's atom order.
                           local_assignments=local, adduct="[M]+"))
        if len(states) == max_states:
            metadata["feature_selection_capped"] = len(chemical) > len(states)
            break
    metadata["retained_states"] = len(states)
    return finish(None if states else "native_featurization_failed", states,
                  "ok" if states else "failed")


