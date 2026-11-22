"""Root-molecule graphs in MAGMa's own atom order.

MAGMa (``magma_workflow.magma.FragmentEngine``) does not fragment the
input SMILES directly. It round-trips SMILES -> InChI -> tautomer-canonical
molecule -> canonical SMILES, and ``parent_atom_indices`` refer to atoms of
*that* molecule. ``magma_workflow.intensity_dataset._root_graph`` instead
parses the original library SMILES, whose atom order (and sometimes tautomer)
differs for about half of the spectra. Candidate atoms then point at the wrong
root atoms. This module rebuilds the root graph from the engine's SMILES and
verifies, candidate by candidate, that indices and formulas agree.
"""
from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import CACHE_DIR

ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")
HYBRIDIZATIONS = ("SP", "SP2", "SP3")


def _engine_smiles(smiles):
    """Exact SMILES whose atom order MAGMa's parent_atom_indices refer to."""
    from magma_workflow.magma import FragmentEngine

    try:
        return FragmentEngine(smiles, max_tree_depth=0, max_broken_bonds=0).smiles
    except Exception as exc:  # pragma: no cover - reported to caller
        return f"ERROR: {type(exc).__name__}: {exc}"


def engine_smiles_map(smiles_list, cache_dir=CACHE_DIR, num_workers=8):
    """{library SMILES: MAGMa engine SMILES}, cached as JSON."""
    path = Path(cache_dir) / "magma_engine_smiles.json"
    mapping = json.loads(path.read_text()) if path.exists() else {}
    todo = sorted(set(smiles_list) - set(mapping))
    if todo:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for smi, eng in zip(todo, pool.map(_engine_smiles, todo, chunksize=32)):
                mapping[smi] = eng
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping))
    return mapping


def mol_graph(smiles):
    """Atom/bond arrays plus chemistry flags used by the features module."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    n = mol.GetNumAtoms()
    element = np.array([a.GetSymbol() for a in mol.GetAtoms()])
    hyb = [str(a.GetHybridization()) for a in mol.GetAtoms()]
    atom = {
        "element_id": np.array([ELEMENTS.index(e) + 1 if e in ELEMENTS else 0 for e in element], np.int64),
        "hyb_id": np.array([HYBRIDIZATIONS.index(h) + 1 if h in HYBRIDIZATIONS else 0 for h in hyb], np.int64),
        "total_h": np.array([a.GetTotalNumHs() for a in mol.GetAtoms()], np.int64),
        "degree": np.array([a.GetDegree() for a in mol.GetAtoms()], np.int64),
        "formal_charge": np.array([a.GetFormalCharge() for a in mol.GetAtoms()], np.int64),
        "aromatic": np.array([a.GetIsAromatic() for a in mol.GetAtoms()], bool),
        "in_ring": np.array([a.IsInRing() for a in mol.GetAtoms()], bool),
    }
    # Carbonyl carbons (C=O / C=S) and likely protonation sites.
    carbonyl = np.zeros(n, bool)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            a, b = bond.GetBeginAtom(), bond.GetEndAtom()
            for x, y in ((a, b), (b, a)):
                if x.GetSymbol() == "C" and y.GetSymbol() in ("O", "S"):
                    carbonyl[x.GetIdx()] = True
    basic_n = np.zeros(n, bool)   # sp3 amine N not attached to carbonyl/aromatic/S(=O)
    arom_n = np.zeros(n, bool)    # pyridine-like aromatic N (no H)
    amide_n = np.zeros(n, bool)
    for a in mol.GetAtoms():
        if a.GetSymbol() != "N":
            continue
        i = a.GetIdx()
        neighbors = list(a.GetNeighbors())
        if a.GetIsAromatic():
            arom_n[i] = a.GetTotalNumHs() == 0 and a.GetDegree() == 2
        elif any(carbonyl[nb.GetIdx()] or nb.GetSymbol() == "S" for nb in neighbors):
            amide_n[i] = True
        elif str(a.GetHybridization()) == "SP3" and not any(nb.GetIsAromatic() for nb in neighbors):
            basic_n[i] = True
    atom.update(carbonyl=carbonyl, basic_n=basic_n, arom_n=arom_n, amide_n=amide_n)

    bonds = list(mol.GetBonds())
    bond = {
        "i": np.array([b.GetBeginAtomIdx() for b in bonds], np.int64),
        "j": np.array([b.GetEndAtomIdx() for b in bonds], np.int64),
        "order": np.array([b.GetBondTypeAsDouble() for b in bonds], np.float32),
        "aromatic": np.array([b.GetIsAromatic() for b in bonds], bool),
        "conjugated": np.array([b.GetIsConjugated() for b in bonds], bool),
        "in_ring": np.array([b.IsInRing() for b in bonds], bool),
    }
    return {"element": element, "atom": atom, "bond": bond, "n_atoms": n}


def verify_alignment(graph, candidates):
    """Count candidates whose atoms disagree with their formula or are disconnected.

    candidates: iterable of (ionic_formula dict, parent_atom_indices array).
    A correct mapping gives the same heavy-atom composition and one connected
    induced subgraph for every candidate.
    """
    element = graph["element"]
    adjacency = [[] for _ in range(graph["n_atoms"])]
    for i, j in zip(graph["bond"]["i"], graph["bond"]["j"]):
        adjacency[i].append(j)
        adjacency[j].append(i)
    bad = 0
    for formula, idx in candidates:
        heavy = {k: v for k, v in formula.items() if k != "H" and v}
        if dict(Counter(element[idx].tolist())) != heavy:
            bad += 1
            continue
        keep = set(int(i) for i in idx)
        stack, seen = [int(idx[0])], {int(idx[0])}
        while stack:
            for nb in adjacency[stack.pop()]:
                if nb in keep and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        bad += len(seen) != len(keep)
    return bad
