"""Elemental formulas, exact masses and canonical heavy-atom graphs.

Formula masses follow the pinned ms-pred convention (RDKit most-common-isotope
masses); graph records are {"atoms": [...], "bonds": [...]} dicts.
"""
import re

from rdkit import Chem

# Pinned ms-pred common/chem_utils.py uses RDKit most-common-isotope masses.
# These are the values in the working prototype (RDKit 2025.03.6).
ELEMENT_TO_MASS = {
    "C": 12.0, "N": 14.003074, "P": 30.97376163, "O": 15.99491462,
    "S": 31.972071, "Si": 27.97692653, "I": 126.904473,
    "H": 1.007825032, "Cl": 34.96885268, "F": 18.99840322,
    "Br": 78.9183371, "B": 11.0093054, "Se": 79.9165213,
    "Fe": 55.9349375, "Co": 58.933195, "As": 74.9215965,
    "Na": 22.98976928, "K": 38.96370668,
}
ELECTRON_MASS = 0.00054858


def formula_counts(formula):
    """Parse an elemental formula (optionally with a terminal charge suffix)."""
    if isinstance(formula, dict):
        result = {}
        for symbol, count in formula.items():
            if isinstance(count, bool) or int(count) != count or count < 0:
                raise ValueError("Invalid elemental count")
            if count:
                result[symbol] = int(count)
    elif isinstance(formula, str):
        clean = re.sub(r"[+-]\d*$", "", formula.strip())
        pieces = re.findall(r"([A-Z][a-z]?)(\d*)", clean)
        if not pieces or "".join(s + n for s, n in pieces) != clean:
            raise ValueError("Unsupported formula notation")
        result = {}
        for symbol, count in pieces:
            if symbol in result:
                raise ValueError("Repeated formula element")
            n = int(count or "1")
            if n <= 0:
                raise ValueError("Invalid elemental count")
            result[symbol] = n
    else:
        raise ValueError("Formula must be text or elemental counts")
    if any(s not in ELEMENT_TO_MASS for s in result):
        raise ValueError("Unsupported formula element")
    return dict(sorted(result.items()))


def ionic_mass(formula, charge=1):
    counts = formula_counts(formula)
    if charge == 0:
        raise ValueError("Ionic m/z requires nonzero charge")
    return (sum(ELEMENT_TO_MASS[s] * n for s, n in counts.items())
            - charge * ELECTRON_MASS) / abs(charge)


def exact_mass(formula):
    """Neutral atomic mass sum; no proton or electron adjustment is applied."""
    return sum(ELEMENT_TO_MASS[s] * n for s, n in formula_counts(formula).items())


def _fresh_graph(atoms, bonds):
    rw = Chem.RWMol()
    for record in atoms:
        atom = Chem.Atom(record["element"])
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(bool(record["aromatic"]))
        rw.AddAtom(atom)
    bond_types = {1.: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC,
                  2.: Chem.BondType.DOUBLE, 3.: Chem.BondType.TRIPLE}
    for record in bonds:
        order = float(record["order"])
        if order not in bond_types:
            raise ValueError("Unsupported bond order")
        i, j = record["i"], record["j"]
        rw.AddBond(i, j, bond_types[order])
        rw.GetBondBetweenAtoms(i, j).SetIsAromatic(bool(record["aromatic"]))
    mol = rw.GetMol()
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    return mol


def _graph_record(mol):
    atoms = [{"element": a.GetSymbol(), "aromatic": bool(a.GetIsAromatic())}
             for a in mol.GetAtoms()]
    bonds = [{"i": min(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
              "j": max(b.GetBeginAtomIdx(), b.GetEndAtomIdx()),
              "order": b.GetBondTypeAsDouble(), "aromatic": bool(b.GetIsAromatic())}
             for b in mol.GetBonds()]
    return {"atoms": atoms, "bonds": sorted(bonds, key=lambda b: (b["i"], b["j"]))}


def _canonical_graph(graph):
    # Fresh atoms are essential: CanonicalRankAtoms otherwise sees original H/Q.
    fresh = _fresh_graph(graph["atoms"], graph["bonds"])
    ranks = Chem.CanonicalRankAtoms(fresh, breakTies=True, includeChirality=False,
                                   includeIsotopes=False, includeAtomMaps=False)
    order = sorted(range(len(ranks)), key=lambda i: ranks[i])
    return _graph_record(Chem.RenumberAtoms(fresh, order))


