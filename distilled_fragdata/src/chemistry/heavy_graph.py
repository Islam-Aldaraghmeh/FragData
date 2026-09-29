"""RDKit heavy-graph records and canonicalization, independent of prediction."""
from rdkit import Chem


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


