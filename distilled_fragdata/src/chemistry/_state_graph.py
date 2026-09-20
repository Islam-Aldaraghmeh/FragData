"""Private fixed-heavy-graph validation and state ranking extracted unchanged."""
from collections import Counter
import math

from rdkit import Chem, rdBase

from src.chemistry.formula import formula_counts, ionic_mass
from src.models.glacier_api import _feature_hash

POLICY_VERSION = "fixed-bonds-closed-shell-min-abs-charge-v1"
MAX_LEAVES = 20_000
MAX_CHEMICAL_STATES = 512
MAX_GRAPH_ATOMS = 100
MAX_DP_STATES = 250_000
MAX_SEARCH_NODES = 250_000
CHARGE_FAMILIES = (1, 3, 5, 7)
# Conservative, explicit closed-shell total-valence templates. Aromatic atoms
# use independent H/Q trials instead: pyrrole and pyridine roles differ.
VALENCES = {
    "B": {-1: (4,), 0: (3,), 1: (2,)},
    "C": {-1: (3,), 0: (4,), 1: (3,)},
    "N": {-1: (2,), 0: (3,), 1: (4,)},
    "O": {-1: (1,), 0: (2,), 1: (3,)},
    "F": {-1: (0,), 0: (1,), 1: (2,)},
    "Si": {-1: (3,), 0: (4,), 1: (3,)},
    "P": {-1: (2, 4, 6), 0: (3, 5), 1: (4,)},
    "S": {-1: (1, 3, 5), 0: (2, 4, 6), 1: (3, 5)},
    "Cl": {-1: (0,), 0: (1,), 1: (2,)},
    "Br": {-1: (0,), 0: (1,), 1: (2,)},
    "I": {-1: (0,), 0: (1,), 1: (2,)},
}



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


def stripped_from_smiles(smiles, ionic_formula, charge, precursor_mz):
    """Return serializable allowed inputs; discard all source local state/stereo.

    Source SMILES only provides the heavy graph. Its H/Q may intentionally differ
    from the ionic formula, as for ordinary neutral MSG SMILES plus [M+H]+.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid source SMILES")
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError("Isotopic labels unsupported")
    mol = Chem.RemoveHs(mol)
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError("Explicit hydrogen nodes unsupported")
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Disconnected graphs unsupported")
    graph = _canonical_graph(_graph_record(mol))
    counts = formula_counts(ionic_formula)
    heavy = Counter(a["element"] for a in graph["atoms"])
    if dict(heavy) != {s: n for s, n in counts.items() if s != "H"}:
        raise ValueError("Heavy elemental formula differs from graph")
    if isinstance(charge, bool) or int(charge) != charge:
        raise ValueError("Charge must be an integer")
    mz = float(precursor_mz)
    if not math.isfinite(mz) or mz <= 0:
        raise ValueError("Invalid precursor m/z")
    return {**graph, "ionic_formula": counts, "charge": int(charge), "precursor_mz": mz}


def _validate_stripped(record):
    if set(record) != {"atoms", "bonds", "ionic_formula", "charge", "precursor_mz"}:
        raise ValueError("Stripped schema contains missing or forbidden fields")
    if not record["atoms"]:
        raise ValueError("Empty graph")
    for atom in record["atoms"]:
        if set(atom) != {"element", "aromatic"}:
            raise ValueError("Atom record contains forbidden local properties")
    for bond in record["bonds"]:
        if set(bond) != {"i", "j", "order", "aromatic"}:
            raise ValueError("Bond record contains forbidden local properties")
    mol = _fresh_graph(record["atoms"], record["bonds"])
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Disconnected graphs unsupported")
    counts = formula_counts(record["ionic_formula"])
    heavy = Counter(a["element"] for a in record["atoms"])
    if dict(heavy) != {s: n for s, n in counts.items() if s != "H"}:
        raise ValueError("Heavy elemental formula differs from graph")
    return mol, counts


def _counts_from_mol(mol):
    counts = Counter(a.GetSymbol() for a in mol.GetAtoms())
    hydrogens = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    if hydrogens:
        counts["H"] += hydrogens
    return dict(sorted(counts.items()))


def _validate_state(mol, graph, counts, charge):
    if _graph_record(mol) != graph:
        return "heavy_graph_or_aromaticity_changed"
    if _counts_from_mol(mol) != counts:
        return "formula_changed"
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != charge:
        return "charge_changed"
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return "radical_unsupported"
    if any(a.GetIsotope() for a in mol.GetAtoms()):
        return "isotope_unsupported"
    return None


def _local_domains(graph):
    valence = [0.] * len(graph["atoms"])
    degree = [0] * len(valence)
    for b in graph["bonds"]:
        for i in (b["i"], b["j"]):
            valence[i] += b["order"]
            degree[i] += 1
    domains = []
    for i, atom in enumerate(graph["atoms"]):
        symbol = atom["element"]
        if symbol not in VALENCES:
            raise ValueError("Unsupported closed-shell element: " + symbol)
        if atom["aromatic"]:
            # Global sanitization determines pyridine/pyrrole electronic roles.
            if symbol == "C":
                # Neutral aromatic carbon cannot redistribute H arbitrarily:
                # degree two has one H; degree three has none. Charged aromatic
                # carbons retain a bounded broader domain for ring validation.
                h_neutral = 3-degree[i]
                options = [(0, h_neutral)] if 0 <= h_neutral <= 1 else []
                options += [(q, h) for q in (-1, 1)
                            for h in range(max(0, 3-degree[i])+1)]
            elif symbol == "N":
                max_h = max(0, 3-degree[i])
                options = [(-1, 0)] + [(q, h) for q in (0, 1) for h in range(max_h+1)]
            else:
                options = [(q, h) for q in (-1, 0, 1) for h in range(max(0, 3-degree[i])+1)]
        else:
            options = []
            for q, totals in VALENCES[symbol].items():
                for total in totals:
                    h = total - valence[i]
                    if 0 <= h <= 4 and h == int(h):
                        options.append((q, int(h)))
        domains.append(tuple(sorted(set(options), key=lambda x: (abs(x[0]), x[0], x[1]))))
    return domains


def _chemistry_rank(mol, smiles):
    # A declared weak site prior, not an energy or proton-affinity calculation.
    positive_penalty = {"N": 0, "O": 1, "S": 2, "P": 2, "C": 3}
    negative_penalty = {"O": 0, "S": 1, "N": 2, "C": 3}
    score = 0
    for atom in mol.GetAtoms():
        q = atom.GetFormalCharge()
        if q > 0:
            score += q * positive_penalty.get(atom.GetSymbol(), 4)
        elif q < 0:
            score -= q * negative_penalty.get(atom.GetSymbol(), 4)
    return (sum(abs(a.GetFormalCharge()) for a in mol.GetAtoms()), score, smiles)

