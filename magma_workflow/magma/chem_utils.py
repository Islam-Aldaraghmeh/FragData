# Subset of ms-pred (https://github.com/samgoldman97/ms-pred), commit
# 67ba0f202305d4bed3134991dd4562fe1edb1763, src/ms_pred/common/chem_utils.py:
# only the names used by fragmentation.py and by magma_workflow.candidates
# (ion2mass). Values and code are unchanged.
#
# MIT License, Copyright (c) 2023 Samuel Goldman (full text in fragmentation.py).
"""Element tables, formula vectors and InChI canonicalization for MAGMa."""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import rdkit.sping
from platformdirs import user_cache_dir
from rdkit import Chem

try:
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer, TautomerTransform
    _RD_TAUTOMER_CANONICALIZER = 'v1'
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator  # newer rdkit
    _RD_TAUTOMER_CANONICALIZER = 'v2'

P_TBL = Chem.GetPeriodicTable()

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C", "N", "P", "O", "S", "Si", "I", "H", "Cl", "F", "Br", "B", "Se", "Fe",
    "Co", "As", "Na", "K",
]

# Monoisotopic masses
VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))

# Map ion to adduct mass, don't use electron
ion2mass = {
    # positive mode
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H-H2O]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M+NH4]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M+H-H3N]+": -ELEMENT_TO_MASS["N"] * 1 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M+H-CH2O2]+": -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M+H-2H2O]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    # negative mode
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M+Cl]-": ELEMENT_TO_MASS["Cl"] + ELECTRON_MASS,
    "[M+CHO2]-": ELEMENT_TO_MASS["C"] + ELEMENT_TO_MASS["O"] * 2 + ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M-H2O-H]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-H2O]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-CO2]-": -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
}

# add equivalent keys: [M]1+ == [M]+, [NH3] == [H3N]
_ori_ions = list(ion2mass.keys())
for ion in _ori_ions:
    adduct, charge = ion.split(']')
    if not charge[0].isnumeric():
        ion2mass[adduct + ']1' + charge] = ion2mass[ion]

_ori_ions = list(ion2mass.keys())
for ion in _ori_ions:
    adduct, charge = ion.split(']')
    if 'H3N' in adduct:
        ion2mass[ion.replace('H3N', 'NH3')] = ion2mass[ion]


@lru_cache(maxsize=1)
def _get_tautomer_canonicalizer():
    from rdkit.Chem.MolStandardize.tautomer import (
        TautomerCanonicalizer, TautomerTransform
    )

    transforms = (
        TautomerTransform(
            '1,3 heteroatom H shift',
            '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'
        ),
        TautomerTransform(
            '1,3 (thio)keto/enol r',
            '[O,S,Se,Te;X2!H0]-[C]=[C]'
        ),
    )
    return TautomerCanonicalizer(transforms=transforms)


@lru_cache(maxsize=1)
def _get_tautomer_enumerator():
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator, CleanupParameters
    tautomer_transform_path = Path(user_cache_dir(f"ms-pred/tautomer_transform_rdkit{rdkit.sping.version}.txt"))
    if not tautomer_transform_path.exists():
        tautomer_transform_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tautomer_transform_path, 'w') as f:
            f.write(
                "// Name SMARTS Bonds Charges\n"
                "1,3 hetero atom H shift\t[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]\n"
                "1,3 (thio)keto/enol r\t[O,S,Se,Te;X2!H0]-[C]=[C]\n"
            )

    p = CleanupParameters()
    p.tautomerTransformsFile = tautomer_transform_path

    return TautomerEnumerator(p)


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec


def vec_to_formula(form_vec):
    """vec_to_formula."""
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def canonical_mol_from_inchi(inchi):
    """Canonicalize mol after Chem.MolFromInchi
    Note that this function may be 50 times slower than Chem.MolFromInchi"""
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == "v1":
        canon = _get_tautomer_canonicalizer()
        mol = canon.canonicalize(mol)
    else:
        enum = _get_tautomer_enumerator()
        mol = enum.Canonicalize(mol)
    return mol
