"""Chemistry and input-semantics checks for the isolated recovery experiment.

These tests perform featurization only; no model forward pass or training runs.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from recovery_lab import baseline_snapshot as baseline
from recovery_lab.aromatic_recovery import (
    prepare_aromatic_recovery, _validate_aromatic_state,
)
from recovery_lab.neutral_reference import prepare_neutral_reference
from src.chemistry.fragment_adapter import FragmentGlacierAdapter, stripped_from_smiles
from src.chemistry.formula import ionic_mass
from src.chemistry._state_graph import _canonical_graph, _graph_record, _counts_from_mol
from src.models.glacier_api import GlacierAPI, FEATURE_CONFIG, build_batch


@pytest.fixture(scope="session", autouse=True)
def pinned_api():
    """Check checkpoint/source identity and select upstream without loading weights."""
    api = GlacierAPI(
        baseline._asset_path(None, "GLACIER_UPSTREAM", "vendor/ms-pred"),
        baseline._asset_path(None, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt"),
    )
    assert api._model is None
    yield api
    assert api._model is None


@pytest.fixture(scope="module")
def pilot():
    return json.loads((Path(__file__).parent / "pilot_candidates.json").read_text())


def stripped(smiles, formula):
    return stripped_from_smiles(smiles, formula, 1, ionic_mass(formula))


def assert_native_mass_semantics(prepared, expected_formula, expected_mz):
    """The native [M]+ category subtracts an electron and adds no hydrogen."""
    for item in prepared["inputs"]:
        assert item["adduct"] == "[M]+"
        records = item["records"]
        counts = {}
        for atom in records["atoms"]:
            counts[atom["symbol"]] = counts.get(atom["symbol"], 0) + 1
            counts["H"] = counts.get("H", 0) + atom["mass_hydrogens"]
        counts = {element: count for element, count in counts.items() if count}
        assert counts == expected_formula
        batch = build_batch([records], [item["collision_energy"]], [item["adduct"]],
                            [item["instrument"]], FEATURE_CONFIG)
        shifts = batch["adduct_mass_shifts"].numpy()[0]
        assert shifts[0] < 0 and shifts[0] == shifts[1]
        actual_mz = batch["masses"].numpy().sum() + shifts[0]
        assert actual_mz == pytest.approx(expected_mz, abs=1e-8)
        for value in (batch["masses"], batch["root_form_vecs"],
                      batch["graphormer_input"]["x"]):
            assert np.isfinite(value.numpy()).all()


@pytest.mark.parametrize("smiles,formula", [
    ("CCO", "C2H6O"), ("c1ccccc1", "C6H6"), ("NCC(=O)O", "C2H5NO2"),
])
def test_neutral_references_preserve_exact_graph_formula_and_mass(smiles, formula):
    candidate = stripped(smiles, formula)
    before = deepcopy(candidate)
    ordinary = FragmentGlacierAdapter().prepare(candidate, 20.)
    assert ordinary["reason"] == "odd_electron_formula_outside_closed_shell_domain"
    result = prepare_neutral_reference(candidate, 20.)
    assert result["status"] == "ok"
    assert result["experimental"] is True
    assert result["metadata"]["accuracy_validated"] is False
    assert result["metadata"]["ion_electron_parity"] == "odd"
    original_graph = _canonical_graph({k: candidate[k] for k in ("atoms", "bonds")})
    for state in result["states"]:
        mol = Chem.MolFromSmiles(state["reference_smiles"])
        assert _canonical_graph(_graph_record(mol)) == original_graph
        assert _counts_from_mol(mol) == candidate["ionic_formula"]
        assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 0
        assert not any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms())
        assert state["charge"] == 1 and state["reference_charge"] == 0
    assert_native_mass_semantics(result, candidate["ionic_formula"], ionic_mass(formula))
    assert candidate == before


def test_neutral_route_rejects_even_electron_ion_instead_of_changing_hydrogens():
    candidate = stripped("c1ccccc1", "C6H5")
    result = prepare_neutral_reference(candidate, 20.)
    assert result["reason"] == "not_an_odd_electron_ion_formula"
    assert result["states"] == result["inputs"] == []


def test_exact_failed_pilot_candidate_recovers_only_aromatic_representation(pilot):
    candidate = next(c for c in pilot["candidates"]
                     if c["candidate_id"] == "candidate_c1c3f8c0a3b30523b61738af55df6fee44a5f04aba93b77ccaf1b67d7e415d0e")
    before = deepcopy(candidate)
    source = candidate["stripped"]
    assert FragmentGlacierAdapter().prepare(candidate, 30.)["reason"] == "no_valid_state_in_declared_domain"
    result = prepare_aromatic_recovery(candidate, 30.)
    assert result["status"] == "ok"
    assert result["experimental"] is True
    assert result["audit"]["performance_validated"] is False
    assert result["audit"]["exact_original_graph_preserved"] is False
    assert [s["ionic_smiles"] for s in result["states"]] == ["[C+]1=CC=CC=C1"]
    for state in result["states"]:
        represented = state["represented_graph_original_atom_order"]
        assert [a["element"] for a in represented["atoms"]] == [a["element"] for a in source["atoms"]]
        old_edges = {(b["i"], b["j"]) for b in source["bonds"]}
        assert {(b["i"], b["j"]) for b in represented["bonds"]} == old_edges
        assert all(b["order"] in (1., 2.) and not b["aromatic"] for b in represented["bonds"])
        mol = Chem.MolFromSmiles(state["ionic_smiles"])
        assert _counts_from_mol(mol) == source["ionic_formula"]
        assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 1
        assert not any(a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms())
    assert_native_mass_semantics(result, source["ionic_formula"], ionic_mass(source["ionic_formula"]))
    assert candidate == before


@pytest.mark.parametrize("prepare", [prepare_neutral_reference, prepare_aromatic_recovery])
@pytest.mark.parametrize("invalid", ["charge", "mass", "disconnected", "formula", "nonfinite_ce"])
def test_recovery_rejects_invalid_candidate_semantics_without_mutation(prepare, invalid):
    formula = "C6H6" if prepare is prepare_neutral_reference else "C6H5"
    candidate = stripped("c1ccccc1", formula)
    if invalid == "charge":
        candidate["charge"] = 2
    elif invalid == "mass":
        candidate["precursor_mz"] += 1.
    elif invalid == "disconnected":
        # Remove both edges incident on one vertex, preserving its atom/formula.
        candidate["bonds"] = [b for b in candidate["bonds"] if 0 not in (b["i"], b["j"])]
    elif invalid == "formula":
        candidate["ionic_formula"]["N"] = 1
    before = deepcopy(candidate)
    if invalid == "nonfinite_ce":
        with pytest.raises(ValueError, match="finite"):
            prepare(candidate, float("nan"))
    else:
        result = prepare(candidate, 20.)
        assert result["status"] != "ok"
        assert result["states"] == result["inputs"] == []
    assert candidate == before


def test_aromatic_route_has_no_nonaromatic_or_odd_electron_shortcut():
    result = prepare_aromatic_recovery(stripped("CCO", "C2H7O"), 20.)
    assert result["reason"] == "no_original_aromatic_bonds"
    result = prepare_aromatic_recovery(stripped("c1ccccc1", "C6H6"), 20.)
    assert result["reason"] == "odd_electron_formula_outside_closed_shell_domain"


@pytest.mark.parametrize("change,expected", [
    ("element", "element_order_changed"),
    ("edge", "heavy_connectivity_changed"),
    ("nonaromatic_order", "nonaromatic_bond_changed"),
    ("hydrogens", "formula_changed"),
    ("charge", "charge_changed"),
])
def test_aromatic_validator_rejects_changes_beyond_aromatic_representation(change, expected):
    mol = Chem.MolFromSmiles("[C+]1=CC=CC=C1")
    original = _graph_record(mol)
    for atom in original["atoms"]:
        atom["aromatic"] = True
    for bond in original["bonds"]:
        bond.update(order=1.5, aromatic=True)
    formula = {"C": 6, "H": 5}
    charge = 1
    assert _validate_aromatic_state(mol, original, formula, charge) is None
    if change == "element":
        original["atoms"][0]["element"] = "N"
    elif change == "edge":
        original["bonds"].pop()
    elif change == "nonaromatic_order":
        original["bonds"][0].update(order=3., aromatic=False)
    elif change == "hydrogens":
        formula["H"] += 1
    else:
        charge = 0
    assert _validate_aromatic_state(mol, original, formula, charge) == expected


def test_aromatic_validator_rejects_new_aromatic_atoms_and_radicals():
    mol = Chem.MolFromSmiles("c1ccccc1")
    graph = _graph_record(mol)
    graph["atoms"][0]["aromatic"] = False
    assert _validate_aromatic_state(mol, graph, {"C": 6, "H": 6}, 0) == "new_aromatic_atom"
    radical = Chem.MolFromSmiles("[c]1ccccc1")
    assert _validate_aromatic_state(radical, _graph_record(radical), {"C": 6, "H": 5}, 0) == "radical_unsupported"
