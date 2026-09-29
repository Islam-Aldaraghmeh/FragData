"""Focused checks for direct radical-cation recovery; no model weights loaded."""
from collections import Counter
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
from recovery_lab.radical import (
    enumerate_radical_states, prepare_radical_candidate, predict_radical_candidate,
)
from src.chemistry.fragment_adapter import FragmentGlacierAdapter, stripped_from_smiles
from src.chemistry.formula import ionic_mass
from src.chemistry._state_graph import (
    _canonical_graph, _counts_from_mol, _fresh_graph, _graph_record,
)
from src.models.glacier_api import GlacierAPI, FEATURE_CONFIG, build_batch


@pytest.fixture(scope="session", autouse=True)
def pinned_api():
    api = GlacierAPI(
        baseline._asset_path(None, "GLACIER_UPSTREAM", "vendor/ms-pred"),
        baseline._asset_path(None, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt"),
    )
    assert api._model is None
    yield api
    assert api._model is None


PILOT = json.loads((Path(__file__).parent / "pilot_candidates.json").read_text())
ACTUAL = [row for row in PILOT["candidates"]
          if row["stripped"]["ionic_formula"] == {"C": 6, "H": 9, "N": 3, "O": 3, "S": 1}]


def stripped(smiles, formula):
    return stripped_from_smiles(smiles, formula, 1, ionic_mass(formula))


@pytest.mark.parametrize("row", ACTUAL, ids=lambda row: row["candidate_id"][-8:])
def test_actual_example_preserves_ionic_chemistry_and_explicit_local_state(row):
    source = row["stripped"]
    before = deepcopy(source)
    assert source["precursor_mz"] == pytest.approx(203.036804)
    assert FragmentGlacierAdapter().prepare(source, 30.)["reason"] == (
        "odd_electron_formula_outside_closed_shell_domain")
    result = enumerate_radical_states(source)
    assert result["status"] == "ok", result
    assert 1 <= len(result["states"]) <= 4
    graph = {key: source[key] for key in ("atoms", "bonds")}
    for state in result["states"]:
        assert "reference_smiles" not in state
        assert state["charge"] == 1
        assert state["ionic_formula"] == source["ionic_formula"]
        assert state["radical_electrons"] == 1
        assert state["adduct"] == "[M]+"
        assert state["formula_mz"] == ionic_mass(source["ionic_formula"], 1)
        assert abs(state["formula_mz"] - source["precursor_mz"]) <= max(
            .005, source["precursor_mz"] * 20e-6)
        mol = Chem.MolFromSmiles(state["ionic_smiles"])
        assert mol is not None
        assert _canonical_graph(_graph_record(mol)) == _canonical_graph(graph)
        assert _counts_from_mol(mol) == source["ionic_formula"]
        assert sum(a.GetTotalNumHs() for a in mol.GetAtoms()) == 9
        assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 1
        assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 1
        assert [a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons()] == (
            state["radical_atom_indices"])
        assert all(a.GetNumRadicalElectrons() in (0, 1) and not a.GetIsotope()
                   and a.GetFormalCharge() in (-1, 0, 1) for a in mol.GetAtoms())
        # Rebuild from the audited assignments, not inferred SMILES hydrogens.
        assignments = state["local_assignments"]
        assert len(assignments) == len(source["atoms"])
        assert sum(item[0] for item in assignments) == 1
        assert sum(item[1] for item in assignments) == 9
        assert sum(item[2] for item in assignments) == 1
        rebuilt = _fresh_graph(source["atoms"], source["bonds"])
        for atom, (charge, hydrogens, radicals) in zip(rebuilt.GetAtoms(), assignments):
            assert charge in (-1, 0, 1) and radicals in (0, 1)
            atom.SetFormalCharge(charge)
            atom.SetNumExplicitHs(hydrogens)
            atom.SetNumRadicalElectrons(radicals)
            atom.SetNoImplicit(True)
        assert all(a.GetNoImplicit() for a in rebuilt.GetAtoms())
        Chem.SanitizeMol(rebuilt)
        assert _graph_record(rebuilt) == graph
        assert _counts_from_mol(rebuilt) == source["ionic_formula"]
        # RDKit may toggle an aromatic N flag, but must not invent implicit H.
        assert all(a.GetNumImplicitHs() == 0 for a in rebuilt.GetAtoms())
        assert sum(a.GetNumRadicalElectrons() for a in rebuilt.GetAtoms()) == 1
        assert Chem.MolToSmiles(rebuilt, isomericSmiles=False) == state["ionic_smiles"]
    assert source == before


def test_native_input_builder_preserves_formula_and_adds_no_proton():
    source = ACTUAL[0]["stripped"]
    result = prepare_radical_candidate(source, 30., allow_radical=True)
    assert result["route"] == "direct_radical_cation"
    assert result["experimental"] is True
    assert len(result["inputs"]) == len(result["states"]) > 0
    for model_input in result["inputs"]:
        assert model_input["adduct"] == "[M]+"
        assert model_input["collision_energy"] == 30.
        records = model_input["records"]
        counts = Counter(atom["symbol"] for atom in records["atoms"])
        counts["H"] = sum(atom["mass_hydrogens"] for atom in records["atoms"])
        assert dict(counts) == source["ionic_formula"]
        batch = build_batch([records], [30.], ["[M]+"], ["Orbitrap"], FEATURE_CONFIG)
        shifts = batch["adduct_mass_shifts"].numpy()[0]
        assert shifts[0] == shifts[1] < 0
        assert batch["masses"].numpy().sum() + shifts[0] == pytest.approx(
            ionic_mass(source["ionic_formula"]), abs=1e-8)


def test_recovery_requires_opt_in_and_default_returns_original_parity_failure():
    candidate = ACTUAL[0]
    ordinary = FragmentGlacierAdapter().prepare(candidate, 30.)
    result = prepare_radical_candidate(candidate, 30.)
    assert result["route"] == "baseline" and result["experimental"] is False
    assert result["reason"] == ordinary["reason"]
    assert result["states"] == result["inputs"] == []


@pytest.mark.parametrize("change", ["charge", "mass", "formula", "local_state"])
def test_invalid_candidates_never_enter_radical_route(change):
    source = deepcopy(ACTUAL[0]["stripped"])
    if change == "charge":
        source["charge"] = 2
    elif change == "mass":
        source["precursor_mz"] += 1.
    elif change == "formula":
        source["ionic_formula"]["C"] += 1
    else:
        source["atoms"][0]["isotope"] = 13
    before = deepcopy(source)
    original = FragmentGlacierAdapter().prepare(source, 30.)
    result = prepare_radical_candidate(source, 30., allow_radical=True)
    assert result["route"] == "baseline"
    assert result["reason"] == original["reason"]
    assert not result["states"] and not result["inputs"]
    assert source == before


def test_aromatic_repair_stays_separate():
    source = next(row["stripped"] for row in PILOT["candidates"]
                  if row["baseline_reason"] == "no_valid_state_in_declared_domain")
    result = prepare_radical_candidate(source, 30., allow_radical=True)
    assert result["route"] == "baseline"
    assert result["reason"] == "no_valid_state_in_declared_domain"
    assert result["states"] == result["inputs"] == []


def test_direct_enumerator_does_not_accept_even_electron_formula():
    source = stripped("CCO", "C2H7O")
    result = enumerate_radical_states(source)
    assert result["status"] != "ok"
    assert not result["states"]


def test_max_states_and_ranking_are_deterministic_and_ignore_ms3():
    source = ACTUAL[0]
    default = prepare_radical_candidate(source, 30., allow_radical=True)
    limited = prepare_radical_candidate(source, 30., allow_radical=True, max_states=1)
    assert default["metadata"]["max_states"] == 4
    assert len(limited["states"]) == 1
    assert limited["states"] == default["states"][:1]
    ranks = [state["chemistry_rank"] for state in default["states"]]
    assert ranks == sorted(ranks)
    assert len({state["feature_hash"] for state in default["states"]}) == len(ranks)
    changed = deepcopy(source)
    changed["metadata"]["ms3_spectra"] = [{"peaks": [[1., 999.]], "collision_energy": 999}]
    changed["ms3_spectra"] = [{"peaks": [[200., 1.]]}]
    repeated = prepare_radical_candidate(changed, 99., allow_radical=True)
    assert repeated["states"] == default["states"]


@pytest.mark.parametrize("cap", ["max_states", "max_leaves", "max_chemical_states", "max_dp_states", "max_search_nodes"])
def test_nonpositive_caps_rejected(cap):
    with pytest.raises(ValueError, match="positive"):
        enumerate_radical_states(ACTUAL[0]["stripped"], **{cap: 0})


@pytest.mark.parametrize("smiles,formula", [("CCO", "C2H7O"), ("NCCO", "C2H8NO")])
def test_supported_baseline_states_inputs_and_predictions_unchanged(smiles, formula):
    import torch

    class DeterministicModel:
        def predict_records(self, **model_input):
            assert model_input["adduct"] == "[M]+"
            return {"spec": torch.tensor([[10., 2.], [10.001, 1.], [20., 3.]])}

    source = stripped(smiles, formula)
    model = DeterministicModel()
    adapter = FragmentGlacierAdapter(model)
    original = adapter.prepare(source, 30.)
    assert original["status"] == "ok" and original["states"]
    result = prepare_radical_candidate(source, 30., model=model, allow_radical=True)
    assert result["route"] == "baseline" and result["experimental"] is False
    assert result["states"] == original["states"]
    assert result["inputs"] == original["inputs"]
    expected = adapter.predict(source, 30.).as_array()
    actual = predict_radical_candidate(model, source, 30., allow_radical=True)
    assert actual["route"] == "baseline"
    assert np.array_equal(actual["spectrum"].as_array(), expected)


def test_unavailable_prediction_does_not_call_model():
    class MustNotRun:
        def predict_records(self, **kwargs):
            raise AssertionError("Unavailable candidate reached model")

    result = predict_radical_candidate(MustNotRun(), ACTUAL[0], 30.)
    assert result["spectrum"] is None
    assert result["route"] == "baseline"
