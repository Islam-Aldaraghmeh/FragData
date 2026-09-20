"""Frozen and live comparisons with the exact working ionic-state prototype."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest
from rdkit import Chem

from src.chemistry import fragment_adapter as adapter
from src.chemistry.formula import ELEMENT_TO_MASS, formula_counts, ionic_mass, exact_mass, mass_matches
from src.models.glacier_api import build_batch, tensor_fingerprint

FIXTURE = json.loads((Path(__file__).parent / "fixtures/fragment_states.json").read_text())


def without_timing(result):
    result = deepcopy(result)
    result["metadata"].pop("elapsed_seconds")
    return result


def test_retained_states_and_model_inputs_match_frozen_and_prototype(glacier, prototype_src):
    old = importlib.import_module("microstates") if prototype_src else None
    old_direct = importlib.import_module("direct_input") if prototype_src else None
    for case in FIXTURE["cases"]:
        if "input_hashes_at_ce20" not in case:
            continue
        prepared = adapter.FragmentGlacierAdapter(**case["settings"]).prepare(case["stripped"], 20.)
        inputs = prepared.pop("inputs")
        assert without_timing(prepared) == case["enumeration"]
        hashes = [tensor_fingerprint(build_batch([item["records"]], [20.], [item["adduct"]],
                                                [item["instrument"]], glacier.model))
                  for item in inputs]
        assert hashes == case["input_hashes_at_ce20"]
        if old:
            assert without_timing(old.enumerate_microstates(case["stripped"], **case["settings"])) == case["enumeration"]
            for state, item in zip(prepared["states"], inputs):
                assert item["records"] == old_direct.records_from_smiles(state["ionic_smiles"])


def test_caps_and_failure_statuses_match_prototype(glacier, prototype_src):
    old = importlib.import_module("microstates") if prototype_src else None
    for case in FIXTURE["cases"]:
        if "input_hashes_at_ce20" in case:
            continue
        result = adapter.enumerate_microstates(case["stripped"], **case["settings"])
        assert without_timing(result) == case["enumeration"]
        if old:
            assert without_timing(result) == without_timing(old.enumerate_microstates(case["stripped"], **case["settings"]))
    for change, reason in [("charge", "native_direct_route_requires_charge_plus_one"),
                           ("mass", "precursor_formula_mass_mismatch"),
                           ("local_state", "invalid_stripped_record")]:
        record = deepcopy(FIXTURE["cases"][0]["stripped"])
        if change == "charge":
            record["charge"] = 2
        elif change == "mass":
            record["precursor_mz"] += 1
        else:
            record["atoms"][0]["formal_charge"] = 1
        result = adapter.enumerate_microstates(record, max_states=4)
        assert result["reason"].startswith(reason)
        if old:
            assert without_timing(result) == without_timing(old.enumerate_microstates(record, max_states=4))


def test_graph_formula_charge_and_target_independent_default(glacier):
    from src.data.candidate_bags import FragmentCandidate

    case = FIXTURE["cases"][6]
    record = case["stripped"]
    candidate = FragmentCandidate("fixture", {k: record[k] for k in ["atoms", "bonds"]},
                                  record["ionic_formula"], record["charge"], record["precursor_mz"])
    prepared = adapter.FragmentGlacierAdapter().prepare(candidate, 20.)
    assert prepared["metadata"]["max_states"] == 4
    assert len(prepared["states"]) == 4
    for state in prepared["states"]:
        mol = Chem.MolFromSmiles(state["ionic_smiles"])
        assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == record["charge"]
        assert all(a.GetNumRadicalElectrons() == 0 for a in mol.GetAtoms())
        assert adapter.stripped_from_smiles(state["ionic_smiles"], case["formula"], 1,
                                            record["precursor_mz"]) == record
        assert state["formula_mz"] == ionic_mass(case["formula"])
    expected = FIXTURE["cases"][0]["stripped"]
    for smiles in ["NCCO", "OCCN", "[NH3+]CCO", "NCC[OH2+]"]:
        assert adapter.stripped_from_smiles(smiles, "C2H8NO", 1, expected["precursor_mz"]) == expected


def test_optional_projection_and_exact_mass_conventions(glacier, prototype_src):
    old = importlib.import_module("microstates") if prototype_src else None
    for smiles, expected in FIXTURE["projections"].items():
        assert adapter.project_neutral_references(smiles) == expected
        if old:
            assert old.project_neutral_references(smiles) == expected
    assert formula_counts("C2H8NO+") == {"C": 2, "H": 8, "N": 1, "O": 1}
    assert ionic_mass("C2H8NO") == exact_mass("C2H8NO") - .00054858
    assert mass_matches("C2H8NO", 1, ionic_mass("C2H8NO") + .004)
    assert not mass_matches("C2H8NO", 1, ionic_mass("C2H8NO") + .006)
    if old:
        assert ELEMENT_TO_MASS == dict(old.common.ELEMENT_TO_MASS)
        for formula in ["C2H8NO", "NaCl", {"C": 2, "H": 8, "N": 1, "O": 1}]:
            assert formula_counts(formula) == old.formula_counts(formula)
            for charge in [-2, -1, 1, 2]:
                assert ionic_mass(formula, charge) == old.ionic_mass(formula, charge)
    for bad in ["CH3CH3", "(CH3)2", "C0", "Li", {"C": True}]:
        with pytest.raises(ValueError):
            formula_counts(bad)
