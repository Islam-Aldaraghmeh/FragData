"""Independent model recovery switches; featurization only, no weight loading."""
import ast
from copy import deepcopy
import inspect
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from rdkit import Chem

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "distilled_fragdata"))

from src.chemistry.fragment_adapter import FragmentGlacierAdapter
from src.chemistry.formula import ionic_mass
from src.chemistry._state_graph import _counts_from_mol
from src.models.glacier_api import GlacierAPI


FLAGS = list(itertools.product((False, True), repeat=2))
PARITY_REASON = "odd_electron_formula_outside_closed_shell_domain"
AROMATIC_REASON = "no_valid_state_in_declared_domain"
PILOT = json.loads((REPO_ROOT / "recovery_lab/pilot_candidates.json").read_text())
SUPPORTED = [row for row in PILOT["candidates"] if row["baseline_status"] == "ok"]
PHENYL = next(row for row in PILOT["candidates"] if row["baseline_reason"] == AROMATIC_REASON)
ODD_EXAMPLES = [row for row in PILOT["candidates"]
                if row["stripped"]["ionic_formula"] == {"C": 6, "H": 9, "N": 3, "O": 3, "S": 1}]


@pytest.fixture(scope="module")
def models():
    models = {flags: GlacierAPI(aromatic_fix=flags[0], radical_fix=flags[1])
              for flags in FLAGS}
    yield models
    assert all(model._model is None for model in models.values())


@pytest.mark.parametrize("name", ["aromatic_fix", "radical_fix"])
@pytest.mark.parametrize("value", [None, 0, 1, "false", "true", [], np.bool_(True)])
def test_flags_require_python_bools_before_asset_verification(monkeypatch, name, value):
    def unexpected_verification(self):
        pytest.fail("Invalid configuration reached asset verification")

    monkeypatch.setattr(GlacierAPI, "_verify_assets", unexpected_verification)
    with pytest.raises(TypeError, match=name):
        GlacierAPI("/unused/upstream", "/unused/checkpoint", **{name: value})


def test_flags_are_keyword_only_and_configuration_is_read_only(monkeypatch, tmp_path):
    monkeypatch.setattr(GlacierAPI, "_verify_assets", lambda self: None)
    monkeypatch.setattr(GlacierAPI, "_select_upstream", lambda self: None)
    signature = inspect.signature(GlacierAPI)
    for name in ("aromatic_fix", "radical_fix"):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert signature.parameters[name].default is False
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.touch()
    model = GlacierAPI(upstream, checkpoint, radical_fix=True)
    assert model.aromatic_fix is False and model.radical_fix is True
    for name in ("aromatic_fix", "radical_fix"):
        with pytest.raises(AttributeError):
            setattr(model, name, True)
    settings = model.recovery_settings
    settings["aromatic_fix"] = True
    assert model.recovery_settings == {"aromatic_fix": False, "radical_fix": True}
    assert model.recovery_settings is not settings


@pytest.mark.parametrize("flags", FLAGS)
def test_four_ablation_settings_select_only_their_own_failure_route(models, flags):
    model = models[flags]
    expected_settings = dict(zip(("aromatic_fix", "radical_fix"), flags))
    cases = [(PHENYL, flags[0], "experimental_aromatic_kekule_reference"),
             *((row, flags[1], "direct_radical_cation") for row in ODD_EXAMPLES)]
    for row, enabled, route in cases:
        before = deepcopy(row)
        prepared = FragmentGlacierAdapter(model).prepare(row, 30.)
        assert prepared.get("baseline_reason", prepared["reason"]) == row["baseline_reason"]
        assert prepared.get("recovery_settings", model.recovery_settings) == expected_settings
        assert bool(prepared["states"]) is enabled
        assert prepared.get("experimental", False) is enabled
        assert prepared.get("route", "baseline") == (route if enabled else "baseline")
        assert len(prepared["states"]) == len(prepared["inputs"])
        if enabled:
            assert prepared["status"] == "ok"
            assert 1 <= len(prepared["states"]) <= 4
            for state, model_input in zip(prepared["states"], prepared["inputs"]):
                mol = Chem.MolFromSmiles(state["ionic_smiles"])
                assert "reference_smiles" not in state
                assert _counts_from_mol(mol) == row["stripped"]["ionic_formula"]
                assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == 1
                radical_count = sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
                assert radical_count == (1 if route == "direct_radical_cation" else 0)
                assert state["formula_mz"] == ionic_mass(row["stripped"]["ionic_formula"])
                assert model_input["adduct"] == "[M]+"
                assert model_input["collision_energy"] == 30.
        else:
            assert prepared["reason"] == row["baseline_reason"]
        assert row == before


def test_supported_states_and_inputs_are_exact_for_every_configuration(models):
    for row in SUPPORTED:
        before = deepcopy(row)
        original = FragmentGlacierAdapter().prepare(row, 30.)
        assert original["states"] == row["baseline_states"]
        for flags, model in models.items():
            prepared = FragmentGlacierAdapter(model).prepare(row, 30.)
            assert prepared["status"] == original["status"] == "ok"
            assert prepared["reason"] == prepared.get("baseline_reason") is None
            assert prepared["states"] == original["states"]
            assert prepared["inputs"] == original["inputs"]
            assert prepared.get("route", "baseline") == "baseline"
            assert prepared.get("experimental", False) is False
            assert prepared.get("recovery_settings", model.recovery_settings) == dict(
                zip(("aromatic_fix", "radical_fix"), flags))
            if not any(flags):
                assert prepared.keys() == original.keys()
        assert row == before


def test_models_do_not_share_settings_or_recovery_results(models):
    enabled = models[True, True]
    disabled = models[False, False]
    recovered = FragmentGlacierAdapter(enabled, max_states=1).prepare(ODD_EXAMPLES[0], 20.)
    assert len(recovered["states"]) == 1
    recovered["recovery_settings"]["radical_fix"] = False
    assert enabled.radical_fix is True
    original = FragmentGlacierAdapter(disabled).prepare(ODD_EXAMPLES[0], 20.)
    assert original["reason"] == PARITY_REASON and not original["states"]
    assert original.get("route", "baseline") == "baseline"
    assert disabled.recovery_settings == {"aromatic_fix": False, "radical_fix": False}


@pytest.mark.parametrize("invalid", ["mass", "charge", "formula"])
def test_other_failures_do_not_trigger_either_recovery(models, invalid):
    source = deepcopy(ODD_EXAMPLES[0]["stripped"])
    if invalid == "mass":
        source["precursor_mz"] += 1.
    elif invalid == "charge":
        source["charge"] = 2
    else:
        source["ionic_formula"]["C"] += 1
    before = deepcopy(source)
    baseline = FragmentGlacierAdapter().prepare(source, 30.)
    prepared = FragmentGlacierAdapter(models[True, True]).prepare(source, 30.)
    assert prepared["reason"] == baseline["reason"]
    assert prepared["baseline_reason"] == baseline["reason"]
    assert prepared["route"] == "baseline" and prepared["experimental"] is False
    assert prepared["states"] == prepared["inputs"] == []
    assert source == before


def test_explicit_baseline_ignores_enabled_recovery_and_preserves_audit_shape(models):
    adapter = FragmentGlacierAdapter(models[True, True])
    for row in (SUPPORTED[0], PHENYL, ODD_EXAMPLES[0]):
        original = FragmentGlacierAdapter().prepare(row, 30.)
        strict = adapter.prepare_baseline(row, 30.)
        original["metadata"].pop("elapsed_seconds")
        strict["metadata"].pop("elapsed_seconds")
        assert strict == original
        assert not {"route", "experimental", "baseline_reason", "recovery_settings"} & strict.keys()


@pytest.mark.parametrize("row", [PHENYL, ODD_EXAMPLES[0]], ids=["aromatic", "radical"])
def test_recovery_respects_adapter_search_caps(models, row):
    prepared = FragmentGlacierAdapter(models[True, True], max_states=2, max_leaves=1).prepare(row, 30.)
    assert prepared["metadata"]["max_states"] == 2
    assert prepared["metadata"]["max_leaves"] == 1
    assert prepared["metadata"]["leaves_checked"] <= 1
    assert len(prepared["states"]) <= 2


def test_failed_radical_search_never_chains_into_aromatic_recovery(models, monkeypatch):
    from src.chemistry import _aromatic_recovery

    def unexpected_recovery(*args, **kwargs):
        pytest.fail("A radical search failure must not trigger aromatic recovery")

    monkeypatch.setattr(_aromatic_recovery, "prepare_aromatic_recovery", unexpected_recovery)
    prepared = FragmentGlacierAdapter(models[True, True], max_dp_states=1).prepare(ODD_EXAMPLES[0], 30.)
    assert prepared["metadata"]["max_dp_states"] == 1
    assert prepared["route"] == "direct_radical_cation"
    assert prepared["baseline_reason"] == PARITY_REASON
    assert prepared["states"] == prepared["inputs"] == []


@pytest.mark.parametrize("flags", FLAGS)
def test_notebook_loader_passes_independent_flags_and_reuses_model(flags):
    """Exercise the shared loader definition without loading real weights."""
    source = (REPO_ROOT / "magma_workflow/glacier.py").read_text()
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == "load_glacier_model")
    calls = []

    class FakeAPI:
        def __init__(self, upstream, checkpoint, **kwargs):
            calls.append((upstream, checkpoint, kwargs))
            self.loads = 0

        @property
        def model(self):
            self.loads += 1
            return self

    namespace = {"GlacierAPI": FakeAPI, "_asset_path": lambda explicit, *args: explicit}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "notebook-loader", "exec"), namespace)
    loader = namespace["load_glacier_model"]
    for name in ("aromatic_fix", "radical_fix"):
        assert inspect.signature(loader).parameters[name].default is False
        assert inspect.signature(loader).parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    settings = dict(zip(("aromatic_fix", "radical_fix"), flags))
    result = loader("upstream", "checkpoint", **settings)
    assert calls == [("upstream", "checkpoint", settings)]
    assert result.loads == 1
