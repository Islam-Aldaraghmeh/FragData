"""Regression against both the working prototype and its frozen raw outputs."""
import json
from pathlib import Path
import platform

import numpy as np

from src.models.glacier_api import (
    CHECKPOINT_SHA256, COMMIT, GlacierAPI, build_batch, records_from_smiles, tensor_fingerprint,
)

FIXTURE_PATH = Path(__file__).with_name("fixtures")
FROZEN = json.loads((FIXTURE_PATH / "glacier_fixture.json").read_text())


def _same_environment():
    import rdkit
    import torch

    actual = {"python": platform.python_version(), "system": platform.system(),
              "machine": platform.machine(), "numpy": np.__version__,
              "torch": torch.__version__, "rdkit": rdkit.__version__}
    return actual == FROZEN["environment"]


def _assert_equal(left, right):
    import torch

    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    else:
        assert left == right


def _assert_frozen(result, prefix):
    with np.load(FIXTURE_PATH / "glacier_fixture.npz", allow_pickle=False) as frozen:
        for key, value in result.items():
            actual = value.detach().cpu().numpy()
            expected = frozen[f"{prefix}_{key}"]
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            if _same_environment() or key == "frag":
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_primitive_records_and_all_model_inputs_match_prototype(model, prototype_src):
    for fixture in FROZEN["fixtures"]:
        records = records_from_smiles(fixture["smiles"])
        assert records == fixture["records"]
        metadata = ([fixture["collision_energy"]], [fixture["adduct"]], [fixture["instrument"]])
        batch = build_batch([records], *metadata, model)
        if _same_environment():
            assert tensor_fingerprint(batch) == fixture["tensor_fingerprint"]
        if prototype_src is not None:
            import direct_input
            from run_equivalence import capture_normal

            old_records = direct_input.records_from_smiles(fixture["smiles"])
            assert records == old_records
            _assert_equal(batch, direct_input.build_batch([old_records], *metadata, model))
            original_inputs, _ = capture_normal(
                model, fixture["smiles"], fixture["collision_energy"],
                fixture["adduct"], fixture["instrument"])
            _assert_equal(original_inputs, batch)


def test_normal_and_direct_predictions_match_original_and_frozen(glacier_paths, model, prototype_src):
    glacier = GlacierAPI(*glacier_paths)
    assert glacier._model is None
    assert glacier.checkpoint_identity == {"ms_pred_commit": COMMIT, "checkpoint_sha256": CHECKPOINT_SHA256}
    assert glacier.checkpoint_sha256 == FROZEN["checkpoint_sha256"]
    assert not any(parameter.requires_grad for parameter in glacier.model.parameters())
    assert not glacier.model.training
    for index, fixture in enumerate(FROZEN["fixtures"]):
        args = (fixture["collision_energy"], fixture["adduct"], fixture["instrument"])
        normal = glacier.predict_molecule(fixture["smiles"], *args)
        records = records_from_smiles(fixture["smiles"])
        direct = glacier.predict_records(records, *args)
        _assert_equal(normal, direct)
        _assert_frozen(normal, f"single_{index}")
        if prototype_src is not None:
            import direct_input

            original_normal = model.predict_mol(
                fixture["smiles"], args[0], args[1], "cpu", instrument=args[2])
            original_direct = direct_input.predict_records(records, *args, model)
            _assert_equal(normal, original_normal)
            _assert_equal(direct, original_direct)


def test_mixed_batch_and_prebuilt_tensor_predictions_match_prototype(glacier, prototype_src):
    fixtures = FROZEN["fixtures"]
    records = [records_from_smiles(fixture["smiles"]) for fixture in fixtures]
    metadata = tuple([fixture[key] for fixture in fixtures]
                     for key in ("collision_energy", "adduct", "instrument"))
    batch = build_batch(records, *metadata, glacier.model)
    if _same_environment():
        assert tensor_fingerprint(batch) == FROZEN["batch_tensor_fingerprint"]
    direct = glacier.predict_batch(records, *metadata)
    _assert_equal(direct, glacier.predict_tensors(batch))
    normal = glacier.predict_molecule([fixture["smiles"] for fixture in fixtures], *metadata)
    _assert_equal(normal, direct)
    for index in range(len(fixtures)):
        _assert_frozen({key: values[index] for key, values in direct.items()}, f"batch_{index}")
    if prototype_src is not None:
        import direct_input
        from run_equivalence import capture_normal

        original_inputs, original = capture_normal(
            glacier.model, [fixture["smiles"] for fixture in fixtures], *metadata)
        _assert_equal(batch, original_inputs)
        _assert_equal(batch, direct_input.build_batch(records, *metadata, glacier.model))
        _assert_equal(original, direct)
