"""Scientific invariants for nearest-assigned, CE-conditioned intensities."""
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import intensity_model as original
import intensity_model_v2 as improved
from magma_workflow.candidates import FragmentCandidate
from src.chemistry.formula import ionic_mass


def fragment(name="carbon", hydrogens=3):
    return FragmentCandidate(
        name, {"atoms": [{"element": "C", "aromatic": False}], "bonds": []},
        {"C": 1, "H": hydrogens}, 1, ionic_mass({"C": 1, "H": hydrogens}),
        {"origins": [{"boundary_cut_count": 1, "tree_depth": 1}]},
    )


def bags():
    first, second = fragment(), fragment("second", 5)
    return [(first.precursor_mz, [first], []),
            (second.precursor_mz, [second], []), (30., [], [])]


def test_features_use_nearest_assignment_and_actual_mass_error():
    candidate = fragment()
    mass = candidate.precursor_mz
    nearby = [(mass - .003, [candidate], []),
              (mass + .0001, [deepcopy(candidate)], [])]
    values, indices = improved.candidate_features("CC", 20., nearby)
    assert values.shape == (1, improved.FEATURE_SIZE)
    np.testing.assert_array_equal(indices, [1])
    feature = improved.DESCRIPTOR_START + improved.DESCRIPTORS.index("mass_error_ppm/50")
    assert values[0, feature] == pytest.approx(.0001 / mass * 1e6 / 50)
    assert values[0, feature - 1] == pytest.approx(abs(values[0, feature]))
    # Feature construction neither removes nor mutates the caller's candidates.
    assert all(len(bag[1]) == 1 for bag in nearby)
    assert "mass_error_ppm" not in candidate.metadata


def test_feature_inputs_ignore_intensities_quality_ids_and_ms3():
    supplied = bags()
    before = deepcopy(supplied)
    values, indices = improved.candidate_features("CC", 40., supplied)
    assert supplied == before
    for _, candidates, observations in supplied:
        observations.append("unused MS3")
        for candidate in candidates:
            candidate.metadata.update(intensity=1e20, peak_rank=1,
                                      quality_explained_intensity=.99)
            candidate.candidate_id = "arbitrary-source-id"
            candidate.ms3_spectra = ["also unused"]
    changed, changed_indices = improved.candidate_features("CC", 40., supplied)
    np.testing.assert_array_equal(changed, values)
    np.testing.assert_array_equal(changed_indices, indices)


def test_duplicate_identity_is_removed_but_different_formula_stays_latent():
    first, other = fragment(), fragment("other-formula", 5)
    duplicate = deepcopy(first)
    duplicate.candidate_id = "different-id"
    values, indices = improved.candidate_features(
        "CC", 40., [(first.precursor_mz, [first, duplicate, other], [])])
    assert len(values) == 2
    np.testing.assert_array_equal(indices, [0, 0])
    assert not np.array_equal(values[0], values[1])
    scores = torch.tensor([.2, .8], requires_grad=True)
    pooled = improved.pool_candidate_scores(scores, torch.as_tensor(indices), 2)
    torch.testing.assert_close(pooled, torch.tensor([.8, 0.]))
    pooled.sum().backward()
    torch.testing.assert_close(scores.grad, torch.tensor([0., 1.]))


def test_ce_replacement_changes_only_energy_and_matches_rebuilt_features():
    supplied = bags()
    low, indices = improved.candidate_features("CC", 20., supplied)
    low_copy = low.copy()
    high = improved.with_collision_energy(low, 60.)
    rebuilt, rebuilt_indices = improved.candidate_features("CC", 60., supplied)
    np.testing.assert_array_equal(high, rebuilt)
    np.testing.assert_array_equal(indices, rebuilt_indices)
    np.testing.assert_array_equal(low, low_copy)
    np.testing.assert_array_equal(low[:, :improved.CE_INDEX], high[:, :improved.CE_INDEX])
    np.testing.assert_allclose(low[:, improved.CE_INDEX], .4)
    np.testing.assert_allclose(high[:, improved.CE_INDEX], 1.2)


@pytest.mark.parametrize("energy", [-1., np.nan, np.inf])
def test_invalid_energy_rejected(energy):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        improved.candidate_features("CC", energy, bags())
    with pytest.raises(ValueError, match="finite and nonnegative"):
        improved.with_collision_energy(np.zeros((0, improved.FEATURE_SIZE)), energy)


@pytest.mark.parametrize("fingerprints", [False, True])
def test_prediction_handles_empty_peaks_and_restores_model_mode(fingerprints):
    torch.manual_seed(7)
    model = improved.EnergyConditionedMLP(use_fingerprints=fingerprints)
    supplied = bags()
    prediction = improved.predict_spectrum(model, "CC", 40., supplied)
    assert model.training
    np.testing.assert_array_equal(prediction[:, 0], [bag[0] for bag in supplied])
    assert prediction.shape == (3, 2) and prediction[2, 1] == 0
    assert prediction[:, 1].max() == pytest.approx(1.)
    assert np.isfinite(prediction).all()
    assert improved.predict_spectrum(model, "CC", 40., []).shape == (0, 2)
    np.testing.assert_array_equal(
        improved.predict_spectrum(model, "CC", 40., [(30., [], [])]), [[30., 0.]])


def test_descriptor_only_ignores_fingerprint_branch_exactly():
    values, _ = improved.candidate_features("CC", 40., bags())
    model = improved.EnergyConditionedMLP(use_fingerprints=False).eval()
    changed = values.copy()
    changed[:, :improved.DESCRIPTOR_START] = 1 - changed[:, :improved.DESCRIPTOR_START]
    with torch.no_grad():
        torch.testing.assert_close(model(torch.tensor(values)), model(torch.tensor(changed)),
                                   rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) < 10_000
    assert sum(p.numel() for p in improved.EnergyConditionedMLP().parameters()) < 10_000


def test_tiny_conditioned_head_learns_energy_dependent_peak_rank_inversion():
    # Identical candidates/support at both energies; only CE changes. A global
    # CE multiplier cannot fit these opposite target shapes under cosine loss.
    features = np.zeros((2, improved.FEATURE_SIZE), dtype=np.float32)
    features[:, improved.DESCRIPTOR_START + 1] = [.95, .2]
    low = torch.tensor(improved.with_collision_energy(features, 20.))
    high = torch.tensor(improved.with_collision_energy(features, 60.))
    targets = (torch.tensor([1., .1]), torch.tensor([.1, 1.]))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(17)
        model = improved.EnergyConditionedMLP(use_fingerprints=False, dropout=0.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=1e-3)
        for _ in range(140):
            predictions = [model(x) for x in (low, high)]
            loss = torch.stack([1 - F.cosine_similarity(p, y, dim=0)
                                for p, y in zip(predictions, targets)]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predicted_low, predicted_high = model(low), model(high)
            assert F.cosine_similarity(predicted_low, targets[0], dim=0) > .99
            assert F.cosine_similarity(predicted_high, targets[1], dim=0) > .99
            assert F.cosine_similarity(predicted_low, predicted_high, dim=0) < .4
            assert predicted_low.argmax() == 0 and predicted_high.argmax() == 1
    finally:
        torch.set_num_threads(previous_threads)


def test_inference_matches_batched_max_and_checkpoint_roundtrip(tmp_path):
    from train_intensity import predict_examples
    torch.manual_seed(7)
    model = improved.EnergyConditionedMLP()
    supplied = bags()
    features, indices = improved.candidate_features("CC", 40., supplied)
    example = dict(features=features, peak_indices=indices, target=np.ones(3))
    predicted = improved.predict_spectrum(model, "CC", 40., supplied)
    np.testing.assert_array_equal(predict_examples(model, [example])[0], predicted[:, 1])
    checkpoint = tmp_path / "model.pt"
    torch.save(dict(schema=improved.SCHEMA, feature_size=improved.FEATURE_SIZE,
                    config=model.config, state_dict=model.state_dict()), checkpoint)
    loaded = improved.load_model(checkpoint)
    np.testing.assert_array_equal(
        improved.predict_spectrum(loaded, "CC", 40., supplied), predicted)


def test_using_improved_model_leaves_old_features_and_predictions_exact():
    torch.manual_seed(7)
    model = original.IntensityMLP()
    candidate = fragment()
    supplied = [(candidate.precursor_mz - .001, [candidate], []),
                (candidate.precursor_mz + .002, [deepcopy(candidate)], [])]
    before_features, before_indices = original.candidate_features("CC", 40., supplied)
    before_prediction = original.predict_spectrum(model, "CC", 40., supplied)
    improved.predict_spectrum(improved.EnergyConditionedMLP(), "CC", 40., supplied)
    after_features, after_indices = original.candidate_features("CC", 40., supplied)
    np.testing.assert_array_equal(after_features, before_features)
    np.testing.assert_array_equal(after_indices, before_indices)
    np.testing.assert_array_equal(original.predict_spectrum(model, "CC", 40., supplied),
                                  before_prediction)
    assert len(before_features) == 2  # Legacy all-match behavior remains intact.


def test_import_does_not_load_glacier_or_ionic_enumeration():
    subprocess.run([sys.executable, "-c", """
import sys
import intensity_model_v2
for name in ('src.models.glacier_api', 'src.chemistry.fragment_adapter',
             'src.chemistry._enumeration', 'src.chemistry._radical_enumeration',
             'src.chemistry._state_graph'):
    assert name not in sys.modules, name
"""], cwd=ROOT, check=True)


def test_diagnostics_credit_exact_ties_fractionally_and_zero_predictions_never_win():
    from train_intensity_v2 import compare_metrics
    examples = [dict(target=np.asarray([2., 2., 1.])),
                dict(target=np.asarray([1., 0., 0.]))]
    scores = compare_metrics(examples, [np.ones(3), np.zeros(3)])
    assert scores["strongest_peak_accuracy"] == pytest.approx((2/3 + 0)/2)
    assert scores["predicted_effective_peak_count"] == pytest.approx((3 + 0)/2)
    assert scores["experimental_effective_peak_count"] == pytest.approx((25/9 + 1)/2)
    assert scores["mean_cosine"] == pytest.approx((5/(3*np.sqrt(3)) + 0)/2)


def test_energy_diagnostic_changes_only_ce_on_fixed_axis_and_excludes_degenerate_support():
    from train_intensity_v2 import energy_response

    class EnergyProbe(torch.nn.Module):
        def forward(self, values):
            return torch.exp((values[:, improved.CE_INDEX] - .8)
                             * (values[:, improved.DESCRIPTOR_START] - .5) * 6)

    values = np.zeros((2, improved.FEATURE_SIZE), dtype=np.float32)
    values[:, improved.DESCRIPTOR_START] = [0., 1.]
    values[:, improved.CE_INDEX] = .8
    examples = [dict(features=values[:count].copy(),
                     peak_indices=np.arange(count, dtype=np.int64),
                     target=np.asarray([1., .5, .25], dtype=np.float32))
                for count in (0, 1, 2)]
    original_features = [example["features"].copy() for example in examples]
    response = energy_response(EnergyProbe(), examples)
    relative = np.exp(-2.4)
    assert response["spectra_with_at_least_two_supported_peaks"] == 1
    assert response["excluded_zero_or_single_supported_peak_spectra"] == 2
    assert response["mean_prediction_cosine"] == pytest.approx(
        2*relative/(1+relative**2), rel=1e-5)
    assert response["mean_absolute_normalized_intensity_change"] == pytest.approx(
        2*(1-relative)/3, rel=1e-5)
    for example, saved_features in zip(examples, original_features):
        np.testing.assert_array_equal(example["features"], saved_features)
    equal = energy_response(None, examples, equal=True)
    assert equal["mean_prediction_cosine"] == pytest.approx(1.)
    assert equal["mean_absolute_normalized_intensity_change"] == 0
    degenerate = energy_response(EnergyProbe(), examples[:2])
    assert degenerate["mean_prediction_cosine"] is None
    assert degenerate["mean_absolute_normalized_intensity_change"] is None


def test_legacy_views_preserve_targets_and_select_requested_frozen_model_features():
    from train_intensity_v2 import legacy_examples
    example = dict(target=np.asarray([1., .5]), features=np.zeros((1, 277)),
                   peak_indices=np.asarray([0]), old_features=np.ones((2, 1032)),
                   old_peak_indices=np.asarray([0, 1]),
                   old_nearest_features=np.full((1, 1032), 2.),
                   old_nearest_peak_indices=np.asarray([1]))
    legacy = legacy_examples([example])[0]
    nearest = legacy_examples([example], nearest=True)[0]
    assert legacy["features"] is example["old_features"]
    assert nearest["features"] is example["old_nearest_features"]
    assert legacy["target"] is nearest["target"] is example["target"]
    np.testing.assert_array_equal(legacy["peak_indices"], [0, 1])
    np.testing.assert_array_equal(nearest["peak_indices"], [1])
    assert example["features"].shape == (1, 277)


@pytest.mark.parametrize("fingerprints", [False, True])
def test_two_epoch_training_selects_validation_checkpoint_and_is_reproducible(fingerprints):
    from train_intensity import predict_examples
    from train_intensity_v2 import fit_model
    supplied = bags()
    features, indices = improved.candidate_features("CC", 40., supplied)
    example = dict(features=features, peak_indices=indices,
                   target=np.asarray([1., .2, .05], dtype=np.float32))
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        first, history, epoch = fit_model([example]*4, [example], seed=17,
            use_fingerprints=fingerprints, epochs=2, patience=2, batch_size=2)
        second, same_history, same_epoch = fit_model([example]*4, [example], seed=17,
            use_fingerprints=fingerprints, epochs=2, patience=2, batch_size=2)
        assert history == same_history and epoch == same_epoch
        expected_epoch = max(history, key=lambda row: row["validation_mean_cosine"])["epoch"]
        assert epoch == expected_epoch
        score = original.spectrum_metrics([example["target"]], predict_examples(first, [example]))
        assert score["mean_cosine"] == pytest.approx(history[epoch-1]["validation_mean_cosine"])
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(deterministic)
