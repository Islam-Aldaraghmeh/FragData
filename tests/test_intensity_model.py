"""Scientific invariants for the small latent-candidate intensity experiment."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import intensity_model as experiment
from magma_workflow.candidates import FragmentCandidate


def candidate(name="carbon", *, hydrogens=3, mz=15.02):
    return FragmentCandidate(
        name, {"atoms": [{"element": "C", "aromatic": False}], "bonds": []},
        {"C": 1, "H": hydrogens}, 1, mz,
        metadata={"origins": [{"boundary_cut_count": 1, "tree_depth": 1}]},
    )


def test_peak_max_keeps_alternatives_latent_and_routes_gradient():
    scores = torch.tensor([0.2, 0.8, 0.6, 0.1], requires_grad=True)
    peak_ids = torch.tensor([0, 0, 1, 1])
    pooled = experiment.pool_candidate_scores(scores, peak_ids, 3)
    torch.testing.assert_close(pooled, torch.tensor([0.8, 0.6, 0.0]))
    pooled.sum().backward()
    torch.testing.assert_close(scores.grad, torch.tensor([0., 1., 1., 0.]))
    duplicated = experiment.pool_candidate_scores(
        torch.tensor([0.2, 0.8, 0.8, 0.6, 0.1]), torch.tensor([0, 0, 0, 1, 1]), 3)
    torch.testing.assert_close(pooled.detach(), duplicated)


def test_features_ignore_experimental_intensities_and_do_not_mutate():
    fragment = candidate()
    bags = [(15.02, [fragment], [])]
    original = deepcopy(fragment)
    features, peaks = experiment.candidate_features("CC", 40., bags)
    assert features.dtype == np.float32
    assert peaks.dtype == np.int64
    assert features.shape == (1, experiment.FEATURE_SIZE)
    assert fragment == original
    fragment.metadata.update(intensity=1e15, normalized_intensity=0.999,
                             peak_rank=1, explained_intensity=0.99)
    fragment.ms3_spectra = ["not a feature"]
    changed, changed_peaks = experiment.candidate_features("CC", 40., bags)
    np.testing.assert_array_equal(changed, features)
    np.testing.assert_array_equal(changed_peaks, peaks)


def test_global_hydrogen_formula_distinguishes_identical_heavy_graphs():
    first, second = candidate("first", hydrogens=3), candidate("second", hydrogens=5)
    np.testing.assert_array_equal(experiment.fingerprint_fragment(first),
                                  experiment.fingerprint_fragment(second))
    features, _ = experiment.candidate_features("CC", 40., [(15., [first, second], [])])
    assert len(features) == 2
    assert not np.array_equal(features[0], features[1])


def test_dedup_uses_exact_structure_not_candidate_id_or_atom_order():
    first = candidate("source-a")
    first.graph = {
        "atoms": [{"element": symbol, "aromatic": False} for symbol in "CNO"],
        "bonds": [{"i": 0, "j": 1, "order": 2., "aromatic": False},
                  {"i": 1, "j": 2, "order": 1., "aromatic": False}],
    }
    first.ionic_formula = {"C": 1, "H": 3, "N": 1, "O": 1}
    duplicate = deepcopy(first)
    duplicate.candidate_id = "different-source-id"
    duplicate.graph = {
        "atoms": list(reversed(first.graph["atoms"])),
        "bonds": [{"i": 1, "j": 2, "order": 2., "aromatic": False},
                  {"i": 0, "j": 1, "order": 1., "aromatic": False}],
    }
    np.testing.assert_array_equal(experiment.fingerprint_fragment(first),
                                  experiment.fingerprint_fragment(duplicate))
    single, single_peaks = experiment.candidate_features("CC", 40., [(60., [first], [])])
    repeated, repeated_peaks = experiment.candidate_features(
        "CC", 40., [(60., [first, duplicate, first], [])])
    np.testing.assert_array_equal(single, repeated)
    np.testing.assert_array_equal(single_peaks, repeated_peaks)
    # Dedup is per peak; identical graphs supplied at two locations remain
    # independently reconstructable peak hypotheses.
    across_peaks, indices = experiment.candidate_features(
        "CC", 40., [(60., [first], []), (60.001, [duplicate], [])])
    assert len(across_peaks) == 2
    np.testing.assert_array_equal(indices, [0, 1])


def test_raw_fragment_fingerprint_handles_invalid_valence_without_enumeration():
    # Oxygen bonded to three carbons is an intentionally unassigned raw heavy
    # graph: fingerprinting must not first solve a chemically valid ion state.
    fragment = candidate("raw")
    fragment.graph = {
        "atoms": [{"element": symbol, "aromatic": False} for symbol in "OCCC"],
        "bonds": [{"i": 0, "j": j, "order": 1., "aromatic": False}
                  for j in (1, 2, 3)],
    }
    fragment.ionic_formula = {"C": 3, "H": 9, "O": 1}
    fp = experiment.fingerprint_fragment(fragment)
    assert fp.ndim == 1 and np.isfinite(fp).all() and fp.sum() > 0


def test_prediction_is_nonnegative_and_preserves_unsupported_peaks():
    torch.manual_seed(7)
    model = experiment.IntensityMLP()
    bags = [(15.02, [candidate()], []), (25., [], []),
            (29., [candidate("other", hydrogens=5, mz=29.)], [])]
    predicted = experiment.predict_spectrum(model, "CC", 40., bags)
    assert predicted.shape == (3, 2)
    np.testing.assert_array_equal(predicted[:, 0], [15.02, 25., 29.])
    assert predicted[1, 1] == 0
    assert np.isfinite(predicted).all()
    assert (predicted[[0, 2], 1] > 0).all()
    assert predicted[:, 1].max() == pytest.approx(1.)
    assert model.training  # Inference does not change the caller's model mode.
    np.testing.assert_array_equal(
        experiment.predict_spectrum(model, "CC", 40., [(25., [], [])]), [[25., 0.]])
    assert experiment.predict_spectrum(model, "CC", 40., []).shape == (0, 2)


def test_metrics_use_full_peak_axis_and_basepeak_normalization():
    # The equal-supported-intensity baseline has [1,1,0] on this full axis.
    # Its unsupported third peak still contributes to both cosine and MAE.
    target = np.asarray([2., 1., 2.])
    baseline = np.asarray([1., 1., 0.])
    metrics = experiment.spectrum_metrics([target], [baseline])
    expected_cosine = 1.5 / (1.5 * np.sqrt(2))
    assert metrics == pytest.approx({
        "mean_cosine": expected_cosine, "median_cosine": expected_cosine,
        "mean_absolute_error": .5, "fraction_cosine_gt_0.5": 1.,
        "fraction_cosine_gt_0.7": 1.,
    })
    zero = experiment.spectrum_metrics([target], [np.zeros(3)])
    assert zero["mean_cosine"] == 0
    assert zero["mean_absolute_error"] == pytest.approx(2.5 / 3)


def test_metrics_macro_average_instead_of_weighting_long_spectra():
    metrics = experiment.spectrum_metrics(
        [np.asarray([1., 0.]), np.ones(10)],
        [np.asarray([0., 1.]), np.ones(10)],
    )
    assert metrics == pytest.approx({
        "mean_cosine": .5, "median_cosine": .5, "mean_absolute_error": .5,
        "fraction_cosine_gt_0.5": .5, "fraction_cosine_gt_0.7": .5,
    })


def test_model_import_does_not_load_glacier_or_ionic_enumerators():
    subprocess.run([sys.executable, "-c", """
import sys
import intensity_model
for module in ('src.models.glacier_api', 'src.chemistry.fragment_adapter',
               'src.chemistry._enumeration', 'src.chemistry._radical_enumeration',
               'src.chemistry._state_graph'):
    assert module not in sys.modules, module
"""], cwd=ROOT, check=True)


def test_equal_peak_baseline_is_independent_of_candidate_multiplicity():
    from train_intensity import equal_peak_baseline
    targets = np.asarray([1., .5, .25])
    examples = [dict(target=targets, peak_indices=np.asarray(indices, dtype=np.int64))
                for indices in ([0, 2], [0, 0, 0, 2], [])]
    first, repeated, unsupported = equal_peak_baseline(examples)
    np.testing.assert_array_equal(first, [1., 0., 1.])
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(unsupported, [0., 0., 0.])


def test_group_split_is_disjoint_complete_and_reproducible():
    from intensity_data import split_molecules
    keys = [f"molecule-{i:03d}" for i in range(100)]
    splits = split_molecules(keys, seed=7)
    assert splits == split_molecules(list(reversed(keys)) + keys[:3], seed=7)
    assert {name: len(group) for name, group in splits.items()} == {
        "train": 70, "validation": 15, "test": 15}
    groups = [set(group) for group in splits.values()]
    assert set.union(*groups) == set(keys)
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all(split_molecules(keys[:3], seed=7).values())


def test_index_groups_equivalent_precursors_and_ignores_labels(tmp_path):
    from intensity_data import _index
    def record(smiles, ce):
        return dict(smiles=smiles, precursor_mz=100., ms_level=2,
                    merge_type="SINGLE_BEST_SCAN", collision_energy=ce,
                    charge=1, adduct="[M+H]+", peaks=[[20., 99999.]])
    records = [record("CCO", 20.), record("OCC", 40.),
               record("C[C@H](O)F", 20.), record("C[C@@H](O)F", 40.),
               record("CCC", [20., 40.])]
    records.append({**record("CCN", 20.), "merge_type": "ALL_MSN_TO_PSEUDO_MS2"})
    path = tmp_path / "ordinary.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    groups, exclusions, _ = _index(path)
    assert len(groups) == 2
    assert sorted(map(len, groups.values())) == [2, 2]
    assert exclusions == {"non_scalar_or_invalid_collision_energy": 1,
                          "not_ordinary_single_scan_ms2": 1}
    for row in records:
        row["peaks"] = [[1., 0.]]
        row["quality_explained_intensity"] = 0.
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    changed, changed_exclusions, _ = _index(path)
    assert set(changed) == set(groups)
    assert {key: [ce for _, ce in rows] for key, rows in changed.items()} == {
        key: [ce for _, ce in rows] for key, rows in groups.items()}
    assert changed_exclusions == exclusions


def test_tiny_training_is_deterministic_and_inference_matches_batch():
    from train_intensity import fit_model, predict_examples
    features = np.zeros((2, experiment.FEATURE_SIZE), dtype=np.float32)
    features[0, 0], features[1, 1] = 1., 1.
    example = dict(features=features, peak_indices=np.asarray([0, 1], dtype=np.int64),
                   target=np.asarray([1., .2, .05], dtype=np.float32))
    previous_threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        first, history, epoch = fit_model([example] * 4, [example], seed=17,
                                          epochs=6, patience=6, batch_size=2)
        repeated, same_history, same_epoch = fit_model([example] * 4, [example], seed=17,
                                                       epochs=6, patience=6, batch_size=2)
        assert history == same_history and epoch == same_epoch
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0, atol=0)
        assert history[-1]["validation_mean_cosine"] > history[0]["validation_mean_cosine"]
        predictions = predict_examples(first, [example] * 3, batch_size=2)
        assert len(predictions) == 3
        assert all(p[2] == 0 and np.isfinite(p).all() for p in predictions)

        bags = [(15.02, [candidate()], []), (25., [], [])]
        values, indices = experiment.candidate_features("CC", 40., bags)
        prepared = dict(features=values, peak_indices=indices, target=np.ones(2))
        via_batch = predict_examples(first, [prepared])[0]
        via_inference = experiment.predict_spectrum(first, "CC", 40., bags)[:, 1]
        np.testing.assert_array_equal(via_batch, via_inference)
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(deterministic)
