"""Analytic checks of the opt-in gallery mixture, independent of Glacier."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "distilled_fragdata"))
from src.spectra import mixture
from src.spectra.mixture import fit_spectral_mixture


@pytest.mark.parametrize("fractions", [[.25, .75], [.2, .3, .5]])
def test_known_two_and_three_candidate_mixtures(fractions):
    masses = 50.0 * np.arange(1, len(fractions) + 1)
    predictions = [np.array([[mass, 10 * (i + 1)]]) for i, mass in enumerate(masses)]
    target = np.column_stack((masses, fractions))
    original = [p.copy() for p in predictions]
    original_target = target.copy()
    result = fit_spectral_mixture(predictions, target)
    assert result["status"] == "ok"
    np.testing.assert_allclose(result["weights"], fractions)
    np.testing.assert_allclose(result["spectrum"], target)
    assert result["cosine"] == pytest.approx(1)
    assert result["best_single_cosine"] < result["cosine"]
    assert np.all((result["weights"] >= 0) & (result["weights"] <= 1))
    for before, after in zip(original, predictions):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(original_target, target)


def test_unmatched_prediction_and_target_peaks_penalize_cosine():
    predictions = [[[50, 1], [150, 1]], [[100, 1], [200, 1]]]
    exact_target = [[50, 1], [100, 1]]
    result = fit_spectral_mixture(predictions, exact_target)
    np.testing.assert_allclose(result["weights"], [.5, .5])
    assert result["cosine"] == pytest.approx(1 / np.sqrt(2))
    assert len(result["spectrum"]) == 4
    with_unmatched_target = fit_spectral_mixture(predictions, exact_target + [[250, 1]])
    np.testing.assert_allclose(with_unmatched_target["weights"], result["weights"])
    assert with_unmatched_target["cosine"] == pytest.approx(1 / np.sqrt(3))


def test_missing_predictions_keep_full_length_zero_weights():
    result = fit_spectral_mixture([None, [[50, 1]], [], [[100, 1]]], [[50, 1], [100, 3]])
    assert result["usable_indices"] == [1, 3]
    np.testing.assert_allclose(result["weights"], [0, .25, 0, .75])
    assert result["cosine"] == pytest.approx(1)


@pytest.mark.parametrize("predictions,target,status", [
    ([[[50, 1]], None], [[50, 1]], "insufficient_candidates"),
    ([[[50, 1]], [[100, 1]]], [], "empty_target"),
    ([[[50, 1]], [[100, 1]]], [[200, 1]], "no_overlap"),
])
def test_uninformative_fits_do_not_invent_a_mixture(predictions, target, status):
    result = fit_spectral_mixture(predictions, target)
    assert result["status"] == status
    assert result["reason"]
    assert result["spectrum"].shape == (0, 2)
    assert not result["weights"].any()
    if status == "no_overlap":
        assert result["cosine"] == result["best_single_cosine"] == 0


def test_duplicate_candidates_have_stable_mixture_not_unique_weights():
    result = fit_spectral_mixture([[[50, 2]], [[50, 3]]], [[50, 7]])
    assert result["status"] == "ok"
    assert result["weights"].sum() == pytest.approx(1)
    np.testing.assert_allclose(result["spectrum"], [[50, 1]])
    assert result["cosine"] == pytest.approx(1)


def test_fixed_anchor_bins_do_not_chain_merge_or_move_with_target():
    predictions = [[[50, 1], [50.008, 1]], [[50.004, 1]]]
    first = fit_spectral_mixture(predictions, [[50, 1], [50.008, 1]])
    second = fit_spectral_mixture(predictions, [[50, 1], [50.008, 3]])
    np.testing.assert_array_equal(first["spectrum"][:, 0], [50, 50.008])
    np.testing.assert_array_equal(second["spectrum"][:, 0], [50, 50.008])
    assert first["cosine"] == pytest.approx(1)


def test_target_and_mixed_spectrum_keep_all_support():
    masses = np.arange(50.0, 170.0)
    predictions = [np.column_stack((masses[:60], np.ones(60))),
                   np.column_stack((masses[60:], np.ones(60)))]
    target = np.column_stack((masses, np.ones(120)))
    result = fit_spectral_mixture(predictions, target)
    assert len(result["spectrum"]) == 120
    assert result["cosine"] == pytest.approx(1)


@pytest.mark.parametrize("minimum", [True, np.bool_(False), 1, 0, 2.5, "2"])
def test_candidate_minimum_is_validated(minimum):
    with pytest.raises(ValueError, match="integer >= 2"):
        fit_spectral_mixture([], [], min_candidates=minimum)


def test_configurable_three_candidate_minimum():
    result = fit_spectral_mixture([[[50, 1]], [[100, 1]]], [[50, 1]], min_candidates=3)
    assert result["status"] == "insufficient_candidates"


@pytest.mark.parametrize("bad", [[[0, 1]], [[50, np.nan]], [[np.inf, 1]], [50, 1]])
def test_invalid_arrays_are_rejected(bad):
    with pytest.raises(ValueError):
        fit_spectral_mixture([bad, [[100, 1]]], [[50, 1]])
    with pytest.raises(ValueError):
        fit_spectral_mixture([[[50, 1]], [[100, 1]]], bad)


def test_solver_failure_is_explicit(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("iteration limit")
    monkeypatch.setattr(mixture, "nnls", fail)
    result = fit_spectral_mixture([[[50, 1]], [[100, 1]]], [[50, 1]])
    assert result["status"] == "solver_failed"
    assert "iteration limit" in result["reason"]
    assert result["spectrum"].shape == (0, 2)
    assert not result["weights"].any()
