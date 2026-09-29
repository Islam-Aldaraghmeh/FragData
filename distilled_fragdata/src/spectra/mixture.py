"""Small, opt-in nonnegative mixture fit for the notebook's MS3 gallery."""
from numbers import Integral

import numpy as np
from scipy.optimize import nnls

from .spectrum import _array, mass_tolerance, match_peaks, prepare_spectrum


def _unit_tic(peaks):
    """Validate/copy peaks and normalize without overflowing their TIC."""
    peaks = _array(peaks)
    if len(peaks):
        peaks[:, 1] /= peaks[:, 1].max()
        peaks[:, 1] /= peaks[:, 1].sum()
    return peaks


def _prediction_axis(predictions):
    """Fixed-anchor bins depend only on predictions, never on target/weights."""
    entries = sorted((float(mz), column, float(intensity))
                     for column, peaks in enumerate(predictions)
                     for mz, intensity in peaks)
    masses, rows = [], []
    for mz, column, intensity in entries:
        if not masses or mz - masses[-1] > mass_tolerance(masses[-1], mz):
            masses.append(mz)
            rows.append(np.zeros(len(predictions), dtype=float))
        rows[-1][column] += intensity
    return np.asarray(masses), np.asarray(rows)


def fit_spectral_mixture(predictions, target, *, min_candidates=2):
    """Maximize cosine to MS3 on a fixed tolerance-aligned peak axis.

    Each nonempty prediction is normalized to unit total intensity. Prediction
    peaks are pooled by sum into fixed-anchor bins using max(0.005 Da, 20 ppm);
    a bin's mass is its lowest predicted mass. Experimental peaks are prepared
    by sum with no threshold or peak cap, then greedily matched to those bins.
    All unmatched predicted and experimental peaks remain in the objective.

    Nonnegative least squares projects the target onto the cone of candidate
    spectra, whose direction maximizes cosine on this fixed axis. Rescaling
    coefficients to sum to one leaves cosine unchanged and puts each in [0, 1].
    The returned mixture uses these exact weights and bin masses, with no peak
    truncation. Weights are relative contributions, not abundance probabilities;
    collinear candidates can have nonunique optimal weights.

    This experimental aligned-fit cosine can differ from the existing sparse
    cosine near tolerance boundaries. ``best_single_cosine`` uses the same fit
    axis for a meaningful comparison. Neither score changes the baseline metric.

    Inputs are N x 2 arrays; None/empty predictions are unavailable and receive
    zero weight. Nonpositive intensities are ignored, as in the existing sparse
    utilities; invalid shapes, nonfinite values, and nonpositive masses raise.
    No input is modified. Failed/uninformative fits return an empty spectrum.
    """
    if isinstance(min_candidates, (bool, np.bool_)) or not isinstance(min_candidates, Integral):
        raise ValueError("min_candidates must be an integer >= 2")
    if min_candidates < 2:
        raise ValueError("min_candidates must be an integer >= 2")
    predictions = list(predictions)
    prepared = [_unit_tic(p) if p is not None else np.empty((0, 2))
                for p in predictions]
    target = _unit_tic(prepare_spectrum(
        _unit_tic(target), relative_threshold=0, max_peaks=None, aggregation="sum"))
    usable = [i for i, peaks in enumerate(prepared) if len(peaks)]
    result = {
        "status": "insufficient_candidates", "reason": "",
        "weights": np.zeros(len(predictions), dtype=float),
        "usable_indices": usable, "spectrum": np.empty((0, 2), dtype=float),
        "cosine": None, "best_single_cosine": None,
    }
    if len(usable) < min_candidates:
        result["reason"] = f"Need at least {min_candidates} usable candidate predictions."
        return result
    if not len(target):
        result.update(status="empty_target", reason="MS3 has no positive-intensity peaks.")
        return result

    masses, matrix = _prediction_axis([prepared[i] for i in usable])
    pairs = match_peaks(np.column_stack((masses, np.ones(len(masses)))), target)
    aligned_target = np.zeros(len(masses))
    matched_target = set()
    for predicted_index, target_index in pairs:
        aligned_target[predicted_index] = target[target_index, 1]
        matched_target.add(target_index)
    unmatched = [target[i, 1] for i in range(len(target)) if i not in matched_target]
    if unmatched:
        matrix = np.vstack((matrix, np.zeros((len(unmatched), len(usable)))))
        aligned_target = np.concatenate((aligned_target, unmatched))

    target_norm = np.linalg.norm(aligned_target)
    single_scores = matrix.T @ aligned_target / (np.linalg.norm(matrix, axis=0) * target_norm)
    result["best_single_cosine"] = float(np.clip(single_scores.max(), 0, 1))
    if not pairs:
        result.update(status="no_overlap", reason="No shared m/z peaks with this MS3.", cosine=0.0)
        return result
    try:
        coefficients, _ = nnls(matrix, aligned_target, maxiter=max(300, 30 * len(usable)))
        total = coefficients.sum()
        if not np.isfinite(coefficients).all() or not np.isfinite(total) or total <= 0:
            raise RuntimeError("Optimizer returned no finite nonzero mixture.")
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
        result.update(status="solver_failed", reason=f"Mixture optimizer failed: {error}")
        return result

    coefficients /= total
    result["weights"][usable] = coefficients
    mixed = matrix @ coefficients
    result["cosine"] = float(np.clip(
        np.dot(mixed, aligned_target) / (np.linalg.norm(mixed) * target_norm), 0, 1))
    intensities = mixed[:len(masses)]
    positive = intensities > 0
    result.update(status="ok", reason="", spectrum=np.column_stack((
        masses[positive], intensities[positive])))
    return result
