"""Unchanged prototype tolerance-matched spectrum scores."""
import numpy as np

from .spectrum import _array, match_peaks, prepare_spectrum


def score_spectra(predicted, target, *, prepared_predicted: bool = False,
                  prepared_target: bool = False, atol: float = 0.005,
                  ppm: float = 20.0) -> dict:
    """Raw-intensity cosine, 1-JSD/ln(2), TIC recall and predicted TIC precision.

    Empty predictions or targets score zero for all similarities and coverage;
    two empty spectra are not evidence of a successful spectrum prediction.
    """
    p = _array(predicted) if prepared_predicted else prepare_spectrum(predicted, atol=atol, ppm=ppm)
    t = _array(target) if prepared_target else prepare_spectrum(target, atol=atol, ppm=ppm, aggregation="sum")
    pairs = match_peaks(p, t, atol=atol, ppm=ppm)
    result = {"cosine": 0.0, "entropy_similarity": 0.0, "intensity_recall": 0.0,
              "predicted_intensity_precision": 0.0, "matched_peaks": len(pairs),
              "predicted_peaks": len(p), "target_peaks": len(t)}
    if not len(p) or not len(t):
        return result
    dot = sum(p[i, 1] * t[j, 1] for i, j in pairs)
    result["cosine"] = float(np.clip(dot / (np.linalg.norm(p[:, 1]) * np.linalg.norm(t[:, 1])), 0, 1))
    result["intensity_recall"] = float(sum(t[j, 1] for _, j in pairs) / t[:, 1].sum())
    result["predicted_intensity_precision"] = float(sum(p[i, 1] for i, _ in pairs) / p[:, 1].sum())
    # Aligned union distribution, including every unmatched peak.
    pn, tn = p[:, 1] / p[:, 1].sum(), t[:, 1] / t[:, 1].sum()
    jsd = np.log(2.0)
    for i, j in pairs:
        x, y = pn[i], tn[j]
        midpoint = (x + y) / 2
        jsd += 0.5 * (x * np.log(x / midpoint) + y * np.log(y / midpoint)
                      - (x + y) * np.log(2.0))
    result["entropy_similarity"] = float(np.clip(1.0 - jsd / np.log(2.0), 0, 1))
    return result


def cosine_similarity(predicted, target, **settings):
    return score_spectra(predicted, target, **settings)["cosine"]


def entropy_similarity(predicted, target, **settings):
    return score_spectra(predicted, target, **settings)["entropy_similarity"]


def intensity_recall(predicted, target, **settings):
    return score_spectra(predicted, target, **settings)["intensity_recall"]


def intensity_precision(predicted, target, **settings):
    return score_spectra(predicted, target, **settings)["predicted_intensity_precision"]


# Future candidate-discriminative metrics belong here as separate functions;
# keep their result names distinct from these regression-locked metrics.
