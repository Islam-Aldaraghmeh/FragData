"""Score an existing candidate bag against one uniquely linked MS3 spectrum."""
import math

import numpy as np

from src.data.candidate_bags import load_candidate_bag
from src.spectra.metrics import score_spectra
from src.spectra.spectrum import Spectrum, normalize_spectrum, prepare_spectrum

METRICS = ("cosine", "entropy_similarity")


def support_summary(scores):
    """Linear normalized similarity weights, not calibrated probabilities.

    An all-zero vector has uniform support. A singleton has no second-best
    margin; primary ranking analyses require fully covered ambiguous bags.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A finite, nonempty score vector is required")
    if (values < 0).any() or (values > 1 + 1e-10).any():
        raise ValueError("Similarities must be in [0,1]")
    weights = values / values.sum() if values.sum() > 0 else np.full(len(values), 1 / len(values))
    nonzero = weights[weights > 0]
    entropy = float(-np.dot(nonzero, np.log(nonzero)))
    ranked = np.sort(values)[::-1]
    return {"best_similarity": float(ranked[0]),
            "best_second_margin": float(ranked[0] - ranked[1]) if len(values) > 1 else None,
            "normalized_support_entropy": entropy / math.log(len(values)) if len(values) > 1 else 0.,
            "effective_candidate_count": float(np.exp(entropy)),
            "support_weights": weights.tolist(),
            "best_tied_indices": np.flatnonzero(np.isclose(values, ranked[0], atol=1e-12, rtol=0)).tolist(),
            "candidate_count": len(values), "all_zero": bool(values.sum() == 0)}


def score_bag(spectra, target):
    rows = [score_spectra(s, target, prepared_predicted=True, prepared_target=True) for s in spectra]
    return {"candidate_scores": rows,
            "support": {metric: support_summary([r[metric] for r in rows]) for metric in METRICS}}


def candidate_support(bag, adapter, experimental_ms3, *, linkage, collision_energy,
                      instrument="Orbitrap", prepared_target=False):
    """Predict every existing candidate; fail explicitly on incomplete coverage.

    ``collision_energy`` is already the GLACIER scalar. Use the data module's
    NCE conversion explicitly when its acquisition assumptions apply.
    """
    if linkage.get("status") != "unique" or len(linkage.get("matches", [])) != 1:
        raise ValueError("Candidate support requires a unique MS2/MS3 link")
    match = linkage["matches"][0]
    if bag.linked_isolation_id and bag.linked_isolation_id != match["linked_isolation_id"]:
        raise ValueError("Candidate bag belongs to a different linked MS2 peak")
    if not bag.candidates:
        raise ValueError("No existing candidates in this bag")
    target = experimental_ms3.as_array() if isinstance(experimental_ms3, Spectrum) else experimental_ms3
    if not prepared_target:
        target = normalize_spectrum(prepare_spectrum(target, relative_threshold=0,
                                                     max_peaks=None, aggregation="sum"))
    if not len(target):
        raise ValueError("Empty experimental spectrum")
    spectra = [adapter.predict(c, collision_energy, instrument=instrument).as_array()
               for c in bag.candidates]
    return {"bag_id": bag.bag_id, "candidate_ids": [c.candidate_id for c in bag.candidates],
            "linked_isolation_id": match["linked_isolation_id"], **score_bag(spectra, target)}


def score_candidate_file(path, adapter, experimental_ms3, *, bag_id=None, **settings):
    return candidate_support(load_candidate_bag(path, bag_id=bag_id), adapter,
                             experimental_ms3, **settings)
