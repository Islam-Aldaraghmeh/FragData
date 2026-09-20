"""Frozen and live prototype regression for peaks, ensemble and candidate scores."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments.ms3_candidate_support import candidate_support, score_bag
from src.spectra.metrics import score_spectra
from src.spectra.spectrum import (
    Spectrum, clean_raw, match_peaks, mean_normalized_spectrum,
    normalize_spectrum, prepare_spectrum,
)

FROZEN = json.loads((Path(__file__).parent / "fixtures/spectra.json").read_text())


def test_postprocessing_matches_frozen_and_live_prototype(prototype_src):
    for case in FROZEN["cases"]:
        result = prepare_spectrum(case["raw"], **case["settings"])
        np.testing.assert_array_equal(result, np.asarray(case["prepared"]).reshape(-1, 2))
        np.testing.assert_array_equal(normalize_spectrum(result),
                                      np.asarray(case["normalized"]).reshape(-1, 2))
        if prototype_src:
            import spectral_metrics as old
            np.testing.assert_array_equal(result, old.prepare_spectrum(case["raw"], **case["settings"]))
    raw = np.array([[0, 3], [-1, 2], [10, float("nan")], [20, 1], [30, 0]])
    np.testing.assert_array_equal(clean_raw(raw), [[20, 1]])
    np.testing.assert_array_equal(prepare_spectrum([[100, 1], [100, 2]]), [[100, 2]])


def test_microstate_average_matches_prototype_and_keeps_full_support(prototype_src):
    actual = mean_normalized_spectrum(FROZEN["states"])
    np.testing.assert_array_equal(actual, FROZEN["ensemble"])
    if prototype_src:
        import spectral_metrics as old
        np.testing.assert_array_equal(actual, old.mean_normalized_spectrum(FROZEN["states"]))
    states = [[[100+i, 100] for i in range(100)], [[300+i, 1] for i in range(100)]]
    averaged = Spectrum.average([Spectrum.from_peaks(s) for s in states])
    assert len(averaged.mz) == 200
    np.testing.assert_allclose(averaged.intensity, np.repeat(1/200, 200))


def test_metric_matching_and_empty_cases(prototype_src):
    cases = [([[100, 3], [200, 4]], [[100, 2], [300, 1]]), ([], []),
             ([[1000, 1]], [[1000.019, 1]]), ([[1000, 1]], [[1000.022, 1]])]
    for prediction, target in cases:
        if prototype_src:
            import spectral_metrics as old
            assert score_spectra(prediction, target) == old.score_spectra(prediction, target)
    assert match_peaks([[100, 1], [100.004, 1]], [[100.003, 1]]) == [(1, 0)]
    assert score_spectra([], [])["cosine"] == 0
    assert score_spectra(*cases[0])["cosine"] == 6/(5*np.sqrt(5))


def test_candidate_scores_and_high_level_flow_match_frozen_prototype(prototype_src):
    result = score_bag(FROZEN["candidates"], FROZEN["target"])
    assert result == FROZEN["scores"]
    if prototype_src:
        import ms3_analysis as old
        assert result == old.score_bag(FROZEN["candidates"], FROZEN["target"])
    bag = SimpleNamespace(bag_id="frozen", linked_isolation_id="peak-1",
                          candidates=[SimpleNamespace(candidate_id=str(i)) for i in range(3)])
    adapter = SimpleNamespace(predict=lambda c, ce, instrument:
                              Spectrum.from_peaks(FROZEN["candidates"][int(c.candidate_id)]))
    link = {"status": "unique", "matches": [{"linked_isolation_id": "peak-1"}]}
    actual = candidate_support(bag, adapter, FROZEN["raw_target"], linkage=link, collision_energy=20)
    assert actual["candidate_scores"] == result["candidate_scores"]
    assert actual["support"] == result["support"]
    with pytest.raises(ValueError, match="Empty experimental"):
        candidate_support(bag, adapter, [], linkage=link, collision_energy=20)
    candidates = bag.candidates
    bag.candidates = []
    with pytest.raises(ValueError, match="No existing candidates"):
        candidate_support(bag, adapter, FROZEN["raw_target"], linkage=link, collision_energy=20)
    bag.candidates = candidates
    with pytest.raises(ValueError, match="unique"):
        candidate_support(bag, adapter, [], linkage={"status": "ambiguous"}, collision_energy=20)
    bag.linked_isolation_id = "different"
    with pytest.raises(ValueError, match="different"):
        candidate_support(bag, adapter, [], linkage=link, collision_energy=20)


def test_actual_ionic_ensemble_matches_prototype(glacier, prototype_src):
    if prototype_src is None:
        pytest.skip("Live ensemble differential requires original prototype")
    from direct_input import records_from_smiles, predict_records
    from microstates import stripped_from_smiles, ionic_mass, enumerate_microstates
    import spectral_metrics as old
    from src.chemistry.fragment_adapter import FragmentGlacierAdapter

    candidate = stripped_from_smiles("NCCO", "C2H8NO", 1, ionic_mass("C2H8NO"))
    states = enumerate_microstates(candidate, max_states=4)["states"]
    assert len(states) > 1
    raw = [predict_records(records_from_smiles(s["ionic_smiles"]), 20., "[M]+", "Orbitrap",
                           glacier.model)["spec"].cpu().numpy() for s in states]
    spectra = [old.normalize_spectrum(old.prepare_spectrum(clean_raw(s))) for s in raw]
    expected = old.mean_normalized_spectrum(spectra, prepared=True)
    actual = FragmentGlacierAdapter(glacier, max_states=4).predict(candidate, 20.).as_array()
    np.testing.assert_array_equal(actual, expected)
