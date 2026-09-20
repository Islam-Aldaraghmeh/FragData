"""Cache identity includes every input context and saves raw rows exactly."""
import numpy as np
import pytest

from src.utils.prediction_cache import PredictionCache


def test_prediction_cache_roundtrip_and_input_separation(tmp_path):
    cache = PredictionCache(tmp_path)
    inputs = dict(candidate_id="c1", state_id="CC[OH2+]", formula="C2H7O", charge=1,
                  collision_energy=20., instrument="Orbitrap", checkpoint_id="public-sha",
                  input_fingerprint="all-tensors", settings={"torch": "2.6.0", "device": "cpu"})
    key = cache.key(**inputs)
    assert cache.get(key) is None
    peaks = np.array([[100., 1.], [100., 2.]], dtype=np.float32)
    cache.put(key, peaks)
    np.testing.assert_array_equal(cache.get(key), peaks)
    assert cache.get(key).dtype == peaks.dtype
    changes = dict(candidate_id="c2", state_id="C[CH]O", formula="C2H5O", charge=-1,
                   collision_energy=21., instrument=None, checkpoint_id="another-sha",
                   input_fingerprint="different", settings={"torch": "other"})
    for field, value in changes.items():
        assert cache.key(**{**inputs, field: value}) != key
    with pytest.raises(ValueError, match="SHA256"):
        cache.get("../outside")
