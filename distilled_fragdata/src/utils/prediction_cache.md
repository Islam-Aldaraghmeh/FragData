# Prediction cache

Purpose: store one raw predicted spectrum per complete input identity, without
stage, manifest or resume machinery. `PredictionCache(directory)` creates a
directory; `key(...)` hashes canonical JSON; `get(key)` returns the stored N×2
array (original dtype) or None; `put(key, peaks)` atomically replaces one NPZ.
NumPy pickle loading is disabled. Corrupt files raise; callers decide whether to
delete and recompute. There is no implicit eviction or locking across workers.

```python
from src.utils.prediction_cache import PredictionCache
cache = PredictionCache("predictions")
key = cache.key(candidate_id, state_id, formula, charge, ce, instrument,
                checkpoint_id, input_fingerprint=all_tensor_hash,
                settings={"torch": "2.6.0", "device": "cpu", "dtype": "native"})
raw = cache.get(key)
if raw is None:
    raw = run_one_prediction()
    cache.put(key, raw)
```

Source concept: `run_retention.py::Predictor.predict` input-tensor/checkpoint/runtime
key and NPZ `spec` payload; `ms3_resume_io.py` atomic cache replacement. This
small I/O implementation is intentionally new, not a claim of old cache-format
equivalence. Old cache directories are not migrated or read. Numerical arrays
are stored without cleaning, pooling or normalization.

Scientific hazards: candidate/state labels alone do not identify model inputs.
The required `input_fingerprint` must cover every actual model tensor (use
`models.glacier_api.tensor_fingerprint`), and checkpoint identity should contain
its SHA256 and pinned upstream commit. Include numerical runtime/settings when
sharing a directory across environments. Cache raw states individually; if
storing ensembles, include retained state identities, caps and averaging policy
in settings. Targets must never enter this key or select states. Integration is
explicit and optional; `FragmentGlacierAdapter` does not hide cache behavior.
Roundtrip/input-separation regression: `tests/test_prediction_cache.py`.
