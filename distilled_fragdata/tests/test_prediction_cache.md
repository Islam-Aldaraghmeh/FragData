# Cache regression

Purpose: verify exact dtype/array NPZ roundtrip, cache misses, and identity
separation across candidate, state, formula, charge, CE, instrument, checkpoint,
input tensors and settings. Inputs are tiny raw duplicate prediction rows and a
temporary directory; output is test assertions. No prototype cache is modified.

Run: `python -m pytest tests/test_prediction_cache.py -q`. New test for the
deliberately simplified `PredictionCache`; the original key concept comes from
`run_retention.py::Predictor.predict`. It tests the cache contract, not old cache
filename compatibility. Assumption: the caller supplies a complete fingerprint;
tests cannot validate fingerprints invented outside the model wrapper.
