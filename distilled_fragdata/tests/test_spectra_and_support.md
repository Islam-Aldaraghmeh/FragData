# Spectrum and scoring regressions

Purpose: compare extracted numerical behavior with frozen outputs produced by
the original `spectral_metrics.py` / `ms3_analysis.py`, plus live originals when
available. Tests cover duplicate max pooling, fixed-anchor grouping, top-k,
normalization, empty states, ensemble support above 100 peaks, matching, cosine,
entropy/support, the high-level flow, link guards, and an actual ionic ensemble.

Inputs: `fixtures/spectra.json` (source hashes, inputs, full expected outputs),
and the multistate C2H8NO/NCCO fixture for the optional actual-model differential. Assertions
use exact array/dict equality against the source outputs. The small analytic
uniform-ensemble check uses numerical tolerance. No research cohort is scored.

Run: `python -m pytest tests/test_spectra_and_support.py -q`. This is new test
code; scientific oracles are the old functions named above and direct-input
prediction plus the `IonicPredictor.predict` ensemble rule. The live ensemble
test explicitly skips without the prototype/model; frozen tests still run.
Frozen predictions in the high-level flow isolate scoring; they are not evidence
that candidate ranking generalizes. Actual checkpoint inference is separately
tested in this file and `test_glacier_api.py`.
