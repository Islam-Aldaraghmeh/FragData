# Spectrum metrics

Purpose: standalone prototype cosine, entropy similarity, target intensity
recall and predicted intensity precision. `score_spectra(predicted, target,
prepared_predicted=False, prepared_target=False, atol=0.005, ppm=20.0)` consumes
N×2 peaks and returns those four scores plus matched/predicted/target counts.
Small named wrappers return an individual score.

Cosine uses untransformed intensities. Entropy similarity is 1−JSD/ln(2) on the
L1-normalized matched union, including unmatched peaks; there is no extra
entropy weighting. Recall is matched target TIC / all target TIC, precision is
matched predicted TIC / all predicted TIC. Empty prediction or target gives zero
similarity and coverage, including two empty spectra. Tolerance is explicitly
max(atol, larger mass × ppm × 1e−6); matching is the spectrum module's exact
greedy rule.

```python
from src.spectra.metrics import score_spectra
scores = score_spectra(ensemble.as_array(), prepared_target,
                      prepared_predicted=True, prepared_target=True,
                      atol=0.005, ppm=20.0)
```

Source: `fragment_glacier_pretrained_adapter/src/spectral_metrics.py::score_spectra`,
unchanged function body. Named wrappers and the comment reserving a place for
future custom metrics are new; no scientific behavior changed. Bootstrap and
cohort statistics were deliberately omitted.

Scientific hazards: default preparation caps both inputs at 100 peaks. The
prototype MS3 workflow instead prepares its experimental target with no cutoff
or top-k, then passes both `prepared_*` flags true. The experiment module retains
that workflow. These are the prototype's sparse metrics, not the paper's full
native binned evaluation or calibrated candidate probabilities. Fixed example
scores and live source equivalence are tested in `tests/test_spectra_and_support.py`.
