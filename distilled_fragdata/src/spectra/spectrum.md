# Spectrum operations

Purpose: retain the prototype's sparse postprocessing exactly, with an optional
`Spectrum(mz, intensity)` container. Array functions consume and return N×2
NumPy arrays; the container exposes `from_peaks`, `as_array`, `normalized`,
`filtered`, `merged`, `match` and `average`.

`clean_raw` filters nonfinite, nonpositive-mass and nonpositive-intensity raw
GLACIER rows. `prepare_spectrum` merges before base-peak thresholding (0.001)
and top-100 filtering. Predicted intensities use **max** pooling within a
prediction; experimental/ensemble merging uses **sum**. Both use the original
intensity-weighted centroid, even with max pooling. The default tolerance is
max(0.005 Da, 20 ppm of the larger mass). Groups use a fixed lowest-mass anchor,
so close masses do not merge transitively. Top-k ties use increasing mass.

`normalize_spectrum` uses L1 normalization. `mean_normalized_spectrum` prepares
each state, normalizes each separately, gives each state equal weight, sums
matching support, and normalizes the final ensemble. It does not truncate the
ensemble to 100 peaks. Empty states contribute zero before final normalization;
an entirely empty ensemble remains empty. `match_peaks` greedily matches
globally closest admissible unused peaks, with deterministic ties; it is not a
maximum-cardinality assignment.

```python
from src.spectra.spectrum import Spectrum, clean_raw, prepare_spectrum
predicted = Spectrum.from_peaks(prepare_spectrum(clean_raw(raw_prediction)))
ensemble = Spectrum.average([predicted, another_prediction], prepared=True)
```

Sources: `spectral_metrics.py` functions `_array`, `mass_tolerance`,
`merge_peaks`, `prepare_spectrum`, `normalize_spectrum`,
`mean_normalized_spectrum`, `match_peaks`, and `run_retention.py::clean_raw`.
All refer to `fragment_glacier_pretrained_adapter/src/`. These function bodies
are preserved. The container is a new convenience API; no scientific change.

Scientific hazards: never sum duplicate GLACIER rows; never normalize raw states
together before averaging; never accidentally apply top-100 filtering again to
the ensemble. `_array` rejects invalid masses/nonfinite values; only the explicit
raw-model cleaning helper silently removes those raw output rows, as before.
`merged` defaults to sum, so predicted raw rows should use `filtered` or explicitly
request `aggregation="max"`. Frozen/live regression is in
`tests/test_spectra_and_support.py`.
