# Notebook lag after running all cells

The live Experiment_1 kernel was using about one CPU core while converting a
large list to text. Native samples show `PyObject_Str → list_repr → array_repr`;
the generated `Spectrum` dataclass representation included all peaks and metadata.
The loaded collection contained 346,248 spectra across 9,544 SMILES groups.
System memory pressure was healthy, and the sampled work was string formatting,
not Glacier inference.

Both installed IPython inspection and the VS Code variable helper build a full
string before truncating the displayed preview. This explains how inspecting a
large dataset can occupy the kernel and queue even a subsequent `print()`.
The exact UI action that submitted the original request was not captured.

The fix in [Spectrum_Classes.py](../../Spectrum_Classes.py) gives spectra and
forests compact representations. `SpectrumCollection` and `SpectraBySmiles`
preserve list/dict operations while reporting only their sizes in `repr`, `str`,
and IPython pretty display. No spectrum data or model inputs are truncated.

[Experiment_1.ipynb](../../Experiment_1.ipynb) now loads into these containers,
filters records before constructing discarded spectra, and checks compatibility
using the first qualifying anchor record instead of loading the entire anchor.

The running session was repaired too. Two interrupts cleared pending formatting
work; the first patch request waited behind another preview and its client timed
out, but the queued patch subsequently executed. A separate client verified its
result in [live_probe.json](live_probe.json). The repair preserved every spectrum
object, the `spectra is raw_spectra` alias, and the loaded model identity.
The model was still a `GlacierAPI` in the subsequent check.

Kernel CPU fell from about 100% to 0%. A captured `print("ok")` completed in
4.4 microseconds; the complete client request took 3.2 milliseconds, including
communication and result inspection. See [print_probe.json](print_probe.json)
and the before/after native samples. No kernel restart or model reload occurred.

Validation: 43 tests passed across the preview/loader regressions and existing
notebook MS3/gallery tests. The preview tests use nested objects that raise when
rendered, proving collection previews do not traverse data; they do not rely on
flaky timing thresholds.

```bash
.venv/bin/python -m pytest tests/test_notebook_previews.py tests/test_experiment_1_ms3.py -q
```

Other old sessions need a restart to load the updated classes, or the included
`apply_preview_fix.py` can patch an available kernel using its connection file.
If a preview is already blocking it, close the Variables panel and interrupt
first. That helper never interrupts or restarts kernels itself.
