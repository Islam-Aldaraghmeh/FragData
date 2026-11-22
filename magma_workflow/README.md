# magma_workflow

MSnLib loading, MAGMa candidate generation and the cached intensity dataset.

| Module | Purpose |
| --- | --- |
| `dataset.py` | MSnLib loading (`parse_dataset`), compact `Spectrum` containers, and `SpectrumForest` / `parse_spectrum_forest` for MSn parent-peak links |
| `magma/` | MAGMa fragmentation engine (`FragmentEngine`) and the part of ms-pred's `chem_utils` it uses, vendored from ms-pred commit `67ba0f2` (MIT) |
| `chemistry.py` | Elemental formulas, exact/ionic masses, canonical heavy-atom graphs |
| `candidates.py` | `FragmentCandidate`, `get_candidate_bags` (MAGMa → per-peak candidate bags), nearest-peak assignment, MS3 linkage |
| `intensity_dataset.py` | Parallel, cached build of per-spectrum intensity training examples (`data/intensity_dataset*/examples/*.pt`) |
| `plotting.py` | `plot_smiles` and `plot_fragment_bags` (candidate fragment drawings) |

```python
from magma_workflow.dataset import parse_dataset
from magma_workflow.candidates import get_candidate_bags, set_spectrum_source

spectra, spectra_by_smiles = parse_dataset("msnlib/20241003_enamdisc_pos_msn.json")
set_spectrum_source(spectra, spectra_by_smiles)  # optional default for MS3 lookup
bags = get_candidate_bags(spectrum)              # [(peak_mz, [FragmentCandidate], [MS3 Spectrum])]

# Intensity-only use needs no MS3 observations:
bags = get_candidate_bags(spectrum, spectra=[], peak_assignment="nearest", include_ms3=False)
```

An explicit `spectra=` or `forest=` overrides the configured source;
`set_spectrum_source()` with no arguments clears it.

Do not rename the classes in `intensity_dataset.py`: the cached `.pt` examples
are pickles that refer to them by module path (`intensity_lab.raw` can read
them without the classes, but `IntensityDataset` cannot).
