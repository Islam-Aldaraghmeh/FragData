# Shared experiment workflow

`Experiment_1.ipynb` imports these implementations instead of embedding copies:

| Module | Purpose |
| --- | --- |
| `dataset.py` | MSnLib loading and compact spectrum containers |
| `candidates.py` | MAGMa generation, formula filtering and MS3 linkage |
| `plotting.py` | Fragment galleries, spectrum comparisons, mirror plots and mixtures |
| `glacier.py` | Optional existing Glacier loading and prediction |

Candidate generation has no Glacier or ionic-state enumeration dependency.
Importing the package, dataset or candidates does not import Torch. Calling
MAGMa still uses its pinned upstream checkout and that checkout's dependencies.

```python
from magma_workflow.dataset import parse_dataset
from magma_workflow.candidates import get_candidate_bags, set_spectrum_source

raw_spectra, spectra_by_smiles = parse_dataset("msnlib/20241003_enamdisc_pos_msn.json")
set_spectrum_source(raw_spectra, spectra_by_smiles)  # optional notebook default
bags = get_candidate_bags(spectrum)

# Intensity-only experiments need no MS3 observations or global dataset context:
bags = get_candidate_bags(spectrum, spectra=[])
```

An explicit `spectra=` or `forest=` overrides notebook context.
`set_spectrum_source()` clears the configured source. Calls preserve the existing
bag structure, candidate ordering, formulas, mass matching and plotting defaults.
Changing a module in a running notebook requires reloading the module or
restarting the kernel; simply rerunning an import does not reload Python source.

RDKit heavy-graph canonicalization lives in
`distilled_fragdata/src/chemistry/heavy_graph.py`, shared with the existing adapter.
The frozen `recovery_lab/baseline_snapshot.py` remains an archival baseline.
