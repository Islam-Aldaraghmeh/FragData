# Fragment-to-GLACIER adapter

**Purpose and API.** `FragmentGlacierAdapter(glacier=None, max_states=4, **caps)`
enumerates a small target-independent ensemble. `.prepare(candidate,
collision_energy, instrument="Orbitrap")` returns the full enumeration audit plus
`inputs`, one primitive-record GLACIER argument dictionary per retained state.
`.predict(...)` returns the equally weighted, normalized ensemble as a `Spectrum`.
Prediction requires a configured `GlacierAPI`. Preparing tensors/feature hashes
also requires its pinned upstream featurizer, but no checkpoint forward pass.

Accepted candidates are `FragmentCandidate` objects, legacy prototype dictionaries
containing `stripped`, or exact stripped records. Each record contains
`atoms=[{element, aromatic}]`, `bonds=[{i, j, order, aromatic}]`, `ionic_formula`,
`charge`, and `precursor_mz`. Graph order is preserved for existing stripped inputs.
`stripped_from_smiles` is available when a source SMILES supplies the heavy graph.

```python
from src.chemistry.fragment_adapter import FragmentGlacierAdapter, stripped_from_smiles
from src.chemistry.formula import ionic_mass

record = stripped_from_smiles("NCCO", "C2H8NO", 1, ionic_mass("C2H8NO"))
adapter = FragmentGlacierAdapter(glacier, max_states=4)
prepared = adapter.prepare(record, collision_energy=20.0)
spectrum = adapter.predict(record, collision_energy=20.0)
```

**Extraction.** Enumeration, graph handling, rank and `project_neutral_references`
come directly from `fragment_glacier_pretrained_adapter/src/microstates.py`.
The same state-SMILES reparse used by `direct_input.records_from_smiles` supplies
model inputs. `.predict` follows `run_ms3_exploratory.IonicPredictor.predict`:
`run_retention.clean_raw`, `spectral_metrics.prepare_spectrum`, TIC normalization,
then `mean_normalized_spectrum(prepared=True)`. The model result's raw `spec`
tensor is converted to a CPU NumPy array, as in `run_retention.Predictor.predict`.
The same finite-output assertion runs before raw-row cleaning.

**Assumptions and scientific hazards.** Charge +1, connected, closed-shell ions
are supported. Formula, charge, heavy graph and aromaticity must be conserved;
formula m/z must match within `max(0.005 Da, 20 ppm)`. `[M]+` supplies the electron
mass correction without adding a proton. CE is already the model's scalar: the
adapter never guesses NCE or applies a second conversion. Each single-state
spectrum max-pools coincident peaks, retains the prototype's threshold/top-100
filter, then normalizes. The ensemble sums normalized state contributions and
retains its full support. Read the returned `metadata` for truncation.

`project_neutral_references(ionic_smiles)` is an optional representation diagnostic:
it removes one H+ at each valid site and pairs the result with `[M+H]+`. It is
never called by the main adapter. A zero-net-charge reference can still contain
separated charges; permanent cations are not silently neutralized.

**Intentional changes.** The requested default retained-state cap is 4 instead
of 16; pass the same explicit cap to recover prototype behavior. The new public
wrapper rejects nonfinite CE values at its boundary. This is input validation
for the new API, with no alteration of scientific calculations for valid inputs.
No target spectrum is accepted or used to choose states.
