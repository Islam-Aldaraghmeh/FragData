# Isolated Glacier recovery experiment

For the full diagnosis, algorithms, validation evidence, and limitations, read
[Missing Glacier fragment spectra: diagnosis, recovery, and validation](../GLACIER_RECOVERY_EXPLAINED.md).

The recovery switches are now integrated into the main notebook loader:

```python
model = load_glacier_model(aromatic_fix=True, radical_fix=True)
predicted_bags = predict_candidate_bags(model, bags, collision_energy=30)
print(model.recovery_settings)
```

Both keyword arguments default to `False` and are independent. The same keywords
are available on `GlacierAPI(...)`. Use all four combinations for ablation;
keep bags, energy, `max_states`, and checkpoint fixed. The properties are read-only:
create a model for each configuration. Successful baseline candidates retain
identical states and predictions. `radical_fix` uses direct +1 one-radical ions;
it does not enable the older neutral-reference experiment. The adapter does
not chain aromatic recovery after a failed radical search.

[ablation_regression.json](ablation_regression.json) records a small live check
of all four configurations on nine baseline candidates, one aromatic failure,
and two C6H9N3O3S candidates. All nine baseline spectra matched previously saved
outputs exactly in every configuration. Store `model.recovery_settings` with
run results and cache settings; checkpoint identity itself stays unchanged.

The tested algorithms now live in `distilled_fragdata/src/chemistry/`.
The lab module paths remain compatibility exports/wrappers, with standalone
experiments explicitly using `prepare_baseline()` before their own opt-in rules.
The original source manifest and saved results remain historical records.
[compatible_source_manifest.json](compatible_source_manifest.json) records the
regression-tested integration revision; `verify_production_files()` accepts
exact matches to either recorded revision and still rejects arbitrary changes.
Restart a running notebook kernel to load the updated model and adapter modules.

The newer [Experiment_3_Direct_Radical.ipynb](../Experiment_3_Direct_Radical.ipynb)
tests a separate direct radical-cation route in [radical.py](radical.py). It
extends local states to `(charge, H, radical_electrons)`, preserves the exact
heavy graph and formula, and requires net charge +1 and exactly one radical
electron. It activates only after the baseline's odd-electron parity rejection
and explicit `allow_radical=True`; it never calls neutral-reference or aromatic
recovery. The small audit uses the saved 74 odd-electron candidates, the two
actual C6H9N3O3S examples, and nine exact baseline regressions. See
[radical_results.json](radical_results.json) for its results and [test_radical.py](test_radical.py)
for focused tests. The saved run represents **74/74** odd-electron candidates
directly, with **0 failures** in that set, **9/9 exact baseline regressions**,
and **19 focused tests passed**. The unchanged model builder has no explicit radical-count
feature, so representability does not establish radical-spectrum accuracy.

The remainder of this README describes the earlier neutral-reference and
aromatic-recovery experiment.

Open [Experiment_2_Glacier_Recovery.ipynb](../Experiment_2_Glacier_Recovery.ipynb)
with the **MAGMA / Glacier (Python 3.10)** kernel, backed by this repository's
`.venv/bin/python`. The pinned model source is in `vendor/ms-pred/` and the
checkpoint is in `checkpoints/best.ckpt`; no sibling repository is required.
The original experiment used the earlier sibling environment. Saved reports
and source manifests retain that historical provenance; current commands and
loaders use the local copies. See the [development quickstart](../README.md).

This experiment investigates
missing fragment predictions without editing `Experiment_1.ipynb`, the
production adapter, or the frozen checkpoint. It uses a saved candidate pilot
and notebook-definition snapshot so work in the original notebook can continue.

The safe default is `allow_experimental=False`: candidates that already work
use the original adapter, and unsupported candidates remain unavailable.
Experimental recovery requires an explicit opt-in and reports its route in each
result. Existing valid candidates always retain their baseline route.

The pilot contains **84 candidates** from seven same-formula bags associated
with parent scan 297, evaluated at model collision energy 30. The preparation
audit recovered all 75 previously unavailable candidates: 74 through neutral
references and one through aromatic representation repair. The nine baseline
predictions matched the original adapter exactly (`numpy.array_equal`). These
are coverage and regression results, not evidence of unchanged accuracy for
the newly available predictions.

The two experimental routes have different meanings:

- **`neutral_reference_[M]+`** enumerates a neutral, closed-shell reference
  for an odd-electron positive ion, then uses Glacier's existing `[M]+` adduct.
  It preserves the candidate's exact elemental formula, total hydrogen count,
  heavy-atom connectivity, bond orders, aromatic labels, and ion mass. It does
  not add an arbitrary hydrogen or pretend that a neutral reference is the
  ionic molecule. Local hydrogen assignments and charge features can change,
  so the model receives a different representation whose accuracy requires
  validation.
- **`experimental_aromatic_kekule_reference`** permits RDKit to express
  inherited aromatic edges as single/double bonds when sanitizing a valid ion.
  The pilot example becomes `[C+]1=CC=CC=C1`. This preserves connectivity,
  formula, total charge, and ion mass, while explicitly changing the aromatic
  representation. It forbids changes to nonaromatic bonds and rejects radicals.

The full recovery policy is deliberately permissive: 17 neutral-reference
candidates need no formal charges, while 57 require separated positive/negative
formal charges (net zero). These are valid within the declared rules, not proven
likely fragment structures. Across all 84 candidates, 40 searches hit the
512-state chemical cap and 64 cap retained feature states at four. Keeping only
the nine baseline cases, one aromatic repair, and 17 neutral references without
charge separation gives a conservative **27-candidate inspection subset**;
its new predictions still require accuracy validation.

Neither fallback modifies the input candidate. Both retain bounded searches,
sanitization, formula/mass checks, and the same pretrained featurizer and
prediction postprocessing. **22 invariant tests passed**, covering conserved
chemistry, invalid-input rejection, and the aromatic-repair boundaries.

The recovered bags have **no linked experimental MS3 observations** in this
pilot. The observed MS3 children belong to the baseline bags at m/z 184.087005
and 185.070999. Consequently this pilot cannot measure recovered prediction
accuracy against experimental MS3. The inspected local MSG `labels.tsv`
contains 119,029 `[M+H]+` records and no `[M]+` records; this describes that
local label file, not proven membership of the checkpoint's exact training
set. An available `[M]+` input category alone does not establish training
coverage or equivalent accuracy.

Run this from the repository root to reproduce preparation, predict every
available candidate, and compare all nine baseline predictions:

```bash
.venv/bin/python -m recovery_lab.experiment
```

Run the invariant checks with:

```bash
.venv/bin/python -m pytest recovery_lab/test_recovery.py -q
```

Use the helper independently after initializing the model:

```python
from recovery_lab.experiment import (
    load_model, load_pilot, candidate_from_pilot, predict_candidate,
)

model = load_model()
pilot = load_pilot()
row = next(row for row in pilot["candidates"] if row["baseline_reason"])
candidate = candidate_from_pilot(row)

# Default behavior preserves the original availability policy.
baseline = predict_candidate(model, candidate, collision_energy=30)

# Deliberately opt in to the experiment and retain its provenance.
trial = predict_candidate(
    model, candidate, collision_energy=30, allow_experimental=True,
)
trial["route"], trial["experimental"], trial["spectrum"]
```

The implementation and evidence are kept together:

| File | Purpose |
| --- | --- |
| [experiment.py](experiment.py) | Baseline-first wrapper, prediction runner, and regression audit |
| [neutral_reference.py](neutral_reference.py) | Exact-formula neutral-reference search |
| [aromatic_recovery.py](aromatic_recovery.py) | Explicit aromatic representation fallback |
| [test_recovery.py](test_recovery.py) | Chemistry and input invariants |
| [baseline_snapshot.py](baseline_snapshot.py) | Frozen original notebook definitions; no demo cells execute |
| [pilot_candidates.json](pilot_candidates.json) | Saved 84-candidate pilot and MS3 availability |
| [results.json](results.json) | Most recent preparation, forward-prediction, and regression results |
| [snapshot_manifest.json](snapshot_manifest.json) | Provenance and hashes of monitored production files |

`results.json` records the number of recovered forward predictions actually
tested. The full runner uses every available candidate by default;
`run_pilot(sample_per_route=2)` provides a smaller forward pass when desired.
Increasing the number of predictions checked establishes execution coverage,
not chemical correctness or predictive accuracy. Promotion to the working
notebook should depend on a separate held-out accuracy evaluation.

The environment and assets now resolve inside this repository. Historical
manifests remain unchanged; [local_source_manifest.json](local_source_manifest.json)
records the tested path-resolution revision. The source integrity check also
accepts that exact revision. [local_runtime_report.json](local_runtime_report.json)
records local module/asset locations, exact reproduction of 84 MAGMa candidates,
and nine unchanged baseline spectra with sibling-file access blocked during
the check. Use the current local kernel after restarting any older session.
