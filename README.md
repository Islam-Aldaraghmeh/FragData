# MAGMA Research

MAGMa fragment candidate generation, frozen GLACIER prediction, and MS2/MS3
inspection notebooks, plus a small intensity-only learning experiment. The working Python environment, pinned model source,
checkpoint, and research data live inside this repository. Normal use does not
require `MAGMA_Research_Agentic` or another sibling checkout.

## Development quickstart

Open the repository folder in your notebook editor and choose the interpreter
at `.venv/bin/python` or the **MAGMA / Glacier (Python 3.10)** kernel. To register
or refresh the kernel, run from this directory:

```sh
.venv/bin/python -m ipykernel install --user --name magma-glacier --display-name 'MAGMA / Glacier (Python 3.10)'
```

Restart open kernels after switching interpreters. The main workflow is
[Experiment_1.ipynb](Experiment_1.ipynb). Its loader resolves the local model
assets automatically:

```python
model = load_glacier_model(aromatic_fix=True, radical_fix=True)
predicted_spectra = predict_candidate_bags(model, bags, collision_energy=30)
```

Both fixes default to `False` and can be enabled independently for ablation.
Keep candidate bags, collision energy, `max_states`, and checkpoint fixed across
comparisons; record `model.recovery_settings` with the results. Representation
recovery does not by itself establish prediction accuracy.

[Experiment_2.ipynb](Experiment_2.ipynb) trains a **7,139-parameter MLP** to predict
intensities at MAGMa-supported observed peak masses, without Glacier or ionic-state
enumeration. Each exact fragment identity goes to its nearest matching centroid;
alternative fragment scores are combined with **max** per peak. A small
regularized fingerprint branch and structural descriptors predict fragment-specific
coefficients of collision energy, so energy explicitly changes candidate scores.

The revised experiment samples 1,500 fresh molecules, split into 1,050 training,
225 validation, and 225 test groups. All original 300 molecules and equivalent
InChI connectivity identities are excluded. It compares the new scorer, a
descriptor-only scorer, the frozen old model, and equal intensity on the same
fresh held-out spectra. The notebook loads saved metrics and figures by default,
without loading the full library or feature cache. Set `RETRAIN=True` to repeat
training, or run:

```sh
.venv/bin/python train_intensity_v2.py --molecules 1500 --seed 20260923 --workers 2 --epochs 100 --patience 15
```

MAGMa features are cached locally; validation alone selects checkpoints.
Read the [updated results summary](results/experiment_2_v2/SUMMARY.md) for metrics,
CE20/CE60 sensitivity, effective peak counts, strongest-peak accuracy, and plots.
The [original summary](results/experiment_2/SUMMARY.md) and
[error analysis](results/experiment_2/ERROR_ANALYSIS.md) remain available with
the original checkpoint. Their earlier test cohort differs from the fresh cohort.

Inference requires only the trained model, precursor, energy and candidate bags:

```python
from magma_workflow.candidates import get_candidate_bags
from intensity_model_v2 import load_model, predict_spectrum

bags = get_candidate_bags(spectrum, spectra=[], peak_assignment="nearest")
model = load_model("results/experiment_2_v2/model.pt")
predicted = predict_spectrum(model, spectrum.smiles,
                             spectrum.metadata["collision_energy"], bags)
```

`predict_spectrum` also enforces nearest assignment when given existing candidate
bags. It preserves the supplied m/z order and returns base-peak-normalized
intensities, including zero for unsupported peaks. Experimental intensities are
training targets only; candidate generation and scoring do not read them.

Shared Experiment 1 implementations now live in [magma_workflow](magma_workflow/README.md).
Import its `dataset`, `candidates`, `plotting` and optional `glacier` modules directly;
notebooks no longer need to execute one another's definition cells.

Run the regression suites from this directory:

```sh
.venv/bin/python -m pytest tests recovery_lab distilled_fragdata/tests -q
```

## Local layout

| Path | Purpose |
| --- | --- |
| `.venv/` | Working Python 3.10 environment and notebook kernel dependencies. |
| `requirements-lock.txt` | Installed package snapshot for the working macOS arm64 environment. |
| `vendor/ms-pred/` | Pinned GLACIER upstream checkout, including Git metadata needed for verification. |
| `checkpoints/best.ckpt` | Verified frozen model weights. |
| `best.ckpt` | Compatibility symlink to `checkpoints/best.ckpt` within this repository. |
| `msnlib/` | Local MSnLib datasets. |
| `distilled_fragdata/` | Reusable chemistry, model, spectra, linkage, and scoring code. |
| `magma_workflow/` | Shared dataset, MAGMa candidate and plotting functions extracted from Experiment 1. |
| `intensity_model_v2.py` | Small CE-conditioned MLP, structural/mass-error features, nearest assignment and inference. |
| `intensity_data_v2.py` | Expanded fresh sample, molecule/connectivity split audits, two-worker MAGMa preparation and local cache. |
| `train_intensity_v2.py` | Cosine training, descriptor-only control, frozen-old-model comparisons and CE diagnostics. |
| `results/experiment_2_v2/` | Revised checkpoints, metrics, manifest, summary and comparison figures. |
| `intensity_model.py`, `intensity_data.py`, `train_intensity.py` | Original experiment retained for reproducibility; shared normalization, max aggregation and plotting helpers. |
| `results/experiment_2/` | Preserved original checkpoint, original-cohort results and error analysis. |
| `Experiment_2_export.py` | Preserved historical Python notebook export, renamed from `code.py` to avoid shadowing Python's standard-library `code` module. |
| `recovery_lab/` | Recovery experiments, focused tests, and saved pilot evidence. |
| `Experiment_2_Glacier_Recovery.ipynb` | Earlier neutral-reference/aromatic recovery investigation. |
| `Experiment_3_Direct_Radical.ipynb` | Direct radical-cation representation experiment. |

`GlacierAPI()` locates the pinned checkout and checkpoint relative to this
repository, independently of the working directory. Explicit constructor paths
and `GLACIER_UPSTREAM` / `GLACIER_CHECKPOINT` are optional overrides. Remove stale
overrides from a shell or editor configuration when using the local defaults.
`DISTILLED_PROTOTYPE_SRC` is only an opt-in comparison against historical source;
the ordinary and frozen-reference tests do not require it.

## Environment and reproducibility

The environment was copied from the working **macOS arm64, Python 3.10.20, CPU**
setup. The virtual environment uses an installed base Python and system
libraries, as virtual environments normally do; it is not a standalone Python
distribution. Moving to another machine or architecture requires recreating
the environment and rechecking predictions. Native dependencies such as DGL
and torch-scatter need platform-compatible builds.

The environment, upstream checkout, checkpoints, and datasets are Git-ignored
local assets. They are present in this working directory but are not supplied
by a fresh Git clone. Preserve or restore them separately. Use
[the environment and asset instructions](distilled_fragdata/requirements.md)
for exact versions, source commit, checkpoint hash, and restoration details.

Saved experiment reports and historical source-attribution documents may still
name the original prototype or its old paths. Those entries record provenance;
current execution uses this repository's local assets. See
[the recovery explanation](GLACIER_RECOVERY_EXPLAINED.md) and
[recovery lab notes](recovery_lab/README.md) for the experimental methods and
their accuracy limitations.
