# Local environment and pinned model assets

The verified environment is **Python 3.10.20 on macOS arm64, CPU**. Code requires
Python ≥3.10. Pure parsing/candidate/cache/spectrum tools do not load GLACIER;
formula parsing and masses are standalone. Chemistry requires RDKit; state
deduplication requires the pinned model featurizer and its dependencies even
without loading weights.

The working environment is now stored at the repository root in `.venv/`.
The pinned upstream checkout and weights are stored in `vendor/ms-pred/` and
`checkpoints/best.ckpt`. They were copied from the working prototype setup;
no sibling repository is needed for normal execution. Like an ordinary virtual
environment, `.venv` still uses an installed base Python and operating-system
libraries. It is a local macOS arm64 environment, not a portable Python bundle.

| Dependency | Verified version / role |
| --- | --- |
| NumPy | 2.2.6; spectra, scores, cache |
| RDKit | 2025.03.6; chemistry/descriptors; frozen exact mass convention |
| PyTorch | 2.6.0; CPU inference |
| DGL | 1.1.3; upstream graph features/random walks |
| torch-scatter | 2.1.2; upstream extension |
| pytorch-lightning | 2.5.6; unchanged checkpoint loader |
| pygmtools | 0.6.0; imported by upstream GLACIER |
| LinSATNet | 0.1.3; imported by upstream GLACIER |
| pytest | 9.1.1; regression suite |

Upstream imports also need its ordinary support packages, including SciPy,
pandas, scikit-learn, pathos, einops, omegaconf, h5py, matplotlib, seaborn,
CairoSVG, pubchempy and msbuddy. The pinned checkout's `pyproject.toml` describes
the full upstream dependency set; its training/retrieval extras are not code
extracted here. The repository-root [requirements-lock.txt](../requirements-lock.txt)
records the installed local environment. It is a snapshot of the working
macOS arm64 Python 3.10 environment, not a claim that every package has a wheel
or identical numerical behavior on other platforms. The local environment was
copied; this migration is not a test of a clean installation from the lock.

## Pinned source and checkpoint

Official upstream: `https://github.com/coleygroup/ms-pred`, commit
`67ba0f202305d4bed3134991dd4562fe1edb1763`.

The checkout already lives at `vendor/ms-pred/` in the working repository.
If restoring a fresh clone, run from the repository root:

```sh
git clone --filter=blob:none --no-checkout https://github.com/coleygroup/ms-pred.git vendor/ms-pred
git -C vendor/ms-pred checkout --detach 67ba0f202305d4bed3134991dd4562fe1edb1763
```

Use the public MassSpecGym GLACIER `best.ckpt`, linked from that pinned README:
[public checkpoint archive](https://www.dropbox.com/scl/fo/ta99j0mp1w7qzek5zvy3w/AFCLQNO8W2EP7ZDMOj9nxWI?rlkey=563zxtyvvhwfu1uyz6n10n2ui&dl=0).
This URL/provenance is taken from the local prototype audit; no new download was
needed for the migration. For a fresh clone, download it and extract `best.ckpt`
into `checkpoints/` in the repository root. Keep the upstream checkout's `.git`
metadata: the API verifies the pinned commit and tracked source cleanliness.

- `best.ckpt` SHA256:
  `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11`
- Original archive SHA256:
  `7e0fa24a19163513fe7cbeea3bf51a48cdd33bf2dbe8ac7a0457cbbf28d1258b`

`GlacierAPI()` uses the repository-local locations by default. To deliberately
select other copies, set `GLACIER_UPSTREAM=/your/path/ms-pred` and
`GLACIER_CHECKPOINT=/your/path/best.ckpt`, or pass both paths to `GlacierAPI`.
The API inserts that verified checkout's `src` into Python's import path; an
editable install is unnecessary. It rejects already imported `ms_pred` from a
different checkout. All weights stay frozen in eval mode; no retraining or
checkpoint rewriting occurs. Lightning's old-metadata upgrade occurs in memory.

## Apple Silicon

The prototype successfully used DGL 1.1.3's Apple Silicon wheel and compiled
torch-scatter 2.1.2 locally against installed torch 2.6.0. When reproducing that
build, install torch, NumPy and setuptools first, then build torch-scatter with
`--no-build-isolation`. Native extension availability depends on your Python and
platform. Upstream CPU extras are not by themselves a tested Apple Silicon lock.

The wrapper intentionally preserves CPU execution, native dtypes, seed 20260919
and four torch threads. MPS/CUDA behavior is not validated here. Switching device,
RDKit or numerical libraries can change tensors or discrete predicted fragments.
Live old/new comparisons are exact on the same environment. Frozen GLACIER
spectra permit only rtol=1e−6 / atol=1e−7 when the recorded environment differs;
fragment masks stay exact. Chemistry retained-state fingerprints stay exact and
may deliberately fail after a dependency change.

## Running locally

```sh
cd /your/research/repo/distilled_fragdata
export PYTHONPATH="$PWD"
../.venv/bin/python -m pytest tests -q
```

Optionally set `DISTILLED_PROTOTYPE_SRC` to the old prototype's `src` for live
differential checks. Its `runtime.py` expects its own original asset layout.
Frozen fixtures need no prototype checkout. Tests never download model assets or
run saved MS3 experiments.

For notebooks, select **MAGMA / Glacier (Python 3.10)** or the interpreter at
`.venv/bin/python`. From the repository root, register or refresh that kernel
after moving the repository:

```sh
.venv/bin/python -m ipykernel install --user --name magma-glacier --display-name 'MAGMA / Glacier (Python 3.10)'
```

Restart open notebook kernels after switching environments. `.venv/`,
`vendor/ms-pred/`, and `checkpoints/` are Git-ignored local assets; Git alone
does not carry them to another machine. The root `best.ckpt` compatibility link
points to `checkpoints/best.ckpt` inside this repository.
