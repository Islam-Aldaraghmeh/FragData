# MAGMA Research

MAGMa fragment candidates for MSnLib spectra, and an intensity-only model that
predicts how a spectrum's intensity is distributed over those candidates.

## Layout

| Path | Purpose |
| --- | --- |
| `magma_workflow/` | MSnLib loading, vendored MAGMa engine, candidate bags, MS3 linkage, intensity dataset builder ([README](magma_workflow/README.md)) |
| `intensity_lab/` | Intensity model: features, splits, baselines, models, training, metrics, inference ([README](intensity_lab/README.md)) |
| `Experiment_1_agentic.ipynb` | Main notebook; generated from `intensity_lab/tools/notebook_cells.py` |
| `msnlib/` | MSnLib JSON-lines datasets (Git-ignored) |
| `data/intensity_dataset_v1/`, `data/intensity_parallel_test/` | Cached MAGMa intensity examples (inputs to `intensity_lab.raw`) |
| `data/agentic_cache/` | Derived caches (raw join, features, splits); rebuilt automatically if deleted |
| `results/experiment_1_agentic/` | Trained runs, figures, final model |
| `Docs/`, `MSnLib_stats_report_files/`, `presentation_visuals/` | MSnLib statistics report and figures |
| `report.md`, `continue.txt` | Findings so far and the remaining steps |
| `tests/` | MAGMa nearest-peak assignment tests |

## Environment

Use the interpreter at `.venv/bin/python` (Python 3.10, macOS arm64). Package
versions are listed in `requirements-lock.txt`.

```sh
.venv/bin/python -m pytest tests -q
```
