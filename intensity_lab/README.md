# intensity_lab

Intensity prediction for pre-assigned MAGMa candidate bags, as used by
[`Experiment_1_agentic.ipynb`](../Experiment_1_agentic.ipynb).

| module | purpose |
| --- | --- |
| `raw.py` | Reads cached `build_intensity_dataset` pickles without their dataclasses (any schema version), joins each with its MSnLib record by USI hash, and resolves scans annotated with several structures. `find_spectrum(usi)` returns one `Spectrum`. |
| `chemistry.py` | Root graphs in **MAGMa's own atom order** (`FragmentEngine` canonical SMILES) and `verify_alignment`, which checks every candidate's atoms against its formula. |
| `features.py` | 88 candidate features (formula, H-shift, broken-bond types, charge-site chemistry, neutral losses, mass error, bag context, CE) plus graph arrays, stored as flat CSR-style arrays. `featurize_candidate_bags` does the same for fresh `get_candidate_bags` output. |
| `splits.py` | Molecule-disjoint and scaffold-disjoint splits, leakage audit, nearest-training-neighbour similarity. |
| `metrics.py` | Per-spectrum cosine (supported / full / sqrt), top-k, rank correlation, effective peak counts, molecule-cluster bootstrap, replicate and other-CE reference cosines. |
| `baselines.py` | Uniform, candidate count, fewest broken bonds, LightGBM on bag aggregates. |
| `model.py` | Candidate scorers (linear / MLP / root-graph GNN) with sum-of-abundance (`logsumexp`) peak aggregation and per-spectrum softmax; KL or cosine losses. |
| `train.py` | Training with validation early stopping; runs cached by config hash in `results/experiment_1_agentic/runs/`. |
| `experiments.py` | Registry of every configuration. `python -m intensity_lab.experiments all` pre-trains them. |
| `diagnostics.py` | Overfit test, permutation importance, CE counterfactuals, precursor calibration, sharpening curve, error slices. |
| `inference.py` | Ensemble save/load and `predict_spectrum(predictor, spectrum, bags)`. |
| `plots.py` | Figures. |

```python
from magma_workflow.candidates import get_candidate_bags
from intensity_lab.inference import load_predictor, predict_spectrum

predictor = load_predictor()   # results/experiment_1_agentic/final_model (written by the notebook)
bags = get_candidate_bags(spectrum, spectra=[], peak_assignment="nearest", include_ms3=False)
intensities = predict_spectrum(predictor, spectrum, bags)   # per spectrum.mz, base peak = 1, 0 = no candidate
```

Caches (safe to delete; rebuilt automatically): `data/agentic_cache/` (raw join, MAGMa engine
SMILES, features, splits) and `results/experiment_1_agentic/runs/` (trained runs).
