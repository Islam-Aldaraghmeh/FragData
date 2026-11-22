# Intensity predictor for MAGMa candidates: what I found, did and achieved

*Status: work paused on request (2026-10-20). All numbers below are on the **validation** set.
The held-out **test** set has deliberately not been used yet. It is evaluated once, at the end
(see `continue.txt`).*

## The short version

- **Two data bugs were silently hurting your models.** The biggest one: for **52% of spectra**, the
  atoms each candidate fragment pointed to were the **wrong atoms** of the molecule graph. Your GNN
  was pooling noise for half the data. That is fixed and verified (0 of 3.47M candidates misaligned now).
- **Important information was being thrown away before the model saw it**, mainly *which bonds
  break* to make a fragment. I added 88 chemistry features computed from the corrected graph.
- **The training pipeline was a hidden bottleneck.** It used one pickle load per spectrum, Python
  loops per spectrum in the loss, and trained on 100 spectra. Now the full 24k training spectra take
  ~10–20 s per epoch.
- **You now have reference points to judge numbers.** Trivial baselines (the "floor"), agreement
  between repeated measurements (the "ceiling"), and debugging checks that prove the model learns
  real signal.
- **Best model so far:** a small GNN on the *corrected* graph plus the new features. It reaches
  **0.781** mean cosine on validation, versus **0.484** for predicting equal intensities and
  **0.660** for a model with your original inputs and the bug.

## 1. What the task is (in plain terms)

For each MS2 spectrum, MAGMa has already proposed candidate fragments for each observed peak (a
"bag"). We know *where* the peaks are. The model has to say *how tall* each peak is, given the
molecule, its candidate fragments and the collision energy (CE). The score is the **cosine
similarity** between predicted and measured peak heights: 1 = identical shape, 0 = unrelated.

Data used: 29,819 spectra of 9,527 molecules (Enamine library, CE 15–75), about 621k peaks with
candidates and 3.47M candidate fragments. Splits are by molecule: every spectrum of a molecule, at
every CE, is in the same split (train 80% / validation 10% / test 10%).

## 2. Problems I found (the "unexpected bottlenecks")

### 2.1 Wrong atoms for half the candidates (the big one)
MAGMa does not fragment your SMILES as written. It first converts it to a canonical form (via
InChI and tautomer standardization), which **renumbers the atoms** and sometimes moves hydrogens
and double bonds. The candidate's `parent_atom_indices` use MAGMa's numbering, but
`intensity_dataset._root_graph()` built the graph from your original SMILES.

Result: in **52% of spectra (33.5% of all candidates)**, "atom 5 of the fragment" meant a
*different atom* in the graph the GNN saw. I check this with two tests a correct mapping must
pass: the fragment's atoms match its formula, and they form one connected piece.

- Fix: `intensity_lab/chemistry.py` rebuilds the graph from MAGMa's own canonical SMILES. Result:
  **0 misaligned** candidates.
- Measured cost of the bug on the best model: **0.781 → 0.741** validation cosine (−0.040).

### 2.2 Same scan, two molecules
MSnLib sometimes annotates one scan with two compounds (e.g. co-eluting isomers). Both got the
same cache file name, so whichever finished last was kept, and a simple lookup can attach the
wrong molecule's labels. The join now picks the record whose structure matches the cached graph
exactly. 230 such scans are flagged.

### 2.3 Your big cache could not be read correctly
`data/intensity_parallel_test` (the complete one, 29,819 spectra) was written with an older version
of the dataclasses, without `root_smiles`, so the current `IntensityDataset` gives broken objects.
`intensity_lab/raw.py` reads the pickles without depending on the class definitions, so every cache
version works. `data/intensity_dataset_v1` (4,168 spectra) has identical bags where they overlap.

### 2.4 Information thrown away before the model
`_to_intensity_candidate` kept only the atom indices. The model never saw how many and which
bonds were broken, whether hydrogens moved, whether the ion is even- or odd-electron, or where the
basic nitrogens (the likely charge sites) went. These are exactly what determines fragment
abundance in mass spectrometry. The model now has them (`intensity_lab/features.py`).

### 2.5 Training pipeline speed
Before: `DataLoader` → one `torch.load` per spectrum → Python collate → Python loop per spectrum
in the loss, trained on a 100-spectrum subset. Now everything is pre-computed once into flat
arrays held in memory (cached on disk, loads in ~1 s), and the losses are vectorized. CPU is
faster than MPS here (scatter operations are slow on MPS), so use `DEVICE = "cpu"`.

### 2.6 Evaluation traps
- Spectra with only one supported peak score 1.0 for *any* model; they are now excluded.
- Spectra of the same molecule are correlated, so confidence intervals resample *molecules*.
- Without a floor and a ceiling, "0.7" means nothing (section 4).

## 3. What I built

| piece | what it is |
|---|---|
| `intensity_lab/` | New package: data loading, the bug fixes, features, splits, metrics, baselines, models, training, diagnostics, inference. `intensity_lab/README.md` lists every module. |
| `Experiment_1_agentic.ipynb` | Notebook that runs everything top to bottom and explains each result. Generated from `intensity_lab/tools/notebook_cells.py`. |
| Caching | Data in `data/agentic_cache/`; each trained model in `results/experiment_1_agentic/runs/`, named by its settings, so it is never retrained. Interrupted training resumes from the last epoch with identical results. |
| Inference | `intensity_lab.inference.predict_spectrum(predictor, spectrum, bags)` works directly on `get_candidate_bags(...)` output. Tested: running MAGMa live gives features identical to the cache. |

**How the model works (simple version):** each candidate fragment gets a score from a small
neural network; exp(score) is how abundant that ion is. A peak's height is the **sum** over its
candidates, because a peak *is* all ions of that mass together. Heights are then normalized within
the spectrum. This also tells you which candidate the model thinks produced each peak.

## 4. How to know whether a number is good

**Floors** (any useful model must beat these), validation cosine:

| simple rule | cosine |
|---|---|
| more MAGMa candidates → taller peak | 0.374 |
| all peaks equal | 0.484 |
| fewer broken bonds → taller (MAGMa's own prior) | 0.521 |

**Ceilings** (from the data itself, full-spectrum cosine on test molecules):
- **Repeat measurements** of the same molecule at the same CE: median **0.996**, but **29%** of
  pairs disagree strongly (< 0.6). Part of the library contains label noise (likely isomers or
  co-isolated compounds), so a library-wide mean of about 0.74 is roughly the noise level.
- **Same molecule at a different CE:** CE20 vs CE30 0.81, CE20 vs CE60 only **0.35**. CE changes
  spectra a lot, so a good model must use it.
- MAGMa's peak coverage limits the full-spectrum cosine to **0.975** on average, even with perfect
  intensities.

**Generalization check:** test molecules are genuinely new. Their median highest Tanimoto
similarity to any training molecule is 0.45, and only 0.1% are above 0.7. A scaffold split
(unseen chemotypes) is also prepared.

## 5. Results so far (validation, higher is better)

| model | what it uses | cosine |
|---|---|---|
| equal intensities | nothing | 0.484 |
| shuffled-label control | real features, *scrambled* targets | 0.495 |
| linear | new features | 0.611 |
| ≈ old Experiment 1 setup | original inputs, misaligned graph, fragment pooling | 0.660 |
| big MLP (420k params) | new features | 0.711 |
| LightGBM (gradient-boosted trees) | new features, per peak | 0.713 |
| small MLP (78k params) | new features | 0.717 |
| GNN, misaligned graph | new features + buggy graph | 0.741 |
| GNN, fragment-only pooling | like old pooling, corrected graph | 0.751 |
| GNN, peak = max instead of sum | | 0.780 |
| **GNN (current best)** | new features + corrected graph, pooled over fragment and broken-bond atoms | **0.781** |

What this means:
- **Representation matters most.** New features: +0.13 over equal intensities for a linear model,
  +0.23 for an MLP. The corrected graph adds +0.04 over the misaligned one. Pooling the atoms around
  the **broken bonds** adds +0.03 over pooling the fragment only.
- **Architecture size matters little.** The 420k-parameter MLP is *worse* than the 78k one (it
  memorizes more), and trees reach the same level as the MLP. Better inputs beat bigger networks.
- **Sum vs max aggregation:** practically the same (0.781 vs 0.780). Sum is the physically correct one.
- **Loss:** KL divergence was better than cosine loss in exploration (0.711 vs 0.707 for the MLP).

## 6. Debugging checks (and what they prove)

| check | result | what it tells you |
|---|---|---|
| Overfit 32 spectra | cosine 0.989 | Model, loss and gradients are wired correctly. If this fails, the bug is in the plumbing, not in generalization. |
| Shuffled labels | 0.495 ≈ equal intensities 0.484 | No leakage: high scores come from real chemistry. |
| Train vs validation gap | grows after ~15–20 epochs | Early stopping on validation picks the right epoch; more capacity does not help. |
| Gradient norms | small and stable | Optimization is not the problem. |
| Feature importance* (shuffle one group) | broken bonds > formula > H-shift > CE/conditions ≈ fragment chemistry > mass error > in-peak context ≫ named losses | Which information the model really uses. |
| Precursor peak vs CE* | measured vs predicted share: CE20 0.316 vs 0.319, CE60 0.0003 vs 0.0007 | The model learned how the intact molecule survives at each energy. |
| CE20 → CE60 change* | real spectra change more (cos 0.36) than predictions (0.65); predicting CE60 with CE gives 0.67 vs 0.32 without | The model uses CE, but still under-reacts to it: a clear next target. |
| Sharpening predictions | best with no sharpening | Predictions are flatter than real spectra (6.8 vs 4.1 effective peaks), but that is honest uncertainty, not a calibration error. |

\*Measured during exploration on the feature-only MLP. The notebook recomputes them for the final model.

Where the model is weaker: spectra with many peaks (cosine ~0.85 with 2–5 peaks, ~0.58 with more
than 40) and high CE (0.75 at CE20 vs 0.66 at CE60, MLP numbers).

## 7. Answers to your original questions

- **Not generalizing?** Much of it was input noise (the atom bug) and missing information,
  not the model. Splits are now leakage-free and audited, and the shuffled-label control proves the
  signal is real.
- **Which representation / features?** Candidate-level chemistry (broken bonds, H shifts,
  charge sites, ion type) plus a GNN on the **correctly aligned** graph, pooled over the fragment
  and the atoms around each broken bond.
- **Which architecture?** Small. Score each candidate, sum within a peak, normalize per spectrum,
  KL loss, early stopping. Bigger networks did not help.
- **Are my numbers good?** Compare against the floors (0.48–0.52) and ceilings (replicate noise,
  CE-neighbour similarity, 0.975 coverage) in section 4, always on held-out molecules.
- **Hidden bottlenecks?** The atom-order bug, the stale cache schema, discarded MAGMa
  information, and a slow data pipeline (section 2).

## 8. What is left

1. Finish 7 remaining one-change experiments (loss, mean aggregation, graph-only, original inputs
   with the corrected graph, context, deeper, more regularization).
2. Choose the final model on validation, train 3 seeds, and evaluate **once on the test set**,
   with confidence intervals, a scaffold split and a learning curve.
3. Run the notebook end to end and write its final summary from those numbers.

Exact commands: **`continue.txt`**.
