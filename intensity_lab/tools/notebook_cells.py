CELLS = []


def md(t):
    CELLS.append(("markdown", t))


def code(t):
    CELLS.append(("code", t))


def orig(i):
    CELLS.append(("orig", i))


# ============================================================================
md(r"""
# Experiment 1 (agentic): a reliable intensity predictor for pre-assigned MAGMa candidates

**Task.** Given a precursor molecule, its collision energy (CE), and the MAGMa candidate fragments
already assigned to each observed peak (*bags*), predict the relative intensity of every peak.
Peak locations are given; only their intensities are predicted.

**How to run.** *Run All*. Every expensive step is cached:

| Step | First run | Cached |
|---|---|---|
| Read 29.8k MAGMa example pickles + MSnLib join | ~35 s | ~5 s |
| MAGMa canonical SMILES + features | ~1.5 min | ~2 s |
| All model trainings (registry in `intensity_lab/experiments.py`) | ~1.5–2 h on CPU | seconds |

Trained runs live in `results/experiment_1_agentic/runs/` (one file per config hash), so editing
one config retrains only that config. To pre-train everything outside Jupyter:
`.venv/bin/python -m intensity_lab.experiments all`.

**Code layout.** All logic is in the new package [`intensity_lab/`](intensity_lab/__init__.py); this
notebook only calls it and explains the results. It reuses `magma_workflow` (MAGMa candidate generation,
`build_intensity_dataset`, the existing cache) and the pinned MAGMa engine for canonicalization.

**Reading guide.** Sections 1–2 find and fix data problems and define what "good" means.
Section 3 explains the features, section 4 runs debugging checks that should precede any model
comparison, sections 5–6 compare models and design choices on *validation only*, section 7 evaluates
the chosen model *once* on the held-out test set, section 8 interprets the model, and section 9 shows how to use it.
The summary of findings is at the end (section 10).
""")

orig(0)
orig(1)
code(r"""
%load_ext autoreload
%autoreload 2

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 4)

# ---- Run settings -------------------------------------------------------------
FORCE_RETRAIN = False        # True: ignore cached runs and retrain every model
BUILD_MISSING_MAGMA = False  # True: run MAGMa for spectra missing from the example caches (slow)
NUM_WORKERS = min(12, os.cpu_count() or 4)
DEVICE = "cpu"               # benchmarked: CPU is ~2-3x faster than MPS for these scatter ops

from intensity_lab import RESULTS_DIR, CACHE_DIR
from intensity_lab.pipeline import load_data, set_threads
from intensity_lab import experiments as EXP
from intensity_lab import plots as P
from intensity_lab import diagnostics as D
from intensity_lab.metrics import (evaluate_bags, summarize, cluster_bootstrap, paired_bootstrap,
                                   reference_pairs, reference_cosines, bag_index)
from intensity_lab.train import train, FEATURE_GROUPS, group_columns, train_metrics_on
from intensity_lab import baselines as B

FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("torch threads:", set_threads())
""")

# ============================================================================
md(r"""
## 1. Data: what one training example is, and two bugs found on the way

`magma_workflow.intensity_dataset.build_intensity_dataset` ran MAGMa on every supported MS2 spectrum
(`SINGLE_BEST_SCAN`, `[M+H]+`/`[M]+`, nearest peak assignment) and saved one pickle per spectrum.
Two caches exist: `data/intensity_dataset_v1` (4,168 spectra, current schema) and
`data/intensity_parallel_test` (29,819 spectra, an older schema without `root_smiles`).
Their bags are identical where they overlap, so both are used (v1 wins on duplicates).

`intensity_lab.raw.load_raw_examples` reads the pickles **without** the dataclasses. The older
schema then works too, and changing `IntensitySpectrum` later cannot break old caches. Each example
is joined to its MSnLib record through the same USI hash used for the file name. The join adds the
InChIKey (for leakage-free splits) and the full measured peak list (to evaluate against all peaks).

Set `BUILD_MISSING_MAGMA = True` to let your original builder fill in any missing spectra first.
""")
code(r"""
if BUILD_MISSING_MAGMA:
    from magma_workflow.dataset import parse_dataset
    from magma_workflow.intensity_dataset import build_intensity_dataset
    spectra, _ = parse_dataset("msnlib/20241003_enamdisc_pos_msn.json")
    build_intensity_dataset(spectra, cache_dir="data/intensity_dataset_v1", num_workers=NUM_WORKERS)
    del spectra

data = load_data(num_workers=NUM_WORKERS)
fs, splits = data.fs, data.splits
pd.DataFrame([fs.summary()])
""")

md(r"""
### 1.1 Bug: candidate atom indices did not refer to the root graph's atoms (52% of spectra)

MAGMa's `FragmentEngine` does **not** fragment the library SMILES. It round-trips
SMILES → InChI → *tautomer-canonical* molecule → canonical SMILES, and `parent_atom_indices` index the
atoms of that molecule. `intensity_dataset._root_graph()` instead parsed the original SMILES. When
the two atom orders differ, candidate atom *k* is a different atom of the root graph.

Experiment 1's GNN pooled atom embeddings over `candidate.parent_atom_indices`. For half the spectra
it therefore pooled the wrong atoms, sometimes a different tautomer, so "which part of the molecule
this fragment is" was noise. Two checks that any correct mapping must pass: the candidate's atoms have
the same heavy-atom composition as its formula, and they form one connected subgraph.

The fix (`intensity_lab.chemistry`): build the root graph from the SMILES the engine itself produced,
obtained from the pinned `FragmentEngine`, so it is exact by construction.
""")
code(r"""
from intensity_lab.chemistry import mol_graph, verify_alignment

rng = np.random.default_rng(0)
check = rng.choice(len(data.examples), 4000, replace=False)
engine_of = dict(zip(fs.meta["smiles"], fs.meta["engine_smiles"]))
rows = []
for label, pick in (("original library SMILES (old _root_graph)", lambda e: e["smiles"]),
                    ("MAGMa engine SMILES (fixed)", lambda e: engine_of[e["smiles"]])):
    bad_spec = bad_cand = total = 0
    for i in check:
        e = data.examples[i]
        cands = [(f, idx) for b in e["bags"] for f, _, _, idx in b[3]]
        n_bad = verify_alignment(mol_graph(pick(e)), cands)
        bad_spec += n_bad > 0; bad_cand += n_bad; total += len(cands)
    rows.append({"root graph built from": label, "spectra with misaligned candidates": f"{bad_spec / len(check):.1%}",
                 "misaligned candidates": f"{bad_cand / total:.1%}"})
display(pd.DataFrame(rows))
print(f"MAGMa re-ordered or re-tautomerized {fs.meta['engine_reordered'].mean():.1%} of all spectra' molecules")
""")

md(r"""
### 1.2 Bug: one scan can carry several structures

MSnLib sometimes annotates the *same* scan (USI) with more than one compound, for example co-eluting
isomers. Both annotations then map to the same cache file name, so `build_intensity_dataset`
kept whichever finished last. A naive USI join can therefore attach the wrong SMILES or InChIKey to an
example. The join now keeps the record whose SMILES reproduces the cached root graph exactly.
Scans annotated with several structures are flagged (`usi_n_structures > 1`). Their intensities are
ambiguous labels, but they are few.
""")
code(r"""
m = fs.meta
pd.DataFrame({
    "spectra": [len(m)],
    "scans annotated with >1 structure": [int((m["usi_n_structures"] > 1).sum())],
    "chimeric flag (MZmine)": [int((m["quality_chimeric"] != "PASSED").sum())],
    "precursor purity < 0.8": [int((m["precursor_purity"] < 0.8).sum())],
})
""")

md(r"""
### 1.3 What the data looks like

* `support_ceiling = ‖intensity at supported peaks‖ / ‖all intensity‖` is the best possible
  full-spectrum cosine when only MAGMa-supported peaks can be predicted.
* The precursor peak is a large share of the signal at low CE and almost nothing at CE60,
  so the model must use CE.
""")
code(r"""
m = fs.meta.copy()
b_all, seg_all, _ = bag_index(fs, np.arange(fs.n_spectra))
t = fs.bag_int[b_all]
m["support_ceiling"] = np.sqrt(np.bincount(seg_all, weights=t * t)) / m["full_int_l2"]
intact = fs.cand_feature_names.index("is_intact")
is_prec_bag = np.zeros(len(fs.bag_int), bool)
is_prec_bag[fs.cand_bag[fs.cand_feat[:, intact] > 0.5]] = True
m["precursor_share"] = np.bincount(seg_all, weights=t * is_prec_bag[b_all]) / np.bincount(seg_all, weights=t)
m["cands_per_bag"] = m["n_cands"] / m["n_bags"]

display(pd.DataFrame({
    "spectra": [len(m)], "molecules": [m["mol_id"].nunique()],
    "median peaks (measured)": [m["n_peaks_full"].median()], "median supported peaks": [m["n_bags"].median()],
    "median candidates / peak": [m["cands_per_bag"].median()],
    "mean support ceiling": [m["support_ceiling"].mean()],
    "spectra with 1 supported peak": [int((m["n_bags"] == 1).sum())],
}))
display(m.groupby("collision_energy").agg(spectra=("cache_id", "size"), precursor_share=("precursor_share", "mean"),
                                           median_supported_peaks=("n_bags", "median")).T)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
P.histogram(np.clip(m["n_bags"], 0, 120), "supported peaks per spectrum (clipped at 120)", ax=axes[0])
P.histogram(np.clip(m["cands_per_bag"], 0, 30), "MAGMa candidates per peak (mean per spectrum)", ax=axes[1])
P.histogram(m["support_ceiling"], "support ceiling (best achievable full cosine)", ax=axes[2])
fig.tight_layout(); fig.savefig(FIG_DIR / "data_overview.png"); plt.show()
""")

# ============================================================================
md(r"""
## 2. Evaluation protocol: how to know whether a number is good

A cosine of 0.7 means nothing by itself. Three things make it interpretable:

1. **Leakage-free splits.** All spectra of a molecule (every CE, every replicate) sit in the same split.
   Molecules are grouped by InChIKey connectivity *or* canonical SMILES. A second split holds out whole
   Bemis–Murcko scaffolds.
2. **Floors.** Trivial predictors any model must beat: uniform intensity, "more candidates means more
   intensity", and MAGMa's own prior (fewest broken bonds).
3. **Ceilings from the data itself.**
   *Replicates*: two measurements of the same molecule at the same CE. A model cannot be expected to
   agree with a measurement better than a second measurement does. *Neighbouring CE*: the same molecule
   measured at another energy, which shows how much CE matters.

**Metrics** (`intensity_lab.metrics`), computed per spectrum and then averaged:

| metric | meaning |
|---|---|
| `cos` | cosine over supported peaks, the **main metric**; what the model controls |
| `cos_full` | cosine against the complete measured spectrum (unsupported peaks predicted as 0) = `cos × support_ceiling` |
| `cos_sqrt` | cosine on √intensity; rewards getting small peaks right too |
| `top1` | predicted base peak is the measured base peak (ties get fractional credit) |
| `top3` | overlap of the 3 strongest predicted and measured peaks |
| `eff_pred` vs `eff_true` | effective number of peaks; shows whether predictions are too flat or too spiky |

Spectra with a single supported peak score 1.0 for *any* model, so they are excluded from averages.
Confidence intervals resample *molecules*, not spectra, because spectra of one molecule are correlated.
""")
code(r"""
from intensity_lab.splits import audit_split, max_train_similarity

for name in ("molecule", "scaffold"):
    table, leaks = audit_split(fs.meta, splits[name])
    print(f"--- {name} split; overlaps between parts (must all be 0): {leaks}")
    display(table)

sim = {}
for name in ("molecule", "scaffold"):
    path = CACHE_DIR / f"maxsim_{name}.npy"
    sim[name] = np.load(path) if path.exists() else max_train_similarity(fs.meta, splits[name])
    np.save(path, sim[name])
    s = sim[name][splits[name]["test"]]
    print(f"{name} split: test molecules' max Tanimoto to any training molecule: "
          f"median {np.median(s):.2f}, share > 0.7: {np.mean(s > 0.7):.1%}")
""")
md(r"""
Even the random molecule split has no near-duplicates between train and test (median max Tanimoto
about 0.45). This Enamine library is diverse, so test numbers measure generalization to genuinely new molecules.
""")
code(r"""
VAL, TEST = splits["molecule"]["val"], splits["molecule"]["test"]
floor_models = {"uniform": B.uniform, "more candidates = more intensity": B.candidate_count,
                "fewest broken bonds (MAGMa prior)": B.fewest_broken_bonds}
floors_val = {n: evaluate_bags(fs, VAL, f(fs, VAL), n) for n, f in floor_models.items()}
summarize(pd.concat(floors_val.values()))
""")
code(r"""
pairs_rep = reference_pairs(fs.meta, TEST, "replicate")
pairs_ce = reference_pairs(fs.meta, TEST, "other_ce")
rep = reference_cosines(fs.meta, data.peaks, pairs_rep)
oce = reference_cosines(fs.meta, data.peaks, pairs_ce)

print(f"Replicate measurements (same molecule, same CE), {len(rep)} test spectra:")
print(f"  full-spectrum cosine: mean {rep.cos_full.mean():.3f}, median {rep.cos_full.median():.3f}, "
      f"share < 0.6: {(rep.cos_full < 0.6).mean():.0%}")
print("Same molecule measured at the closest *other* CE (full-spectrum cosine):")
display(oce.groupby(["ce", "ref_ce"]).cos_full.agg(["mean", "count"]).query("count >= 10").T)
""")
md(r"""
**How to read these references.**

* Replicates are bimodal. Most agree almost perfectly, but about a quarter disagree badly. Those are
  separate acquisitions that likely contain a different isomer or co-isolated compound. So about 0.75 mean
  full cosine is a realistic *noise ceiling* for a library-wide average, and medians are more robust than means.
* Changing CE changes the spectrum a lot (CE20 vs CE60: cosine about 0.35). A model that ignores CE
  cannot exceed the neighbouring-CE numbers. A model that beats them has learned CE-dependent chemistry.
""")

# ============================================================================
md(r"""
## 3. Representation: which features, and why

Each MAGMa candidate becomes one feature row (`intensity_lab.features`, 88 columns). All columns are computed
from the MAGMa-aligned root graph, the candidate's ionic formula and the observed m/z; no column reads a
measured intensity. Every column below lost information in the old pipeline, either because
`_to_intensity_candidate` kept only atom indices or because it was never computed.

| group | columns | chemistry rationale |
|---|---|---|
| `mass/formula` | fragment & neutral-loss element counts, m/z fraction, RDBE, odd-electron flag | even-electron rule: odd-electron ions are rarely abundant in ESI-HCD |
| `h_shift` | ΔH = H on ion − H on its atoms (−3…+4) | MAGMa enumerates H-shift hypotheses; +1 = plain protonated cleavage |
| `bond_cuts` | number and type of broken bonds (ring, aromatic, C–N, amide, ester, benzylic, conjugated) | which bonds are weak; cleaving rings needs ≥2 cuts |
| `fragment_chemistry` | basic / aromatic / amide N, carbonyls, aromatic atoms in fragment vs neutral | charge retention: the fragment keeping the most basic site keeps the proton |
| `named_losses` | H2O, NH3, CO, CO2, HNCO, … | common neutral losses |
| `mass_error` | ppm error of candidate vs peak | separates true from coincidental matches |
| `bag_context` | candidates in this peak, is this the minimal-cut candidate | competition inside a bag |
| `conditions` | CE, CE², precursor m/z, size, adduct, number of basic sites | energy dependence |

Features are standardized with the mean and SD of all candidate rows. These are unlabeled feature statistics,
with no intensities, and identical for every run, so they leak no target information. `TensorStore` keeps everything on the device as one
flat array, so a batch is index arithmetic, not 128 pickle loads plus a Python collate loop (the old
`DataLoader` path).
""")
code(r"""
names = fs.cand_feature_names
rows = [{"group": g, "n": len(group_columns(names, g)),
         "columns": ", ".join(np.array(names)[group_columns(names, g)][:8]) + (" …" if len(group_columns(names, g)) > 8 else "")}
        for g in FEATURE_GROUPS]
assigned = sorted(set(sum((group_columns(names, g) for g in FEATURE_GROUPS), [])))
display(pd.DataFrame(rows))
X = fs.cand_feat
print(f"{len(assigned)}/{len(names)} columns assigned to a group | non-finite values: {int((~np.isfinite(X)).sum())} | "
      f"constant columns: {[names[i] for i in np.flatnonzero(X.std(0) == 0)]}")
""")

# ============================================================================
md(r"""
## 4. Debugging before trusting any training curve

A model can train smoothly and still be wrong, for example from leakage, a broken loss, or misaligned
inputs. These checks take minutes and catch most such problems:

1. **Overfit a tiny batch.** With no regularization, the model must fit 32 spectra almost perfectly.
   If it cannot, the problem is the plumbing (features, loss, aggregation), not generalization.
2. **Shuffled-label control.** Permute intensities among the peaks of each training spectrum. Validation
   should fall to about the uniform baseline. If it stays high, information is leaking somewhere.
3. **Train vs validation gap and gradient norms** (section 5): spot overfitting and unstable optimization.
""")
code(r"""
tiny = D.overfit_tiny(fs, splits["molecule"]["train"][:32], EXP.get("gnn"), steps=300, device=DEVICE)
ax = P.lines(tiny["step"], {"cosine on the 32 training spectra": tiny["cos"]}, "optimizer step", "mean cosine",
             title="Overfit test: the model can represent these spectra", markers=False)
ax.set_ylim(0, 1.02); plt.show()
print(f"final cosine on the tiny batch: {tiny['cos'].iloc[-1]:.4f}")
""")
code(r"""
def fit(name_or_cfg, verbose=False):
    cfg = EXP.get(name_or_cfg) if isinstance(name_or_cfg, str) else name_or_cfg
    return train(fs, splits[cfg.split], cfg, data.data_key(cfg.split), device=DEVICE,
                 force=FORCE_RETRAIN, verbose=verbose)

def val_metrics(run, ids=VAL):
    return evaluate_bags(fs, ids, run.predict(fs, ids, DEVICE), run.cfg.name)

shuffled = fit("mlp_shuffled_labels")
summarize(pd.concat([val_metrics(shuffled), floors_val["uniform"]]))
""")
md(r"""
With intensities shuffled within each spectrum, the model learns the shape of the intensity distribution
(a few big peaks, many small ones) but cannot place it. Validation cosine collapses to roughly the uniform
baseline. This is the expected outcome: the scores below come from real structure-to-intensity signal.
""")

# ============================================================================
md(r"""
## 5. Model ladder: how much capacity does the task need?

All models score candidates with the same physical aggregation (`intensity_lab.model`):

$$\text{abundance}(c)=e^{s_c},\qquad I(\text{peak})=\sum_{c\in\text{peak}} e^{s_c},\qquad
\hat p(\text{peak})=\frac{I(\text{peak})}{\sum_{\text{peaks}} I}$$

so a peak explained by several candidate ions receives their *summed* abundance (`logsumexp`), and each
prediction decomposes into per-candidate contributions. The loss is the KL divergence between measured
and predicted intensity distributions over supported peaks. Selection uses the validation mean cosine with
early stopping, and the test set is not used in this section.
""")
code(r"""
ladder = {name: fit(name) for name in ("linear", "mlp_large", "mlp", "gnn")}
lgbm_predict, lgbm_booster = B.lightgbm_bags(fs, splits["molecule"], data.data_key(), num_threads=8,
                                             force=FORCE_RETRAIN)
val_tables = [floors_val["uniform"], floors_val["fewest broken bonds (MAGMa prior)"]]
val_tables += [val_metrics(r) for r in ladder.values()]
val_tables.append(evaluate_bags(fs, VAL, lgbm_predict(fs, VAL), "lightgbm (bag aggregates)"))
ladder_table = summarize(pd.concat(val_tables))
ladder_table.insert(0, "params", [np.nan, np.nan] + [r.n_params() for r in ladder.values()] + [np.nan])
ladder_table
""")
code(r"""
fig = P.training_curves(ladder)
fig.tight_layout(); fig.savefig(FIG_DIR / "training_curves.png"); plt.show()

gap = []
for name, r in ladder.items():
    tr = summarize(train_metrics_on(r, fs, splits["molecule"]["train"], DEVICE)).iloc[0]
    va = ladder_table.loc[name]
    best = max(r.history, key=lambda h: h["val_cos"])
    gap.append({"model": name, "train cos": tr["cos"], "val cos": va["cos"], "gap": tr["cos"] - va["cos"],
                "selected epoch": best["epoch"], "epochs run": len(r.history),
                "median grad norm": np.median([h["grad_norm"] for h in r.history]),
                "max grad norm": max(h["grad_norm_max"] for h in r.history)})
pd.DataFrame(gap).set_index("model")
""")
md(r"""
**Reading the ladder** (numbers in the two tables above):

* The **linear** scorer already beats every floor by a wide margin, so the engineered features carry real signal.
* The **MLP** adds a large step on the same features, mostly from CE × structure interactions that a
  linear score cannot express. **LightGBM** on bag-level aggregates of the same features lands at about
  the same place. Two very different model families agreeing means that this is roughly the limit of
  *these features*, not of the model class.
* The **large MLP** has about 5× the parameters but is no better on validation. It just has a bigger
  train–val gap, so capacity is not what is missing.
* The **GNN** breaks through that ceiling. It gets the same features plus a message-passing embedding of
  the MAGMa-aligned root graph, pooled over the fragment's atoms, over the atoms on each side of every
  broken bond, and over the whole molecule. It learns the local chemistry around the cleaved bonds that
  hand-made counts cannot describe.
* Gradient norms are small and stable (clipping at 5 never binds), so optimization is not a bottleneck.
  Early stopping on validation picks the epoch before the gap grows.
""")

# ============================================================================
md(r"""
## 6. Which design choices matter? (validation only)

Each row changes *one* thing relative to the default `gnn`:

* **loss**: KL (default) vs cosine loss vs both
* **aggregation of candidates in a peak**: sum of abundances (`lse`, default) vs max vs mean
* **representation**: misaligned root graph (the old bug); fragment-only pooling (as in Experiment 1);
  graph without engineered features; the original Experiment 1 inputs; a close replica of the old model
* **capacity / regularization / context**: deeper GNN, more dropout, spectrum-level context
""")
code(r"""
ablation_runs = {name: fit(name) for name in EXP.ABLATIONS}
abl = summarize(pd.concat([val_metrics(ladder["gnn"])] + [val_metrics(r) for r in ablation_runs.values()]))
abl.insert(0, "Δcos vs gnn", abl["cos"] - abl.loc["gnn", "cos"])
abl.insert(0, "changes", ["default"] + [EXP.DESCRIPTIONS[n] for n in ablation_runs])
abl.insert(2, "params", [ladder["gnn"].n_params()] + [r.n_params() for r in ablation_runs.values()])
abl
""")
code(r"""
P.bars(abl["Δcos vs gnn"].drop("gnn").sort_values(), "Δ validation cosine vs default gnn",
       title="One change at a time", fmt="{:+.3f}")
plt.savefig(FIG_DIR / "ablations.png", bbox_inches="tight"); plt.show()
""")
md(r"""
ABLATION_PLACEHOLDER
""")

# ============================================================================
md(r"""
## 7. Final model and a single test-set evaluation

The final configuration is chosen on validation among `EXP.FINAL_CANDIDATES`, ordered simplest first.
A more complex variant has to beat a simpler one by more than `EXP.SIMPLICITY_MARGIN` to be chosen.
It is trained with three seeds and averaged (an ensemble reduces variance cheaply). The test set is used
here for the first time, and nothing is tuned afterwards.
""")
code(r"""
cand_scores = {n: summarize(val_metrics(fit(n))).iloc[0]["cos"] for n in EXP.FINAL_CANDIDATES}
FINAL_NAME = EXP.pick_final(cand_scores)   # simplest candidate within EXP.SIMPLICITY_MARGIN of the best
print("validation cosine of final candidates:", {k: round(v, 4) for k, v in cand_scores.items()})
print("selected:", FINAL_NAME)

from intensity_lab.inference import EnsemblePredictor, save_predictor
members = [fit(cfg) for cfg in EXP.seeds(FINAL_NAME)]
final = EnsemblePredictor(members)
print("member validation cosines:", [round(summarize(val_metrics(r)).iloc[0]["cos"], 4) for r in members])
print("ensemble validation cosine:", round(summarize(evaluate_bags(fs, VAL, final.predict(fs, VAL))).iloc[0]["cos"], 4))
""")
md(r"""
**Calibration check.** Predictions spread intensity over more peaks than real spectra (`eff_pred > eff_true`).
Is that a defect that post-hoc sharpening (p → p^a, a > 1) would fix? Validation says no. Cosine is best
near a = 1, so the KL-trained distribution already balances big and small peaks. The extra spread is
honest uncertainty about which peak dominates, not miscalibration, and no post-processing is applied.
""")
code(r"""
D.sharpening_curve(fs, VAL, final.predict(fs, VAL, DEVICE))
""")
code(r"""
test_models = {
    "uniform": B.uniform(fs, TEST),
    "fewest broken bonds (MAGMa prior)": B.fewest_broken_bonds(fs, TEST),
    "linear": ladder["linear"].predict(fs, TEST, DEVICE),
    "lightgbm (bag aggregates)": lgbm_predict(fs, TEST),
    "mlp (features only)": ladder["mlp"].predict(fs, TEST, DEVICE),
    "≈ old Experiment 1 (its inputs, misaligned graph)": ablation_runs["gnn_original_inputs_misaligned"].predict(fs, TEST, DEVICE),
    f"FINAL: {FINAL_NAME} ×{len(members)} seeds": final.predict(fs, TEST, DEVICE),
}
test_df = {n: evaluate_bags(fs, TEST, p, n) for n, p in test_models.items()}
test_table = summarize(pd.concat(test_df.values()))
ci = {n: cluster_bootstrap(d, "cos") for n, d in test_df.items()}
test_table.insert(1, "cos 95% CI", [f"[{ci[n][1]:.3f}, {ci[n][2]:.3f}]" for n in test_table.index])
display(test_table)

final_key = list(test_df)[-1]
diffs = []
for n in list(test_df)[:-1]:
    for col in ("cos", "cos_sqrt", "top1"):
        mean, lo, hi = paired_bootstrap(test_df[final_key], test_df[n], col)
        diffs.append({"final minus": n, "metric": col, "difference [95% CI]": f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"})
print("Paired differences on the same test spectra (molecule-cluster bootstrap):")
pd.DataFrame(diffs).pivot(index="final minus", columns="metric", values="difference [95% CI]")
""")
code(r"""
rep_model = test_df[final_key].set_index("spec").loc[rep["spec"]]
ax = P.model_comparison(test_table, ci, "cos", title="Held-out test molecules (supported-peak cosine)")
plt.savefig(FIG_DIR / "test_comparison.png", bbox_inches="tight"); plt.show()

print("Apples-to-apples with measurement noise, on the test spectra that have a replicate "
      f"(n={len(rep)}), full-spectrum cosine:")
display(pd.DataFrame({"replicate vs replicate": [rep["cos_full"].mean(), rep["cos_full"].median()],
                      "final model vs measured": [rep_model["cos_full"].mean(), rep_model["cos_full"].median()]},
                     index=["mean", "median"]))
""")
md(r"""
### 7.1 Robustness: unseen scaffolds, similarity to training, and how much data helps

* **Scaffold split**: retrain the final configuration with whole Murcko scaffolds held out.
* **Similarity curve**: test cosine vs the molecule's maximum Tanimoto similarity to any training molecule.
  A flat curve means the model does not depend on near neighbours.
* **Learning curve**: train on 10–100% of the training molecules. If validation still rises at 100%,
  more data would help. If it is flat, better inputs are needed, not more of the same data.
""")
code(r"""
scaf_cfg = EXP.scaffold_config(FINAL_NAME)
scaf_run = fit(scaf_cfg)
S_TEST = splits["scaffold"]["test"]
scaf = {"uniform": evaluate_bags(fs, S_TEST, B.uniform(fs, S_TEST), "uniform"),
        "final config (scaffold split)": evaluate_bags(fs, S_TEST, scaf_run.predict(fs, S_TEST, DEVICE), "final config (scaffold split)")}
scaf_table = summarize(pd.concat(scaf.values()))
scaf_table.insert(1, "cos 95% CI", [f"[{a:.3f}, {b:.3f}]" for a, b in (cluster_bootstrap(d)[1:] for d in scaf.values())])
scaf_table
""")
code(r"""
dfm = test_df[final_key].copy()
dfm["similarity"] = pd.cut(sim["molecule"][dfm["spec"]], [0, 0.3, 0.4, 0.5, 0.6, 1.0],
                           labels=["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", ">0.6"])
by_sim = dfm[dfm.n_bags >= 2].groupby("similarity", observed=True).agg(cos=("cos", "mean"), n=("cos", "size"))

fractions = EXP.SCALING_FRACTIONS
curve = []
for f in fractions:
    r = fit(EXP.scaling_config(FINAL_NAME, f)) if f < 1 else members[0]
    curve.append(summarize(val_metrics(r)).iloc[0]["cos"])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
P.bars(by_sim["cos"], "mean test cosine", title="Test cosine by similarity to training set", ax=axes[0])
P.lines([f * 100 for f in fractions], {"validation cosine": curve}, "% of training molecules used",
        "mean cosine", title="Learning curve", ax=axes[1])
fig.tight_layout(); fig.savefig(FIG_DIR / "robustness.png"); plt.show()
display(by_sim.T)
""")

# ============================================================================
md(r"""
## 8. Interpreting the model

* **Permutation importance**: shuffle one feature group across candidates and measure the drop in
  validation cosine. This shows what the trained model actually relies on.
* **Linear model coefficients**: a transparent, standardized view of which properties raise or lower a
  fragment's abundance.
* **Collision energy**: does the precursor survive at the right rate at each CE, and do predictions move
  with CE as much as real spectra do?
* **Error slices**: where the model is weak.
* **Examples**: chosen by quantile of test cosine (not cherry-picked), with the candidate fragments the
  model credits for each peak.
""")
code(r"""
imp, base_cos = D.permutation_importance(members[0], fs, VAL, DEVICE)
P.bars(imp.set_index("group")["cos_drop"], "drop in validation cosine when shuffled",
       title=f"Permutation importance ({members[0].cfg.name}, base {base_cos:.3f})")
plt.savefig(FIG_DIR / "importance.png", bbox_inches="tight"); plt.show()

lin = ladder["linear"]
coef = pd.Series(lin.model.scorer.weight.detach().numpy().ravel(), index=np.array(names)[lin.mask])
top = pd.concat([coef.nlargest(10), coef.nsmallest(10)]).sort_values()
P.bars(top, "standardized coefficient (log-abundance per SD)", title="Linear scorer: strongest effects", fmt="{:+.2f}")
plt.show()
""")
code(r"""
calib = D.precursor_calibration(members[0], fs, VAL, DEVICE, preds=final.predict(fs, VAL, DEVICE))
pairs = D.ce_pair_response(members[0], fs, data.peaks, VAL)
pairs = pairs[pairs["has_signal_b"] & (pairs["n_bags"] >= 2)]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
P.lines(calib.index, {"measured": calib["observed"], "predicted": calib["predicted"]}, "collision energy",
        "mean share of intensity in the precursor peak", title="Precursor survival vs CE", ax=axes[0])
P.scatter_xy(pairs["measured_shift"], pairs["predicted_shift"], "measured cos(CE20, CE60)",
             "predicted cos(CE20, CE60)", title="How much spectra change from CE20 to CE60", ax=axes[1])
fig.tight_layout(); fig.savefig(FIG_DIR / "collision_energy.png"); plt.show()
display(calib.T)
display(pairs[["measured_shift", "predicted_shift", "cf_accuracy", "static_accuracy"]].mean().to_frame("mean").T)
""")
md(r"""
In the CE table, `cf_accuracy` scores the model's prediction **at CE60** for the CE20 candidates
against the measured CE60 intensities on those same peaks. `static_accuracy` reuses the CE20 prediction
unchanged. The gap between them is skill the model gets from CE alone. In the scatter, points above the
diagonal are pairs where the model changes the spectrum *less* than reality: predictions are smoother
across energies than real spectra.
""")
code(r"""
uni_test = test_df["uniform"]
slices_model = D.error_slices(test_df[final_key], fs)
slices_base = D.error_slices(uni_test, fs)
fig = P.slices(slices_model, slices_base)
fig.savefig(FIG_DIR / "error_slices.png", bbox_inches="tight"); plt.show()
""")
code(r"""
from rdkit import Chem

def candidate_table(spec, cand_share, top=6):
    # The fragments the model credits most for this spectrum's predicted intensity.
    smi = fs.meta["engine_smiles"].values[spec]
    mol = Chem.MolFromSmiles(smi)
    a, b = fs.spec_cand_ptr[spec], fs.spec_cand_ptr[spec + 1]
    order = np.argsort(-cand_share)[:top]
    rows = []
    for k in order:
        c = a + k
        atoms = fs.cand_atoms[fs.cand_atom_ptr[c]:fs.cand_atom_ptr[c + 1]].astype(int) - fs.spec_atom_ptr[spec]
        f = dict(zip(names, fs.cand_feat[c]))
        rows.append({"peak m/z": fs.bag_mz[fs.cand_bag[c]], "share of predicted signal": cand_share[k],
                     "fragment (heavy atoms)": Chem.MolFragmentToSmiles(mol, atomsToUse=[int(x) for x in atoms], kekuleSmiles=False),
                     "broken bonds": int(f["n_cut"]), "ΔH": int(f["dH"])})
    return pd.DataFrame(rows)

d = test_df[final_key]
d = d[d.n_bags >= 5].sort_values("cos").reset_index(drop=True)
for q in (0.1, 0.5, 0.9):
    row = d.iloc[int(q * (len(d) - 1))]
    spec = int(row["spec"])
    bag_p, cand_share = final.predict_with_candidates(fs, np.array([spec]), DEVICE)
    meta = fs.meta.iloc[spec]
    P.mirror(fs, data.peaks, spec, bag_p,
             title=f"{int(q * 100)}th percentile: {meta['compound_name']} | CE {meta['collision_energy']:.0f} | "
                   f"cos {row['cos']:.2f} (full {row['cos_full']:.2f}) | top1 {row['top1']:.0f}")
    plt.savefig(FIG_DIR / f"example_q{int(q * 100)}.png", bbox_inches="tight"); plt.show()
    display(candidate_table(spec, cand_share))
""")

# ============================================================================
md(r"""
## 9. Using the predictor in your workflow

The final ensemble is saved to `results/experiment_1_agentic/final_model/`. Inference needs only a
`Spectrum` and the bags from `get_candidate_bags`. Below, MAGMa is rerun **live** on a test spectrum,
the features are checked to be identical to the cached ones, and a prediction is made.
""")
code(r"""
import json
from intensity_lab.inference import load_predictor, predict_spectrum
from intensity_lab.raw import find_spectrum
from magma_workflow.candidates import get_candidate_bags

final_metrics = {"selected_config": FINAL_NAME, "test": test_table.drop(columns=["cos 95% CI"]).loc[final_key].to_dict(),
                 "test_cos_ci": ci[final_key], "validation_candidates": cand_scores}
save_predictor(final, metrics=final_metrics)
(RESULTS_DIR / "metrics.json").write_text(json.dumps({"test": test_table.drop(columns=["cos 95% CI"]).to_dict(orient="index"),
                                                      "final": final_metrics}, indent=2, default=float))
predictor = load_predictor()

spec = int(d.iloc[len(d) // 2]["spec"])
usi = next(e["usi"] for e in data.examples if e["cache_id"] == fs.meta["cache_id"].values[spec])
spectrum = find_spectrum(usi)
bags = get_candidate_bags(spectrum, spectra=[], peak_assignment="nearest", include_ms3=False)
predicted, live_fs = predict_spectrum(predictor, spectrum, bags, return_store=True)

a, b = fs.spec_cand_ptr[spec], fs.spec_cand_ptr[spec + 1]
print("live MAGMa features identical to cache:", np.array_equal(live_fs.cand_feat, fs.cand_feat[a:b]))
print("reloaded predictor reproduces the in-memory ensemble:",
      np.allclose(predictor.predict(fs, np.array([spec])), final.predict(fs, np.array([spec])), atol=1e-6))
measured = spectrum.intensity / spectrum.intensity.max()
display(pd.DataFrame({"m/z": spectrum.mz, "measured": measured, "predicted": predicted}).round(3).head(15))
""")

# ============================================================================
md(r"""
## 10. Summary

SUMMARY_PLACEHOLDER
""")

md(r"""
---
## Appendix: MSnLib field reference (from Experiment_1.ipynb)
""")
orig(2)
orig(3)
