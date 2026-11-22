"""Figures for the supervisor brainstorm (validation data only; test untouched)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from intensity_lab.pipeline import load_data, set_threads
from intensity_lab import experiments as EXP, plots as P, diagnostics as D, baselines as B
from intensity_lab.train import train
from intensity_lab.metrics import evaluate_bags, summarize, cluster_bootstrap, matched_cosine, bag_index
from intensity_lab.splits import molecule_groups

OUT = Path(__file__).parent
set_threads(10); P.style()
d = load_data(verbose=False); fs = d.fs; sp = d.splits["molecule"]; V = sp["val"]; peaks = d.peaks
fit = lambda n: train(fs, sp, EXP.get(n), d.data_key(), verbose=False)
gnn = fit("gnn")
p_gnn, s_gnn = gnn.predict(fs, V, return_candidates=True)
df_gnn = evaluate_bags(fs, V, p_gnn, "gnn")
stats = {}

# ---- Example at cos ~0.78 -----------------------------------------------------
c = df_gnn[(df_gnn.n_bags >= 10) & (df_gnn.n_bags <= 30)].copy()
row = c.iloc[(c["cos"] - 0.78).abs().argsort().iloc[0]]
spec = int(row["spec"]); m = fs.meta.iloc[spec]
bags, seg, L = bag_index(fs, V); k = list(V).index(spec)
a = int(np.concatenate([[0], np.cumsum(L)])[k]); pred = p_gnn[a:a + L[k]]
fig = plt.figure(figsize=(13, 4.2)); gs = fig.add_gridspec(1, 3)
ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(Draw.MolToImage(Chem.MolFromSmiles(m["smiles"]), size=(420, 360))); ax0.axis("off")
ax0.set_title(f"{m['compound_name']}", loc="left", fontsize=9)
ax1 = fig.add_subplot(gs[0, 1:])
P.mirror(fs, peaks, spec, pred, ax=ax1, label="predicted (GNN)",
         title=f"CE {m['collision_energy']:.0f} | cosine {row['cos']:.2f} on supported peaks (full {row['cos_full']:.2f}) | "
               f"base peak correct: {'yes' if row['top1'] > 0.5 else 'no'}")
fig.tight_layout(); fig.savefig(OUT / "example_cos078.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(6.5, 3.4))
dd = df_gnn[df_gnn.n_bags >= 2]
ax.hist(dd["cos"], bins=40, color=P.BLUE, edgecolor="white", linewidth=0.5)
for v, lab in ((dd["cos"].median(), f"median {dd['cos'].median():.2f}"), (row["cos"], "example")):
    ax.axvline(v, color=P.INK2, linestyle="--", linewidth=1); ax.text(v, ax.get_ylim()[1] * 0.95, " " + lab, fontsize=8.5, color=P.INK2)
ax.set(xlabel="per-spectrum cosine (validation, GNN)", ylabel="spectra", title="")
ax.set_title("How cosines are distributed across spectra", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "cosine_distribution.png", dpi=150); plt.close(fig)
stats["example"] = {"compound": m["compound_name"], "ce": float(m["collision_energy"]), "cos": float(row["cos"]),
                    "cos_full": float(row["cos_full"]), "n_peaks": int(row["n_bags"])}
stats["val_cos_quartiles"] = [float(x) for x in dd["cos"].quantile([0.25, 0.5, 0.75])]

# ---- Q1: label noise and CE dependence (measured spectra only) -------------
groups = molecule_groups(fs.meta); ce = fs.meta["collision_energy"].values; cid = fs.meta["cache_id"].values
by = pd.Series(np.arange(len(groups))).groupby(groups).apply(list)
CES = [15, 20, 30, 45, 60]; sums = {}; reps = []
for ids in by:
    if len(ids) < 2: continue
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            x, y = ids[i], ids[j]
            if ce[x] not in CES or ce[y] not in CES: continue
            v = matched_cosine(peaks[cid[x]], peaks[cid[y]])
            key = tuple(sorted((ce[x], ce[y]))); sums.setdefault(key, []).append(v)
            if ce[x] == ce[y]: reps.append(v)
M = np.full((5, 5), np.nan)
for (u, w), v in sums.items():
    i, j = CES.index(u), CES.index(w); M[i, j] = M[j, i] = np.mean(v) if len(v) >= 10 else np.nan
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9))
a1.hist(reps, bins=30, color=P.BLUE, edgecolor="white", linewidth=0.5)
a1.set(xlabel="full-spectrum cosine between two measurements", ylabel="pairs")
a1.set_title(f"Same molecule, same CE (n={len(reps)}): median {np.median(reps):.2f}, {np.mean(np.array(reps) < 0.6):.0%} < 0.6",
             loc="left", fontsize=9.5)
im = a2.imshow(M, cmap="Blues", vmin=0, vmax=1)
a2.set_xticks(range(5), CES); a2.set_yticks(range(5), CES); a2.grid(False)
for i in range(5):
    for j in range(5):
        if not np.isnan(M[i, j]): a2.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                                          color="white" if M[i, j] > 0.6 else P.INK)
a2.set(xlabel="collision energy", ylabel="collision energy"); a2.set_title("Mean cosine between CEs of the same molecule", loc="left", fontsize=9.5)
fig.colorbar(im, ax=a2, fraction=0.046); fig.tight_layout(); fig.savefig(OUT / "q1_label_noise_ce.png", dpi=150); plt.close(fig)
stats["replicates"] = {"n": len(reps), "median": float(np.median(reps)), "mean": float(np.mean(reps)), "frac_below_0.6": float(np.mean(np.array(reps) < 0.6))}

# ---- Q2: ambiguity of the MAGMa candidate space by intensity rank -----------
f = fs.cand_feat; names = fs.cand_feature_names
cols = [names.index(f"frag_{e}") for e in ("C", "H", "N", "O", "S", "F", "Cl", "Br")]
key = np.zeros(len(f), np.int64)
for cc in cols: key = key * 128 + np.rint(f[:, cc] * 10).astype(np.int64)
pairs = pd.DataFrame({"bag": fs.cand_bag, "key": key}).drop_duplicates()
n_form = np.bincount(pairs["bag"], minlength=len(fs.bag_int)); n_cand = np.diff(fs.bag_cand_ptr)
rank = pd.DataFrame({"s": fs.bag_spec, "i": fs.bag_int}).groupby("s")["i"].rank(ascending=False, method="first").values
rb = np.minimum(rank, 11).astype(int)
t = pd.DataFrame({"rank": rb, "n_cand": n_cand, "n_form": n_form}).groupby("rank").agg(
    amb=("n_cand", lambda x: np.mean(x > 1)), multi_formula=("n_form", lambda x: np.mean(x > 1)),
    med=("n_cand", "median"), q1=("n_cand", lambda x: x.quantile(.25)), q3=("n_cand", lambda x: x.quantile(.75)))
x = t.index.values; lab = [str(i) for i in x[:-1]] + [">10"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7))
for ax in (a1, a2): ax.axvspan(0.5, 5.5, color=P.GRID, zorder=0); ax.text(3, 0.02, "ranks usually fragmented again (MS3)", ha="center", fontsize=8, color=P.INK2, transform=ax.get_xaxis_transform())
a1.fill_between(x, t.q1, t.q3, color=P.BLUE, alpha=0.18, linewidth=0); a1.plot(x, t.med, color=P.BLUE, lw=2, marker="o", ms=6, mec="white")
a1.set_xticks(x, lab); a1.set(xlabel="peak intensity rank in its spectrum", ylabel="MAGMa candidates per peak")
a1.set_title("Candidates per peak (median, IQR)", loc="left", fontsize=10)
P.lines(x, {"peaks with >1 candidate": t.amb, "peaks with >1 distinct formula": t.multi_formula}, "peak intensity rank in its spectrum", "share of peaks", ax=a2)
a2.set_xticks(x, lab); a2.set_ylim(0, 1.02); a2.set_title("How ambiguous are peaks?", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "q2_ambiguity.png", dpi=150); plt.close(fig)
stats["ambiguity"] = {"share_multi_candidate": float(np.mean(n_cand > 1)), "share_multi_formula": float(np.mean(n_form > 1)),
                      "median_candidates": float(np.median(n_cand))}

# ---- Q3: representation / architecture ladder (validation, 95% CI) ----------
lgb_pred, _ = B.lightgbm_bags(fs, sp, d.data_key())
models = {"uniform": B.uniform(fs, V), "fewest broken bonds (MAGMa prior)": B.fewest_broken_bonds(fs, V),
          "linear (new features)": fit("linear").predict(fs, V), "LightGBM (new features)": lgb_pred(fs, V),
          "MLP (new features)": fit("mlp").predict(fs, V),
          "≈ old Exp. 1: GNN, old inputs, misaligned": fit("gnn_original_inputs_misaligned").predict(fs, V),
          "GNN, misaligned graph": fit("gnn_misaligned").predict(fs, V),
          "GNN, fragment-only pooling": fit("gnn_fragment_pool").predict(fs, V), "GNN (current best)": p_gnn}
dfs = {n: evaluate_bags(fs, V, p, n) for n, p in models.items()}
summ = summarize(pd.concat(dfs.values())); ci = {n: cluster_bootstrap(v) for n, v in dfs.items()}
ax = P.model_comparison(summ, ci, "cos", title="Validation cosine by model (molecule-disjoint)")
ax.set_xlabel("mean cosine on supported peaks (validation, 95% molecule-bootstrap CI)")
plt.savefig(OUT / "q3_model_ladder.png", dpi=150, bbox_inches="tight"); plt.close()
stats["ladder"] = {n: round(float(summ.loc[n, "cos"]), 4) for n in summ.index}

# ---- Q4: collision energy ------------------------------------------------------
calib = D.precursor_calibration(gnn, fs, V, preds=p_gnn)
pr = D.ce_pair_response(gnn, fs, peaks, V); pr = pr[pr.has_signal_b & (pr.n_bags >= 2)]
byce = df_gnn[df_gnn.n_bags >= 2].groupby("ce")["cos"].mean()
uce = evaluate_bags(fs, V, B.uniform(fs, V)); uce = uce[uce.n_bags >= 2].groupby("ce")["cos"].mean()
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 3.8))
calib = calib[calib.n >= 50]
P.lines(calib.index, {"measured": calib.observed, "predicted": calib.predicted}, "collision energy", "share of intensity in intact precursor", ax=a1, title="Precursor survival")
P.scatter_xy(pr.measured_shift, pr.predicted_shift, "measured cos(CE20, CE60) on same peaks", "predicted cos(CE20, CE60)", ax=a2, title=f"CE20→60 change (n={len(pr)}); above diagonal = under-reacts")
ces = [e for e in byce.index if e in calib.index]
P.lines(ces, {"GNN": byce.loc[ces], "uniform": uce.loc[ces]}, "collision energy", "mean cosine (validation)", ax=a3, title="Accuracy by CE")
fig.tight_layout(); fig.savefig(OUT / "q4_collision_energy.png", dpi=150); plt.close(fig)
stats["ce"] = {"measured_shift": float(pr.measured_shift.mean()), "predicted_shift": float(pr.predicted_shift.mean()),
               "cf_accuracy": float(pr.cf_accuracy.mean()), "static_accuracy": float(pr.static_accuracy.mean()),
               "cos_by_ce": {str(k): round(float(v), 3) for k, v in byce.items()}}

# ---- Q5: how sure is the model about WHICH candidate explains a peak? ------
cand, Lc = D._cand_ranges(fs, V)
cb = fs.cand_bag[cand]
sh = pd.DataFrame({"bag": cb, "s": s_gnn}); sh["w"] = np.exp(sh.s - sh.groupby("bag").s.transform("max"))
sh["share"] = sh.w / sh.groupby("bag").w.transform("sum")
top = sh.groupby("bag").share.max(); nb = sh.groupby("bag").size()
tb = pd.DataFrame({"top": top, "n": nb, "rank": rb[top.index.values]}); tb = tb[tb.n > 1]
g = tb.groupby("rank").top.agg(["median", lambda x: x.quantile(.25), lambda x: x.quantile(.75)]); g.columns = ["med", "q1", "q3"]
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.axvspan(0.5, 5.5, color=P.GRID, zorder=0); ax.text(3, 0.03, "MS3 usually available here", ha="center", fontsize=8, color=P.INK2, transform=ax.get_xaxis_transform())
ax.fill_between(g.index, g.q1, g.q3, color=P.BLUE, alpha=0.18, linewidth=0)
ax.plot(g.index, g.med, color=P.BLUE, lw=2, marker="o", ms=6, mec="white", label="share of the peak given to its top candidate")
ax.plot(g.index, 1 / tb.groupby("rank").n.median(), color=P.MUTED, lw=1.5, ls="--", label="what an undecided model would give (1 / candidates)")
ax.set_xticks(g.index, [str(i) for i in g.index[:-1]] + [">10"]); ax.set_ylim(0, 1.02)
ax.set(xlabel="peak intensity rank in its spectrum", ylabel="share (median, IQR)"); ax.legend(fontsize=8, loc="upper right")
ax.set_title("Ambiguous peaks: how decisive is the model's attribution?", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "q5_attribution_confidence.png", dpi=150); plt.close(fig)
stats["attribution"] = {"median_top_share_rank1to5": float(tb[tb["rank"] <= 5].top.median()),
                        "share_bags_top_over_0.9": float(np.mean(tb.top > 0.9))}
(OUT / "stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
