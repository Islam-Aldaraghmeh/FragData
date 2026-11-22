"""Simple figures for the five open questions (validation molecules only)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from intensity_lab.pipeline import load_data, set_threads
from intensity_lab import experiments as EXP, plots as P, diagnostics as D
from intensity_lab.train import train
from intensity_lab.metrics import evaluate_bags, summarize
from intensity_lab.raw import DEFAULT_MSNLIB
import magma_workflow  # noqa
from Spectrum_Classes import Spectrum
from magma_workflow.candidates import find_ms3_children, get_candidate_bags

OUT = Path(__file__).parent; set_threads(10); P.style()
d = load_data(verbose=False); fs = d.fs; sp = d.splits["molecule"]; V = sp["val"]
fit = lambda n: train(fs, sp, EXP.get(n), d.data_key(), verbose=False)
gnn = fit("gnn"); stats = {}

def shares(scores, cand_bag):
    s = pd.DataFrame({"bag": cand_bag, "s": scores}); s["w"] = np.exp(s.s - s.groupby("bag").s.transform("max"))
    return (s.w / s.groupby("bag").w.transform("sum")).values

def atoms_of(c, spec):
    return set((fs.cand_atoms[fs.cand_atom_ptr[c]:fs.cand_atom_ptr[c + 1]] - fs.spec_atom_ptr[spec]).tolist())

# ---- Find a validation molecule with a real MSn tree --------------------------
usi_of = {e["cache_id"]: e["usi"] for e in d.examples}
want = {fs.meta["feature_id"].values[s] for s in V}
recs = {}
with open(DEFAULT_MSNLIB) as fh:
    for line in fh:
        if not any(x in line for x in ('"ms_level": 2', '"ms_level": 3')) and '"ms_level":' in line:
            pass
        r = json.loads(line)
        if r.get("merge_type") == "SINGLE_BEST_SCAN" and r.get("feature_id") in want and int(r["ms_level"]) in (2, 3):
            recs.setdefault(r["feature_id"], []).append(Spectrum.from_record(r))
chosen = None
for s in V:
    m = fs.meta.iloc[s]; related = recs.get(m["feature_id"], [])
    ms2 = [x for x in related if x.ms_level == 2 and x.metadata["usi"][0] == usi_of[m["cache_id"]]]
    if not ms2: continue
    kids = find_ms3_children(ms2[0], spectra=related)
    a, b = fs.spec_bag_ptr[s], fs.spec_bag_ptr[s + 1]
    pk = {int(p): bi for bi, p in zip(range(a, b), fs.bag_peak_index[a:b])}
    linked = [p for p in kids if p in pk]
    if len(linked) < 3: continue
    for p in sorted(linked, key=lambda p: -fs.bag_int[pk[p]]):
        bag = pk[p]; cands = range(fs.bag_cand_ptr[bag], fs.bag_cand_ptr[bag + 1])
        if len({frozenset(atoms_of(c, s)) for c in cands}) >= 2 and max(len(k.peaks) for k in kids[p]) >= 6:
            chosen = (s, ms2[0], kids, linked, p, bag); break
    if chosen: break
s, ms2, kids, linked, p_sel, bag_sel = chosen
m = fs.meta.iloc[s]

# ---- Q1: one molecule's MSn tree --------------------------------------------
show = sorted(linked, key=lambda p: -ms2.intensity[p])[:3]
fig, axes = plt.subplots(1 + len(show), 1, figsize=(9, 2.0 + 1.7 * len(show)),
                         gridspec_kw={"height_ratios": [1.6] + [1] * len(show)})
I = ms2.intensity / ms2.intensity.max() * 100
axes[0].vlines(ms2.mz, 0, I, color=P.ORANGE, lw=1.4)
axes[0].plot(ms2.mz[show], I[show] + 6, "v", color=P.BLUE, ms=7, label="peak isolated again → MS3")
axes[0].set_ylim(0, 115); axes[0].set_ylabel("MS2 (%)"); axes[0].legend(fontsize=8, loc="upper left")
axes[0].set_title(f"One molecule, one tree: {m['compound_name'][:70]} (MS2 at CE {m['collision_energy']:.0f})", loc="left", fontsize=10)
for ax, p in zip(axes[1:], show):
    k = max(kids[p], key=lambda x: len(x.peaks))
    ax.vlines(k.mz, 0, k.intensity / k.intensity.max() * 100, color=P.BLUE, lw=1.4)
    ax.axvline(ms2.mz[p], color=P.MUTED, ls="--", lw=1)
    ax.set_xlim(axes[0].get_xlim()); ax.set_ylim(0, 110); ax.set_ylabel("MS3 (%)")
    ax.text(0.01, 0.82, f"MS3 of m/z {ms2.mz[p]:.3f} (CE {k.metadata.get('collision_energy')}); dashed = isolated ion",
            transform=ax.transAxes, fontsize=8.5, color=P.INK2)
axes[-1].set_xlabel("m/z"); fig.tight_layout(); fig.savefig(OUT / "q1_msn_tree.png", dpi=150); plt.close(fig)

# ---- Q2: hypothesis space: covers the mass, not the structure ----------------
f = fs.cand_feat; names = fs.cand_feature_names
cols = [names.index(f"frag_{e}") for e in ("C", "H", "N", "O", "S", "F", "Cl", "Br")]
key = np.zeros(len(f), np.int64)
for c in cols: key = key * 128 + np.rint(f[:, c] * 10).astype(np.int64)
nf = np.bincount(pd.DataFrame({"b": fs.cand_bag, "k": key}).drop_duplicates()["b"], minlength=len(fs.bag_int))
nc = np.diff(fs.bag_cand_ptr); w = fs.bag_int / np.bincount(fs.bag_spec, weights=fs.bag_int)[fs.bag_spec]
cats = {"1 candidate": nc == 1, "several candidates,\nsame formula (isomers)": (nc > 1) & (nf == 1), "several formulas": nf > 1}
bag_ceiling = np.sqrt(np.bincount(fs.bag_spec, weights=fs.bag_int ** 2)) / fs.meta["full_int_l2"].values
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
P.histogram(bag_ceiling, "share of each spectrum MAGMa can explain by mass (L2)", ax=a1,
            title=f"By mass: mean {bag_ceiling.mean():.3f}")
x = np.arange(3); wpk = [np.mean(v) for v in cats.values()]; wint = [np.average(v, weights=w) for v in cats.values()]
a2.bar(x - 0.2, wpk, 0.38, color=P.BLUE, label="share of peaks"); a2.bar(x + 0.2, wint, 0.38, color=P.AQUA, label="share of intensity")
for xx, v1, v2 in zip(x, wpk, wint):
    a2.text(xx - 0.2, v1, f"{v1:.0%}", ha="center", va="bottom", fontsize=8.5, color=P.INK2)
    a2.text(xx + 0.2, v2, f"{v2:.0%}", ha="center", va="bottom", fontsize=8.5, color=P.INK2)
a2.set_xticks(x, list(cats), fontsize=8.5); a2.set_ylim(0, 1); a2.legend(fontsize=8); a2.grid(axis="x", visible=False)
a2.set_title("…but which structure? peaks by ambiguity", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "q2_hypothesis_space.png", dpi=150); plt.close(fig)
stats["q2"] = {"ceiling": float(bag_ceiling.mean()), "peaks": dict(zip(cats, map(float, wpk))), "intensity": dict(zip(cats, map(float, wint)))}

# ---- Q3: membership test on a real MS3 spectrum --------------------------------
_, sc = gnn.predict(fs, np.array([s]), return_candidates=True)
ca, cb = fs.spec_cand_ptr[s], fs.spec_cand_ptr[s + 1]
sh = shares(sc, fs.cand_bag[ca:cb])
in_bag = [c for c in range(ca, cb) if fs.cand_bag[c] == bag_sel]
uniq = {}
for c in sorted(in_bag, key=lambda c: -sh[c - ca]):
    uniq.setdefault(frozenset(atoms_of(c, s)), c)
A, Bc = list(uniq.values())[:2]
k = max(kids[p_sel], key=lambda x: len(x.peaks))
fake = Spectrum(smiles=ms2.smiles, precursor_mz=ms2.precursor_mz, ms_level=2, peaks=k.peaks, metadata=dict(ms2.metadata))
ms3_bags = get_candidate_bags(fake, spectra=[], peak_assignment="all", include_ms3=False)
setA, setB = atoms_of(A, s), atoms_of(Bc, s)
expl = []
for mz, cands, _ in ms3_bags:
    sets = [set(o["parent_atom_indices"]) for c3 in cands for o in c3.metadata["origins"]]
    expl.append((any(x <= setA for x in sets), any(x <= setB for x in sets)))
expl = np.array(expl, bool).reshape(-1, 2)
iso = np.abs(k.mz - ms2.mz[p_sel]) < 0.01
I3 = k.intensity / k.intensity.max() * 100; tot = k.intensity[~iso].sum()
fa, fb = k.intensity[~iso & expl[:, 0]].sum() / tot, k.intensity[~iso & expl[:, 1]].sum() / tot
fig, ax = plt.subplots(figsize=(9, 3.8))
groups = [("explained by both", expl[:, 0] & expl[:, 1], P.MUTED), ("only candidate A", expl[:, 0] & ~expl[:, 1], P.BLUE),
          ("only candidate B", ~expl[:, 0] & expl[:, 1], P.ORANGE), ("by neither", ~expl[:, 0] & ~expl[:, 1], P.INK)]
for lab, msk, col in groups:
    msk = msk & ~iso
    if msk.any(): ax.vlines(k.mz[msk], 0, I3[msk], color=col, lw=1.8, label=f"{lab} ({k.intensity[msk].sum() / tot:.0%})")
ax.vlines(k.mz[iso], 0, I3[iso], color=P.MUTED, lw=1, linestyles="dotted", label="isolated ion itself")
ax.set(xlabel="m/z", ylabel="MS3 relative intensity (%)"); ax.set_ylim(0, 110); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"Membership test: MS3 of m/z {ms2.mz[p_sel]:.3f} | GNN credits A {sh[A - ca] / sh[[c - ca for c in in_bag]].sum():.0%}, "
             f"B {sh[Bc - ca] / sh[[c - ca for c in in_bag]].sum():.0%} | MS3 intensity explained: A {fa:.0%}, B {fb:.0%}", loc="left", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "q3_membership_test.png", dpi=150); plt.close(fig)
from rdkit import Chem
mol = Chem.MolFromSmiles(m["engine_smiles"])
stats["q3"] = {"molecule": m["compound_name"], "peak_mz": float(ms2.mz[p_sel]), "n_candidates_in_bag": len(in_bag),
               "A": Chem.MolFragmentToSmiles(mol, sorted(setA)), "B": Chem.MolFragmentToSmiles(mol, sorted(setB)),
               "explained_A": float(fa), "explained_B": float(fb)}

# ---- Q4: same score, different story --------------------------------------------
mx = fit("gnn_agg_max")
p1, s1 = gnn.predict(fs, V, return_candidates=True); p2, s2 = mx.predict(fs, V, return_candidates=True)
c1 = summarize(evaluate_bags(fs, V, p1)).iloc[0]["cos"]; c2 = summarize(evaluate_bags(fs, V, p2)).iloc[0]["cos"]
cand, _ = D._cand_ranges(fs, V); cbag = fs.cand_bag[cand]
t = pd.DataFrame({"bag": cbag, "s1": s1, "s2": s2, "key": key[cand]})
top1 = t.loc[t.groupby("bag").s1.idxmax(), ["bag", "key"]].set_index("bag")
amb = t.groupby("bag").size() > 1
agree_idx = t.loc[t.groupby("bag").s1.idxmax()].index.values == t.loc[t.groupby("bag").s2.idxmax()].index.values
agree = pd.Series(agree_idx, index=t.groupby("bag").s1.idxmax().index)[amb]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.2]})
P.bars(pd.Series({"peak = sum of candidates": c1, "peak = max candidate": c2}), "validation cosine", ax=a1, title="Same accuracy…")
a1.set_xlim(0, 1)
P.bars(pd.Series({"same top candidate": agree.mean(), "different top candidate": 1 - agree.mean()}), "share of ambiguous peaks",
       ax=a2, title="…different explanations of the same peaks", fmt="{:.0%}", color=P.AQUA)
a2.set_xlim(0, 1); fig.tight_layout(); fig.savefig(OUT / "q4_same_score_different_story.png", dpi=150); plt.close(fig)
stats["q4"] = {"cos_sum": float(c1), "cos_max": float(c2), "agree_top_candidate": float(agree.mean())}

# ---- Q5: where supervision exists (numbers from Docs/MSnLib_Analysis.pdf) ------
vals = pd.Series({"molecules with ≥1 MS3-linked peak": 0.9903, "MS2 intensity on linked peaks\n(active spectra)": 0.7659,
                  "MS2 spectra that are MS3 parents": 0.3475, "linked peaks among active spectra": 0.1765,
                  "MS2 peaks with MS3 (all)": 0.0562})
P.bars(vals, "share", title="Where MS3 supervision exists (ENAMDISC, your MSnLib analysis)", fmt="{:.0%}")
plt.xlim(0, 1.12); plt.savefig(OUT / "q5_supervision_availability.png", dpi=150, bbox_inches="tight"); plt.close()
(OUT / "stats_v2.json").write_text(json.dumps(stats, indent=2)); print(json.dumps(stats, indent=2))
