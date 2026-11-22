"""Per-spectrum evaluation, reference ceilings and cluster-bootstrap intervals.

Every model returns one non-negative score per *bag* (supported peak) for a set
of spectra, concatenated in spectrum order. ``evaluate_bags`` turns that into
per-spectrum metrics:

    cos         cosine(pred, observed) over supported peaks          (main metric)
    cos_full    cosine against the *complete* measured spectrum; unsupported
                peaks are predicted as zero.  cos_full = cos * support_ceiling
    cos_sqrt    cosine of square-rooted intensities (weights small peaks more)
    top1        the predicted strongest peak is the observed strongest peak
    top3        |top-3 predicted ∩ top-3 observed| / 3
    spearman    rank correlation over supported peaks
    eff_pred / eff_true   effective number of peaks  (Σp)² / Σp²
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _segments(ptr, spec_ids):
    starts, ends = ptr[spec_ids], ptr[spec_ids + 1]
    lengths = ends - starts
    seg = np.repeat(np.arange(len(spec_ids)), lengths)
    idx = np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths) + np.arange(lengths.sum())
    return idx, seg, lengths


def bag_index(fs, spec_ids):
    """Global bag ids for spec_ids (in order) and their local spectrum segment."""
    return _segments(fs.spec_bag_ptr, np.asarray(spec_ids))


def _seg_sum(x, seg, n):
    return np.bincount(seg, weights=x, minlength=n)


def evaluate_bags(fs, spec_ids, pred, name=None):
    spec_ids = np.asarray(spec_ids)
    bags, seg, lengths = bag_index(fs, spec_ids)
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    if len(pred) != len(bags):
        raise ValueError(f"expected {len(bags)} bag predictions, got {len(pred)}")
    t = fs.bag_int[bags].astype(np.float64)
    n = len(spec_ids)
    eps = 1e-30

    def cosine(a, b):
        return _seg_sum(a * b, seg, n) / np.sqrt(_seg_sum(a * a, seg, n) * _seg_sum(b * b, seg, n) + eps)

    cos = cosine(pred, t)
    cos_sqrt = cosine(np.sqrt(pred), np.sqrt(t))
    meta = fs.meta.iloc[spec_ids]
    ceiling = np.sqrt(_seg_sum(t * t, seg, n)) / meta["full_int_l2"].values
    eff = lambda x: _seg_sum(x, seg, n) ** 2 / (_seg_sum(x * x, seg, n) + eps)

    top1 = np.zeros(n)
    top3 = np.zeros(n)
    spear = np.full(n, np.nan)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    from scipy.stats import rankdata
    for k, (s, L) in enumerate(zip(starts, lengths)):
        p, y = pred[s:s + L], t[s:s + L]
        # Fractional credit on ties so constant predictions are not rewarded.
        best = np.flatnonzero(p == p.max())
        top1[k] = np.mean(best == np.argmax(y)) if p.max() > 0 else 0.0
        kk = min(3, L)
        top3[k] = len(set(np.argsort(-p, kind="stable")[:kk]) & set(np.argsort(-y, kind="stable")[:kk])) / kk
        if L >= 3 and np.ptp(p) > 0:
            rp, ry = rankdata(p), rankdata(y)
            spear[k] = np.corrcoef(rp, ry)[0, 1]
    df = pd.DataFrame({
        "spec": spec_ids, "mol_id": meta["mol_id"].values, "ce": meta["collision_energy"].values,
        "n_bags": lengths, "cos": cos, "cos_full": cos * ceiling, "cos_sqrt": cos_sqrt,
        "top1": top1, "top3": top3, "spearman": spear, "eff_pred": eff(pred), "eff_true": eff(t),
        "support_ceiling": ceiling,
    })
    if name is not None:
        df.insert(0, "model", name)
    return df


SUMMARY_COLS = ("cos", "cos_full", "cos_sqrt", "top1", "top3", "spearman", "eff_pred", "eff_true")


def summarize(df, min_bags=2, by=None):
    """Mean of each metric over spectra with >= min_bags supported peaks.

    Single-peak spectra score 1.0 for *any* model on the supported metrics, so
    they are excluded by default to keep comparisons honest.
    """
    d = df[df["n_bags"] >= min_bags]
    cols = [c for c in SUMMARY_COLS if c in d]
    keys = ["model"] if "model" in d else []
    if by:
        keys = keys + ([by] if isinstance(by, str) else list(by))
    if keys:
        out = d.groupby(keys, sort=False)[cols].mean()
        out["n_spectra"] = d.groupby(keys, sort=False).size()
        return out
    return d[cols].mean().to_frame().T


def cluster_bootstrap(df, col="cos", n_boot=1000, seed=0, min_bags=2):
    """95% CI of the mean, resampling molecules (all their spectra together)."""
    d = df[df["n_bags"] >= min_bags]
    groups = d.groupby("mol_id")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    sums, counts = groups["sum"].values, groups["count"].values
    means = sums[pick].sum(1) / counts[pick].sum(1)
    return float(d[col].mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(df_a, df_b, col="cos", n_boot=1000, seed=0, min_bags=2):
    """Mean(a - b) with a molecule-cluster bootstrap CI (same spectra required)."""
    a = df_a.set_index("spec")
    b = df_b.set_index("spec").loc[a.index]
    d = pd.DataFrame({"mol_id": a["mol_id"], "n_bags": a["n_bags"], "diff": a[col] - b[col]})
    return cluster_bootstrap(d.reset_index(), "diff", n_boot, seed, min_bags)


# ---------------------------------------------------------------------------
# Reference points from the measured spectra themselves
# ---------------------------------------------------------------------------

def matched_cosine(peaks_a, peaks_b, abs_tol=0.005, ppm=20.0, sqrt=False):
    """Cosine of two measured spectra with greedy one-to-one m/z matching."""
    ma, ia = peaks_a[:, 0].astype(float), peaks_a[:, 1].astype(float)
    mb, ib = peaks_b[:, 0].astype(float), peaks_b[:, 1].astype(float)
    if sqrt:
        ia, ib = np.sqrt(ia), np.sqrt(ib)
    tol = np.maximum(abs_tol, ppm * 1e-6 * ma)[:, None]
    close = np.abs(ma[:, None] - mb[None, :]) <= tol
    ai, bi = np.nonzero(close)
    score = ia[ai] * ib[bi]
    order = np.argsort(-score)
    used_a, used_b, dot = set(), set(), 0.0
    for k in order:
        if ai[k] in used_a or bi[k] in used_b:
            continue
        used_a.add(ai[k])
        used_b.add(bi[k])
        dot += score[k]
    return dot / (np.linalg.norm(ia) * np.linalg.norm(ib) + 1e-30)


def reference_pairs(meta, spec_ids, kind="replicate"):
    """Pairs (target, reference) of spectra of the same molecule.

    replicate: same collision energy, different scan (measurement noise ceiling)
    other_ce:  the closest *different* collision energy of the same molecule
    """
    from .splits import molecule_groups

    groups = molecule_groups(meta)
    sub = pd.DataFrame({"spec": spec_ids, "group": groups[spec_ids],
                        "ce": meta["collision_energy"].values[spec_ids]})
    pairs = []
    for _, g in sub.groupby("group"):
        if len(g) < 2:
            continue
        for row in g.itertuples():
            others = g[g["spec"] != row.spec]
            if kind == "replicate":
                same = others[others["ce"] == row.ce]
                if len(same):
                    pairs.append((row.spec, int(same["spec"].iloc[0])))
            else:
                diff = others[others["ce"] != row.ce]
                if len(diff):
                    j = int(np.argmin(np.abs(diff["ce"].values - row.ce)))
                    pairs.append((row.spec, int(diff["spec"].iloc[j])))
    return pairs


def reference_cosines(meta, peaks_by_id, pairs, sqrt=False):
    ids = meta["cache_id"].values
    return pd.DataFrame([{
        "spec": a, "ref": b, "ce": meta["collision_energy"].values[a],
        "ref_ce": meta["collision_energy"].values[b],
        "cos_full": matched_cosine(peaks_by_id[ids[a]], peaks_by_id[ids[b]], sqrt=sqrt),
    } for a, b in pairs])
