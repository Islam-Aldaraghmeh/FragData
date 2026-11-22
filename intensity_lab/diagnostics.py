"""Debugging tools for trained intensity networks.

Each function answers one question that otherwise gets guessed at:

    overfit_tiny             can the model + loss fit a handful of spectra at all?
    permutation_importance   which feature groups does the trained model rely on?
    ce_counterfactual        does the prediction respond to collision energy, and how
                             does that response compare with real spectra?
    precursor_calibration    is the surviving-precursor share right at every CE?
    error_slices             where (CE, peak count, similarity to train...) does it fail?
    weight_report            per-layer weight / gradient health
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .metrics import bag_index, evaluate_bags, summarize
from .model import IntensityModel, TensorStore, loss_fn
from .train import FEATURE_GROUPS, group_columns


def overfit_tiny(fs, ids, cfg, steps=300, device="cpu", seed=0):
    """Train on a few spectra only; a correct model/loss should approach cos ~1."""
    torch.manual_seed(seed)
    names = list(fs.cand_feature_names)
    store = TensorStore(fs, device, graph=cfg.kind == "gnn")
    model = IntensityModel(len(names), kind=cfg.kind, hidden=cfg.hidden, n_layers=cfg.n_layers, dropout=0.0,
                           agg=cfg.agg, atom_dim=fs.atom_feat.shape[1], bond_dim=fs.bond_feat.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    b = store.batch(np.asarray(ids))
    curve = []
    for step in range(steps):
        model.train()
        log_p, _ = model(b, names.index("ce"))
        loss = loss_fn(cfg.loss, log_p, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 10 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                p = model(b, names.index("ce"))[0].exp().cpu().numpy()
            cos = summarize(evaluate_bags(fs, ids, p), min_bags=1).iloc[0]["cos"]
            curve.append({"step": step, "loss": loss.item(), "cos": cos})
    return pd.DataFrame(curve)


def permutation_importance(run, fs, ids, device="cpu", seed=0, groups=None):
    """Drop in mean cosine when one feature group is shuffled across candidates."""
    store = run.store(fs, device)
    base = summarize(evaluate_bags(fs, ids, run.predict(fs, ids, device, store=store))).iloc[0]["cos"]
    names = list(np.array(fs.cand_feature_names)[run.mask])
    rng = np.random.default_rng(seed)
    rows = []
    original = store.cand_x.clone()
    # Only shuffle among the candidates of the evaluated spectra.
    cand, _ = _cand_ranges(fs, ids)
    cand_t = torch.from_numpy(cand)
    for group in groups or FEATURE_GROUPS:
        cols = [names.index(n) for n in np.array(fs.cand_feature_names)[group_columns(fs.cand_feature_names, group)]
                if n in names]
        if not cols:
            continue
        perm = torch.from_numpy(rng.permutation(len(cand)))
        block = store.cand_x[cand_t][:, cols]
        store.cand_x[cand_t[:, None], torch.tensor(cols)[None, :]] = block[perm]
        cos = summarize(evaluate_bags(fs, ids, run.predict(fs, ids, device, store=store))).iloc[0]["cos"]
        store.cand_x.copy_(original)
        rows.append({"group": group, "n_features": len(cols), "cos_drop": base - cos})
    return pd.DataFrame(rows).sort_values("cos_drop", ascending=False), base


def _cand_ranges(fs, ids):
    ids = np.asarray(ids)
    starts, ends = fs.spec_cand_ptr[ids], fs.spec_cand_ptr[ids + 1]
    lengths = ends - starts
    return np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths) + np.arange(lengths.sum()), lengths


def _set_ce(store, run, fs, ids, ce):
    names = list(np.array(fs.cand_feature_names)[run.mask])
    cand, _ = _cand_ranges(fs, ids)
    cand_t = torch.from_numpy(cand)
    ce_s = (ce - 35.0) / 20.0
    for col, value in (("ce", ce_s), ("ce_sq", ce_s ** 2)):
        if col in names:
            j = names.index(col)
            store.cand_x[cand_t, j] = float((value - run.mean[j]) / run.std[j])


def ce_counterfactual(run, fs, ids, ce_values=(15, 20, 30, 45, 60), device="cpu"):
    """Predict every spectrum's candidates at several CEs (all else fixed)."""
    store = run.store(fs, device)
    original = store.cand_x.clone()
    preds = {}
    for ce in ce_values:
        _set_ce(store, run, fs, ids, ce)
        preds[ce] = run.predict(fs, ids, device, store=store)
        store.cand_x.copy_(original)
    return preds


def precursor_calibration(run, fs, ids, device="cpu", preds=None):
    """Observed vs predicted share of the intact-precursor peak, by CE."""
    ids = np.asarray(ids)
    bags, seg, _ = bag_index(fs, ids)
    intact_col = fs.cand_feature_names.index("is_intact")
    is_prec = np.zeros(len(fs.bag_int), bool)
    is_prec[fs.cand_bag[fs.cand_feat[:, intact_col] > 0.5]] = True
    pred = run.predict(fs, ids, device) if preds is None else preds
    t = fs.bag_int[bags]
    obs = np.bincount(seg, weights=t * is_prec[bags], minlength=len(ids)) / np.bincount(seg, weights=t, minlength=len(ids))
    prd = np.bincount(seg, weights=pred * is_prec[bags], minlength=len(ids)) / np.bincount(seg, weights=pred, minlength=len(ids))
    df = pd.DataFrame({"ce": fs.meta["collision_energy"].values[ids], "observed": obs, "predicted": prd})
    return df.groupby("ce")[["observed", "predicted"]].mean().assign(n=df.groupby("ce").size())


def _intensity_at(peaks, mz, abs_tol=0.005, ppm=20.0):
    """Measured intensity at each m/z (nearest peak within tolerance, else 0)."""
    out = np.zeros(len(mz))
    if len(peaks):
        j = np.abs(peaks[:, 0][None, :] - mz[:, None]).argmin(1)
        ok = np.abs(peaks[j, 0] - mz) <= np.maximum(abs_tol, ppm * 1e-6 * mz)
        out[ok] = peaks[j[ok], 1]
    return out


def ce_pair_response(run, fs, peaks_by_id, ids, ce_a=20.0, ce_b=60.0, device="cpu"):
    """Does the prediction move with collision energy as much as real spectra do?

    For molecules measured at both energies, take the CE_a spectrum's supported
    peaks and compare, on exactly those peaks:
        measured_shift   cos(measured CE_a, measured CE_b)
        predicted_shift  cos(prediction at CE_a, prediction at CE_b)   (same candidates)
        cf_accuracy      cos(prediction at CE_b, measured CE_b)        (counterfactual skill)
        static_accuracy  cos(prediction at CE_a, measured CE_b)        (ignoring the CE change)
    A model that ignores CE has predicted_shift ~ 1 and cf_accuracy ~ static_accuracy.
    """
    meta = fs.meta
    ce, mol, cache = meta["collision_energy"].values, meta["mol_id"].values, meta["cache_id"].values
    partner = {}
    for s in ids:
        if ce[s] == ce_b:
            partner.setdefault(mol[s], int(s))
    spec_a = np.array([int(s) for s in ids if ce[s] == ce_a and mol[s] in partner], dtype=np.int64)
    preds = ce_counterfactual(run, fs, spec_a, (ce_a, ce_b), device)
    bags, seg, lengths = bag_index(fs, spec_a)
    obs_b = np.concatenate([_intensity_at(peaks_by_id[cache[partner[mol[s]]]],
                                          fs.bag_mz[fs.spec_bag_ptr[s]:fs.spec_bag_ptr[s + 1]]) for s in spec_a])
    obs_a = fs.bag_int[bags]
    n = len(spec_a)

    def cos(x, y):
        return np.bincount(seg, weights=x * y, minlength=n) / np.sqrt(
            np.bincount(seg, weights=x * x, minlength=n) * np.bincount(seg, weights=y * y, minlength=n) + 1e-30)

    return pd.DataFrame({
        "spec": spec_a, "n_bags": lengths, "measured_shift": cos(obs_a, obs_b),
        "predicted_shift": cos(preds[ce_a], preds[ce_b]), "cf_accuracy": cos(preds[ce_b], obs_b),
        "static_accuracy": cos(preds[ce_a], obs_b), "has_signal_b": np.bincount(seg, weights=obs_b, minlength=n) > 0,
    })


def error_slices(df, fs, extra=None, min_bags=2):
    """Mean cosine by collision energy, number of supported peaks, and extras."""
    d = df[df["n_bags"] >= min_bags].copy()
    d["peaks_bin"] = pd.cut(d["n_bags"], [1, 5, 10, 20, 40, 1000], labels=["2-5", "6-10", "11-20", "21-40", ">40"])
    d["ambiguity"] = pd.cut(fs.meta["n_cands"].values[d["spec"]] / d["n_bags"], [0, 2, 4, 8, 1e9],
                            labels=["<=2", "2-4", "4-8", ">8"])
    if extra is not None:
        for name, (values, bins, labels) in extra.items():
            d[name] = pd.cut(values[d["spec"]], bins, labels=labels)
    out = {}
    for col in ["ce", "peaks_bin", "ambiguity"] + list(extra or {}):
        g = d.groupby(col, observed=True)
        out[col] = pd.DataFrame({"cos": g["cos"].mean(), "top1": g["top1"].mean(), "n": g.size()})
    return out


def weight_report(model):
    rows = []
    for name, p in model.named_parameters():
        rows.append({"param": name, "shape": tuple(p.shape), "norm": float(p.detach().norm()),
                     "mean_abs": float(p.detach().abs().mean())})
    return pd.DataFrame(rows)


def sharpening_curve(fs, ids, pred, exponents=(0.6, 0.8, 1.0, 1.25, 1.5, 2.0)):
    """Validation metrics after p -> p**a. If a=1 is best, the KL-trained
    distribution is already calibrated for cosine; a>1 would mean it is too flat."""
    rows = []
    for a in exponents:
        s = summarize(evaluate_bags(fs, ids, np.asarray(pred) ** a)).iloc[0]
        rows.append({"exponent": a, "cos": s["cos"], "cos_sqrt": s["cos_sqrt"], "eff_pred": s["eff_pred"],
                     "eff_true": s["eff_true"]})
    return pd.DataFrame(rows).set_index("exponent")
