"""Reference predictors every learned model must beat.

All return one non-negative score per bag of ``spec_ids`` (concatenated in
spectrum order), the same interface as ``Run.predict``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from . import RESULTS_DIR
from .metrics import bag_index


def uniform(fs, spec_ids):
    bags, _, _ = bag_index(fs, spec_ids)
    return np.ones(len(bags))


def candidate_count(fs, spec_ids):
    """Peaks explained by more MAGMa hypotheses get more intensity."""
    bags, _, _ = bag_index(fs, spec_ids)
    return np.diff(fs.bag_cand_ptr)[bags].astype(float)


def fewest_broken_bonds(fs, spec_ids, temperature=1.0):
    """Each candidate ∝ exp(-n_cut / T); a peak sums its candidates (MAGMa-style prior)."""
    bags, _, _ = bag_index(fs, spec_ids)
    col = fs.cand_feature_names.index("n_cut")
    w = np.exp(-fs.cand_feat[:, col].astype(np.float64) / temperature)
    per_bag = np.bincount(fs.cand_bag, weights=w, minlength=len(fs.bag_int))
    return per_bag[bags]


def _bag_table(fs, spec_ids):
    """Bag-level design matrix: min/mean/max of candidate features + bag size."""
    bags, seg, _ = bag_index(fs, spec_ids)
    starts, ends = fs.bag_cand_ptr[bags], fs.bag_cand_ptr[bags + 1]
    x = fs.cand_feat
    n_c = ends - starts
    idx = np.repeat(starts - np.concatenate([[0], np.cumsum(n_c)[:-1]]), n_c) + np.arange(n_c.sum())
    rows, first = x[idx], np.concatenate([[0], np.cumsum(n_c)[:-1]])
    sums = np.add.reduceat(rows.astype(np.float64), first, axis=0)
    mins = np.minimum.reduceat(rows, first, axis=0)
    maxs = np.maximum.reduceat(rows, first, axis=0)
    table = np.concatenate([sums / n_c[:, None], mins, maxs, np.log1p(n_c)[:, None]], axis=1)
    names = ([f"mean:{n}" for n in fs.cand_feature_names] + [f"min:{n}" for n in fs.cand_feature_names]
             + [f"max:{n}" for n in fs.cand_feature_names] + ["log_n_cand"])
    names = [re.sub(r"[^0-9A-Za-z_]", "_", n) for n in names]  # LightGBM-safe
    return table.astype(np.float32), bags, seg, names


def lightgbm_bags(fs, split, cache_key, num_threads=8, force=False, out_dir=RESULTS_DIR / "runs"):
    """Gradient-boosted trees on bag-level aggregates, target = intensity share.

    Uses LightGBM's cross-entropy objective on the per-spectrum intensity
    fraction, early-stopped on validation. A strong non-neural reference.
    Returns (predict_fn, booster).
    """
    import lightgbm as lgb

    key = hashlib.sha256(json.dumps(["lgbm-v1", cache_key]).encode()).hexdigest()[:12]
    path = Path(out_dir) / f"lightgbm-{key}.txt"

    def data(ids):
        table, bags, seg, names = _bag_table(fs, ids)
        inten = fs.bag_int[bags]
        share = inten / np.bincount(seg, weights=inten)[seg]
        return table, share, names

    if path.exists() and not force:
        booster = lgb.Booster(model_file=str(path))
    else:
        xtr, ytr, names = data(split["train"])
        xva, yva, _ = data(split["val"])
        params = dict(objective="cross_entropy", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                      num_threads=num_threads, verbose=-1, seed=0)
        booster = lgb.train(params, lgb.Dataset(xtr, ytr, feature_name=names), num_boost_round=3000,
                            valid_sets=[lgb.Dataset(xva, yva)],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        path.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(path), num_iteration=booster.best_iteration)

    def predict(fs_, ids):
        table, _, _, _ = _bag_table(fs_, ids)
        return booster.predict(table)

    return predict, booster
