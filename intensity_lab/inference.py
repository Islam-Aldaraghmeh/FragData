"""Use a trained predictor on new spectra / candidate bags.

    from intensity_lab.inference import load_predictor, predict_spectrum
    from magma_workflow.candidates import get_candidate_bags

    predictor = load_predictor()                       # final ensemble saved by the notebook
    bags = get_candidate_bags(spectrum, spectra=[], peak_assignment="nearest", include_ms3=False)
    intensities = predict_spectrum(predictor, spectrum, bags)   # one value per spectrum peak

``intensities`` follows ``spectrum.mz`` order, is scaled so the strongest peak
is 1, and is 0 where MAGMa has no candidate. Measured intensities are not used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import RESULTS_DIR
from .features import featurize_candidate_bags
from .model import IntensityModel
from .train import Run, TrainConfig

FINAL_DIR = RESULTS_DIR / "final_model"


class EnsemblePredictor:
    """Average of the bag distributions of several runs (e.g. seeds)."""

    def __init__(self, runs):
        self.runs = list(runs)

    def predict(self, fs, spec_ids, device="cpu"):
        return np.mean([r.predict(fs, np.asarray(spec_ids), device) for r in self.runs], axis=0)

    def predict_with_candidates(self, fs, spec_ids, device="cpu"):
        """Bag predictions plus per-candidate shares of predicted intensity (ensemble mean)."""
        bag_preds, cand_shares = [], []
        for r in self.runs:
            p, s = r.predict(fs, np.asarray(spec_ids), device, return_candidates=True)
            bag_preds.append(p)
            cand_shares.append(_candidate_shares(fs, np.asarray(spec_ids), s))
        return np.mean(bag_preds, 0), np.mean(cand_shares, 0)


def _candidate_shares(fs, spec_ids, scores):
    """exp(score) normalized within each spectrum = fraction of predicted signal per candidate."""
    lengths = fs.spec_cand_ptr[spec_ids + 1] - fs.spec_cand_ptr[spec_ids]
    seg = np.repeat(np.arange(len(spec_ids)), lengths)
    shifted = scores - np.maximum.reduceat(scores, np.concatenate([[0], np.cumsum(lengths)[:-1]]))[seg]
    w = np.exp(shifted)
    return w / np.bincount(seg, weights=w)[seg]


def save_predictor(predictor, metrics=None, out_dir=FINAL_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"runs": [], "metrics": metrics or {}}
    for i, r in enumerate(predictor.runs):
        target = out_dir / f"member_{i}.pt"
        saved = torch.load(r.path, weights_only=False, map_location="cpu")
        torch.save(saved, target)
        manifest["runs"].append(target.name)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float))
    return out_dir


def load_predictor(path=FINAL_DIR, fs_feature_names=None):
    from .features import ATOM_FEATURE_NAMES, BOND_FEATURE_NAMES, candidate_feature_names

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    names = fs_feature_names or candidate_feature_names()
    runs = []
    for member in manifest["runs"]:
        saved = torch.load(path / member, weights_only=False, map_location="cpu")
        cfg_dict = dict(saved["config"])
        cfg_dict["features"] = tuple(cfg_dict.get("features", ()))
        cfg_dict["drop_groups"] = tuple(cfg_dict.get("drop_groups", ()))
        cfg = TrainConfig(**cfg_dict)
        mask = np.array([n in set(saved["features"]) for n in names])
        model = IntensityModel(int(mask.sum()), kind=cfg.kind, hidden=cfg.hidden, n_layers=cfg.n_layers,
                               dropout=cfg.dropout, agg=cfg.agg, use_features=cfg.use_features,
                               gnn_hidden=cfg.gnn_hidden, gnn_layers=cfg.gnn_layers,
                               atom_dim=len(ATOM_FEATURE_NAMES), bond_dim=len(BOND_FEATURE_NAMES),
                               context=bool(cfg.extra.get("context", False)),
                        pool=cfg.extra.get("pool", "fragment+cuts"))
        model.load_state_dict(saved["state_dict"])
        model.eval()
        runs.append(Run(cfg, model, saved["mean"], saved["std"], mask, saved["history"], path / member))
    return EnsemblePredictor(runs)


def predict_spectrum(predictor, spectrum, candidate_bags, collision_energy=None, return_store=False):
    """Predicted relative intensity for every peak of ``spectrum`` (base peak = 1)."""
    fs = featurize_candidate_bags(spectrum, candidate_bags, collision_energy)
    bag_pred = predictor.predict(fs, np.array([0]))
    out = np.zeros(len(spectrum.mz))
    out[fs.bag_peak_index] = bag_pred
    if out.max() > 0:
        out = out / out.max()
    return (out, fs) if return_store else out
