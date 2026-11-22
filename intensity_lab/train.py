"""Training with validation early stopping and on-disk run caching.

A run is identified by a hash of (config, feature cache, split). If a finished
run with that hash exists under ``results/experiment_1_agentic/runs`` it is
loaded instead of retrained, so the notebook can be re-executed top to bottom
cheaply. Pass ``force=True`` (or delete the file) to retrain.

Every epoch also writes ``<run>.partial.pt`` (model, optimizer, scheduler and RNG
states). Interrupting training (kernel restart, Ctrl-C, closing the laptop) and
calling ``train`` again resumes after the last completed epoch, with the same
result as an uninterrupted run.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import RESULTS_DIR
from .metrics import evaluate_bags, summarize
from .model import IntensityModel, TensorStore, loss_fn

FEATURE_GROUPS = {
    "mass/formula": ("frag_mz_frac", "neutral_loss_mass", "log_neutral_loss_mass", "frag_heavy_frac",
                     "frag_heavy", "is_intact", "rdbe_ion", "odd_electron", "frag_*", "loss_C", "loss_H",
                     "loss_N", "loss_O", "loss_S", "loss_F", "loss_Cl", "loss_Br"),
    "h_shift": ("dH", "dH=*"),
    "named_losses": ("loss=*",),
    "mass_error": ("mass_error_ppm", "abs_mass_error_ppm"),
    "bond_cuts": ("n_cut*", "cut_*"),
    "fragment_chemistry": ("frag_aromatic_atoms", "frag_ring_atoms", "frag_aromatic_frac", "frag_basic_N",
                           "frag_aromatic_N", "frag_amide_N", "loss_basic_N", "loss_aromatic_N",
                           "frag_charge", "frag_carbonyl"),
    "bag_context": ("bag_*",),
    "conditions": ("ce", "ce_sq", "precursor_mz", "root_heavy", "adduct_MH", "root_basic_N", "root_aromatic_N"),
}
# Roughly the candidate information Experiment_1.ipynb gave its model.
ORIGINAL_FEATURES = ("frag_mz_frac", "mass_error_ppm", "frag_heavy_frac", "frag_C", "frag_H", "frag_N",
                     "frag_O", "frag_S", "frag_F", "frag_Cl", "frag_Br", "ce", "precursor_mz", "adduct_MH")


def _match(name, pattern):
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def group_columns(names, group):
    """Column indices of a feature group (names from the FeatureStore).

    Exact names claimed by another group (e.g. ``frag_aromatic_atoms``) are not
    captured by a wildcard such as ``frag_*``.
    """
    exact = {p for g, pats in FEATURE_GROUPS.items() if g != group for p in pats if not p.endswith("*")}
    cols = []
    for i, n in enumerate(names):
        for p in FEATURE_GROUPS[group]:
            if _match(n, p) and (not p.endswith("*") or n not in exact):
                cols.append(i)
                break
    return cols


@dataclass
class TrainConfig:
    name: str
    kind: str = "mlp"                 # linear | mlp | gnn
    hidden: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    agg: str = "lse"                  # lse | max | mean
    loss: str = "kl"                  # kl | cos | cos_sqrt | kl+cos
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 40
    patience: int = 6
    seed: int = 0
    features: tuple = ()              # empty = all; else exact names to keep
    drop_groups: tuple = ()           # feature groups to remove
    use_features: bool = True         # False: GNN sees only CE + graph
    misaligned_graph: bool = False    # reproduce the old root-graph atom-order bug
    gnn_hidden: int = 96
    gnn_layers: int = 3
    train_fraction: float = 1.0       # fraction of training *molecules*
    shuffle_labels: bool = False      # sanity control: permute intensities within spectra
    grad_clip: float = 5.0
    split: str = "molecule"
    extra: dict = field(default_factory=dict)

    def key(self, data_key):
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        payload["data"] = data_key
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=list).encode()).hexdigest()[:12]


def feature_mask(cfg, names):
    keep = np.ones(len(names), bool)
    if cfg.features:
        keep[:] = False
        keep[[names.index(n) for n in cfg.features]] = True
    for g in cfg.drop_groups:
        keep[group_columns(names, g)] = False
    return keep


class Run:
    """A trained model plus everything needed to predict on new spectra."""

    def __init__(self, cfg, model, mean, std, mask, history, path=None):
        self.cfg, self.model, self.mean, self.std, self.mask = cfg, model, mean, std, mask
        self.history = history
        self.path = path
        self.ce_col = None

    def store(self, fs, device):
        return TensorStore(fs, device, graph=self.cfg.kind == "gnn", misaligned_graph=self.cfg.misaligned_graph,
                           feature_mask=self.mask, mean=self.mean, std=self.std)

    @torch.no_grad()
    def predict(self, fs, spec_ids, device="cpu", store=None, batch_size=256, return_candidates=False):
        store = store or self.store(fs, device)
        self.model.eval()
        names = list(np.array(fs.cand_feature_names)[self.mask])
        ce_col = names.index("ce") if "ce" in names else None
        out, cand_out = [], []
        for start in range(0, len(spec_ids), batch_size):
            b = store.batch(spec_ids[start:start + batch_size])
            log_p, s = self.model(b, ce_col)
            out.append(log_p.exp().cpu().numpy())
            if return_candidates:
                cand_out.append(s.cpu().numpy())
        pred = np.concatenate(out) if out else np.zeros(0)
        return (pred, np.concatenate(cand_out)) if return_candidates else pred

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())


def _subsample_molecules(fs, ids, fraction, seed):
    if fraction >= 1:
        return ids
    mols = fs.meta["mol_id"].values[ids]
    uniq = np.unique(mols)
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(uniq, size=max(1, int(round(fraction * len(uniq)))), replace=False))
    return ids[np.array([m in keep for m in mols])]


def _shuffled_intensity(fs, seed):
    rng = np.random.default_rng(seed)
    inten = fs.bag_int.copy()
    for s in range(fs.n_spectra):
        a, b = fs.spec_bag_ptr[s], fs.spec_bag_ptr[s + 1]
        inten[a:b] = rng.permutation(inten[a:b])
    return inten


def train(fs, split, cfg: TrainConfig, data_key, device="cpu", force=False, verbose=True,
          runs_dir=RESULTS_DIR / "runs"):
    """Train (or load) a model. split: {'train': ids, 'val': ids}."""
    names = list(fs.cand_feature_names)
    mask = feature_mask(cfg, names)
    path = Path(runs_dir) / f"{cfg.name}-{cfg.key(data_key)}.pt"
    kept = list(np.array(names)[mask])
    model_kwargs = dict(n_feat=int(mask.sum()), kind=cfg.kind, hidden=cfg.hidden, n_layers=cfg.n_layers,
                        dropout=cfg.dropout, agg=cfg.agg, use_features=cfg.use_features,
                        gnn_hidden=cfg.gnn_hidden, gnn_layers=cfg.gnn_layers,
                        atom_dim=fs.atom_feat.shape[1], bond_dim=fs.bond_feat.shape[1],
                        context=bool(cfg.extra.get("context", False)),
                        pool=cfg.extra.get("pool", "fragment+cuts"))

    if path.exists() and not force:
        saved = torch.load(path, weights_only=False, map_location="cpu")
        model = IntensityModel(**model_kwargs)
        model.load_state_dict(saved["state_dict"])
        run = Run(cfg, model.to(device), saved["mean"], saved["std"], mask, saved["history"], path)
        if verbose:
            best = max(saved["history"], key=lambda h: h["val_cos"])
            print(f"[cached] {cfg.name}: best val cos {best['val_cos']:.4f} at epoch {best['epoch']} "
                  f"({run.n_params():,} params)")
        return run

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    store = TensorStore(fs, device, graph=cfg.kind == "gnn", misaligned_graph=cfg.misaligned_graph,
                        feature_mask=mask)
    mean, std = store.normalization()
    model = IntensityModel(**model_kwargs).to(device)
    ce_col = kept.index("ce") if "ce" in kept else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_ids = _subsample_molecules(fs, np.asarray(split["train"]), cfg.train_fraction, cfg.seed)
    val_ids = np.asarray(split["val"])
    steps_per_epoch = math.ceil(len(train_ids) / cfg.batch_size)
    total = steps_per_epoch * cfg.max_epochs
    warmup = steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total)))
    inten = _shuffled_intensity(fs, cfg.seed) if cfg.shuffle_labels else None
    rng = np.random.default_rng(cfg.seed)
    run = Run(cfg, model, mean, std, mask, [], path)

    # Resume an interrupted run from its last completed epoch (bit-identical continuation).
    partial = path.with_name(path.stem + ".partial.pt")
    history, best_cos, best_state, bad, first_epoch = [], -1.0, None, 0, 1
    if partial.exists() and not force:
        ck = torch.load(partial, weights_only=False, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"])
        history, best_cos, best_state, bad = ck["history"], ck["best_cos"], ck["best_state"], ck["bad"]
        first_epoch = history[-1]["epoch"] + 1
        if verbose:
            print(f"[resume] {cfg.name}: continuing after epoch {history[-1]['epoch']} (best val cos {best_cos:.4f})")
    done = bool(history) and (bad >= cfg.patience or first_epoch > cfg.max_epochs)
    for epoch in range(first_epoch, cfg.max_epochs + 1):
        if done:
            break
        start = time.time()
        model.train()
        order = rng.permutation(train_ids)
        losses, grads = [], []
        for i in range(0, len(order), cfg.batch_size):
            ids = order[i:i + cfg.batch_size]
            if inten is not None:
                bags = np.concatenate([np.arange(fs.spec_bag_ptr[s], fs.spec_bag_ptr[s + 1]) for s in ids])
                b = store.batch(ids, intensity_override=inten[bags])
            else:
                b = store.batch(ids)
            log_p, _ = model(b, ce_col)
            loss = loss_fn(cfg.loss, log_p, b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)))
            opt.step()
            sched.step()
            losses.append(loss.item())
        train_time = time.time() - start

        pred = run.predict(fs, val_ids, device, store=store)
        val = summarize(evaluate_bags(fs, val_ids, pred)).iloc[0]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_cos": float(val["cos"]),
                        "val_cos_sqrt": float(val["cos_sqrt"]), "val_top1": float(val["top1"]),
                        "grad_norm": float(np.median(grads)), "grad_norm_max": float(np.max(grads)),
                        "lr": float(sched.get_last_lr()[0]), "seconds": train_time})
        improved = val["cos"] > best_cos + 1e-4
        if improved:
            best_cos, bad = float(val["cos"]), 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose:
            print(f"{cfg.name} ep {epoch:2d} | loss {history[-1]['train_loss']:.4f} | val cos {val['cos']:.4f} "
                  f"sqrt {val['cos_sqrt']:.4f} top1 {val['top1']:.3f} | grad {history[-1]['grad_norm']:.2f} "
                  f"| {train_time:.1f}s{' *' if improved else ''}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = partial.with_name(partial.name + ".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "history": history,
                    "best_cos": best_cos, "best_state": best_state, "bad": bad}, tmp)
        tmp.replace(partial)  # atomic: an interrupt never leaves a half-written checkpoint
        if bad >= cfg.patience:
            break

    model.load_state_dict(best_state)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "mean": mean, "std": std, "history": history,
                "config": asdict(cfg), "features": kept}, path)
    partial.unlink(missing_ok=True)
    run.history = history
    return run


def train_metrics_on(run, fs, ids, device="cpu", max_spectra=3000, seed=0):
    """Metrics on (a subsample of) training spectra, to measure the generalization gap."""
    ids = np.asarray(ids)
    if len(ids) > max_spectra:
        ids = np.sort(np.random.default_rng(seed).choice(ids, max_spectra, replace=False))
    return evaluate_bags(fs, ids, run.predict(fs, ids, device), run.cfg.name)
