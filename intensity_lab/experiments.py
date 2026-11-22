"""Registry of every configuration the notebook trains (names = cache file names).

Keeping them here means a command-line run and the notebook share cached runs.
Pre-train everything (or one group) outside Jupyter with:

    .venv/bin/python -m intensity_lab.experiments all
    .venv/bin/python -m intensity_lab.experiments ablations 4      # group, torch threads
"""
from __future__ import annotations

import sys
from dataclasses import replace

from .train import ORIGINAL_FEATURES, TrainConfig

# Feature-only MLP baseline (small; a larger one overfits more, see "mlp_large").
MLP = TrainConfig("mlp", kind="mlp", hidden=128, n_layers=2, dropout=0.2, weight_decay=1e-4,
                  loss="kl", agg="lse", max_epochs=50, patience=8)
# Default model: candidate features + GNN over the MAGMa-aligned root graph,
# pooled over the fragment's atoms, the atoms on both sides of each broken bond, and the whole root.
GNN = TrainConfig("gnn", kind="gnn", hidden=256, n_layers=3, dropout=0.1, weight_decay=1e-4,
                  loss="kl", agg="lse", max_epochs=30, patience=6, gnn_hidden=96, gnn_layers=3)

LADDER = {
    "linear": replace(MLP, name="linear", kind="linear", lr=3e-3, max_epochs=20, patience=6),
    "mlp_large": replace(MLP, name="mlp_large", hidden=256, n_layers=3, dropout=0.1, max_epochs=30, patience=6),
    "mlp": MLP,
    "gnn": GNN,
}

# One change at a time relative to the default GNN.
_ABL = {
    "gnn_misaligned": (dict(misaligned_graph=True), "root graph with the old atom-order bug"),
    "gnn_fragment_pool": (dict(extra={"pool": "fragment"}), "pool fragment atoms only (no broken-bond atoms)"),
    "gnn_graph_only": (dict(use_features=False), "graph + CE only, no engineered features"),
    "gnn_original_inputs": (dict(features=ORIGINAL_FEATURES), "Experiment 1 inputs (formula, m/z, error, size, CE) + aligned graph"),
    "gnn_original_inputs_misaligned": (dict(features=ORIGINAL_FEATURES, misaligned_graph=True, extra={"pool": "fragment"}),
                                       "≈ old Experiment 1: its inputs, fragment pooling, misaligned graph"),
    "gnn_agg_max": (dict(agg="max"), "peak = max over candidates instead of sum"),
    "gnn_agg_mean": (dict(agg="mean"), "peak = mean over candidates"),
    "gnn_loss_cos": (dict(loss="cos"), "cosine loss instead of KL"),
    "gnn_context": (dict(extra={"context": True}), "candidates see a summary of all candidates of the spectrum"),
    "gnn_deeper": (dict(gnn_layers=5, gnn_hidden=128), "5 message-passing layers, 128 wide"),
    "gnn_reg": (dict(dropout=0.2, weight_decay=1e-3), "more dropout and weight decay"),
}
ABLATION_CONFIGS = {n: replace(GNN, name=n, **kw) for n, (kw, _) in _ABL.items()}
ABLATIONS = tuple(_ABL)
DESCRIPTIONS = {n: d for n, (_, d) in _ABL.items()}

CONTROLS = {
    "mlp_shuffled_labels": replace(MLP, name="mlp_shuffled_labels", shuffle_labels=True, max_epochs=10, patience=3),
}

# Final model = best validation cosine among these; ordered simplest first, and a
# more complex model must beat a simpler one by > SIMPLICITY_MARGIN to be chosen.
FINAL_CANDIDATES = ("gnn", "gnn_reg", "gnn_deeper", "gnn_context")
SIMPLICITY_MARGIN = 0.002
N_SEEDS = 3
SCALING_FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def all_named():
    return {**LADDER, **ABLATION_CONFIGS, **CONTROLS}


def get(name):
    return all_named()[name]


def pick_final(val_scores):
    """Simplest candidate within SIMPLICITY_MARGIN of the best validation score."""
    best = max(val_scores.values())
    return next(n for n in FINAL_CANDIDATES if val_scores[n] >= best - SIMPLICITY_MARGIN)


def seeds(name):
    base = get(name)
    return [base] + [replace(base, name=f"{name}_seed{k}", seed=k) for k in range(1, N_SEEDS)]


def scaffold_config(name):
    return replace(get(name), name=f"{name}_scaffold", split="scaffold")


def scaling_config(name, fraction):
    return replace(get(name), name=f"{name}_frac{int(round(fraction * 100))}", train_fraction=fraction)


def groups(final_name="gnn"):
    return {
        "ladder": list(LADDER.values()),
        "ablations": list(ABLATION_CONFIGS.values()),
        "controls": list(CONTROLS.values()),
        "final": seeds(final_name)[1:] + [scaffold_config(final_name)]
                 + [scaling_config(final_name, f) for f in SCALING_FRACTIONS if f < 1],
    }


def main(argv):
    """python -m intensity_lab.experiments GROUP_OR_CONFIG[,...] [THREADS] [FINAL_NAME]"""
    from .pipeline import load_data, set_threads
    from .train import train

    set_threads(int(argv[2]) if len(argv) > 2 else None)
    final_name = argv[3] if len(argv) > 3 else "gnn"
    data = load_data(verbose=False)
    table = groups(final_name)
    names = list(table) if len(argv) < 2 or argv[1] == "all" else argv[1].split(",")
    configs = [cfg for n in names for cfg in (table[n] if n in table else [get(n)])]
    for cfg in configs:
        if True:
            run = train(data.fs, data.splits[cfg.split], cfg, data.data_key(cfg.split), verbose=False)
            best = max(run.history, key=lambda h: h["val_cos"])
            print(f"DONE {cfg.name}: val cos {best['val_cos']:.4f} (epoch {best['epoch']}/{len(run.history)})", flush=True)


if __name__ == "__main__":
    main(sys.argv)
