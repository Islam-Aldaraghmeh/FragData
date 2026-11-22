"""Intensity prediction from pre-assigned MAGMa candidates (Experiment_1_agentic).

Modules, in pipeline order:

    raw         schema-robust loading of cached MAGMa examples + MSnLib metadata
    features    chemistry-aware candidate / bag / spectrum features (flat arrays)
    splits      molecule- and scaffold-disjoint splits with leakage audits
    metrics     per-spectrum metrics, baselines' common interface, bootstrap CIs
    baselines   uniform, heuristics, gradient-boosted trees
    model       candidate-scoring networks (linear, MLP, optional root GNN)
    train       batched training with early stopping and checkpoint caching
    diagnostics debugging tools for trained networks
    plots       figures used by the notebook

Everything heavy is cached under ``data/agentic_cache`` and
``results/experiment_1_agentic``; delete those folders to rebuild from scratch.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "agentic_cache"
RESULTS_DIR = REPO_ROOT / "results" / "experiment_1_agentic"
