"""Intensity-only MAGMa proof of concept: graph fingerprints, MLP, latent max.

No candidate identity labels, local ionic states, or experimental intensities
are used as features. Fragment fingerprints describe the supplied heavy graph;
global H count and charge are separate features, not assigned microstates.
"""
from functools import lru_cache
import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import torch
from torch import nn

import magma_workflow  # Registers the existing distilled chemistry package.
from src.chemistry.formula import formula_counts, ionic_mass
from src.chemistry.heavy_graph import _canonical_graph, _fresh_graph

FP_SIZE = 512
EXTRA_FEATURES = ("collision_energy/100", "fragment_mass/precursor_mass",
                  "fragment_mass/500", "charge/2", "formula_H/50",
                  "heavy_atoms/50", "min_boundary_cuts/6", "min_tree_depth/3")
FEATURE_SIZE = 2 * FP_SIZE + len(EXTRA_FEATURES)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)


@lru_cache(maxsize=2048)
def fingerprint_molecule(smiles):
    """512-bit radius-2 Morgan fingerprint of a known precursor structure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid precursor SMILES: {smiles}")
    return _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32)


def fingerprint_fragment(candidate):
    """Fingerprint exact heavy graph without sanitizing/repairing ionic valence."""
    graph = candidate.graph
    mol = _fresh_graph(graph["atoms"], graph["bonds"])
    # _fresh_graph disables implicit H and updates its cache non-strictly.
    # Do not sanitize, neutralize, protonate, or enumerate this descriptor.
    return _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32)


def structure_key(candidate):
    """Exact graph/formula/charge key, independent of IDs or FP collisions."""
    return json.dumps([_canonical_graph(candidate.graph),
                       formula_counts(candidate.ionic_formula), int(candidate.charge)],
                      sort_keys=True)


def candidate_features(precursor_smiles, collision_energy, bags):
    """Return deduplicated candidate rows and their bag indices.

    Only the explicitly named features below are read; bag MS3, observed peak
    intensities, peak rank, candidate IDs, and dataset quality fields are ignored.
    The supplied bag order is retained, including empty bags.
    """
    ce = float(collision_energy)
    if not np.isfinite(ce) or ce < 0:
        raise ValueError("collision_energy must be a finite nonnegative scalar")
    precursor_fp = fingerprint_molecule(precursor_smiles)
    mol = Chem.MolFromSmiles(precursor_smiles)
    from rdkit.Chem import Descriptors
    precursor_mass = Descriptors.ExactMolWt(mol)
    rows, peak_indices = [], []
    for peak_index, bag in enumerate(bags):
        seen = set()
        for fragment in bag[1]:
            key = structure_key(fragment)
            if key in seen:
                continue
            seen.add(key)
            formula = formula_counts(fragment.ionic_formula)
            mass = ionic_mass(formula, fragment.charge)
            origins = fragment.metadata.get("origins", [])
            cuts = min((o.get("boundary_cut_count", 0) for o in origins), default=0)
            depth = min((o.get("tree_depth", 0) for o in origins), default=0)
            extras = [ce / 100, mass / precursor_mass, mass / 500,
                      fragment.charge / 2, formula.get("H", 0) / 50,
                      len(fragment.graph["atoms"]) / 50, cuts / 6, depth / 3]
            rows.append(np.concatenate((precursor_fp, fingerprint_fragment(fragment), extras)))
            peak_indices.append(peak_index)
    return (np.asarray(rows, dtype=np.float32).reshape(-1, FEATURE_SIZE),
            np.asarray(peak_indices, dtype=np.int64))


class IntensityMLP(nn.Module):
    """Small positive scorer; every candidate shares these same weights."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(FEATURE_SIZE, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(),
                                     nn.Linear(64, 1), nn.Softplus())

    def forward(self, features):
        return self.network(features).squeeze(-1)


def pool_candidate_scores(scores, peak_indices, n_peaks):
    """One max per observed peak; unsupported peaks remain zero."""
    return scores.new_zeros(n_peaks).scatter_reduce(
        0, peak_indices, scores, reduce="amax", include_self=True)


def normalize_intensity(values):
    """Base-peak normalization to [0,1], preserving an all-zero vector."""
    values = np.asarray(values, dtype=np.float64)
    return values / max(float(values.max(initial=0)), 1e-12)


@torch.no_grad()
def predict_spectrum(model, precursor_smiles, collision_energy, candidate_bags):
    """Return [m/z, intensity] rows in bag order; no target spectrum is required.

    `candidate_bags` is get_candidate_bags(...) output. Optional MS3 entries
    are ignored. Intensities are base-peak normalized. Empty bags receive zero.
    """
    features, indices = candidate_features(precursor_smiles, collision_energy, candidate_bags)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        scores = model(torch.as_tensor(features, device=device))
        peaks = pool_candidate_scores(scores, torch.as_tensor(indices, device=device),
                                      len(candidate_bags)).cpu().numpy()
    finally:
        model.train(was_training)
    return np.column_stack(([bag[0] for bag in candidate_bags], normalize_intensity(peaks)))


def spectrum_metrics(targets, predictions):
    """Macro-average over spectra on the full observed peak axis.

    MAE uses independent base-peak normalization. Undefined cosine for an
    all-zero prediction is zero; thresholds are strictly greater than .5/.7.
    """
    if len(targets) != len(predictions) or not len(targets):
        raise ValueError("Expected equally sized nonempty target and prediction lists")
    cosine, mae = [], []
    for target, prediction in zip(targets, predictions):
        target, prediction = normalize_intensity(target), normalize_intensity(prediction)
        if target.shape != prediction.shape or not target.size:
            raise ValueError("Prediction and target must share a nonempty peak axis")
        denominator = np.linalg.norm(target) * np.linalg.norm(prediction)
        score = np.dot(target, prediction) / denominator if denominator > 0 else 0.
        cosine.append(float(np.clip(score, 0, 1)))
        mae.append(float(np.mean(np.abs(target - prediction))))
    return {"mean_cosine": float(np.mean(cosine)),
            "median_cosine": float(np.median(cosine)),
            "mean_absolute_error": float(np.mean(mae)),
            "fraction_cosine_gt_0.5": float(np.mean(np.asarray(cosine) > .5)),
            "fraction_cosine_gt_0.7": float(np.mean(np.asarray(cosine) > .7))}
