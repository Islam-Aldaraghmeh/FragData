"""Small energy-conditioned MAGMa intensity scorer; old model stays reproducible."""
from functools import lru_cache

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch
from torch import nn

from intensity_model import (fingerprint_molecule, fingerprint_fragment,
                             normalize_intensity, pool_candidate_scores)
from magma_workflow.candidates import assign_candidates_to_nearest_peak
from src.chemistry.formula import formula_counts, ionic_mass

FP_SIZE = 128
DESCRIPTORS = ("fragment_mass/500", "fragment_mass/parent_mass", "neutral_loss/200",
               "heavy_atom_fraction", "heavy_atoms/50", "formula_H/50", "N/10",
               "O/10", "S/5", "halogens/5", "aromatic_fraction", "cycle_rank/5",
               "min_boundary_cuts/6", "min_tree_depth/3", "intact_heavy_graph",
               "charge/2", "absolute_mass_error_ppm/50", "mass_error_ppm/50",
               "parent_heavy_atoms/50", "parent_mass/500")
DESCRIPTOR_START = 2 * FP_SIZE
CE_INDEX = DESCRIPTOR_START + len(DESCRIPTORS)
FEATURE_SIZE = CE_INDEX + 1
SCHEMA = "nearest_energy_v2"


def _fold(fingerprint):
    return fingerprint.reshape(-1, FP_SIZE).max(axis=0)


@lru_cache(maxsize=4096)
def _parent(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid precursor SMILES")
    return (_fold(fingerprint_molecule(smiles)), Descriptors.ExactMolWt(molecule),
            molecule.GetNumHeavyAtoms())


def candidate_features(precursor_smiles, collision_energy, bags):
    """Deduplicate and assign each theoretical ion to its nearest observed peak.

    Only structural descriptors, measured mass error and CE enter the scorer.
    No experimental intensity, quality, peak rank or MS3 information is read.
    All observed locations remain in the output axis, even if now unsupported.
    """
    ce = float(collision_energy)
    if not np.isfinite(ce) or ce < 0:
        raise ValueError("collision_energy must be finite and nonnegative")
    precursor_fp, parent_mass, parent_size = _parent(precursor_smiles)
    rows, indices = [], []
    for peak_index, bag in enumerate(assign_candidates_to_nearest_peak(bags)):
        for fragment in bag[1]:
            formula = formula_counts(fragment.ionic_formula)
            mass = ionic_mass(formula, fragment.charge)
            atoms, bonds = fragment.graph["atoms"], fragment.graph["bonds"]
            n = len(atoms)
            origins = fragment.metadata.get("origins", [])
            cuts = min((o.get("boundary_cut_count", 0) for o in origins), default=0)
            depth = min((o.get("tree_depth", 0) for o in origins), default=0)
            error_ppm = (float(bag[0]) - mass) / mass * 1e6
            descriptors = [mass/500, mass/parent_mass, (parent_mass-mass)/200,
                n/parent_size, n/50, formula.get("H",0)/50, formula.get("N",0)/10,
                formula.get("O",0)/10, formula.get("S",0)/5,
                sum(formula.get(e,0) for e in ("F","Cl","Br","I"))/5,
                sum(bool(a["aromatic"]) for a in atoms)/max(n,1),
                max(0,len(bonds)-n+1)/5, cuts/6, depth/3, float(n==parent_size),
                fragment.charge/2, abs(error_ppm)/50, error_ppm/50,
                parent_size/50, parent_mass/500]
            rows.append(np.concatenate((precursor_fp, _fold(fingerprint_fragment(fragment)),
                                        descriptors, [ce/50])))
            indices.append(peak_index)
    return (np.asarray(rows,dtype=np.float32).reshape(-1,FEATURE_SIZE),
            np.asarray(indices,dtype=np.int64))


def with_collision_energy(features, collision_energy):
    """Copy features and set CE; interactions are recomputed inside the model."""
    ce = float(collision_energy)
    if not np.isfinite(ce) or ce < 0:
        raise ValueError("collision_energy must be finite and nonnegative")
    changed = features.copy()
    changed[:,CE_INDEX] = ce/50
    return changed


class EnergyConditionedMLP(nn.Module):
    """Learn fragment-specific intercept, energy slope and curvature.

    The optional small fingerprint projection is downweighted and regularized.
    Descriptor-only uses exactly the same conditioning and aggregation.
    """
    def __init__(self, use_fingerprints=True, dropout=0.15):
        super().__init__()
        self.config = dict(use_fingerprints=use_fingerprints, dropout=dropout)
        self.fingerprint_branch = (nn.Sequential(nn.Linear(2*FP_SIZE,16), nn.Tanh(),
                                                nn.Dropout(dropout)) if use_fingerprints else None)
        inputs = len(DESCRIPTORS) + (16 if use_fingerprints else 0)
        self.coefficients = nn.Sequential(nn.Linear(inputs,48),nn.SiLU(),
                                          nn.Linear(48,24),nn.SiLU(),nn.Linear(24,3))

    def forward(self, features):
        descriptors = features[:,DESCRIPTOR_START:CE_INDEX]
        if self.fingerprint_branch is not None:
            fingerprints = .25 * self.fingerprint_branch(features[:,:DESCRIPTOR_START])
            descriptors = torch.cat((descriptors,fingerprints),dim=1)
        coefficients = self.coefficients(descriptors)
        energy = (features[:,CE_INDEX]*50-40)/20
        log_score = coefficients[:,0] + energy*coefficients[:,1] + energy.square()*coefficients[:,2]
        return torch.nn.functional.softplus(log_score)


@torch.no_grad()
def predict_spectrum(model, precursor_smiles, collision_energy, candidate_bags):
    """N×2 m/z/base-peak-normalized intensity, with nearest assignment and max."""
    features, indices = candidate_features(precursor_smiles,collision_energy,candidate_bags)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        scores = model(torch.as_tensor(features,device=device))
        peaks = pool_candidate_scores(scores,torch.as_tensor(indices,device=device),
                                      len(candidate_bags)).cpu().numpy()
    finally:
        model.train(was_training)
    return np.column_stack(([b[0] for b in candidate_bags],normalize_intensity(peaks)))


def load_model(path):
    checkpoint = torch.load(path,map_location="cpu",weights_only=True)
    if checkpoint["schema"] != SCHEMA or checkpoint["feature_size"] != FEATURE_SIZE:
        raise ValueError("Incompatible intensity checkpoint")
    model = EnergyConditionedMLP(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()
