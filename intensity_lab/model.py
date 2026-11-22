"""Candidate-scoring intensity models.

Physical picture: each MAGMa candidate ion c gets a non-negative abundance
exp(s_c). A peak's intensity is the *sum* over the candidate ions at that m/z,
so the bag logit is logsumexp(s_c) ("lse"), and the predicted spectrum is the
softmax of bag logits within the spectrum. This keeps per-candidate
attributions (which fragment explains how much of a peak) for interpretation.

Scorers, from simplest to richest:
    linear  s_c = w . x_c                      (interpretable coefficients)
    mlp     s_c = MLP(x_c)                     (feature interactions, incl. CE)
    gnn     s_c = MLP([x_c, pooled root-GNN embeddings of fragment / cut atoms])
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Batching: gather a set of spectra from the flat FeatureStore arrays
# ---------------------------------------------------------------------------

def _ranges(starts, ends):
    lengths = ends - starts
    offsets = np.repeat(starts - np.concatenate([[0], np.cumsum(lengths)[:-1]]), lengths)
    return offsets + np.arange(lengths.sum()), lengths


@dataclass
class Batch:
    cand_x: torch.Tensor
    cand_bag: torch.Tensor          # local bag id per candidate
    bag_spec: torch.Tensor          # local spectrum id per bag
    target: torch.Tensor            # observed intensity normalized per spectrum (sum 1)
    n_spec: int
    n_bag: int
    # Graph parts (None unless a graph model is used)
    atom_x: torch.Tensor | None = None
    edge: torch.Tensor | None = None
    edge_x: torch.Tensor | None = None
    atom_spec: torch.Tensor | None = None
    ca_atom: torch.Tensor | None = None   # candidate-atom pairs
    ca_cand: torch.Tensor | None = None
    cut_in: torch.Tensor | None = None
    cut_out: torch.Tensor | None = None
    cut_cand: torch.Tensor | None = None
    cand_spec: torch.Tensor | None = None


class TensorStore:
    """FeatureStore arrays as tensors (on `device`) + fast batch assembly."""

    def __init__(self, fs, device="cpu", graph=False, misaligned_graph=False,
                 feature_mask=None, mean=None, std=None):
        self.fs = fs
        self.device = torch.device(device)
        x = fs.cand_feat
        if feature_mask is not None:
            x = x[:, feature_mask]
        self.mean = x.mean(0) if mean is None else mean
        self.std = (x.std(0) + 1e-6) if std is None else std
        self.cand_x = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32)).to(self.device)
        self.graph = graph
        if graph:
            suffix = "_orig" if misaligned_graph else ""
            self.atom_x = torch.from_numpy(getattr(fs, "atom_feat" + suffix)).to(self.device)
            self.edge_np = getattr(fs, "edge" + suffix)
            self.bond_x = torch.from_numpy(getattr(fs, "bond_feat" + suffix)).to(self.device)

    def normalization(self):
        return self.mean, self.std

    def batch(self, spec_ids, intensity_override=None):
        fs = self.fs
        spec_ids = np.asarray(spec_ids)
        bags, n_bags = _ranges(fs.spec_bag_ptr[spec_ids], fs.spec_bag_ptr[spec_ids + 1])
        cands, n_cands = _ranges(fs.spec_cand_ptr[spec_ids], fs.spec_cand_ptr[spec_ids + 1])
        bag_spec = np.repeat(np.arange(len(spec_ids)), n_bags)
        # Map global bag ids to local positions (bags of a spectrum are contiguous).
        bag_shift = np.repeat(fs.spec_bag_ptr[spec_ids] - np.concatenate([[0], np.cumsum(n_bags)[:-1]]), n_cands)
        cand_bag = fs.cand_bag[cands] - bag_shift
        inten = fs.bag_int[bags] if intensity_override is None else intensity_override
        totals = np.bincount(bag_spec, weights=inten, minlength=len(spec_ids))
        target = inten / totals[bag_spec]
        dev = self.device
        b = Batch(
            cand_x=self.cand_x[torch.from_numpy(cands).to(dev)],
            cand_bag=torch.from_numpy(cand_bag).to(dev),
            bag_spec=torch.from_numpy(bag_spec).to(dev),
            target=torch.from_numpy(target.astype(np.float32)).to(dev),
            n_spec=len(spec_ids), n_bag=len(bags),
            cand_spec=torch.from_numpy(np.repeat(np.arange(len(spec_ids)), n_cands)).to(dev),
        )
        if self.graph:
            atoms, n_atoms = _ranges(fs.spec_atom_ptr[spec_ids], fs.spec_atom_ptr[spec_ids + 1])
            atom_shift = fs.spec_atom_ptr[spec_ids] - np.concatenate([[0], np.cumsum(n_atoms)[:-1]])
            edges, n_edges = _ranges(fs.spec_edge_ptr[spec_ids], fs.spec_edge_ptr[spec_ids + 1])
            e = self.edge_np[:, edges] - np.repeat(atom_shift, n_edges)[None, :]
            ca, n_ca = _ranges(fs.cand_atom_ptr[cands], fs.cand_atom_ptr[cands + 1])
            ca_cand = np.repeat(np.arange(len(cands)), n_ca)
            cand_spec = np.repeat(np.arange(len(spec_ids)), n_cands)
            ca_atom = fs.cand_atoms[ca] - atom_shift[cand_spec[ca_cand]]
            cut, n_cut = _ranges(fs.cand_cut_ptr[cands], fs.cand_cut_ptr[cands + 1])
            cut_cand = np.repeat(np.arange(len(cands)), n_cut)
            shift = atom_shift[cand_spec[cut_cand]]
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(dev)
            b.atom_x = self.atom_x[t(atoms)]
            b.edge = t(np.concatenate([e, e[::-1]], axis=1))
            bx = self.bond_x[t(edges)]
            b.edge_x = torch.cat([bx, bx], 0)
            b.atom_spec = t(np.repeat(np.arange(len(spec_ids)), n_atoms))
            b.ca_atom, b.ca_cand = t(ca_atom), t(ca_cand)
            b.cut_in, b.cut_out = t(fs.cut_in[cut] - shift), t(fs.cut_out[cut] - shift)
            b.cut_cand = t(cut_cand)
        return b


# ---------------------------------------------------------------------------
# Scatter helpers (index_add based; work on CPU, CUDA and MPS)
# ---------------------------------------------------------------------------

def scatter_sum(x, index, n):
    out = x.new_zeros((n,) + x.shape[1:])
    return out.index_add_(0, index, x)


def scatter_mean(x, index, n):
    count = scatter_sum(torch.ones_like(index, dtype=x.dtype), index, n).clamp_min(1)
    return scatter_sum(x, index, n) / count.view(-1, *([1] * (x.dim() - 1)))


def scatter_max(x, index, n):
    out = torch.full((n,), -1e30, dtype=x.dtype, device=x.device)
    return out.scatter_reduce(0, index, x, reduce="amax", include_self=True)


def scatter_logsumexp(x, index, n):
    m = scatter_max(x, index, n).detach()
    return torch.log(scatter_sum(torch.exp(x - m[index]), index, n).clamp_min(1e-30)) + m


def spectrum_log_softmax(bag_logit, bag_spec, n_spec):
    return bag_logit - scatter_logsumexp(bag_logit, bag_spec, n_spec)[bag_spec]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ResidualMLP(nn.Module):
    def __init__(self, n_in, hidden, n_layers, dropout):
        super().__init__()
        self.inp = nn.Linear(n_in, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(),
                          nn.Dropout(dropout), nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.inp(x)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h).squeeze(-1)


class RootGNN(nn.Module):
    """Small GINE-style message passing over the (MAGMa-aligned) root graph."""

    def __init__(self, atom_dim, bond_dim, hidden, n_layers, dropout):
        super().__init__()
        self.inp = nn.Linear(atom_dim, hidden)
        self.edge = nn.ModuleList([nn.Linear(bond_dim, hidden) for _ in range(n_layers)])
        self.upd = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            for _ in range(n_layers)
        ])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])

    def forward(self, atom_x, edge, edge_x):
        h = self.inp(atom_x)
        src, dst = edge
        for lin_e, upd, norm in zip(self.edge, self.upd, self.norm):
            msg = torch.relu(h[src] + lin_e(edge_x))
            h = norm(h + upd(h + scatter_sum(msg, dst, h.shape[0])))
        return h


class SpectrumContext(nn.Module):
    """Two-stage DeepSets: candidates see a summary of all candidates of their spectrum.

    Lets the score of one fragment depend on its competitors (e.g. consecutive
    fragmentation at high energy) instead of being computed in isolation.
    """

    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.SiLU())
        self.n_out = 3 * hidden

    def forward(self, x, cand_spec, n_spec):
        h = self.enc(x)
        mean = scatter_mean(h, cand_spec, n_spec)
        mx = h.new_full((n_spec, h.shape[1]), -1e30).scatter_reduce(
            0, cand_spec[:, None].expand_as(h), h, reduce="amax", include_self=True)
        return torch.cat([h, mean[cand_spec], mx[cand_spec]], -1)


class IntensityModel(nn.Module):
    def __init__(self, n_feat, kind="mlp", hidden=256, n_layers=3, dropout=0.1, agg="lse",
                 use_features=True, gnn_hidden=96, gnn_layers=3, atom_dim=None, bond_dim=None,
                 context=False, pool="fragment+cuts"):
        super().__init__()
        self.kind, self.agg, self.use_features, self.pool = kind, agg, use_features, pool
        n_in = n_feat if use_features else 0
        if kind == "gnn":
            self.gnn = RootGNN(atom_dim, bond_dim, gnn_hidden, gnn_layers, dropout)
            # fragment mean + root mean (+ mean over atoms inside / outside each broken bond)
            n_in += (4 if pool == "fragment+cuts" else 2) * gnn_hidden
            if not use_features:
                n_in += 1  # collision energy must always be available
        self.context = SpectrumContext(n_in, hidden, dropout) if context else None
        if context:
            n_in = n_in + self.context.n_out
        if kind == "linear":
            self.scorer = nn.Linear(n_in, 1)
        else:
            self.scorer = ResidualMLP(n_in, hidden, n_layers, dropout)

    def candidate_scores(self, b: Batch, ce_col=None):
        parts = [b.cand_x] if self.use_features else []
        if self.kind == "gnn":
            h = self.gnn(b.atom_x, b.edge, b.edge_x)
            n_c = b.cand_x.shape[0]
            frag = scatter_mean(h[b.ca_atom], b.ca_cand, n_c)
            root = scatter_mean(h, b.atom_spec, b.n_spec)[b.cand_spec]
            parts.append(frag)
            if self.pool == "fragment+cuts":  # order must stay frag, cut_in, cut_out, root (checkpoints)
                parts += [scatter_mean(h[b.cut_in], b.cut_cand, n_c), scatter_mean(h[b.cut_out], b.cut_cand, n_c)]
            parts.append(root)
            if not self.use_features:
                parts.append(b.cand_x[:, [ce_col]])
        x = torch.cat(parts, -1) if len(parts) > 1 else parts[0]
        if self.context is not None:
            x = torch.cat([x, self.context(x, b.cand_spec, b.n_spec)], -1)
        s = self.scorer(x)
        return s.squeeze(-1) if s.dim() > 1 else s

    def forward(self, b: Batch, ce_col=None):
        s = self.candidate_scores(b, ce_col)
        if self.agg == "lse":
            bag_logit = scatter_logsumexp(s, b.cand_bag, b.n_bag)
        elif self.agg == "max":
            bag_logit = scatter_max(s, b.cand_bag, b.n_bag)
        elif self.agg == "mean":
            bag_logit = scatter_mean(s, b.cand_bag, b.n_bag)
        else:
            raise ValueError(self.agg)
        return spectrum_log_softmax(bag_logit, b.bag_spec, b.n_spec), s


# ---------------------------------------------------------------------------
# Losses on log-probabilities over the supported peaks of each spectrum
# ---------------------------------------------------------------------------

def loss_fn(name, log_p, b: Batch):
    t = b.target
    if name == "kl":
        per = scatter_sum(-(t * log_p), b.bag_spec, b.n_spec)
        ent = scatter_sum(-(t * torch.log(t.clamp_min(1e-12))), b.bag_spec, b.n_spec)
        return (per - ent).mean()
    p = log_p.exp()
    if name in ("cos", "cos_sqrt"):
        if name == "cos_sqrt":
            p, t = torch.sqrt(p + 1e-12), torch.sqrt(t)
        dot = scatter_sum(p * t, b.bag_spec, b.n_spec)
        norm = torch.sqrt(scatter_sum(p * p, b.bag_spec, b.n_spec) * scatter_sum(t * t, b.bag_spec, b.n_spec))
        return (1 - dot / norm.clamp_min(1e-12)).mean()
    if name == "kl+cos":
        return loss_fn("kl", log_p, b) + loss_fn("cos", log_p, b)
    raise ValueError(name)
