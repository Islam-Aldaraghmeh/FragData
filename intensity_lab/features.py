"""Flat, cached feature arrays for candidate-level intensity models.

Layout (all spectra concatenated, contiguous per spectrum):

    spectrum  -> bags        spec_bag_ptr[s] : spec_bag_ptr[s+1]
    bag       -> candidates  bag_cand_ptr[b] : bag_cand_ptr[b+1]
    spectrum  -> root atoms  spec_atom_ptr[s] : spec_atom_ptr[s+1]
    candidate -> its atoms   cand_atom_ptr[c] : cand_atom_ptr[c+1]  (global atom ids)
    candidate -> cut bonds   cand_cut_ptr[c]  : cand_cut_ptr[c+1]   (inside/outside atom ids)

Every candidate feature is derived from the MAGMa-aligned root graph
(``chemistry.mol_graph`` of the engine SMILES), the candidate's ionic formula and
the observed peak. No feature reads a measured intensity.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import CACHE_DIR
from .chemistry import ELEMENTS, HYBRIDIZATIONS, engine_smiles_map, mol_graph

FEATURE_VERSION = "feat-v1"
FORMULA_ELEMENTS = ("C", "H", "N", "O", "S", "F", "Cl", "Br")
HALOGENS = ("F", "Cl", "Br", "I")

# Named neutral losses (precursor ion formula minus fragment ion formula).
NAMED_LOSSES = {
    "H2O": {"H": 2, "O": 1}, "NH3": {"N": 1, "H": 3}, "CO": {"C": 1, "O": 1},
    "CO2": {"C": 1, "O": 2}, "H2O+CO": {"C": 1, "H": 2, "O": 2}, "CH3OH": {"C": 1, "H": 4, "O": 1},
    "CH2O": {"C": 1, "H": 2, "O": 1}, "HCN": {"C": 1, "H": 1, "N": 1},
    "HNCO": {"C": 1, "H": 1, "N": 1, "O": 1}, "CH5N": {"C": 1, "H": 5, "N": 1},
    "C2H2O": {"C": 2, "H": 2, "O": 1}, "C2H4": {"C": 2, "H": 4}, "HF": {"H": 1, "F": 1},
    "HCl": {"H": 1, "Cl": 1}, "H2S": {"H": 2, "S": 1}, "SO2": {"S": 1, "O": 2},
}
DH_BINS = tuple(range(-3, 5))  # H on ion minus H on its atoms; +1 = plain protonated cleavage

# Cut-bond categories (counts of broken bonds of each type).
CUT_PAIRS = ("C-C", "C-N", "C-O", "C-S", "C-X", "N-S", "O-S", "other")


def candidate_feature_names():
    names = [
        "frag_mz_frac", "neutral_loss_mass", "log_neutral_loss_mass", "frag_heavy_frac",
        "frag_heavy", "is_intact", "dH", *[f"dH={d}" for d in DH_BINS], "rdbe_ion",
        "odd_electron", *[f"frag_{e}" for e in FORMULA_ELEMENTS],
        *[f"loss_{e}" for e in FORMULA_ELEMENTS], *[f"loss={n}" for n in NAMED_LOSSES],
        "mass_error_ppm", "abs_mass_error_ppm",
        "n_cut", "n_cut_ring", "n_cut_aromatic", "n_cut_multiple", *[f"cut_{p}" for p in CUT_PAIRS],
        "cut_amide_CN", "cut_ester_CO", "cut_carbonyl_adjacent", "cut_benzylic", "cut_conjugated",
        "frag_aromatic_atoms", "frag_ring_atoms", "frag_aromatic_frac",
        "frag_basic_N", "frag_aromatic_N", "frag_amide_N", "loss_basic_N", "loss_aromatic_N",
        "frag_charge", "frag_carbonyl",
        "bag_log_n_cand", "bag_is_min_cut", "bag_cut_minus_min",
        "ce", "ce_sq", "precursor_mz", "root_heavy", "adduct_MH", "root_basic_N", "root_aromatic_N",
    ]
    return names


ATOM_FEATURE_NAMES = (
    *[f"el={e}" for e in ("other", *ELEMENTS)], *[f"hyb={h}" for h in ("other", *HYBRIDIZATIONS)],
    "total_h", "degree", "formal_charge", "aromatic", "in_ring", "carbonyl", "basic_n",
    "arom_n", "amide_n",
)
BOND_FEATURE_NAMES = ("single", "double", "triple", "aromatic", "conjugated", "in_ring")


def _atom_matrix(g):
    a = g["atom"]
    el = np.eye(len(ELEMENTS) + 1, dtype=np.float32)[a["element_id"]]
    hy = np.eye(len(HYBRIDIZATIONS) + 1, dtype=np.float32)[a["hyb_id"]]
    rest = np.stack([a["total_h"] / 3.0, a["degree"] / 4.0, a["formal_charge"], a["aromatic"],
                     a["in_ring"], a["carbonyl"], a["basic_n"], a["arom_n"], a["amide_n"]],
                    axis=1).astype(np.float32)
    return np.concatenate([el, hy, rest], axis=1)


def _bond_matrix(g):
    b = g["bond"]
    order = b["order"]
    return np.stack([(order == 1) & ~b["aromatic"], order == 2, order == 3, b["aromatic"],
                     b["conjugated"], b["in_ring"]], axis=1).astype(np.float32)


def _formula_vector(formula, elements=FORMULA_ELEMENTS):
    return np.array([formula.get(e, 0) for e in elements], dtype=np.float32)


def _count_matmul(a, b):
    """Integer counts via matmul. numpy's Accelerate backend can emit spurious
    floating-point warnings here, so compute in float64 and verify exactness."""
    with np.errstate(all="ignore"):
        out = a @ b.astype(np.float64)
    rounded = np.rint(out)
    if not np.allclose(out, rounded):
        raise AssertionError("count matmul produced non-integer values")
    return rounded


def _featurize_one(args):
    """Features for one spectrum (runs in a worker process)."""
    ex, engine_smiles = args
    g = mol_graph(engine_smiles)
    g_orig = mol_graph(ex["smiles"])
    n = g["n_atoms"]
    el = g["element"]
    atom = g["atom"]
    bi, bj = g["bond"]["i"], g["bond"]["j"]

    # Root / precursor ion formula.
    root = {e: int((el == e).sum()) for e in set(el)}
    root["H"] = int(atom["total_h"].sum())
    adduct_mh = ex["adduct"] == "[M+H]+"
    prec = dict(root)
    prec["H"] += int(adduct_mh)
    prec_vec = _formula_vector(prec)

    cands = [(bag_i, c) for bag_i, bag in enumerate(ex["bags"]) for c in bag[3]]
    n_c = len(cands)
    cand_bag = np.array([b for b, _ in cands], np.int64)
    idx_list = [c[3].astype(np.int64) for _, c in cands]
    lengths = np.array([len(i) for i in idx_list], np.int64)
    rows = np.repeat(np.arange(n_c), lengths)
    flat_idx = np.concatenate(idx_list)
    member = np.zeros((n_c, n), bool)
    member[rows, flat_idx] = True

    # Cut bonds: exactly one endpoint inside the fragment.
    in_i, in_j = member[:, bi], member[:, bj]
    cut = in_i ^ in_j
    bond = g["bond"]
    ei, ej = el[bi], el[bj]
    pair = np.array(["-".join(sorted((x, y))) for x, y in zip(ei, ej)])
    halogen = np.isin(ei, HALOGENS) | np.isin(ej, HALOGENS)
    pair_masks = {
        "C-C": pair == "C-C", "C-N": pair == "C-N", "C-O": pair == "C-O", "C-S": pair == "C-S",
        "C-X": halogen & ((ei == "C") | (ej == "C")), "N-S": pair == "N-S", "O-S": pair == "O-S",
    }
    pair_masks["other"] = ~np.any(np.stack(list(pair_masks.values())), axis=0) if len(bi) else np.zeros(0, bool)
    carb = atom["carbonyl"]
    single = (bond["order"] == 1) & ~bond["aromatic"]
    amide = single & ((carb[bi] & (ej == "N")) | (carb[bj] & (ei == "N")))
    ester = single & ((carb[bi] & (ej == "O")) | (carb[bj] & (ei == "O")))
    carb_adj = single & (carb[bi] | carb[bj])
    arom = atom["aromatic"]
    benzylic = ~bond["aromatic"] & (arom[bi] ^ arom[bj])
    cutf = cut.astype(np.float64)
    bond_cols = [bond["in_ring"], bond["aromatic"], bond["order"] >= 2,
                 *[pair_masks[p] for p in CUT_PAIRS], amide, ester, carb_adj, benzylic,
                 bond["conjugated"]]
    bond_counts = _count_matmul(cutf, np.stack(bond_cols, axis=1)) if len(bi) else \
        np.zeros((n_c, len(bond_cols)))
    n_cut = cutf.sum(1)

    # Atom-level counts in fragment and in the neutral remainder.
    memf = member.astype(np.float64)
    atom_cols = np.stack([arom, atom["in_ring"], atom["basic_n"], atom["arom_n"], atom["amide_n"],
                          atom["formal_charge"], carb, atom["total_h"]], axis=1)
    frag_counts = _count_matmul(memf, atom_cols)
    root_counts = atom_cols.sum(0)
    loss_counts = root_counts[None, :] - frag_counts

    # Formula-level features.
    ion = np.stack([_formula_vector(c[0]) for _, c in cands])
    loss = prec_vec[None, :] - ion
    halo = ion[:, FORMULA_ELEMENTS.index("F")] + ion[:, FORMULA_ELEMENTS.index("Cl")] + \
        ion[:, FORMULA_ELEMENTS.index("Br")]
    rdbe = ion[:, 0] - (ion[:, 1] + halo) / 2 + ion[:, 2] / 2 + 1
    odd = (np.round(2 * rdbe) % 2 == 0).astype(np.float32)
    named = np.zeros((n_c, len(NAMED_LOSSES)), np.float32)
    for k, f in enumerate(NAMED_LOSSES.values()):
        named[:, k] = np.all(loss == _formula_vector(f)[None, :], axis=1)
    dH = ion[:, 1] - frag_counts[:, 7]
    dh_onehot = (np.clip(dH, DH_BINS[0], DH_BINS[-1])[:, None] == np.array(DH_BINS)[None, :]).astype(np.float32)

    theo = np.array([c[2] for _, c in cands], np.float64)
    peak_mz = np.array([ex["bags"][b][1] for b in cand_bag], np.float64)
    prec_mz = ex["precursor_mz"]
    ppm = (peak_mz - theo) / theo * 1e6
    heavy = lengths.astype(np.float32)
    nl_mass = np.clip(prec_mz - theo, 0, None)

    # In-bag context.
    n_bags = len(ex["bags"])
    bag_n = np.bincount(cand_bag, minlength=n_bags)
    bag_min_cut = np.full(n_bags, np.inf)
    np.minimum.at(bag_min_cut, cand_bag, n_cut)
    cut_minus_min = n_cut - bag_min_cut[cand_bag]

    ce = ex["collision_energy"]
    ce_s = (ce - 35.0) / 20.0
    root_heavy = float(n)
    spec_cols = np.array([ce_s, ce_s ** 2, prec_mz / 500.0, root_heavy / 30.0, float(adduct_mh),
                          root_counts[2], root_counts[3]], np.float32)

    feats = np.concatenate([
        np.stack([theo / prec_mz, nl_mass / 100.0, np.log1p(nl_mass) / 5.0, heavy / n, heavy / 30.0,
                  (lengths == n).astype(np.float32), dH], axis=1),
        dh_onehot, rdbe[:, None] / 10.0, odd[:, None], ion / 10.0, loss / 10.0, named,
        np.stack([ppm / 10.0, np.abs(ppm) / 10.0, n_cut], axis=1), bond_counts,
        frag_counts[:, [0, 1]] / 10.0, (frag_counts[:, 0] / heavy)[:, None],
        frag_counts[:, [2, 3, 4]], loss_counts[:, [2, 3]], frag_counts[:, [5, 6]],
        np.stack([np.log1p(bag_n[cand_bag]), (cut_minus_min == 0).astype(np.float32),
                  cut_minus_min], axis=1),
        np.broadcast_to(spec_cols, (n_c, len(spec_cols))),
    ], axis=1).astype(np.float32)

    # Cut-bond endpoints for graph models.
    crow, cbond = np.nonzero(cut)
    inside_is_i = in_i[crow, cbond]
    cut_in = np.where(inside_is_i, bi[cbond], bj[cbond])
    cut_out = np.where(inside_is_i, bj[cbond], bi[cbond])

    return {
        "cand_feat": feats, "cand_bag": cand_bag, "cand_atoms": flat_idx.astype(np.int32),
        "cand_atom_len": lengths, "cut_row": crow.astype(np.int64), "cut_in": cut_in.astype(np.int32),
        "cut_out": cut_out.astype(np.int32),
        "atom_feat": _atom_matrix(g), "edge": np.stack([bi, bj]).astype(np.int32),
        "bond_feat": _bond_matrix(g),
        "atom_feat_orig": _atom_matrix(g_orig), "edge_orig": np.stack([g_orig["bond"]["i"], g_orig["bond"]["j"]]).astype(np.int32),
        "bond_feat_orig": _bond_matrix(g_orig),
        "bag_mz": np.array([b[1] for b in ex["bags"]], np.float64),
        "bag_int": np.array([b[2] for b in ex["bags"]], np.float64),
        "bag_peak_index": np.array([b[0] for b in ex["bags"]], np.int32),
    }


class FeatureStore:
    """All arrays for the dataset plus per-spectrum metadata (pandas DataFrame)."""

    def __init__(self, arrays, meta, key=""):
        self.__dict__.update(arrays)
        self.meta = meta
        self.key = key
        self.cand_feature_names = candidate_feature_names()

    @property
    def n_spectra(self):
        return len(self.spec_bag_ptr) - 1

    def summary(self):
        return {
            "spectra": self.n_spectra, "molecules": int(self.meta["mol_id"].nunique()),
            "bags": int(len(self.bag_int)), "candidates": int(len(self.cand_bag)),
            "candidate_features": int(self.cand_feat.shape[1]), "root_atoms": int(len(self.atom_feat)),
        }


def _concat(parts, key):
    return np.concatenate([p[key] for p in parts], axis=0)


def _ptr(lengths):
    return np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)


def _assemble(parts, examples, engine):
    """Concatenate per-spectrum feature parts into global CSR-style arrays."""
    import pandas as pd

    n_bags = np.array([len(p["bag_mz"]) for p in parts])
    n_cands = np.array([len(p["cand_bag"]) for p in parts])
    n_atoms = np.array([len(p["atom_feat"]) for p in parts])
    bag_off, cand_off, atom_off = _ptr(n_bags), _ptr(n_cands), _ptr(n_atoms)

    cand_bag = np.concatenate([p["cand_bag"] + bag_off[i] for i, p in enumerate(parts)])
    cand_atoms = np.concatenate([p["cand_atoms"] + atom_off[i] for i, p in enumerate(parts)]).astype(np.int32)
    cut_row = np.concatenate([p["cut_row"] + cand_off[i] for i, p in enumerate(parts)])
    cut_in = np.concatenate([p["cut_in"] + atom_off[i] for i, p in enumerate(parts)]).astype(np.int32)
    cut_out = np.concatenate([p["cut_out"] + atom_off[i] for i, p in enumerate(parts)]).astype(np.int32)
    edge = np.concatenate([p["edge"] + atom_off[i] for i, p in enumerate(parts)], axis=1)
    edge_orig = np.concatenate([p["edge_orig"] + atom_off[i] for i, p in enumerate(parts)], axis=1)
    n_edges = np.array([p["edge"].shape[1] for p in parts])

    arrays = {
        "spec_bag_ptr": bag_off, "spec_cand_ptr": cand_off, "spec_atom_ptr": atom_off,
        "spec_edge_ptr": _ptr(n_edges),
        "bag_cand_ptr": _ptr(np.bincount(cand_bag, minlength=int(bag_off[-1]))),
        "bag_spec": np.repeat(np.arange(len(parts)), n_bags),
        "bag_mz": _concat(parts, "bag_mz"), "bag_int": _concat(parts, "bag_int"),
        "bag_peak_index": _concat(parts, "bag_peak_index"),
        "cand_feat": _concat(parts, "cand_feat"), "cand_bag": cand_bag,
        "cand_atom_ptr": _ptr(np.concatenate([p["cand_atom_len"] for p in parts])),
        "cand_atoms": cand_atoms,
        "cand_cut_ptr": _ptr(np.bincount(cut_row, minlength=int(cand_off[-1]))),
        "cut_in": cut_in, "cut_out": cut_out,
        "atom_feat": _concat(parts, "atom_feat"), "edge": edge, "bond_feat": _concat(parts, "bond_feat"),
        "atom_feat_orig": _concat(parts, "atom_feat_orig"), "edge_orig": edge_orig,
        "bond_feat_orig": _concat(parts, "bond_feat_orig"),
    }
    if not np.all(np.diff(cut_row) >= 0):
        raise AssertionError("cut rows must be sorted by candidate")

    # Spectrum-level metadata, including the full measured spectrum's norms.
    rows = []
    for e in examples:
        I = e["peaks"][:, 1].astype(np.float64)
        rows.append({
            "cache_id": e["cache_id"], "mol_id": e["inchikey"][:14], "inchikey": e["inchikey"],
            "smiles": e["smiles"], "engine_smiles": engine[e["smiles"]],
            "compound_name": e["compound_name"], "feature_id": e["feature_id"],
            "collision_energy": e["collision_energy"], "precursor_mz": e["precursor_mz"],
            "adduct": e["adduct"], "n_atoms": len(e["atoms"]), "n_peaks_full": len(I),
            "full_int_sum": I.sum(), "full_int_l2": np.linalg.norm(I),
            "precursor_purity": e["precursor_purity"], "quality_chimeric": e["quality_chimeric"],
            "usi_n_structures": e["usi_n_structures"],
            "engine_reordered": e["smiles"] != engine[e["smiles"]],
        })
    meta = pd.DataFrame(rows)
    meta["n_bags"] = n_bags
    meta["n_cands"] = n_cands

    return arrays, meta


def build_feature_store(examples, cache_dir=CACHE_DIR, num_workers=8, force=False, verbose=True):
    import pandas as pd

    key = hashlib.sha256(json.dumps({
        "version": FEATURE_VERSION, "names": candidate_feature_names(),
        "ids": [e["cache_id"] for e in examples],
    }).encode()).hexdigest()[:16]
    path = Path(cache_dir) / f"features_{key}.npz"
    meta_path = Path(cache_dir) / f"features_{key}_meta.parquet"
    meta_path = meta_path.with_suffix(".pkl")
    if path.exists() and meta_path.exists() and not force:
        start = time.time()
        with np.load(path) as data:
            arrays = {k: data[k] for k in data.files}
        meta = pd.read_pickle(meta_path)
        if verbose:
            print(f"Loaded cached features in {time.time() - start:.1f}s ({path.name})")
        return FeatureStore(arrays, meta, key)

    start = time.time()
    engine = engine_smiles_map([e["smiles"] for e in examples], num_workers=num_workers)
    jobs = [(e, engine[e["smiles"]]) for e in examples]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        parts = list(pool.map(_featurize_one, jobs, chunksize=32))
    if verbose:
        print(f"Featurized {len(parts):,} spectra in {time.time() - start:.1f}s")

    arrays, meta = _assemble(parts, examples, engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    meta.to_pickle(meta_path)
    if verbose:
        print(f"Saved features to {path.name} ({path.stat().st_size / 1e9:.2f} GB)")
    return FeatureStore(arrays, meta, key)


def featurize_candidate_bags(spectrum, candidate_bags, collision_energy=None):
    """FeatureStore for one spectrum straight from ``get_candidate_bags`` output.

    Use this at inference time: ``spectrum`` is a ``Spectrum`` (MS2, with
    ``metadata['adduct']``), ``candidate_bags`` the list returned by
    ``magma_workflow.candidates.get_candidate_bags(..., peak_assignment='nearest')``.
    Measured intensities are only copied for plotting/evaluation; features never
    read them. Empty bags are skipped (they cannot receive intensity).
    """
    from .chemistry import _engine_smiles, verify_alignment

    ce = float(spectrum.metadata["collision_energy"] if collision_energy is None else collision_energy)
    bags = []
    for mz, cands, *_ in candidate_bags:
        if not cands:
            continue
        peak_index = int(cands[0].metadata["peak_index"])
        bags.append((peak_index, float(mz), float(spectrum.intensity[peak_index]), [
            (dict(c.ionic_formula), int(c.charge), float(c.metadata["theoretical_mz"]),
             np.asarray(c.metadata["origins"][0]["parent_atom_indices"], dtype=np.int16)) for c in cands]))
    if not bags:
        raise ValueError("No candidate bag has a MAGMa candidate")
    engine = _engine_smiles(spectrum.smiles)
    graph = mol_graph(engine)
    if verify_alignment(graph, [(f, idx) for b in bags for f, _, _, idx in b[3]]):
        raise AssertionError("MAGMa atom indices do not match the engine root graph")
    ex = {"cache_id": "inference", "smiles": spectrum.smiles, "inchikey": spectrum.metadata.get("inchikey", ""),
          "compound_name": spectrum.metadata.get("compound_name"), "feature_id": spectrum.metadata.get("feature_id"),
          "collision_energy": ce, "precursor_mz": float(spectrum.precursor_mz),
          "adduct": spectrum.metadata["adduct"], "atoms": [None] * graph["n_atoms"],
          "peaks": np.asarray(spectrum.peaks, dtype=np.float32), "precursor_purity": np.nan,
          "quality_chimeric": None, "usi_n_structures": 1, "bags": bags}
    part = _featurize_one((ex, engine))
    arrays, meta = _assemble([part], [ex], {spectrum.smiles: engine})
    return FeatureStore(arrays, meta, "inference")
