"""MS3-derived contribution scores for the MAGMa candidates of an MS2 peak.

An MS2 peak with several MAGMa candidates is a mixture of ions. Its MS3 scans
are the fragments of that mixture, so the measured MS3 is modelled as a
non-negative mix of one predicted MS3 per candidate, and the mixing weights are
the contribution scores.

Predicted MS3 of candidate k (no new model is trained):

    1. Run MAGMa on the root molecule against the MS3 peak list, and score every
       sub-fragment d with the trained MS2 intensity model (abundance exp(s_d),
       collision energy = the MS3 stage energy).
    2. Keep only sub-fragments *nested* in k: the atoms of d are a subset of the
       atoms of k and d has no more hydrogens than k.
    3. Sum the kept abundances per MS3 peak and normalize.

The isolated ion itself (and anything above it) is removed first, because every
candidate has that mass. All MS3 scans of a peak (usually 20/40/60 eV) share one
weight vector, which is what makes the weights identifiable.

Candidates whose predicted MS3 are (almost) identical cannot be separated by
MS3. They are merged into a group; MS3 decides the group weight and the MS2
model's own shares split it inside the group.

    .venv/bin/python -m intensity_lab.ms3_attribution            # all validation molecules
    .venv/bin/python -m intensity_lab.ms3_attribution --limit 50  # quick look

Outputs (``results/experiment_1_agentic/ms3_attribution/``): ``candidates.csv``
(one row per candidate, ``contribution`` is the score), ``peaks.csv`` (one row
per MS2 peak with its fit quality), ``scans.csv`` (one row per MS3 scan) and
``summary.json``.

Limits to keep in mind: a candidate that does not fragment is invisible outside
the removed precursor peak and gets weight 0; MAGMa only reaches sub-fragments
within its bond-break limits from the root; and the MS3 isolation window can
let neighbouring MS2 peaks in (``coisolated_fraction``).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from . import RESULTS_DIR
from .raw import DEFAULT_MSNLIB

OUT_DIR = RESULTS_DIR / "ms3_attribution"
PRECURSOR_MARGIN = 0.5  # Da below the isolated m/z; peaks above this are not fragments
GROUP_COSINE = 0.98     # predicted MS3 this similar are treated as indistinguishable


def _cos(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 0 else 0.0


def _unit_columns(m):
    total = m.sum(0, keepdims=True)
    return m / np.where(total > 0, total, 1.0)


def _groups(columns, threshold):
    """Label candidates whose stacked predicted MS3 have cosine >= threshold; -1 = no support."""
    norm = np.linalg.norm(columns, axis=0)
    unit = columns / np.where(norm > 0, norm, 1.0)
    with np.errstate(all="ignore"):  # numpy's Accelerate matmul emits spurious warnings (see features._count_matmul)
        similar = (unit.T @ unit) >= threshold
    label = -np.ones(columns.shape[1], int)
    n_groups = 0
    for k in range(columns.shape[1]):
        if norm[k] == 0 or label[k] >= 0:
            continue
        label[k] = n_groups
        stack = [k]
        while stack:
            for j in np.flatnonzero(similar[stack.pop()] & (label < 0) & (norm > 0)):
                label[j] = n_groups
                stack.append(j)
        n_groups += 1
    return label, n_groups


def _subgraph(graph, atoms):
    """The part of a ``chemistry.mol_graph`` on ``atoms``; atoms keep their root features."""
    atoms = np.asarray(atoms)
    pos = -np.ones(graph["n_atoms"], int)
    pos[atoms] = np.arange(len(atoms))
    bond = graph["bond"]
    keep = (pos[bond["i"]] >= 0) & (pos[bond["j"]] >= 0)
    sub = {key: value[keep] for key, value in bond.items()}
    sub["i"], sub["j"] = pos[bond["i"][keep]], pos[bond["j"][keep]]
    return {"element": graph["element"][atoms], "atom": {k: v[atoms] for k, v in graph["atom"].items()},
            "bond": sub, "n_atoms": len(atoms)}


def _candidate_spectra(run, jobs):
    """Model heights for fragments used as the input molecule. jobs: [(graph, ion formula, m/z, ce, bags)]."""
    from .features import FeatureStore, _assemble, _featurize_one

    exs = [{"cache_id": "candidate", "smiles": "", "inchikey": "", "compound_name": None, "feature_id": None,
            "collision_energy": ce, "precursor_mz": mz, "adduct": "[M]+", "atoms": [None] * graph["n_atoms"],
            "peaks": np.zeros((1, 2), np.float32), "precursor_purity": np.nan, "quality_chimeric": None,
            "usi_n_structures": 1, "bags": bags, "graph": graph, "precursor_formula": formula}
           for graph, formula, mz, ce, bags in jobs]
    arrays, meta = _assemble([_featurize_one((ex, None)) for ex in exs], exs, {"": ""})
    fs = FeatureStore(arrays, meta, "inference")
    pred = run.predict(fs, np.arange(len(exs)))
    return [pred[fs.spec_bag_ptr[i]:fs.spec_bag_ptr[i + 1]] for i in range(len(exs))]


def _mix(columns, measured):
    """Non-negative weights (normalized to sum 1) and the unnormalized fit."""
    with np.errstate(all="ignore"):
        weights, _ = nnls(columns, measured)
        fit = columns @ weights
    if not (np.isfinite(weights).all() and np.isfinite(fit).all()):
        raise AssertionError("non-finite mixture fit")
    total = weights.sum()
    return (weights / total if total > 0 else weights), fit


def attribute_spectrum(run, ms2, kids, margin=PRECURSOR_MARGIN, group_cos=GROUP_COSINE, heights="model",
                       spectra="nested"):
    """Contribution scores for every MS3-linked peak of one MS2 spectrum.

    ``kids`` is ``find_ms3_children(ms2, ...)``: {MS2 peak index: [MS3 Spectrum]}.
    ``heights="equal"`` ignores the model's intensities: a candidate's predicted MS3
    is then equal heights on every peak it can explain (the model-free reference).
    ``spectra="candidate"`` gives the model each candidate as its input molecule (its part
    of the root graph, its ion formula, the isolated m/z) and predicts that molecule's
    heights at the MS3 peaks; ``"nested"`` filters the root molecule's prediction instead.
    Returns {"candidates": [...], "peaks": [...], "scans": [...]} as lists of dicts.
    """
    from magma_workflow.candidates import assign_candidates_to_nearest_peak, get_candidate_bags
    from magma_workflow.chemistry import formula_counts
    from magma_workflow.dataset import Spectrum

    from .chemistry import mol_graph
    from .features import featurize_candidate_bags

    order = [(int(p), k) for p, ks in sorted(kids.items()) for k in ks]
    segments = [np.asarray(ms2.peaks, float)] + [np.asarray(k.peaks, float).reshape(-1, 2) for _, k in order]
    bounds = np.cumsum([0] + [len(s) for s in segments])
    # One MAGMa run for the MS2 peaks and every MS3 peak list: all atom indices share one numbering.
    union = Spectrum(ms2.smiles, ms2.precursor_mz, 2, np.concatenate(segments), dict(ms2.metadata))
    all_bags = get_candidate_bags(union, spectra=[], peak_assignment="all", include_ms3=False)

    def score_segment(i, ce):
        local = assign_candidates_to_nearest_peak(all_bags[bounds[i]:bounds[i + 1]])
        flat = []
        for j, (_, cands, _) in enumerate(local):
            for c in cands:
                c.metadata["peak_index"] = j
                flat.append((j, c))
        if not flat:
            return flat, np.zeros(0)
        spec = Spectrum(ms2.smiles, ms2.precursor_mz, 2, segments[i], dict(ms2.metadata))
        fs = featurize_candidate_bags(spec, local, ce)
        _, scores = run.predict(fs, np.array([0]), return_candidates=True)
        return flat, scores

    masks = lambda c: [int(o["fragment_mask"]) for o in c.metadata["origins"]]
    hydrogens = lambda c: formula_counts(c.ionic_formula).get("H", 0)

    ms2_flat, ms2_scores = score_segment(0, float(ms2.metadata["collision_energy"]))
    by_peak = {}
    for (j, c), s in zip(ms2_flat, ms2_scores):
        by_peak.setdefault(j, []).append((c, float(s)))
    ms2_total = float(ms2.intensity.sum())
    base = {"usi": ms2.metadata["usi"][0], "compound_name": ms2.metadata.get("compound_name"),
            "inchikey": ms2.metadata.get("inchikey"), "ms2_ce": float(ms2.metadata["collision_energy"])}
    out = {"candidates": [], "peaks": [], "scans": []}

    for p in sorted(kids):
        hyps = by_peak.get(int(p), [])
        iso = float(ms2.mz[p])
        peak = {**base, "peak_index": int(p), "peak_mz": iso,
                "peak_share_of_ms2": float(ms2.intensity[p]) / ms2_total, "n_candidates": len(hyps)}
        if not hyps:
            out["peaks"].append({**peak, "status": "no_candidate"})
            continue
        cands = [c for c, _ in hyps]
        K = len(cands)
        hyp_masks, hyp_h = [masks(c) for c in cands], [hydrogens(c) for c in cands]
        s = np.array([x for _, x in hyps])
        prior = np.exp(s - s.max())
        prior /= prior.sum()  # the MS2 model's own split of this peak

        scans, jobs = [], []
        if spectra == "candidate":
            root_graph = mol_graph(cands[0].metadata["engine_parent_smiles"])
            atoms_of = [[a for a in range(root_graph["n_atoms"]) if m[0] >> a & 1] for m in hyp_masks]
            local = [{a: n for n, a in enumerate(atoms)} for atoms in atoms_of]
            graphs = [_subgraph(root_graph, atoms) for atoms in atoms_of]
            formulas = [formula_counts(c.ionic_formula) for c in cands]
        for i, (q, child) in enumerate(order, start=1):
            if q != int(p):
                continue
            mz, inten = segments[i][:, 0], segments[i][:, 1]
            keep = np.flatnonzero((mz < iso - margin) & (inten > 0))
            if not len(keep):
                continue
            row = {int(j): r for r, j in enumerate(keep)}
            ce = float(child.metadata["collision_energy"])
            flat, scores = score_segment(i, ce)
            nested, unrestricted = np.zeros((len(keep), K)), np.zeros(len(keep))
            pieces = [{} for _ in range(K)]  # candidate -> MS3 row -> its sub-fragments there
            if len(scores):
                for (j, d), a in zip(flat, np.exp(scores - scores.max())):
                    r = row.get(j)
                    if r is None:
                        continue
                    unrestricted[r] += a
                    d_masks, d_h = masks(d), hydrogens(d)
                    for k in range(K):
                        if spectra == "candidate":  # sub-fragments inside the candidate's own atoms
                            inside = [t for t in d_masks if t & hyp_masks[k][0] == t]
                            if d_h <= hyp_h[k] and inside:
                                idx = np.array([local[k][x] for x in range(inside[0].bit_length()) if inside[0] >> x & 1], np.int16)
                                pieces[k].setdefault(r, []).append(
                                    (formula_counts(d.ionic_formula), int(d.charge), float(d.metadata["theoretical_mz"]), idx))
                        elif d_h <= hyp_h[k] and any(t & m == t for t in d_masks for m in hyp_masks[k]):
                            nested[r, k] += a
            for k in range(K):
                if pieces[k]:
                    rows = sorted(pieces[k])
                    bags = [(r, float(mz[keep[r]]), float(inten[keep[r]]), pieces[k][r]) for r in rows]
                    jobs.append((len(scans), k, rows, (graphs[k], formulas[k], iso, ce, bags)))
            if heights == "equal":
                nested = (nested > 0).astype(float)
            scans.append({"ce": ce, "y": inten[keep] / inten[keep].sum(), "nested": nested,
                          "unrestricted": unrestricted, "usi": child.metadata["usi"][0]})
        if not scans:
            out["peaks"].append({**peak, "status": "no_ms3_fragments"})
            continue
        if jobs:
            for (s_i, k, rows, _), column in zip(jobs, _candidate_spectra(run, [j[-1] for j in jobs])):
                scans[s_i]["nested"][rows, k] = column

        y = np.concatenate([sc["y"] for sc in scans])
        label, n_groups = _groups(np.concatenate([_unit_columns(sc["nested"]) for sc in scans]), group_cos)
        member = np.stack([label == g for g in range(n_groups)], axis=1).astype(float) if n_groups else np.zeros((K, 0))
        for sc in scans:  # one column per group: mean of its members' predicted MS3
            with np.errstate(all="ignore"):
                sc["cols"] = _unit_columns(sc["nested"]) @ member / np.maximum(member.sum(0), 1)
            sc["supported"] = sc["nested"].sum(1) > 0
        with np.errstate(all="ignore"):
            group_prior = prior @ member
        if n_groups:
            weight, fit = _mix(np.concatenate([sc["cols"] for sc in scans]), y)
        else:
            weight, fit = np.zeros(0), np.zeros(len(y))
        informative = n_groups > 0 and weight.sum() > 0
        top = int(np.argmax(weight)) if informative else -1

        # Agreement between independent MS3 scans (each fitted alone).
        tops, per_scan = [], []
        start = 0
        for sc in scans:
            n = len(sc["y"])
            sc["fit"] = fit[start:start + n]
            start += n
            w_scan, _ = _mix(sc["cols"], sc["y"]) if n_groups else (np.zeros(0), None)
            ok = n_groups > 0 and w_scan.sum() > 0
            sc["top"] = int(np.argmax(w_scan)) if ok else -1
            if ok:
                tops.append(sc["top"])
                per_scan.append(w_scan)
        pairs = [(a, b) for a in range(len(tops)) for b in range(a + 1, len(tops))]
        scan_top_agree = float(np.mean([tops[a] == tops[b] for a, b in pairs])) if pairs else np.nan
        scan_weight_tv = float(np.mean([0.5 * np.abs(per_scan[a] - per_scan[b]).sum() for a, b in pairs])) if pairs else np.nan

        # MS2-only attributions to compare against: the model's shares and MAGMa's fewest-cuts rule.
        cuts = np.array([min(o["boundary_cut_count"] for o in c.metadata["origins"]) for c in cands])
        fewest = {int(label[k]) for k in np.flatnonzero(cuts == cuts.min())}
        unsupported_prior = float(prior[label < 0].sum())
        model_top = int(np.argmax(group_prior)) if n_groups and group_prior.max() >= unsupported_prior else -1
        half_window = float((child.metadata.get("msn_isolation_windows") or [0, 2.0])[-1]) / 2
        near = np.abs(ms2.mz - iso) <= half_window
        status = ("unsupported" if not informative else "single_candidate" if K == 1
                  else "indistinguishable" if n_groups == 1 and (label >= 0).all() else "resolvable")
        out["peaks"].append({
            **peak, "status": status, "n_groups": n_groups, "n_unsupported": int((label < 0).sum()),
            "n_ms3_scans": len(scans), "ms3_ces": "/".join(f"{sc['ce']:g}" for sc in scans),
            "fit_cosine": _cos(fit, y), "unexplained_fraction": float(np.mean([sc["y"][~sc["supported"]].sum() for sc in scans])),
            "top_weight": float(weight[top]) if informative else np.nan,
            "scan_top_agree": scan_top_agree, "scan_weight_tv": scan_weight_tv,
            "model_top_agrees": float(model_top == top) if informative else np.nan,
            "model_share_on_ms3_top": float(group_prior[top]) if informative else np.nan,
            "fewest_cuts_agrees": (float(top in fewest) / len(fewest)) if informative else np.nan,
            "chance_agreement": 1.0 / (n_groups + bool((label < 0).any())) if informative else np.nan,
            "coisolated_fraction": float(1 - ms2.intensity[p] / ms2.intensity[near].sum()),
        })
        for sc in scans:
            sup = sc["supported"]
            out["scans"].append({
                "usi": base["usi"], "peak_index": int(p), "ms3_usi": sc["usi"], "ms3_ce": sc["ce"], "status": status,
                "n_fragment_peaks": len(sc["y"]), "supported_fraction": float(sc["y"][sup].sum()),
                "cos_mixture": _cos(sc["fit"], sc["y"]), "cos_equal_heights": _cos(sup.astype(float), sc["y"]),
                "cos_unrestricted_model": _cos(sc["unrestricted"], sc["y"]),
                "cos_mixture_supported": _cos(sc["fit"][sup], sc["y"][sup]) if sup.sum() > 1 else np.nan,
                "cos_equal_heights_supported": _cos(np.ones(int(sup.sum())), sc["y"][sup]) if sup.sum() > 1 else np.nan,
                "top_group": sc["top"],
            })

        from rdkit import Chem

        mol = Chem.MolFromSmiles(cands[0].metadata["engine_parent_smiles"])
        for k, c in enumerate(cands):
            g = int(label[k])
            ms3_weight = float(weight[g]) if informative and g >= 0 else 0.0
            contribution = ms3_weight * prior[k] / group_prior[g] if informative and g >= 0 else \
                (float(prior[k]) if not informative else 0.0)
            formula = formula_counts(c.ionic_formula)
            out["candidates"].append({
                "usi": base["usi"], "peak_index": int(p), "peak_mz": iso, "candidate_id": c.candidate_id,
                "fragment_smiles": Chem.MolFragmentToSmiles(mol, list(c.metadata["origins"][0]["parent_atom_indices"])),
                "ion_formula": "".join(f"{e}{n if n > 1 else ''}" for e, n in sorted(formula.items())),
                "broken_bonds": int(cuts[k]), "group": g,
                "group_size": int((label == g).sum()) if g >= 0 else 0,
                "ms3_group_weight": ms3_weight, "ms2_model_share": float(prior[k]),
                "contribution": float(contribution), "ms3_informative": bool(informative), "status": status,
            })
    return out


def candidate_spectra(run, ms2, ms3, peak_index, margin=PRECURSOR_MARGIN):
    """One predicted spectrum per MAGMa candidate of an MS2 peak, on the measured MS3's m/z values.

    Each candidate is given to the model as its input molecule (its part of the root
    graph, its ion formula, the isolated m/z, the MS3 collision energy). Returns
    ``(candidates, mz, measured, spectra)``: ``spectra[k]`` are candidate k's heights at
    ``mz``, the MS3 peaks below the isolated ion; ``measured`` is the MS3 there. Both sum to 1.
    A candidate with no sub-fragment at any of these peaks gets an all-zero spectrum.

    The sub-fragments are enumerated from each candidate itself (three atom-removal
    steps from the candidate, with no limit counted from the root molecule).
    """
    from magma_workflow.candidates import get_candidate_bags
    from magma_workflow.chemistry import formula_counts
    from magma_workflow.dataset import Spectrum

    from .chemistry import mol_graph

    candidates = get_candidate_bags(ms2, spectra=[], peak_assignment="nearest", include_ms3=False)[peak_index][1]
    isolated = float(ms2.mz[peak_index])
    ms3_bags = []
    if len(candidates) and len(ms3.peaks):
        as_ms2 = Spectrum(ms2.smiles, ms2.precursor_mz, 2, ms3.peaks, dict(ms2.metadata))
        ms3_bags = get_candidate_bags(
            as_ms2, spectra=[], peak_assignment="nearest", include_ms3=False, max_boundary_cuts=10 ** 6,
            deepen_from=[int(c.metadata["origins"][0]["fragment_mask"]) for c in candidates])
    keep = np.flatnonzero((ms3.mz < isolated - margin) & (ms3.intensity > 0))
    measured = ms3.intensity[keep] / ms3.intensity[keep].sum()
    spectra = np.zeros((len(candidates), len(keep)))
    if not len(candidates) or not len(keep):
        return candidates, ms3.mz[keep], measured, spectra
    graph = mol_graph(candidates[0].metadata["engine_parent_smiles"])
    ce = float(ms3.metadata["collision_energy"])
    jobs = []
    for k, candidate in enumerate(candidates):
        mask = int(candidate.metadata["origins"][0]["fragment_mask"])
        atoms = [a for a in range(graph["n_atoms"]) if mask >> a & 1]
        formula = formula_counts(candidate.ionic_formula)
        model_bags = []
        for row, j in enumerate(keep):
            pieces = []  # sub-fragments of this candidate that land on MS3 peak j
            for piece in ms3_bags[j][1]:
                inside = [m for m in (int(o["fragment_mask"]) for o in piece.metadata["origins"]) if m & mask == m]
                piece_formula = formula_counts(piece.ionic_formula)
                if inside and piece_formula.get("H", 0) <= formula.get("H", 0):
                    index = np.array([n for n, a in enumerate(atoms) if inside[0] >> a & 1], np.int16)
                    pieces.append((piece_formula, int(piece.charge), float(piece.metadata["theoretical_mz"]), index))
            if pieces:
                model_bags.append((row, float(ms3.mz[j]), float(ms3.intensity[j]), pieces))
        if model_bags:
            jobs.append((k, [b[0] for b in model_bags], (_subgraph(graph, atoms), formula, isolated, ce, model_bags)))
    if jobs:
        for (k, rows, _), heights in zip(jobs, _candidate_spectra(run, [j[-1] for j in jobs])):
            spectra[k, rows] = heights
    return candidates, ms3.mz[keep], measured, spectra


# ---------------------------------------------------------------------------
# Batch run over a split
# ---------------------------------------------------------------------------

_RUN = None
_HEIGHTS = "model"
_SPECTRA = "nested"


def _init_worker(run_path, heights="model", spectra="nested"):
    global _RUN, _HEIGHTS, _SPECTRA
    _HEIGHTS, _SPECTRA = heights, spectra
    import torch
    from rdkit import RDLogger

    from .inference import load_run

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)
    _RUN = load_run(run_path)


def _work(job):
    from magma_workflow.candidates import find_ms3_children
    from magma_workflow.dataset import Spectrum

    records, targets = job
    related = [Spectrum.from_record(dict(r)) for r in records]
    out = {"candidates": [], "peaks": [], "scans": [], "parents": 0, "skipped": []}
    for ms2 in related:
        if ms2.ms_level != 2 or ms2.metadata["usi"][0] not in targets:
            continue
        try:
            kids = find_ms3_children(ms2, spectra=[s for s in related if s.smiles == ms2.smiles])
            if not kids:
                continue
            res = attribute_spectrum(_RUN, ms2, kids, heights=_HEIGHTS, spectra=_SPECTRA)
        except (ValueError, AssertionError) as error:  # unsupported adduct, atom alignment, ...
            out["skipped"].append(f"{ms2.metadata['usi'][0]}: {error}")
            continue
        out["parents"] += 1
        for key in ("candidates", "peaks", "scans"):
            out[key] += res[key]
    return out


def collect_jobs(fs, examples, spec_ids, msnlib_path=DEFAULT_MSNLIB, limit=None):
    """[(MSnLib MS2/MS3 records of one feature, {target MS2 USIs})] for the given spectra."""
    usi_of = {e["cache_id"]: e["usi"] for e in examples}
    meta = fs.meta.iloc[np.asarray(spec_ids)]
    targets = {}
    for cache_id, feature_id in zip(meta["cache_id"], meta["feature_id"]):
        targets.setdefault(feature_id, set()).add(usi_of[cache_id])
    if limit:
        targets = dict(list(targets.items())[:limit])
    records = {}
    with Path(msnlib_path).open(encoding="utf-8") as handle:
        for line in handle:
            if '"SINGLE_BEST_SCAN"' not in line:
                continue
            r = json.loads(line)
            if r.get("feature_id") in targets and int(r["ms_level"]) in (2, 3):
                records.setdefault(r["feature_id"], []).append(r)
    return [(records[f], targets[f]) for f in targets if f in records]


def summarize(peaks, scans):
    """Headline numbers; ``peaks`` and ``scans`` are DataFrames from a run."""
    usable = peaks[peaks["status"].isin(["single_candidate", "indistinguishable", "resolvable"])]
    res = usable[usable["status"] == "resolvable"]
    single = scans[scans["status"] == "single_candidate"]
    mean = lambda x: float(np.nanmean(x)) if len(x) else float("nan")
    cols = ["cos_mixture", "cos_equal_heights", "cos_unrestricted_model",
            "cos_mixture_supported", "cos_equal_heights_supported", "supported_fraction"]
    return {
        "linked_peaks": int(len(peaks)),
        "status_counts": {k: int(v) for k, v in peaks["status"].value_counts().items()},
        "fit_cosine": {"median": float(usable["fit_cosine"].median()), "share_at_least_0.7": mean(usable["fit_cosine"] >= 0.7),
                       "mean_unexplained_fraction": mean(usable["unexplained_fraction"])},
        "predictor_on_single_candidate_ms3": {
            "all": {c: mean(single[c]) for c in cols},
            **{f"ce_{ce:g}": {c: mean(g[c]) for c in cols} | {"scans": int(len(g))} for ce, g in single.groupby("ms3_ce")},
        },
        "resolvable_peaks": {
            "n": int(len(res)), "mean_groups": mean(res["n_groups"]),
            "mean_top_weight": mean(res["top_weight"]), "share_top_weight_at_least_0.8": mean(res["top_weight"] >= 0.8),
            "ms3_scans_agree_on_top": mean(res["scan_top_agree"]), "ms3_scan_weight_distance": mean(res["scan_weight_tv"]),
            "ms2_model_agrees_with_ms3": mean(res["model_top_agrees"]),
            "fewest_cuts_agrees_with_ms3": mean(res["fewest_cuts_agrees"]),
            "chance_agreement": mean(res["chance_agreement"]),
            "ms2_model_share_on_ms3_top": mean(res["model_share_on_ms3_top"]),
        },
    }


def main(argv=None):
    import pandas as pd

    from . import experiments as EXP
    from .pipeline import load_data
    from .train import train

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", default="val", choices=["val", "train"], help="the test set stays held out")
    parser.add_argument("--model", default="gnn", help="name in intensity_lab.experiments")
    parser.add_argument("--limit", type=int, default=None, help="only the first N features")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--heights", default="model", choices=["model", "equal"],
                        help="predicted MS3 heights: the model's, or equal on every explainable peak")
    parser.add_argument("--spectra", default="nested", choices=["nested", "candidate"],
                        help="candidate: each candidate is the model's input molecule; nested: filter the root's prediction")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    suffix = ("_candidate_input" if args.spectra == "candidate" else "") + ("_equal_heights" if args.heights == "equal" else "")
    args.out = args.out or OUT_DIR.with_name(OUT_DIR.name + suffix)

    data = load_data(verbose=False)
    split = data.splits["molecule"]
    run = train(data.fs, split, EXP.get(args.model), data.data_key(), verbose=True)
    start = time.time()
    jobs = collect_jobs(data.fs, data.examples, split[args.split], limit=args.limit)
    print(f"{len(jobs):,} features with MS2/MS3 records ({time.time() - start:.0f}s)")
    del data

    start = time.time()
    merged = {"candidates": [], "peaks": [], "scans": [], "parents": 0, "skipped": []}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(run.path, args.heights, args.spectra)) as pool:
        for result in pool.map(_work, jobs, chunksize=4):
            for key, value in result.items():
                merged[key] += value
    print(f"{merged['parents']:,} MS2 parents, {len(merged['peaks']):,} linked peaks, "
          f"{len(merged['skipped'])} skipped ({time.time() - start:.0f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    frames = {key: pd.DataFrame(merged[key]) for key in ("candidates", "peaks", "scans")}
    for key, frame in frames.items():
        frame.to_csv(args.out / f"{key}.csv", index=False)
    summary = {"split": args.split, "model": run.path.name, "heights": args.heights, "spectra": args.spectra, "ms2_parents": merged["parents"],
               "skipped": len(merged["skipped"]), **summarize(frames["peaks"], frames["scans"])}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "skipped.txt").write_text("\n".join(merged["skipped"]))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
