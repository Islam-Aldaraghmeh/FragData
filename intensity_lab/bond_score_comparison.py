"""Compare GLACIER/ms-pred's bond-score peak assignment with the MS3 mixing scores.

The heuristic (ms-pred ``magma/run_magma.py``, "Step 3: Make all assignments"):

    bond score of a fragment = sum over its broken bonds of
        TYPEW[bond type] * HETEROW[bond touches a non-carbon atom]
        TYPEW   = single 1, double 2, aromatic 2, triple 3
        HETEROW = 2 for a C-C bond, 1 otherwise        (lower = easier to break)

    for a peak: keep the fragments with the smallest mass error, then of those
    the ones with the lowest bond score. Every tie is kept.

Our score: exactly the notebook (``mixing_experiment.ipynb``). One predicted
spectrum per candidate on the m/z values of the peak's MS3 scan with the most
peaks, then plain non-negative least squares. Candidates whose predicted
spectrum is identical to the top-scoring one are tied with it, because the
solver puts the weight on any one of them.

    .venv/bin/python -m intensity_lab.bond_score_comparison                  # all validation molecules
    .venv/bin/python -m intensity_lab.bond_score_comparison --molecules 100
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from math import comb

import numpy as np
from scipy.optimize import nnls

from . import RESULTS_DIR
from . import ms3_attribution as MS3

OUT_DIR = RESULTS_DIR / "bond_score_comparison"
METHOD = "v2"           # part of the cache file name; change it when the scoring changes
SAME_SPECTRUM = 0.9999  # cosine above which two predicted spectra count as identical


def score_peak(candidates, peak_mz, measured, spectra):
    """Both scores for the candidates of one peak.

    Returns ``chosen`` (the heuristic's picks, all ties), ``bond_score``, ``score``
    (non-negative weights normalized to sum 1; all zero if nothing fits), ``ours``
    (our top candidate and everything tied with it) and ``mixed`` (the weighted sum).
    """
    n = len(candidates)
    theoretical = np.array([c.metadata["theoretical_mz"] for c in candidates])
    ppm = np.abs(peak_mz - theoretical) / theoretical * 1e6
    bond_score = np.array([min(o["bond_score"] for o in c.metadata["origins"]) for c in candidates])
    closest = np.isclose(ppm, ppm.min(), rtol=0, atol=1e-6)
    chosen = closest & (bond_score == bond_score[closest].min())
    with np.errstate(all="ignore"):  # numpy's Accelerate matmul emits spurious warnings
        weights, _ = nnls(spectra.T, measured) if len(measured) else (np.zeros(n), None)
        norm = np.linalg.norm(spectra, axis=1)
        unit = spectra / np.where(norm > 0, norm, 1.0)[:, None]
        similar = (unit @ unit.T >= SAME_SPECTRUM) & (norm > 0)[:, None] & (norm > 0)[None, :]
    score = weights / weights.sum() if weights.sum() > 0 else weights
    top = (score >= score.max() - 1e-9) & (score > 0)
    return {"chosen": chosen, "bond_score": bond_score, "score": score, "similar": similar,
            "ours": top | similar[:, top].any(1), "mixed": (weights[:, None] * spectra).sum(0)}


def compare_peak(candidates, peak_mz, measured, spectra):
    n = len(candidates)
    row = {"n_candidates": n, "n_ms3_peaks": len(measured),
           # share of the MS3 intensity that sits on peaks some candidate has a piece at
           "ms3_explainable": float(measured[spectra.sum(0) > 0].sum()) if n else 0.0}
    if n == 0:
        return {**row, "status": "no_candidate"}
    s = score_peak(candidates, peak_mz, measured, spectra)
    chosen, ours = s["chosen"], s["ours"]
    if s["score"].sum() <= 0:
        return {**row, "status": "no_score", "n_heuristic": int(chosen.sum())}
    like_chosen = chosen | s["similar"][:, chosen].any(1)
    h, t = int(chosen.sum()), int(ours.sum())
    return {
        **row, "status": "single_candidate" if n == 1 else "compared",
        "n_heuristic": h, "n_ours": t, "match": bool((chosen & ours).any()),
        "chance": 1 - comb(n - t, h) / comb(n, h),   # a random pick of the same size hitting our top set
        "our_score_on_heuristic": float(s["score"][like_chosen].sum()),
        "fit_cosine": float(np.sum(s["mixed"] * measured) / (np.linalg.norm(s["mixed"]) * np.linalg.norm(measured))),
        "min_bond_score": float(s["bond_score"].min()), "bond_score_of_our_top": float(s["bond_score"][ours].min()),
    }


def _work(job):
    from magma_workflow.candidates import find_ms3_children
    from magma_workflow.dataset import Spectrum

    records, targets = job
    related = [Spectrum.from_record(dict(r)) for r in records]
    rows = []
    for ms2 in related:
        if ms2.ms_level != 2 or ms2.metadata["usi"][0] not in targets:
            continue
        try:
            linked = find_ms3_children(ms2, spectra=[s for s in related if s.smiles == ms2.smiles])
            for peak, scans in sorted(linked.items()):
                ms3 = max(scans, key=lambda s: len(s.peaks))
                candidates, _, measured, spectra = MS3.candidate_spectra(MS3._RUN, ms2, ms3, int(peak))
                rows.append({"usi": ms2.metadata["usi"][0], "inchikey": ms2.metadata.get("inchikey"),
                             "peak_index": int(peak), "peak_mz": float(ms2.mz[peak]),
                             "ms3_ce": float(ms3.metadata["collision_energy"]),
                             **compare_peak(candidates, float(ms2.mz[peak]), measured, spectra)})
        except (ValueError, AssertionError):  # unsupported adduct etc.: skip this MS2 spectrum
            continue
    return rows


def comparison_table(data, run, molecules=None, seed=0, workers=8, force=False):
    """One row per MS3-linked MS2 peak of ``molecules`` random validation molecules (None or
    more than there are = all of them). Cached on disk; the test set stays held out."""
    import pandas as pd

    val = np.asarray(data.splits["molecule"]["val"])
    mol = data.fs.meta["mol_id"].values[val]
    unique = np.unique(mol)
    n = len(unique) if molecules is None else min(int(molecules), len(unique))
    path = OUT_DIR / f"peaks_{METHOD}_{n}molecules_seed{seed}.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    picked = np.random.default_rng(seed).choice(unique, n, replace=False)
    jobs = MS3.collect_jobs(data.fs, data.examples, val[np.isin(mol, picked)])
    with ProcessPoolExecutor(max_workers=workers, initializer=MS3._init_worker, initargs=(run.path,)) as pool:
        rows = [row for result in pool.map(_work, jobs, chunksize=4) for row in result]
    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def report(table):
    """Print how often the bond-score pick is (among) the top MS3-scored candidate(s)."""
    compared = table[table["status"] == "compared"]
    tied = compared["n_ours"] > 1
    line = lambda name, t: print(f"  {name:<34}{len(t):>6,} peaks   {t['match'].mean():6.1%}   (chance {t['chance'].mean():.1%})")
    print(f"{table['inchikey'].str[:14].nunique():,} molecules, {len(table):,} MS3-linked MS2 peaks, "
          f"{len(compared):,} with at least 2 candidates and an MS3 score\n")
    print("Lowest bond score is (among) the highest MS3-scored candidate(s):")
    line("all peaks", compared)
    line("our top is a single candidate", compared[~tied])
    line("our top is a tie", compared[tied])


def show_example(run, features, table, agree, seed=0, max_candidates=6):
    """Display one random peak where the bond-score pick and the top MS3 score agree (or not):
    the molecule, its MS2 and MS3, then every candidate with both scores, its predicted
    spectrum and the measured MS3, highest MS3 score first."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display
    from rdkit import rdBase

    from magma_workflow.candidates import find_ms3_children
    from magma_workflow.dataset import Spectrum
    from magma_workflow.plotting import plot_candidate_rows, plot_smiles

    pool = table[(table["status"] == "compared") & (table["match"] == agree) & (table["n_candidates"] <= max_candidates)]
    row = pool.iloc[np.random.default_rng(seed).integers(len(pool))]
    peak = int(row["peak_index"])
    records = next(records for records, targets in features if row["usi"] in targets)
    related = [Spectrum.from_record(dict(r)) for r in records]
    ms2 = next(s for s in related if s.ms_level == 2 and s.metadata["usi"][0] == row["usi"])
    scans = find_ms3_children(ms2, spectra=[s for s in related if s.smiles == ms2.smiles])[peak]
    ms3 = max(scans, key=lambda s: len(s.peaks))
    with rdBase.BlockLogs():  # MAGMa's InChI round trip logs stereo warnings
        candidates, mz, measured, spectra = MS3.candidate_spectra(run, ms2, ms3, peak)
    s = score_peak(candidates, float(ms2.mz[peak]), measured, spectra)

    figure, (top, middle, bottom) = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.5, 1, 1]})
    top.imshow(np.asarray(plot_smiles(ms2.smiles, size=(520, 300))))
    top.axis("off")
    top.set_title(f"{'AGREE' if agree else 'DISAGREE'}  |  {ms2.metadata.get('compound_name')}  |  "
                  f"peak m/z {ms2.mz[peak]:.4f}  |  fit cosine {row['fit_cosine']:.2f}", loc="left", fontsize=10)
    middle.vlines(ms2.mz, 0, ms2.intensity / ms2.intensity.max(), color="black", linewidth=1)
    middle.vlines(ms2.mz[peak], 0, ms2.intensity[peak] / ms2.intensity.max(), color="crimson", linewidth=2.5,
                  label=f"linked peak, m/z {ms2.mz[peak]:.4f}")
    middle.set_ylabel(f"MS2 (CE {ms2.metadata['collision_energy']:g})")
    middle.legend()
    bottom.vlines(ms3.mz, 0, ms3.intensity / ms3.intensity.max(), color="black", linewidth=1)
    bottom.axvline(ms2.mz[peak], color="crimson", linestyle=":", linewidth=1)
    bottom.set_ylabel(f"MS3 (CE {ms3.metadata['collision_energy']:g})")
    bottom.set_xlabel("m/z")
    bottom.set_xlim(middle.get_xlim())
    plt.tight_layout()
    plt.show()

    captions = [[f"bond score {b:g}" + ("  ← lowest" if c else ""), f"MS3 score {w:.2f}" + ("  ← highest" if o else "")]
                for b, c, w, o in zip(s["bond_score"], s["chosen"], s["score"], s["ours"])]
    display(HTML('<div style="height:40px"></div>'))
    display(plot_candidate_rows(candidates, mz, spectra, captions, measured=measured,
                                order=np.argsort(-s["score"], kind="stable")))


def main(argv=None):
    from . import experiments as EXP
    from .pipeline import load_data
    from .train import train

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--molecules", type=int, default=None, help="default: all validation molecules")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args(argv)
    data = load_data(verbose=False)
    run = train(data.fs, data.splits["molecule"], EXP.get("gnn"), data.data_key(), verbose=False)
    report(comparison_table(data, run, args.molecules, args.seed, args.workers, args.force))


if __name__ == "__main__":
    main()
