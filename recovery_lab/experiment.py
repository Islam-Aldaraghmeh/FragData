"""Isolated recovery pilot. Importing this module does not load model weights."""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import baseline_snapshot as baseline
from .aromatic_recovery import prepare_aromatic_recovery
from .neutral_reference import prepare_neutral_reference
from src.chemistry.fragment_adapter import FragmentGlacierAdapter
from src.data.candidate_bags import candidate_from_record
from src.models.glacier_api import GlacierAPI
from src.spectra.spectrum import (
    Spectrum, clean_raw, prepare_spectrum, normalize_spectrum,
    mean_normalized_spectrum,
)

LAB = Path(__file__).resolve().parent


def load_pilot():
    return json.loads((LAB / "pilot_candidates.json").read_text())


def load_model():
    """Use the same verified frozen assets as the original notebook."""
    return GlacierAPI(
        baseline._asset_path(None, "GLACIER_UPSTREAM", "vendor/ms-pred"),
        baseline._asset_path(None, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt"),
    )


def candidate_from_pilot(row):
    return candidate_from_record({k: row[k] for k in
                                  ("candidate_id", "stripped", "metadata")})


def prepare_candidate(candidate, collision_energy, *, model=None,
                      allow_experimental=False, max_states=4,
                      instrument="Orbitrap"):
    """Keep the baseline when available; recovery requires explicit opt-in.

    No observed spectrum, score, or confirmed fragment structure selects states.
    A neutral reference is never labelled as a recovered ionic molecule.
    """
    adapter = FragmentGlacierAdapter(model, max_states=max_states)
    original = adapter.prepare_baseline(candidate, collision_energy, instrument)
    result = {**original, "route": "baseline", "experimental": False,
              "baseline_reason": original["reason"]}
    if original["states"] or not allow_experimental:
        return result
    reason = original["reason"]
    if reason == "odd_electron_formula_outside_closed_shell_domain":
        helper = prepare_neutral_reference
    elif reason == "no_valid_state_in_declared_domain":
        helper = prepare_aromatic_recovery
    else:
        return result
    recovered = helper(candidate, collision_energy, instrument, max_states=max_states)
    return {**recovered, "baseline_reason": reason,
            "baseline_metadata": original["metadata"]}


def predict_prepared(model, prepared):
    """Use exactly the existing adapter's state-spectrum postprocessing."""
    if not prepared["states"]:
        return None
    spectra = []
    for model_input in prepared["inputs"]:
        raw = model.predict_records(**model_input)["spec"].detach().cpu().numpy()
        if not np.isfinite(raw).all():
            raise ValueError("GLACIER produced nonfinite values")
        spectra.append(normalize_spectrum(prepare_spectrum(clean_raw(raw))))
    return Spectrum.from_peaks(mean_normalized_spectrum(spectra, prepared=True))


def predict_candidate(model, candidate, collision_energy, *,
                      allow_experimental=False, max_states=4, instrument="Orbitrap"):
    """Return spectrum plus provenance, retaining None for unavailable inputs."""
    prepared = prepare_candidate(candidate, collision_energy, model=model,
                                 allow_experimental=allow_experimental,
                                 max_states=max_states, instrument=instrument)
    return {**prepared, "spectrum": predict_prepared(model, prepared)}


def verify_production_files():
    """Accept a recorded, regression-tested source revision.

    The historical manifest is immutable. The separate compatible manifest
    records the opt-in model flags and promoted helpers. The local-assets
    manifest records their relocation; arbitrary edits still fail this check.
    Defaults retain the original baseline policy.
    """
    differences = {}
    for name in ("snapshot_manifest.json", "compatible_source_manifest.json",
                 "local_source_manifest.json"):
        manifest_path = LAB / name
        if not manifest_path.exists():
            continue
        hashes = json.loads(manifest_path.read_text())["production_files_sha256"]
        changed = [path for path, expected in hashes.items()
                   if not (LAB.parent / path).exists() or
                   hashlib.sha256((LAB.parent / path).read_bytes()).hexdigest() != expected]
        if not changed:
            return len(hashes)
        differences[name] = changed
    raise AssertionError(f"Production files differ from recorded source snapshots: {differences}")


def run_pilot(model=None, *, sample_per_route=None):
    """Audit all 84 preparations, regress 9 baseline outputs, predict recovery.

    Writes only under recovery_lab. This is a coverage and regression experiment,
    not a held-out accuracy evaluation. Timing reflects this CPU session only.
    """
    if sample_per_route is not None and (
            isinstance(sample_per_route, bool) or not isinstance(sample_per_route, int)
            or sample_per_route < 1):
        raise ValueError("sample_per_route must be None or a positive integer")
    model = model or load_model()
    pilot = load_pilot()
    ce, max_states = pilot["model_collision_energy"], pilot["max_states"]
    prepared_rows, rows = [], []
    started = time.perf_counter()
    for i, source in enumerate(pilot["candidates"]):
        candidate = candidate_from_pilot(source)
        before = deepcopy(candidate.stripped)
        tick = time.perf_counter()
        prepared = prepare_candidate(candidate, ce, model=model,
                                     allow_experimental=True, max_states=max_states)
        assert candidate.stripped == before, "Recovery mutated an input candidate"
        if source["baseline_states"]:
            assert prepared["states"] == source["baseline_states"]
            assert prepared["route"] == "baseline"
        row = {k: source[k] for k in ("bag_index", "candidate_index", "candidate_id")}
        row.update(mz=candidate.precursor_mz, ionic_formula=candidate.ionic_formula,
                   baseline_reason=source["baseline_reason"] or "ok",
                   route=prepared["route"], available=bool(prepared["states"]),
                   reason=prepared["reason"], experimental=prepared["experimental"],
                   state_count=len(prepared["states"]),
                   prepare_seconds=time.perf_counter() - tick,
                   states=prepared["states"], metadata=prepared["metadata"],
                   audit=prepared.get("audit"), input_unchanged=True)
        rows.append(row)
        prepared_rows.append((candidate, prepared))
        if (i + 1) % 10 == 0 or i + 1 == len(pilot["candidates"]):
            print(f"Prepared {i + 1}/{len(pilot['candidates'])}; "
                  f"available {sum(r['available'] for r in rows)}", flush=True)
    preparation_seconds = time.perf_counter() - started
    baseline_indices = [i for i, r in enumerate(rows)
                        if r["available"] and r["route"] == "baseline"]
    sample_indices = []
    for route in sorted({r["route"] for r in rows if r["experimental"]}):
        eligible = [i for i, r in enumerate(rows) if r["route"] == route and r["available"]]
        if sample_per_route is None:
            sample_indices.extend(eligible)
            continue
        # Spread the forward samples across m/z groups when possible.
        selected = []
        for i in eligible:
            if not any(rows[j]["mz"] == rows[i]["mz"] for j in selected):
                selected.append(i)
            if len(selected) >= sample_per_route:
                break
        for i in eligible:
            if len(selected) >= sample_per_route:
                break
            if i not in selected:
                selected.append(i)
        sample_indices.extend(selected)
    adapter = FragmentGlacierAdapter(model, max_states=max_states)
    predictions, regression = [], []
    for i in baseline_indices + sample_indices:
        candidate, prepared = prepared_rows[i]
        tick = time.perf_counter()
        observed = predict_prepared(model, prepared).as_array()
        duration = time.perf_counter() - tick
        assert observed.ndim == 2 and observed.shape[1] == 2
        assert np.isfinite(observed).all()
        assert len(observed) and np.isclose(observed[:, 1].sum(), 1)
        same = None
        if i in baseline_indices:
            expected = adapter.predict(candidate, ce).as_array()
            same = np.array_equal(expected, observed)
            regression.append({"candidate_id": candidate.candidate_id,
                               "array_equal": same, "peak_count": len(observed)})
            assert same, "Existing candidate prediction changed"
        predictions.append({"candidate_id": candidate.candidate_id,
                            "bag_index": rows[i]["bag_index"],
                            "candidate_index": rows[i]["candidate_index"],
                            "mz": candidate.precursor_mz, "route": prepared["route"],
                            "experimental": prepared["experimental"],
                            "states": prepared["states"], "spectrum": observed.tolist(),
                            "forward_seconds": duration, "baseline_array_equal": same})
        print(f"Forward bag {rows[i]['bag_index']} candidate {rows[i]['candidate_index']}: "
              f"{prepared['route']}, {len(observed)} peaks", flush=True)
    labels = model.upstream_path / "data/spec_datasets/msg/labels.tsv"
    with labels.open() as stream:
        adduct_counts = dict(Counter(row["ionization"] for row in
                                    csv.DictReader(stream, delimiter="\t")))
    summary = {
        "total_candidates": len(rows), "baseline_available": len(baseline_indices),
        "recovery_available": sum(r["available"] and r["experimental"] for r in rows),
        "combined_available": sum(r["available"] for r in rows),
        "remaining_unavailable": sum(not r["available"] for r in rows),
        "available_by_route": dict(Counter(r["route"] for r in rows if r["available"])),
        "remaining_reasons": dict(Counter(r["reason"] for r in rows if not r["available"])),
        "preparation_seconds": preparation_seconds,
        "baseline_regression_cases": len(regression),
        "baseline_all_array_equal": all(r["array_equal"] for r in regression),
        "recovered_forward_samples": len(sample_indices),
        "all_available_candidates_forward_tested": len(predictions) == sum(r["available"] for r in rows),
        "all_forward_outputs_finite_nonempty_normalized": True,
        "production_files_unchanged": verify_production_files(),
        "checkpoint": model.checkpoint_identity,
        "local_msg_label_adduct_counts": adduct_counts,
        "recovered_accuracy_validated": False,
        "parent_scan": pilot["parent_scan"], "model_collision_energy": ce,
        "ms3_observations": pilot["bag_observations"],
    }
    report = {"summary": summary, "candidates": rows,
              "baseline_regression": regression, "forward_samples": predictions}
    (LAB / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_pilot()
