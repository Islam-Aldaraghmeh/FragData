"""Summarize finished stress reports; safe to rerun while the probe is running.

Usage: python recovery_lab/summarize_stress.py [directory] [--prefix partial_]
Only this script's summary/CSV outputs are replaced. Started or unreadable
reports are listed separately and never enter finished-cohort statistics.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics


def statistics_of(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0, "sum": 0.0}

    def quantile(p):
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return dict(count=len(values), sum=sum(values), min=values[0],
                median=statistics.median(values), p90=quantile(.9),
                p95=quantile(.95), max=values[-1])


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidate_seconds(row):
    return row.get("prepare_seconds", 0) + row.get("prediction_seconds", 0)


def duplicate_summary(values):
    counts = Counter(values)
    return dict(occurrences=sum(counts.values()), unique=len(counts),
                duplicate_occurrences=sum(n - 1 for n in counts.values()),
                repeated_keys=sum(n > 1 for n in counts.values()),
                maximum_multiplicity=max(counts.values(), default=0))


def bag_availability(reports):
    """Keep full-bag failure claims restricted to fully attempted spectra."""
    observed = {scope: Counter(known_bags=0, observed_bags=0, bags_with_prediction=0,
                              bags_with_candidate_timeout=0, wholly_unattempted_bags=0)
                for scope in ("all", "linked_ms3", "other")}
    complete = {scope: Counter(bags=0, bags_with_prediction=0, all_candidates_unavailable=0,
                              no_prediction_with_timeout_or_error=0, bags_with_candidate_timeout=0)
                for scope in observed}
    unavailable_bags, censored, complete_indices = [], [], []
    for report in reports:
        groups = defaultdict(list)
        for row in report.get("candidate_rows", []):
            groups[row["bag_index"]].append(row)
        fully_attempted = (report.get("status") == "complete"
            and len(report.get("candidate_rows", [])) == report.get("candidates")
            and len(groups) == report.get("homogeneous_bags"))
        if fully_attempted:
            complete_indices.append(report["index"])
        elif "homogeneous_bags" in report:
            censored.append(dict(spectrum_index=report["index"], status=report["status"],
                known_bags=report["homogeneous_bags"], observed_bags=len(groups),
                wholly_unattempted_bags=report["homogeneous_bags"] - len(groups),
                untested_candidates=report["candidates"] - len(report["candidate_rows"]),
                last_observed_bag=max(groups, default=None),
                last_observed_bag_completeness="unknown: individual expected bag sizes not recorded"))
        if "homogeneous_bags" in report:
            totals = dict(all=report["homogeneous_bags"], linked_ms3=report["linked_ms3_bags"],
                          other=report["homogeneous_bags"] - report["linked_ms3_bags"])
            for scope, total in totals.items():
                observed[scope]["known_bags"] += total
        for bag_index, rows in groups.items():
            outcomes = {row.get("outcome") for row in rows}
            linked = bool(rows[0].get("has_ms3"))
            predicted, timed_out = "predicted" in outcomes, "timeout" in outcomes
            unavailable = outcomes == {"unavailable"}
            for scope in ("all", "linked_ms3" if linked else "other"):
                observed[scope]["observed_bags"] += 1
                observed[scope]["bags_with_prediction"] += predicted
                observed[scope]["bags_with_candidate_timeout"] += timed_out
                if fully_attempted:
                    complete[scope]["bags"] += 1
                    complete[scope]["bags_with_prediction"] += predicted
                    complete[scope]["all_candidates_unavailable"] += unavailable
                    complete[scope]["no_prediction_with_timeout_or_error"] += not predicted and not unavailable
                    complete[scope]["bags_with_candidate_timeout"] += timed_out
            if fully_attempted and unavailable:
                unavailable_bags.append(dict(spectrum_index=report["index"], bag_index=bag_index,
                    mz=rows[0]["mz"], has_ms3=linked, candidates=len(rows),
                    reasons=Counter(row.get("reason", "unknown") for row in rows)))
    for values in observed.values():
        values["wholly_unattempted_bags"] = values["known_bags"] - values["observed_bags"]
    return dict(scope="Homogeneous nonempty candidate bags; started spectrum reports excluded.",
        all_finished_spectra=observed,
        full_bag_taxonomy_scope="Only complete spectra with every candidate and bag accounted for; timeout is not unavailable.",
        fully_attempted_spectrum_indices=complete_indices, fully_attempted_spectra=complete,
        fully_unavailable_bags=unavailable_bags, censored_spectrum_bag_coverage=censored,
        spectra_with_unknown_bag_total=sum("homogeneous_bags" not in r for r in reports))


def summarize(directory):
    reports, pending, unreadable = [], [], []
    for path in sorted(directory.glob("spectrum_*.json")):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            unreadable.append(dict(file=path.name, error=str(error)))
            continue
        if report.get("status") == "started":
            pending.append(dict(file=path.name, candidates=report.get("candidates"),
                                recorded_candidates=len(report.get("candidate_rows", []))))
        else:
            reports.append(report)
    rows = [(report, row) for report in reports for row in report.get("candidate_rows", [])]
    outcomes = Counter(row.get("outcome", "unknown") for _, row in rows)
    known_total = sum(report.get("candidates", 0) for report in reports)
    reasons, phases, routes, failures, rescued = Counter(), Counter(), {}, Counter(), Counter()
    reason_details = defaultdict(lambda: dict(candidates=0, zero_leaves=0, positive_leaves=0,
                                            leaves_unreported=0, truncated=0,
                                            stop_reasons=Counter(), rejections=Counter()))
    for report, row in rows:
        outcome, route = row.get("outcome", "unknown"), row.get("route", "unknown")
        route_rows = routes.setdefault(route, dict(candidates=0, outcomes=Counter(),
                                                  preparation_seconds=0., prediction_seconds=0.))
        route_rows["candidates"] += 1
        route_rows["outcomes"][outcome] += 1
        route_rows["preparation_seconds"] += row.get("prepare_seconds", 0)
        route_rows["prediction_seconds"] += row.get("prediction_seconds", 0)
        if row.get("baseline_reason"):
            failures[row["baseline_reason"]] += 1
            if row.get("states") and route != "baseline":
                rescued[route] += 1
        if outcome != "predicted":
            reason = row.get("reason") or row.get("error_type") or outcome
            reasons[reason] += 1
            if row.get("error_phase"):
                phases[row["error_phase"] + ":" + outcome] += 1
            detail = reason_details[reason]
            metadata = row.get("metadata", {})
            leaves = metadata.get("leaves_checked")
            detail["candidates"] += 1
            detail["leaves_unreported" if leaves is None else
                   "zero_leaves" if leaves == 0 else "positive_leaves"] += 1
            detail["truncated"] += bool(metadata.get("truncated"))
            detail["stop_reasons"][metadata.get("stop_reason", "unreported")] += 1
            detail["rejections"].update(metadata.get("rejections", {}))

    groups = {}
    for label, linked in (("linked_ms3", True), ("other", False)):
        selected = [row for _, row in rows if bool(row.get("has_ms3")) == linked]
        groups[label] = dict(candidates=len(selected),
            outcomes=Counter(row.get("outcome", "unknown") for row in selected),
            preparation_seconds=sum(row.get("prepare_seconds", 0) for row in selected),
            prediction_seconds=sum(row.get("prediction_seconds", 0) for row in selected),
            combined_seconds=sum(candidate_seconds(row) for row in selected),
            retained_states=sum(len(row.get("states", [])) for row in selected))

    timing_keys = ("elapsed_seconds", "generation_seconds", "preparation_seconds", "prediction_seconds")
    timings = {"spectrum_" + key: statistics_of(r.get(key) for r in reports) for key in timing_keys}
    timings.update({"candidate_" + key: statistics_of(row.get(key) for _, row in rows)
                    for key in ("prepare_seconds", "prediction_seconds")})
    timings["candidate_combined_seconds"] = statistics_of(candidate_seconds(row) for _, row in rows)
    timings["candidate_mean_prediction_seconds_per_retained_state"] = statistics_of(
        row["prediction_seconds"] / len(row["states"]) for _, row in rows
        if row.get("outcome") == "predicted" and row.get("states"))

    def slow_candidate(item):
        report, row = item
        return dict(spectrum_index=report["index"], dataset=report.get("dataset"),
            feature_id=report.get("feature_id"), scan=report.get("scan"),
            **{key: row.get(key) for key in ("candidate_id", "bag_index", "candidate_index", "mz",
                "has_ms3", "route", "outcome", "reason", "baseline_reason", "error_phase",
                "prepare_seconds", "prediction_seconds", "metadata", "baseline_metadata")},
            total_seconds=candidate_seconds(row),
            ionic_formula=row.get("stripped", {}).get("ionic_formula"),
            heavy_atoms=len(row.get("stripped", {}).get("atoms", [])),
            retained_states=len(row.get("states", [])))

    cohort_path = directory / "cohort.json"
    planned = len(json.loads(cohort_path.read_text()).get("spectra", [])) if cohort_path.exists() else None
    summary = dict(planned_spectra=planned, finished_spectra=len(reports),
        spectrum_statuses=Counter(r.get("status", "unknown") for r in reports),
        pipeline_error_phases=Counter(r.get("error_phase", "unknown") for r in reports
                                      if r.get("status") != "complete"),
        pending_reports=pending, unreadable_reports=unreadable,
        completed_generation_spectra=sum("candidates" in r for r in reports),
        spectra_with_unknown_candidate_total=sum("candidates" not in r for r in reports),
        known_total_candidates=known_total, tested_candidates=len(rows),
        untested_known_candidates=known_total - len(rows), outcomes=outcomes,
        unavailability_and_error_reasons=reasons, candidate_error_phases=phases,
        routes=routes, baseline_failures=failures, rescued_candidates_by_route=rescued,
        failed_candidate_diagnostics=dict(reason_details), comparison_scope=groups,
        bag_availability=bag_availability(reports),
        timings=timings,
        retained_state_count_distribution=Counter(str(len(row.get("states", []))) for _, row in rows),
        preparation_stop_reasons=Counter(row.get("metadata", {}).get("stop_reason", "unreported")
                                         for _, row in rows),
        truncated_preparations=Counter(row.get("outcome", "unknown") for _, row in rows
                                      if row.get("metadata", {}).get("truncated")),
        completed_model_state_calls=sum(r.get("model_state_calls", 0) for r in reports),
        duplicated_exact_stripped=duplicate_summary(fingerprint(row["stripped"]) for _, row in rows),
        duplicated_candidate_ids=duplicate_summary(row["candidate_id"] for _, row in rows),
        duplicated_retained_feature_hashes=duplicate_summary(state["feature_hash"] for _, row in rows
            for state in row.get("states", []) if state.get("feature_hash")),
        duplicated_predicted_feature_hashes=duplicate_summary(state["feature_hash"] for _, row in rows
            if row.get("outcome") == "predicted" for state in row.get("states", [])
            if state.get("feature_hash")),
        slowest_candidates=[slow_candidate(item) for item in sorted(rows,
                            key=lambda item: candidate_seconds(item[1]), reverse=True)[:20]],
        timing_caveats=["Started reports are excluded; finished timeout reports include completed partial work.",
            "Timeouts are censored observations, not proofs that no valid chemical state exists.",
            "Per-state timing is candidate prediction time divided by retained states, not an independent timer.",
            "Feature hashes use fixed-context featurization; reuse additionally requires equal runtime inputs/settings."])
    csv_rows = []
    for report in reports:
        selected = report.get("candidate_rows", [])
        record = {key: report.get(key) for key in ("index", "dataset", "feature_id", "scan", "status",
            "precursor_mz", "candidates", "linked_ms3_candidates", *timing_keys, "model_state_calls")}
        record["tested_candidates"] = len(selected)
        record.update({outcome: sum(row.get("outcome") == outcome for row in selected)
                       for outcome in ("predicted", "unavailable", "timeout", "exception", "invalid_output")})
        for label, linked in (("linked_ms3", True), ("other", False)):
            record[label + "_candidate_seconds"] = sum(candidate_seconds(row) for row in selected
                                                       if bool(row.get("has_ms3")) == linked)
        csv_rows.append(record)
    return summary, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path,
                        default=Path(__file__).resolve().parent / "stress_20260922")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    if Path(args.prefix).name != args.prefix and args.prefix:
        parser.error("--prefix must be a filename prefix, not a path")
    summary, csv_rows = summarize(args.directory)
    buffer = io.StringIO()
    if csv_rows:
        writer = csv.DictWriter(buffer, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    for filename, contents in ((args.prefix + "summary.json", json.dumps(summary, indent=2) + "\n"),
                               (args.prefix + "timings.csv", buffer.getvalue())):
        path = args.directory / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(contents)
        temporary.replace(path)
    print(json.dumps({key: summary[key] for key in ("finished_spectra", "spectrum_statuses",
        "tested_candidates", "outcomes", "comparison_scope")}, indent=2))


if __name__ == "__main__":
    main()
