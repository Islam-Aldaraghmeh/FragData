"""Explain a few recorded adapter failures without model or chemistry changes.

Run with the repository's .venv/bin/python. This checks combinations of the
existing atom-local domains, not RDKit sanitization or chemical plausibility.
"""
import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "distilled_fragdata"))

from src.chemistry._state_graph import CHARGE_FAMILIES, _local_domains
from src.chemistry._radical_enumeration import _radical_domains


# Concrete, distinct constraints observed in the seeded September stress run.
EXAMPLE_IDS = (
    "candidate_b25eb4b57f5bb4423fce041816608835d49a0cd4fe47cb9f5fb2a381ef27a24a",
    "candidate_61930afa67d97518259a605f5392426caff5c9d9e1b56eabf5d1aa6be2a593ab",
    "candidate_13fc33dcbcc6e04ff0419848f25981a139b280d9718199dbbe45fbe41f63ca18",
    "candidate_19df15c134e31821bb7aca73f1f9e253e3db284e89585836eaa3aef0e5ff70ca",
)


def explain(source, spectrum, row):
    """Compute exact local-domain bounds without limiting total absolute Q."""
    stripped, metadata = row["stripped"], row.get("metadata", {})
    required_r = int(metadata.get("required_total_radicals", 0))
    if required_r == 1:
        domains = _radical_domains(stripped)
    else:
        domains = [tuple((q, h, 0) for q, h in domain)
                   for domain in _local_domains(stripped)]
    target_h, target_q = stripped["ionic_formula"].get("H", 0), stripped["charge"]

    # H is nonnegative locally, so removing H>target cannot remove a solution.
    reachable = {(0, 0, 0, 0)}
    for domain in domains:
        reachable = {
            (h + ah, q + aq, absolute_q + abs(aq), r + ar)
            for h, q, absolute_q, r in reachable for aq, ah, ar in domain
            if h + ah <= target_h and r + ar <= required_r
        }
    required_families = sorted({absolute_q for h, q, absolute_q, r in reachable
                               if (h, q, r) == (target_h, target_q, required_r)})

    # Separate minimization retains only minimum H for each net Q/radical sum.
    # It deliberately has neither the target-H bound nor an absolute-Q cap.
    minimum_h = {(0, 0): 0}
    for domain in domains:
        next_minimum = {}
        for (q, r), h in minimum_h.items():
            for aq, ah, ar in domain:
                if r + ar > required_r:
                    continue
                key, value = (q + aq, r + ar), h + ah
                next_minimum[key] = min(next_minimum.get(key, value), value)
        minimum_h = next_minimum
    net_charge_h_min = minimum_h.get((target_q, required_r))
    any_charge_h_min = min((h for (q, r), h in minimum_h.items()
                           if r == required_r), default=None)
    declared = metadata.get("charge_families", list(CHARGE_FAMILIES))
    if net_charge_h_min is None:
        proof = "No local assignment reaches the required net charge/radical count."
    elif net_charge_h_min > target_h:
        proof = (f"Required net charge/radical count needs at least {net_charge_h_min} H; "
                 f"formula has {target_h} H, even without an absolute-charge cap.")
    elif required_families and not set(required_families).intersection(declared):
        proof = (f"Formula/charge/radical sums require total absolute charge in "
                 f"{required_families}; declared families are {declared}.")
    elif not required_families:
        proof = "No local assignment has the exact required H/charge/radical sums."
    else:
        proof = ("Local sums are feasible in a declared family; inspect sanitization "
                 "rejections or search budgets for the recorded failure.")

    return {
        "source_file": source.name, "spectrum_index": spectrum["index"],
        "spectrum_status": spectrum["status"], "scan": spectrum["scan"],
        **{key: row[key] for key in ("candidate_id", "bag_index", "candidate_index", "mz")},
        "has_ms3": row.get("has_ms3", False),
        "formula": stripped["ionic_formula"], "charge": target_q,
        "required_radicals": required_r,
        "atoms": stripped["atoms"], "bonds": stripped["bonds"],
        "reason": row.get("reason"), "baseline_reason": row.get("baseline_reason"),
        "route": row.get("route"), "leaves_checked": metadata.get("leaves_checked"),
        "stop_reason": metadata.get("stop_reason"), "truncated": metadata.get("truncated"),
        "rejections": metadata.get("rejections", {}),
        "declared_charge_families": declared, "declared_max_abs_charge": max(declared),
        "required_abs_charge_families": required_families,
        "minimum_required_abs_charge": min(required_families, default=None),
        "minimum_h_any_net_charge": any_charge_h_min,
        "minimum_h_required_net_charge": net_charge_h_min,
        "proof": proof,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "recovery_lab" / "stress_20260922")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    failures = []
    for source in sorted(args.run_dir.glob("spectrum_*.json")):
        spectrum = json.loads(source.read_text())
        failures.extend((source, spectrum, row) for row in spectrum["candidate_rows"]
                        if row.get("outcome") == "unavailable")
    selected = [next(item for item in failures if item[2]["candidate_id"] == candidate_id)
                for candidate_id in EXAMPLE_IDS
                if any(item[2]["candidate_id"] == candidate_id for item in failures)]
    represented_reasons = {item[2].get("reason") for item in selected}
    for item in failures:
        if len(selected) == 5:
            break
        if item[2].get("reason") not in represented_reasons:
            selected.append(item)
            represented_reasons.add(item[2].get("reason"))
    result = {
        "method": "Exact sums of existing local domains, unrestricted total absolute charge.",
        "limitation": ("No sanitization or prediction is performed. A reachable assignment "
                       "is not proof of a valid, plausible or accurately predicted molecule."),
        "examples": [explain(*item) for item in selected],
    }
    output = args.output or args.run_dir / "domain_examples.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {len(result['examples'])} examples to {output}")
    for row in result["examples"]:
        print(f"{row['source_file']} bag {row['bag_index']} candidate {row['candidate_index']}: "
              f"{row['proof']}")


if __name__ == "__main__":
    main()
