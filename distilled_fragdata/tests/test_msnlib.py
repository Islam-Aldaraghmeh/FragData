"""Frozen prototype records cover parsing, linkage and existing-bag loading."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import pytest

from src.data import msnlib
from src.data.candidate_bags import candidate_bag_from_record, iter_candidate_bags, load_candidate_bag

FIXTURE = json.loads((Path(__file__).parent / "fixtures/msnlib.json").read_text())


def canonical(module, raw, row=1):
    return module.canonical_metadata(raw, source_row=row, **FIXTURE["metadata_kwargs"])


def test_physical_identity_and_canonical_metadata_match_prototype(prototype_src):
    old = importlib.import_module("ms3_data") if prototype_src else None
    for case in FIXTURE["metadata"]:
        actual = canonical(msnlib, case["raw"])
        assert actual == case["expected"]
        if old:
            assert actual == canonical(old, case["raw"])
            assert msnlib.physical_scan_identity(case["raw"]) == old.physical_scan_identity(case["raw"])
    first = FIXTURE["metadata"][0]["expected"]
    assert first["ionic_formula"] is None and first["charge"] is None
    assert first["parent_formula"] == "C20H20" and first["parent_charge"] == 1


def test_all_link_statuses_and_linked_peaks_match_prototype(prototype_src):
    old = importlib.import_module("ms3_data") if prototype_src else None
    summary = set()
    for case in FIXTURE["links"]:
        child = canonical(msnlib, case["child"])
        parents = [canonical(msnlib, row, i + 2) for i, row in enumerate(case["parents"])]
        result = msnlib.link_ms3(child, parents)
        assert result == case["expected"], case["name"]
        if old:
            assert result == old.link_ms3(child, parents), case["name"]
        summary.add(msnlib.link_status(result))
    assert summary == {"unique", "ambiguous", "unresolved", "conflicting"}


def test_metadata_peaks_and_existing_candidate_loaders(tmp_path, prototype_src):
    path = tmp_path / "20241003_enamdisc_pos_msn.json"
    raw = FIXTURE["metadata"][0]["raw"]
    path.write_text(json.dumps(raw) + "\n")
    rows = list(msnlib.iter_metadata(path, include_ms2_peak_mzs=True))
    assert msnlib.load_source_row(rows[0]) == raw
    assert msnlib.load_peaks(rows[0]) == raw["peaks"]
    if prototype_src:
        old = importlib.import_module("ms3_data")
        assert rows == list(old.iter_metadata(path, include_ms2_peak_mzs=True))
        assert msnlib.load_peaks(rows[0]) == old.load_peaks(rows[0])
    path.write_text(json.dumps({**raw, "scan_number": 99}) + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        msnlib.load_source_row(rows[0])

    stripped = {"atoms": [{"element": "N", "aromatic": False}], "bonds": [],
                "ionic_formula": {"N": 1, "H": 4}, "charge": 1, "precursor_mz": 18.0338}
    legacy = {"bag_id": "bag_fixture", "linked_isolation_id": "linked_peak_fixture",
              "isolated_mz": 18.0338, "observation_ids": ["obs_fixture"],
              "candidates": [{"candidate_id": "candidate_fixture", "stripped": stripped,
                              "ionic_formula": "H4N", "origins": [{"origin_id": "frozen"}]}]}
    legacy["candidates"].append({**deepcopy(legacy["candidates"][0]), "candidate_id": "second"})
    expected = candidate_bag_from_record(legacy)
    assert expected.candidates[0].stripped == stripped
    assert [c.candidate_id for c in expected.candidates] == ["candidate_fixture", "second"]
    assert expected.metadata["observation_ids"] == ["obs_fixture"]
    assert expected.candidates[0].metadata["origins"] == [{"origin_id": "frozen"}]
    for name, payload in (("bags.jsonl", json.dumps(legacy) + "\n"),
                          ("bag.json", json.dumps(legacy)), ("list.json", json.dumps([legacy])),
                          ("nested.json", json.dumps({"bags": [legacy]}))):
        candidate_path = tmp_path / name
        candidate_path.write_text(payload)
        assert list(iter_candidate_bags(candidate_path)) == [expected]
        assert load_candidate_bag(candidate_path, bag_id="bag_fixture") == expected
    duplicate = deepcopy(legacy)
    duplicate["candidates"].append(duplicate["candidates"][0])
    with pytest.raises(ValueError, match="Duplicate"):
        candidate_bag_from_record(duplicate)
    for candidate in ({"candidate_id": "flat", **stripped},
                      {"candidate_id": "graph", "graph": {k: stripped[k] for k in ("atoms", "bonds")},
                       **{k: stripped[k] for k in ("ionic_formula", "charge", "precursor_mz")}}):
        adapted = candidate_bag_from_record({"bag_id": "direct", "candidates": [candidate]})
        assert adapted.candidates[0].stripped == stripped


def test_fragment_ce_and_energy_groups_preserve_replicates():
    row = deepcopy(FIXTURE["metadata"][0]["expected"])
    assert msnlib.isolated_fragment_mz(row) == 100. != row["parent_precursor_mz"]
    assert msnlib.ms3_collision_energy(row) == 20.
    assert msnlib.nce_to_msg_scalar(20., msnlib.isolated_fragment_mz(row)) == 4.
    row["linked_isolation_id"] = "linked_peak_fixture"
    duplicate = {**row, "observation_id": "second_scan"}
    higher = {**row, "observation_id": "higher_energy", "collision_energy": 40.}
    groups = msnlib.group_ms3_energies([row, duplicate, higher])
    assert groups == {"linked_peak_fixture": {20.: [row, duplicate], 40.: [higher]}}
    assert msnlib.group_ms3_energies([row]) == {"linked_peak_fixture": {20.: [row]}}
