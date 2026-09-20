"""MSnLib parsing and conservative linkage, preserved from prototype ms3_data.

Top-level molecular annotation is parent data, never an isolated ion formula.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

INDIVIDUAL_MERGE_TYPES = {"SINGLE_BEST_SCAN", "SINGLE_SCAN"}
LINK_MZ_TOLERANCE_DA = 0.01
COLLECTION_FILES = {
    "20241003_enamdisc_pos_msn.json": "ENAMDISC",
    "20241003_mcedrug_pos_msn.json": "MCEDRUG",
    "20250828_mcediv_50k_sub_pos_msn.json": "MCEDIV_50k_sub",
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _list(value):
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _number(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _scalar(value):
    if isinstance(value, list):
        return _number(value[0]) if len(value) == 1 else None
    return _number(value)


def _raw_name(name):
    name = Path(str(name)).name
    return re.sub(r"\.(mzml|mzxml|raw)$", "", name, flags=re.IGNORECASE)


def _scan_tokens(value):
    """Expand exported scan ranges with a finite defensive bound."""
    values = []
    for piece in str(value).split(","):
        if re.fullmatch(r"\d+", piece):
            values.append(int(piece))
        elif re.fullmatch(r"\d+-\d+", piece):
            lo, hi = map(int, piece.split("-"))
            if hi < lo or hi-lo > 100_000:
                raise ValueError("Invalid or oversized scan range")
            values.extend(range(lo, hi+1))
        else:
            raise ValueError("Unsupported physical scan notation")
    return values


def _usi_scans(usi):
    parts = str(usi).split(":")
    if len(parts) == 4 and parts[0] == "mzspec":
        dataset, raw, scans = parts[1:]
    elif len(parts) >= 5 and parts[0] == "mzspec" and parts[3] in {"scan", "scan_number"}:
        dataset, raw, scans = parts[1], parts[2], parts[4]
    else:
        raise ValueError("USI does not identify physical scan numbers")
    return [(dataset, _raw_name(raw), n) for n in _scan_tokens(scans)]


def physical_scan_identity(record):
    """Use source_scan_usi before library accession USIs; cross-check raw scans."""
    source_usis = _list(record.get("source_scan_usi"))
    identities, issues = set(), []
    source = "source_scan_usi" if source_usis else "usi"
    # New export `usi` names a library entry, not a physical acquisition scan.
    usis = source_usis or ([] if record.get("raw_file_name") else _list(record.get("usi")))
    for usi in usis:
        try:
            identities.update(_usi_scans(usi))
        except ValueError as exc:
            issues.append(str(exc))
    raw = record.get("raw_file_name")
    dataset = record.get("dataset_id")
    fallback = set()
    if raw and dataset:
        try:
            for number in _list(record.get("scan_number")):
                fallback.update((str(dataset), _raw_name(raw), n) for n in _scan_tokens(number))
        except ValueError as exc:
            issues.append(str(exc))
    if identities and fallback and identities != fallback:
        issues.append("source_scan_usi_and_raw_scan_disagree")
    if identities and record.get("scan_number") is not None:
        try:
            annotated_scans = set(n for value in _list(record["scan_number"]) for n in _scan_tokens(value))
            if annotated_scans != {n for _, _, n in identities}:
                issues.append("source_usi_and_scan_number_disagree")
        except ValueError as exc:
            issues.append(str(exc))
    if not identities and fallback:
        identities, source = fallback, "raw_file_name+scan_number"
    if not identities:
        issues.append("physical_scan_identity_missing")
    raws = sorted({raw for _, raw, _ in identities})
    scans = sorted({scan for _, _, scan in identities})
    return {"physical_scan_ids": [f"{ds}|{raw}|scan={n}" for ds, raw, n in sorted(identities)],
            "physical_scan_identity_source": source, "physical_identity_issues": sorted(set(issues)),
            "raw_file": raws[0] if len(raws) == 1 else None,
            "scan_number": scans[0] if len(scans) == 1 else None}


def ce_provenance(collection):
    if collection in {"ENAMDISC", "MCEDRUG"}:
        return {"ce_units": "NCE_percent", "ce_unit_status": "verified_collection_method",
                "ce_provenance": "MSnLib Supplementary Table 2: Normalized HCD; outputs/ce_provenance/MSnLib_supplement.pdf",
                "model_ce_compatibility": "conditional_conversion_not_stage_calibrated"}
    return {"ce_units": None, "ce_unit_status": "unresolved",
            "ce_provenance": "Later collection absent from attested original acquisition table",
            "model_ce_compatibility": "excluded_from_unit_verified_stratum"}


def canonical_metadata(record, *, source_path, source_row, source_sha256,
                       byte_offset=None, row_sha256=None, include_ms2_peak_mzs=False):
    """Normalize one source row without assigning isolated-ion formula/charge."""
    path = Path(source_path).resolve()
    collection = COLLECTION_FILES.get(path.name, record.get("description", "UNKNOWN"))
    level = int(record["ms_level"])
    physical = physical_scan_identity(record)
    dataset = str(record.get("dataset_id") or "UNKNOWN_DATASET")
    raw_feature = record.get("featurelist_feature_id", record.get("feature_id"))
    feature_id = None if raw_feature is None else f"{dataset}|{physical['raw_file']}|{raw_feature}"
    ce = _scalar(record.get("collision_energy"))
    raw_path = record.get("msn_collision_energies")
    if isinstance(raw_path, list) and raw_path and isinstance(raw_path[0], list):
        ce_path = raw_path[0] if len(raw_path) == 1 else None
    else:
        ce_path = raw_path
    ce_path = [_number(v) for v in ce_path] if isinstance(ce_path, list) else ([ce] if level == 2 else None)
    precursors = record.get("msn_precursor_mzs")
    precursor_path = [_number(v) for v in precursors] if isinstance(precursors, list) else None
    if level == 2:
        precursor_path = [_number(record.get("precursor_mz"))]
    issues = list(physical["physical_identity_issues"])
    if ce_path is None or len(ce_path) != level-1 or any(v is None for v in ce_path):
        issues.append("missing_or_ambiguous_energy_path")
    if precursor_path is None or len(precursor_path) != level-1 or any(v is None for v in precursor_path):
        issues.append("missing_or_invalid_precursor_path")
    if ce is None:
        issues.append("missing_or_multiple_current_energies")
    if ce_path and ce is not None and ce_path[-1] != ce:
        issues.append("current_energy_disagrees_with_path")
    if feature_id is None:
        issues.append("feature_identity_missing")
    if len(physical["physical_scan_ids"]) != 1:
        issues.append("not_one_physical_scan")
    isolated_mz = precursor_path[-1] if precursor_path and len(precursor_path) == level-1 else None
    parent_charge = _scalar(record.get("charge"))
    parent_charge = int(parent_charge) if parent_charge is not None and parent_charge.is_integer() else None
    charge, charge_origin = None, "isolated_stage_charge_not_exported"
    explicit_charges = record.get("msn_precursor_charges")
    if isinstance(explicit_charges, list) and len(explicit_charges) == level-1:
        stage_charge = _scalar(explicit_charges[-1])
        if stage_charge is not None and stage_charge.is_integer() and stage_charge != 0:
            charge, charge_origin = int(stage_charge), "explicit_msn_precursor_charges"
    elif level == 2 and parent_charge is not None:
        charge, charge_origin = parent_charge, "original_precursor_charge"
    inchikey = record.get("inchikey")
    parent_connectivity = str(inchikey).split("-")[0] if isinstance(inchikey, str) and inchikey else None
    if parent_connectivity is None:
        issues.append("parent_connectivity_missing")
    windows = record.get("msn_isolation_windows")
    current_window = _number(record.get("isolation_window"))
    if current_window is None and isinstance(windows, list) and windows:
        current_window = _number(windows[-1])
    identity = {"feature": feature_id, "physical": physical["physical_scan_ids"], "level": level}
    isolation = {"feature": feature_id, "precursor_path": precursor_path,
                 "ancestor_ce_path": ce_path[:-1] if ce_path else None,
                 "isolation_windows": windows}
    peaks = record.get("peaks", [])
    out = {
        "observation_id": "msn_" + _digest(identity)[:24],
        "source": {"path": str(path), "row": int(source_row), "sha256": source_sha256,
                   "byte_offset": byte_offset, "row_sha256": row_sha256},
        "collection": collection, "dataset_id": dataset,
        "parent_connectivity": parent_connectivity, "parent_inchikey": inchikey,
        "parent_smiles": record.get("smiles"), "parent_formula": record.get("formula"),
        "parent_adduct": record.get("adduct"),
        "parent_precursor_mz": _number(record.get("precursor_mz")), "parent_charge": parent_charge,
        "feature_id": feature_id, "source_feature_id": raw_feature,
        "isolation_id": "isolation_path_" + _digest(isolation)[:24],
        "isolation_id_status": "acquisition_path_group_not_attested_physical_event",
        **physical, "ms_level": level, "merge_type": record.get("merge_type"),
        "individual_scan": record.get("merge_type") in INDIVIDUAL_MERGE_TYPES,
        "isolated_mz": isolated_mz, "ionic_formula": None, "charge": charge,
        "charge_provenance": charge_origin, "collision_energy": ce, **ce_provenance(collection),
        "precursor_mz_path": precursor_path, "collision_energy_path": ce_path,
        "instrument": record.get("instrument"), "instrument_type": record.get("instrument_type"),
        "acquisition_context": {"collection": collection, "dataset_id": dataset,
            "raw_file": physical["raw_file"], "polarity": record.get("polarity"),
            "ion_source": record.get("ion_source"), "acquisition_export_label": record.get("acquisition"),
            "fragmentation_methods": record.get("msn_fragmentation_methods") or [record.get("fragmenation_method")],
            "isolation_window": current_window, "merge_type": record.get("merge_type")},
        "peak_count": len(peaks), "precursor_purity": _number(record.get("precursor_purity")),
        "quality_chimeric": record.get("quality_chimeric"), "metadata_issues": sorted(set(issues)),
        "spectrum_path_digest": _digest({"peaks": peaks, "mz_path": precursor_path,
                                        "ce_path": ce_path, "instrument": record.get("instrument")}),
    }
    if include_ms2_peak_mzs and level == 2:
        out["ms2_peak_mzs"] = [float(p[0]) for p in peaks]
    return out


@lru_cache(maxsize=16)
def _hashed_source(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _hashed_source(str(path), stat.st_size, stat.st_mtime_ns)


def iter_metadata(path, *, ms_levels=(2, 3), individual_only=True, include_ms2_peak_mzs=False):
    """Stream canonical metadata with verified source and exact row locators."""
    path = Path(path).resolve()
    digest = source_sha256(path)
    with path.open("rb") as handle:
        row = 0
        while True:
            offset = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            row += 1
            record = json.loads(raw)
            if record.get("ms_level") not in ms_levels:
                continue
            if individual_only and record.get("merge_type") not in INDIVIDUAL_MERGE_TYPES:
                continue
            yield canonical_metadata(record, source_path=path, source_row=row, source_sha256=digest,
                byte_offset=offset, row_sha256=hashlib.sha256(raw).hexdigest(), include_ms2_peak_mzs=include_ms2_peak_mzs)


def load_source_row(metadata, *, verify_source=True):
    """Load only a requested original row, verifying source and row hashes."""
    locator = metadata["source"]
    path = Path(locator["path"])
    if verify_source and source_sha256(path) != locator["sha256"]:
        raise ValueError("Source file SHA256 changed")
    with path.open("rb") as handle:
        if locator.get("byte_offset") is not None:
            handle.seek(locator["byte_offset"])
            raw = handle.readline()
        else:
            raw = b""
            for _ in range(locator["row"]):
                raw = handle.readline()
    if not raw or (locator.get("row_sha256") and hashlib.sha256(raw).hexdigest() != locator["row_sha256"]):
        raise ValueError("Source row locator or SHA256 changed")
    return json.loads(raw)


def load_peaks(metadata, *, verify_source=True):
    return load_source_row(metadata, verify_source=verify_source).get("peaks", [])


def link_ms3(child, parents, *, mz_tolerance=LINK_MZ_TOLERANCE_DA):
    """Return all allowed (MS2 row, peak) links; never choose an ambiguous best."""
    parents = list(parents)
    matches = []
    parent_annotations = defaultdict(set)
    parent_spectra = defaultdict(set)
    for parent in parents:
        parent_annotations[parent["observation_id"]].add(parent["parent_connectivity"])
        parent_spectra[parent["observation_id"]].add(parent["spectrum_path_digest"])
    if child["ms_level"] != 3 or not child["individual_scan"] or child["metadata_issues"]:
        return {"status": "invalid_child_metadata", "matches": [], "mz_tolerance_da": mz_tolerance}
    child_path, child_ce = child["precursor_mz_path"], child["collision_energy_path"]
    for parent in parents:
        if (parent["ms_level"] != 2 or not parent["individual_scan"] or parent["metadata_issues"]
                or parent["feature_id"] != child["feature_id"] or parent["raw_file"] != child["raw_file"]
                or parent["scan_number"] is None or child["scan_number"] is None
                or parent["scan_number"] >= child["scan_number"]
                or parent["collision_energy_path"] != child_ce[:-1]):
            continue
        path = parent["precursor_mz_path"]
        if len(path) != len(child_path)-1 or any(abs(a-b) > mz_tolerance for a, b in zip(path, child_path[:-1])):
            continue
        masses = parent.get("ms2_peak_mzs")
        if masses is None:
            masses = [p[0] for p in load_peaks(parent)]
        for index, mass in enumerate(masses):
            error = abs(float(mass)-child["isolated_mz"])
            if error <= mz_tolerance:
                matches.append({"parent_observation_id": parent["observation_id"],
                    "parent_source": parent["source"], "parent_physical_scan_ids": parent["physical_scan_ids"],
                    "parent_connectivity": parent["parent_connectivity"], "peak_index": index,
                    "parent_peak_mz": float(mass), "peak_mass_error_da": error,
                    "parent_path_error_da": sum(abs(a-b) for a, b in zip(path, child_path[:-1])),
                    "linked_isolation_id": "linked_peak_" + _digest([parent["observation_id"], index])[:24]})
    # Source annotation duplicates are not independent links; retain provenance
    # only if the calling importer has resolved annotation conflicts explicitly.
    unique = {}
    for match in matches:
        key = (match["parent_observation_id"], match["peak_index"])
        if key not in unique:
            unique[key] = {**match, "parent_source_aliases": [match["parent_source"]]}
        else:
            unique[key]["parent_source_aliases"].append(match["parent_source"])
    matches = sorted(unique.values(), key=lambda x: (x["parent_observation_id"], x["peak_index"]))
    conflicts = [m["parent_observation_id"] for m in matches
                 if len(parent_annotations[m["parent_observation_id"]]) > 1
                 or len(parent_spectra[m["parent_observation_id"]]) > 1]
    status = "annotation_conflict" if conflicts else ("unique" if len(matches) == 1 else ("ambiguous" if matches else "unresolved"))
    return {"status": status, "parent_conflict_observation_ids": sorted(set(conflicts)),
            "matches": matches, "mz_tolerance_da": mz_tolerance,
            "link_semantics": "export_path_and_peak_compatible; no raw spectrumRef attestation"}


def link_status(result):
    """Expose four summary statuses while preserving the original result."""
    return {"annotation_conflict": "conflicting", "invalid_child_metadata": "unresolved"}.get(
        result["status"], result["status"])


def isolated_fragment_mz(metadata):
    """Read the isolated stage m/z from canonical metadata, never parent m/z."""
    return metadata["isolated_mz"]


def ms3_collision_energy(metadata):
    """Read the current scalar energy parsed by canonical_metadata."""
    if metadata["ms_level"] != 3:
        raise ValueError("An MS3 observation is required")
    return metadata["collision_energy"]


def nce_to_msg_scalar(nce, fragment_mz):
    """MassSpecGym convention, not an attested ID-X physical calibration."""
    return float(nce) * float(fragment_mz) / 500.


def group_ms3_energies(observations):
    """Group by linked peak (or acquisition path), then CE; retain replicates."""
    groups = {}
    for row in observations:
        if row["ms_level"] != 3:
            raise ValueError("Only MS3 observations can be grouped")
        identity = row.get("linked_isolation_id") or row["isolation_id"]
        groups.setdefault(identity, {}).setdefault(row["collision_energy"], []).append(row)
    return groups
