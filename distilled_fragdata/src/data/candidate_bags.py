"""Small containers and loaders for existing candidates; no MAGMa generation."""
from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class FragmentCandidate:
    candidate_id: str
    graph: dict
    ionic_formula: str | dict
    charge: int
    precursor_mz: float
    metadata: dict = field(default_factory=dict)
    # Runtime observations shared by candidates in one MS2 peak bag. These are
    # reference spectra, not additional model inputs or confirmed assignments.
    ms3_spectra: list = field(default_factory=list, repr=False, compare=False)

    @property
    def stripped(self):
        """The exact five allowed adapter inputs, with original graph ordering."""
        return deepcopy({**self.graph, "ionic_formula": self.ionic_formula,
                         "charge": self.charge, "precursor_mz": self.precursor_mz})


@dataclass
class CandidateBag:
    bag_id: str
    candidates: list[FragmentCandidate]
    linked_isolation_id: str | None = None
    isolated_mz: float | None = None
    metadata: dict = field(default_factory=dict)


def candidate_from_record(record):
    """Adapt prototype `stripped`, graph+fields, or flat atoms/bonds records."""
    if "stripped" in record:
        stripped = record["stripped"]
    else:
        graph = record.get("graph", {k: record[k] for k in ("atoms", "bonds") if k in record})
        stripped = {**graph, **{k: record[k] for k in ("ionic_formula", "charge", "precursor_mz")}}
    if set(stripped) != {"atoms", "bonds", "ionic_formula", "charge", "precursor_mz"}:
        raise ValueError("Expected exactly atoms, bonds, ionic_formula, charge, precursor_mz")
    known = {"candidate_id", "stripped", "graph", "atoms", "bonds", "ionic_formula",
             "charge", "precursor_mz", "metadata"}
    metadata = {**record.get("metadata", {}), **{k: v for k, v in record.items() if k not in known}}
    if "stripped" in record and "ionic_formula" in record:
        metadata["source_ionic_formula"] = record["ionic_formula"]
    return FragmentCandidate(record["candidate_id"],
        deepcopy({k: stripped[k] for k in ("atoms", "bonds")}),
        deepcopy(stripped["ionic_formula"]), stripped["charge"], stripped["precursor_mz"],
        deepcopy(metadata))


def candidate_bag_from_record(record):
    """Keep candidates in source order and preserve bag provenance as metadata."""
    candidates = [candidate_from_record(candidate) for candidate in record["candidates"]]
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("Duplicate candidate_id within one bag")
    known = {"bag_id", "candidates", "linked_isolation_id", "isolated_mz", "metadata"}
    metadata = {**record.get("metadata", {}), **{k: v for k, v in record.items() if k not in known}}
    return CandidateBag(record["bag_id"], candidates, record.get("linked_isolation_id"),
                        record.get("isolated_mz"), deepcopy(metadata))


def iter_candidate_bags(path):
    """Read one bag per JSONL line, or JSON containing a bag/list/`bags` list."""
    path = Path(path)
    with path.open() as handle:
        if path.suffix.lower() in {".jsonl", ".ndjson"}:
            for line in handle:
                if line.strip():
                    yield candidate_bag_from_record(json.loads(line))
        else:
            value = json.load(handle)
            records = value if isinstance(value, list) else value.get("bags", [value])
            for record in records:
                yield candidate_bag_from_record(record)


def load_candidate_bag(path, *, bag_id=None):
    """Read exactly one matching bag; reject missing or duplicate matches."""
    selected = None
    for bag in iter_candidate_bags(path):
        if bag_id is not None and bag.bag_id != bag_id:
            continue
        if selected is not None:
            raise ValueError("Multiple matching bags; provide a unique bag_id")
        selected = bag
    if selected is None:
        raise ValueError("No matching candidate bag")
    return selected
