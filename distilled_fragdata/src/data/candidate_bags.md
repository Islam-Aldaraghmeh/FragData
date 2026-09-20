# Existing candidate bags

Purpose: convert existing MAGMa bag files to small Python containers. This module
uses only the standard library. It does not invoke MAGMa or generate candidates.

## API and inputs/outputs

- `FragmentCandidate(candidate_id, graph, ionic_formula, charge, precursor_mz,
  metadata={})` stores one hypothesis. `graph` contains `atoms` and `bonds`;
  formula may be a string or element-count dictionary. `.stripped` returns an
  independent copy of exactly the five inputs accepted by the fragment adapter.
- `CandidateBag(bag_id, candidates, linked_isolation_id=None, isolated_mz=None,
  metadata={})` stores candidates in their original order. The optional linked
  isolation ID lets a caller reject comparison with an unrelated MS3 spectrum.
- `candidate_from_record(record)` accepts the prototype candidate schema with
  `record["stripped"]`, a `graph` plus formula/charge/mz schema, or flat atoms and
  bonds plus formula/charge/mz. A `candidate_id` is required in every case.
- `candidate_bag_from_record(record)` adapts a bag containing `bag_id` and a
  `candidates` list. Duplicate candidate IDs within a bag raise an error.
- `iter_candidate_bags(path)` streams `.jsonl`/`.ndjson` files with one bag per
  nonempty line. JSON files may contain one bag, a list of bags, or
  `{"bags": [...]}`; those JSON containers are read in memory.
- `load_candidate_bag(path, bag_id=None)` returns exactly one matching bag and
  raises for zero or multiple matches. It does not silently choose the first.

Prototype `stripped` fields are authoritative when present. The legacy display
formula is retained as `candidate.metadata["source_ionic_formula"]`, and other
provenance, including MAGMa origins and graph IDs, remains in metadata. Graph
order, bond order/aromaticity, formula counts, charge and precursor m/z are copied
without canonicalization, chemistry repair or use of any experimental spectrum.

## Example

```python
from src.data.candidate_bags import load_candidate_bag

bag = load_candidate_bag("my_bags.jsonl", bag_id="bag_of_interest")
candidate = bag.candidates[0]
assert set(candidate.stripped) == {
    "atoms", "bonds", "ionic_formula", "charge", "precursor_mz"
}
# Pass candidate to a configured FragmentGlacierAdapter.prepare(...).
```

## Sources, assumptions and hazards

These dataclasses and loaders are new wrappers over the existing schema emitted
by `fragment_glacier_pretrained_adapter/src/build_ms3_universe.py`, especially
`generate_parent`'s candidate construction (`candidate_id`, `graph_id`, `stripped`, display
formula, theoretical m/z, origins) and `build_bags`. The actual existing
`outputs/ms3_exploratory/universe_bags.jsonl` schema was inspected without running
generation. The allowed stripped fields are defined by
`microstates._validate_stripped`; the fragment adapter still performs chemistry
validation. No generation code from `build_ms3_universe.py` was transferred.

`tests/test_msnlib.py` verifies exact stripped-field preservation, metadata and
candidate order, all four file containers, and duplicate-ID rejection. There
was no prototype dataclass loader to compare; equivalence here means retaining
the inputs consumed by the original adapter without changing their values.

Intentional API additions: dataclasses, multiple explicit file containers and
duplicate-match validation. These do not alter candidate chemistry or invent
missing charge/formula hypotheses. The loader does not derive a fragment formula
from a parent annotation and does not assume top-level parent m/z is fragment
m/z. Formula consistency, graph validity and mass tolerance belong to the
adapter. A candidate score is support for a supplied hypothesis, not proof the
bag contains the true fragment.
