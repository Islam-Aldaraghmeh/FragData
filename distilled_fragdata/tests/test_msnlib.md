# MSnLib and candidate loader regressions

Purpose: four focused tests compare the extracted data behavior with frozen
prototype results and cover the new thin data containers. Inputs are the small
fixed `fixtures/msnlib.json` source records, generated original metadata/link
outputs, and an inline pre-existing-candidate schema example. No real data
cohort, candidate generation or scoring experiment is run.

Run from `distilled_fragdata/`: `python -m pytest tests/test_msnlib.py`.
The tests use the `prototype_src` fixture from `conftest.py`: when the original
prototype exists, complete outputs are also checked against its live
`ms3_data` functions; frozen assertions always run when this folder is copied.

- Canonical metadata and physical identities compare exactly with
  `ms3_data.canonical_metadata` and `physical_scan_identity`, including new-export
  USIs, scan ranges, contradictory scan annotations and ambiguous energy paths.
- Link status and every matched peak/provenance field compare exactly with
  `ms3_data.link_ms3`; unique, ambiguous, unresolved, conflict, invalid and alias
  cases are included. Friendly status mapping retains the original result.
- Source streaming and peak loading compare with `ms3_data.iter_metadata`,
  `load_source_row` and `load_peaks`; changed source bytes must be rejected.
  Candidate files exercise the schema constructed by `build_ms3_universe.py`
  without importing or running MAGMa generation.
- CE conversion checks the isolated fragment m/z, following
  `run_ms3_exploratory.model_ce`. Grouping checks the intentionally new retention
  of all energy replicates and singleton groups instead of the prototype
  experiment's hash representative selection.

Hazards: exact equality on these fixtures protects the listed branches; it does
not attest every possible malformed export or establish acquisition linkage
beyond the prototype's exported-path semantics. The synthetic records do not
assert physical ID-X CE calibration. No scientific parsing/link behavior changes
are expected; new convenience APIs and replicate-preserving grouping are
documented in `src/data/msnlib.md` and `src/data/candidate_bags.md`.
