# Distillation report

Created `distilled_fragdata/` as a standalone source folder. The scientific
algorithms were extracted from the actual working prototype before API cleanup.
Fixed fixtures compare retained states, full model inputs, model outputs, peak
processing, link dictionaries and candidate scores. Source hashes and numerical
provenance are stored with the small frozen fixtures.

The requested default cap changed from 16 to **4 states**. At matching explicit
caps/settings, the regression fixtures retain the prototype's scientific
behavior. This is a tested extraction, not a newly designed fragmentation or
scoring implementation.

## Files created and exact source map

Every Python file listed below has its own adjacent `.md` covering purpose,
functions/classes, inputs/outputs, assumptions, usage, source functions and
scientific hazards. The package has 12 source modules, six test/support Python
files, and five compact frozen data files.

All old source paths in this table are relative to
`fragment_glacier_pretrained_adapter/src/` unless stated otherwise.

| New file | Function and original source |
| --- | --- |
| `src/models/glacier_api.py` | Verified public weights, normal/direct/prebuilt/batch prediction; `runtime.py::sha256/load_model`, pinned upstream `JointModel.predict_mol/predict_inten_frag_batch`; exact `microstates.py::FEATURE_CONFIG/_feature_hash` keeps tensor deduplication inside the model layer |
| `src/models/_direct_input.py` | Exact primitive descriptors and Graphormer input builder; all eight `direct_input.py` functions: `records_from_mol`, `records_from_smiles`, `_onehot`, `_binary`, `graph_tensors`, `build_batch`, `predict_records`, `tensor_fingerprint` |
| `src/chemistry/fragment_adapter.py` | Public candidate preparation/ensemble API; exact `microstates.py::project_neutral_references`; state reparse and inference sequence from `run_retention.py::Predictor.predict` / `run_ms3_exploratory.py::IonicPredictor.predict` |
| `src/chemistry/_state_graph.py` | Original `microstates.py` chemistry constants, `_fresh_graph`, `_graph_record`, `_canonical_graph`, `stripped_from_smiles`, `_validate_stripped`, `_counts_from_mol`, `_validate_state`, `_local_domains`, `_chemistry_rank`; feature hashing delegates to the model layer |
| `src/chemistry/_enumeration.py` | Original `microstates.py::enumerate_microstates`, including minimum-charge family, suffix DP, bounded search, validation, rank, feature deduplication and metadata |
| `src/chemistry/formula.py` | `microstates.py::formula_counts/ionic_mass`; exact pinned `common.ELEMENT_TO_MASS/ELECTRON_MASS`; small neutral-mass and consistency conveniences |
| `src/data/msnlib.py` | Original `ms3_data.py` parsing helpers, physical scan identity, canonical metadata, CE provenance, verified source loading and conservative `link_ms3`; `run_ms3_exploratory.py::model_ce` conversion branch |
| `src/data/candidate_bags.py` | New dataclasses/loaders around the existing bag/candidate `stripped` schema consumed by `IonicPredictor.enumeration` and `evaluate`; no generation code |
| `src/spectra/spectrum.py` | Exact `_array`, `mass_tolerance`, `merge_peaks`, `prepare_spectrum`, `normalize_spectrum`, `mean_normalized_spectrum`, `match_peaks` from `spectral_metrics.py`; `run_retention.py::clean_raw`; new thin `Spectrum` container |
| `src/spectra/metrics.py` | Exact `spectral_metrics.py::score_spectra`, named metric convenience wrappers and a place for future distinct metrics |
| `src/experiments/ms3_candidate_support.py` | Exact `ms3_analysis.py::score_bag/support_summary`; short high-level flow from `run_ms3_exploratory.py::evaluate`; target preparation from `prepare_ms3_exploratory.py::prepare_observation` |
| `src/utils/prediction_cache.py` | New minimal atomic NPZ cache using `run_retention.py::Predictor.predict`'s complete input/checkpoint/runtime identity principle; atomic replacement concept from `ms3_resume_io.py` |
| `tests/conftest.py` | Optional prototype source, external model paths and shared session fixtures; new test wiring |
| `tests/test_glacier_api.py` | Three actual checkpoint tests: descriptors/every tensor, independent extracted loader and normal/direct outputs, mixed batch/prebuilt tensors |
| `tests/test_fragment_adapter.py` | Four tests of frozen/live state selection at identical caps, cap/failure behavior, invariants/default 4, optional projection/formula conventions |
| `tests/test_msnlib.py` | Four tests of full metadata, statuses/linked peaks, verified source loading/CE/grouping, existing candidate loaders |
| `tests/test_spectra_and_support.py` | Five tests of exact prepared peaks, state mean, metrics/matching, frozen candidate scoring/high-level guards, actual multistate ionic ensemble |
| `tests/test_prediction_cache.py` | One cache roundtrip and context-identity test |

Additional files: `README.md` (pipeline/setup/examples), `requirements.md`
(environment/asset instructions), this report, `tests/fixtures/README.md`,
and `.gitignore` (exclude generated Python/test caches).
Frozen data files are `glacier_fixture.json`, `glacier_fixture.npz`,
`fragment_states.json`, `msnlib.json`, and `spectra.json` (about 137 KiB total,
including their explanatory README).

## Preserved scientific behavior

- Same pinned `ms-pred` commit
  `67ba0f202305d4bed3134991dd4562fe1edb1763` and public checkpoint SHA256
  `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11`.
  CPU, seed, thread count, eval/freeze state, native dtypes and forward/assembly
  routes are retained. An independent extracted model load was compared to the
  original loaded model.
- Exact atom order, graph descriptors, weighted random walks, padded tensors,
  formula/mass/H channels and raw `spec`/`frag` outputs. All eight direct builder
  function ASTs match the original.
- Exact graph/formula/charge validation, H/Q domains, chemistry ranking, search
  caps and feature deduplication. Twelve graph/enumeration/projection function
  ASTs match, allowing only the requested default-cap change. Wall-clock elapsed
  time is deliberately excluded from result equality.
- Identical ionic `[M]+` inputs, no added proton, no silent graph repair and no
  target-driven state selection. A real multistate aminoethanol fixture confirms
  the final predicted ensemble against the original sequence.
- Exact max pooling within predictions, filtering order/ties, fixed-anchor
  tolerance grouping, per-state L1 normalization, sum pooling across states,
  and no top-100 cutoff on the ensemble. Experimental MS3 retains all positive
  support after sum merging; cosine/entropy and support arithmetic are unchanged.
- All 16 retained MSnLib parsing/linkage function ASTs match the original.
  Complete linked-peak records and statuses match on eight fixed cases, including
  multiple parents, multiple peaks, conflicts and duplicate aliases.

## Intentional changes

Each change is also documented next to the relevant module.

1. **Default retained-state cap is 4 instead of 16.** This can change the
   ensemble when more than four states are available. Both explicit settings are
   checked against the original; selection is otherwise unchanged.
2. **Model asset paths are explicit/environment-configurable; imports and weight
   loading are lazy.** Descriptive exceptions replace asset assertions and an
   already imported different checkout is rejected. The old global
   `sys.dont_write_bytecode` side effect and report helpers are not transferred.
   Weights and numerical inference behavior are unchanged.
3. **Small convenience APIs were added.** Dataclasses, named metric wrappers,
   neutral exact-mass/mass-consistency helpers and `Spectrum` wrap the preserved
   values/algorithms. Formula masses use the exact original constants rather
   than a newer precision convention. The fragment API rejects nonfinite CE.
4. **Detailed link statuses stay intact.** `link_status` separately maps
   `annotation_conflict` to `conflicting`, and `invalid_child_metadata` to
   `unresolved`; it never selects a link. `group_ms3_energies` retains all
   replicates and singleton groups, omitting the experiment's hash-selected
   representative and requirement for at least two energies.
5. **Experiment orchestration uses simple exceptions.** Incomplete chemistry,
   empty bags/targets, nonunique links or inconsistent stored linked-peak IDs
   fail explicitly. The old stage-status/report framework is absent. A
   represented singleton may be scored; the exploratory ambiguous-bag gate is
   omitted. Metric values for accepted inputs remain unchanged.
6. **Cache storage is deliberately simplified.** New canonical keys include
   candidate/state, formula, charge, CE, instrument, exact tensor fingerprint,
   checkpoint and optional runtime/settings. Atomic raw-spectrum NPZ files
   replace the old layered graph/tensor/prediction/resume machinery. Old cache
   filenames are not compatible. Raw arrays are preserved exactly.

## Validation run

Using the original Python 3.10.20 environment, RDKit 2025.03.6, NumPy 2.2.6,
PyTorch 2.6.0 and verified public checkpoint on macOS arm64:

```sh
PYTHONDONTWRITEBYTECODE=1 fragment_glacier_pretrained_adapter/.venv/bin/python \
  -m pytest distilled_fragdata/tests -q
```

Result: **17 passed**. All five requested scientific-equivalence areas are
covered, including actual pretrained inference. Live comparisons use exact
tensor/array/dictionary equality. Frozen model outputs use exact equality in
the recorded environment; on a different numerical environment only spectra
allow rtol=1e−6 / atol=1e−7, while fragment masks remain exact. This is not a claim
of universal cross-platform bitwise reproducibility.

The complete suite was also run with
`DISTILLED_PROTOTYPE_SRC=/tmp/distilled-no-prototype` while retaining external
model assets: **16 passed, 1 skipped**. The explicitly live-only ionic ensemble
differential skips; all frozen checkpoint/chemistry/data/scoring tests still
run. This verifies the extraction does not require the old source to operate.
Only upstream deprecation/nested-tensor warnings appeared.

All 18 Python files parse and have adjacent Markdown. Runtime import inspection
found no imports from prototype modules. No large experiments, saved MS3 scoring
runs, new MAGMa candidates, training or checkpoint changes were performed.

## Discarded and remaining limits

Discarded: fingerprint MLP, custom small GNN, fragment-native v1/v2 and training;
MAGMa universe generation; 64-bag and 256-parent cohort machinery; donor/null
experiments; gates/bootstrap statistics; resume/stage drivers; manifest systems;
automatic reports; cached experimental outputs and large data/assets.

The exact tensor builder and chemical enumerator could not honestly become a
few-line wrapper without losing the tested independent input route or scientific
selection behavior. Three private helper modules keep these internals out of
the public flow. The direct builder is 253 lines and MSnLib is 366 lines; most
other modules are below 250. A few original scientific functions remain longer
than 30 lines to preserve their control flow and make source comparison easy.

MSnLib parsing retains the prototype's one-JSON-record-per-line format, even for
`.json` exports. Original acquisition provenance labels and paths are preserved,
but their historical evidence PDFs are not bundled. Paths/source hashes still
require the source export to be available for peak loading. Candidate loaders
support the inspected existing schemas; arbitrary MAGMa file layouts may need
a small explicit field adapter.

The finite state family is not an exhaustive thermodynamic protomer ensemble;
its cap and unsupported states remain visible. Neutral projection is diagnostic
only. MS3 NCE conversion remains a preprocessing convention, not physical ID-X
calibration. Linking is export-path/peak compatibility, not raw-parent
attestation. Similarity support is not a probability or ground-truth assignment.

No new environment installation was tested. External pinned upstream is needed
for chemistry feature deduplication; public weights are required for inference;
CPU is the validated target. The package's `src` namespace assumes its own root
is on the import path; if the destination research repository already owns a
different `src` package, relocate the imports explicitly during integration.
