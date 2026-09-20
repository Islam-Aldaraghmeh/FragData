# Candidate support

Purpose: readable orchestration of an existing candidate bag, fragment adapter,
linked experimental MS3 and spectrum metrics. `score_candidate_file` loads one
bag then calls `candidate_support`; the latter accepts a `CandidateBag`, an
adapter exposing `.predict`, raw MS3 peaks (or Spectrum), a unique linkage,
an already chosen GLACIER collision-energy scalar and instrument. It returns
ordered candidate IDs, per-candidate metric dictionaries and support summaries.
`score_bag` scores already prepared predicted/target arrays directly.

```python
result = candidate_support(bag, adapter, observed_ms3,
                           linkage=link, collision_energy=model_ce)
cosines = [row["cosine"] for row in result["candidate_scores"]]
```

Sources: exact `ms3_analysis.py::score_bag` and `support_summary`; the orchestration
is extracted from `run_ms3_exploratory.py::IonicPredictor.predict/evaluate` and
the no-cutoff target preparation in `prepare_ms3_exploratory.py::prepare_observation`.
All old files live in `fragment_glacier_pretrained_adapter/src/`.

Preserved behavior: input candidate order, equal-state spectrum averaging in the
adapter, raw-intensity cosine/entropy, untruncated MS3 target after sum merging
and L1 normalization, and support weights linearly normalized from scores.
All-zero scores give uniform support; ties use the original 1e−12 rule.
These weights are not posterior probabilities.

Intentional interface changes: a nonunique link or mismatched stored
`linked_isolation_id` raises rather than proceeding. Unsupported candidates
raise through the adapter instead of returning an experiment-stage
`incomplete_bag` report. A represented singleton can be scored; the old
exploratory ambiguous-bag gate is omitted. No candidate is silently discarded.
Empty bags and experimental spectra remain ineligible, expressed as exceptions
instead of experiment preparation-status records.
No null donor selection, cohort analysis, report generation or rerunning of
saved experiment results is included.

Assumptions/hazards: callers must supply the MS3 corresponding to the link and
bag; a bag without `linked_isolation_id` cannot be cross-checked automatically.
CE is explicit; this function never assumes NCE units or converts parent m/z.
MS3 targets are fully prepared unless `prepared_target=True`; do not set that
flag on raw duplicate peaks. Small frozen candidate scores are tested against
the original, with separate actual-checkpoint ensemble equivalence.
