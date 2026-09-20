# Fragment regression tests

**Purpose and API.** Four pytest tests verify extracted ionic states, model inputs,
cap/failure statuses, chemical invariants, optional neutral projection and exact
formula masses. Run `python -m pytest tests/test_fragment_adapter.py` from the
distilled folder in the documented dependency environment.

**Inputs and outputs.** `fixtures/fragment_states.json` contains results generated
by the unmodified prototype, its SHA-256 and dependency versions. Cases include
aminoethanol, aromatic nitrogen, intrinsic charge separation, and a molecule with
five retained states at cap 16 but four at cap 4. It also freezes leaf, chemical,
DP and search-node caps and two projection controls. Assertions compare complete
enumeration dictionaries except elapsed time, and actual tensor fingerprints at
CE 20. Available local prototype sources additionally run live differential
comparisons and compare every primitive atom/bond model input.

**Source and assumptions.** References are `microstates.enumerate_microstates`,
`stripped_from_smiles`, `project_neutral_references`, `formula_counts`, `ionic_mass`
and `direct_input.records_from_smiles/build_batch/tensor_fingerprint`. Checkpoint
assets are configured by the shared `glacier` fixture. No MAGMa generation or
experimental MS3 scoring occurs.

**Hazards and intentional changes.** Tensor fingerprints are strict, including
dtype, shape and floating-point bytes; dependency changes can expose drift. This
is deliberate, not permission to replace fixtures. The cap-default change is
tested explicitly. Fixture generation never imports extracted implementations.
Only wall-clock elapsed time is excluded as nondeterministic; all search counts,
failure reasons and retained-state order remain part of the contract.
