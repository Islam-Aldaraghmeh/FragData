# Test assets and optional prototype

Purpose: put this copied folder on the import path and locate local pretrained
assets. `prototype_src` returns an optional old source directory; `glacier_paths`,
`model` and `glacier` provide shared verified actual model assets. Environment
variables are `DISTILLED_PROTOTYPE_SRC`, `GLACIER_UPSTREAM`, and
`GLACIER_CHECKPOINT`; defaults locate this workspace's sibling prototype.

Usage: `python -m pytest tests -q` from this folder. Inputs are local paths;
outputs are session fixtures. Tests do not download assets. Missing model assets
explicitly skip dependent tests; missing old sources still permits frozen
regression. Fixtures outside model tests need only their documented packages.

New test wiring, derived from the old `tests/test_direct_input.py` model fixture
and `runtime.load_model`. Live original loading is used when available; model
tests also independently load the extracted API. Sharing a model saves test
time, not output validation. A skipped checkpoint test is not evidence of model
equivalence. Frozen fixtures remain in `tests/fixtures/` after copying.
