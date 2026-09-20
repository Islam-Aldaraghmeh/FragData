# GLACIER regression fixtures

These three tests validate the exact primitive descriptors and every model input,
normal/direct single predictions, and heterogeneous padded batch predictions.
Outputs include every raw `spec` row and every corresponding `frag` mask.
The stereo fixture is `F[C@H](Cl)Br` at CE 25; the aromatic fixture is
`c1cc[nH]c1` at CE 35. Both use `[M+H]+` and `Orbitrap`.

`fixtures/glacier_fixture.json` stores primitive records, tensor fingerprints,
checkpoint/commit identity, source hashes, and Python/library/platform versions.
`fixtures/glacier_fixture.npz` stores the complete raw single and mixed-batch
outputs generated only by the original prototype. During freezing, all direct
inputs were checked against captured normal-route inputs and all normal/direct
outputs were bitwise equal. The NPZ contains no executable pickles.

## Sources and assumptions

Live comparisons import `fragment_glacier_pretrained_adapter/src/direct_input.py`
and `run_equivalence.py::capture_normal`, with the model loaded using the original
`runtime.py::load_model` when available. The single-prediction test independently
loads a second model through `GlacierAPI.model`, exercising the new lazy loader
as well as comparing its outputs to the original model. The test fixtures in `conftest.py` select
external asset paths and the optional original source checkout. No fake model or
approximated spectrum is used. The normal path calls pinned upstream
`JointModel.predict_mol` directly.

Inputs are fixed and do not depend on any experiment output or held-out target
spectrum. Prototype-vs-distilled comparisons always use exact tensor equality.
Frozen outputs are compared bitwise on their recorded environment. On a different
environment, only floating output values receive `rtol=1e-6, atol=1e-7`; fragment
masks, shapes and dtypes still must match exactly. Frozen tensor hashes are
asserted on the recorded environment. Primitive records remain exact.

## Usage and limits

From `distilled_fragdata`, with the documented dependencies and external model
assets configured:

```sh
python -m pytest tests/test_glacier_api.py -q
```

The source checkout is optional for the frozen checks. Real pretrained assets
are required; a skipped model test is not evidence of model equivalence.
Cross-platform floating tolerance is a testing accommodation, not an intentional
change to inference. No benchmark, MS3 scoring run, training, or candidate
generation is performed. Frozen tests cannot guarantee behavior for every
possible molecule, and unsupported chemistry must still be rejected upstream.
