# Pretrained GLACIER interface

`GlacierAPI` provides the existing normal SMILES, direct primitive-record,
prebuilt tensor, and padded batch prediction routes. It returns upstream raw
`spec` and `frag` tensors unchanged. No training or model modification occurs.

## Inputs, outputs, and usage

```python
from src.models.glacier_api import GlacierAPI, records_from_smiles

glacier = GlacierAPI()  # Repository-local vendor/ms-pred and checkpoints/best.ckpt.
normal = glacier.predict_molecule("CCN", 25.0, "[M+H]+", "Orbitrap")
records = records_from_smiles("CCN")
direct = glacier.predict_records(records, 25.0, "[M+H]+", "Orbitrap")
batch = glacier.predict_batch([records], [25.0], ["[M+H]+"], ["Orbitrap"])
```

The default paths are resolved from the repository location, not the current
working directory. Paths can be overridden explicitly in the constructor or
through `GLACIER_UPSTREAM` and `GLACIER_CHECKPOINT`. No sibling prototype
checkout is required.
The constructor verifies the git commit, absence of tracked modifications, and
checkpoint hash, and selects the checkout's `src` directory. It does not import
torch or load weights. Accessing `.model` or predicting loads weights once on CPU.
The public module itself has only standard-library import dependencies.

- `predict_molecule` delegates directly to upstream `JointModel.predict_mol`.
  Single SMILES gives a dictionary of tensors; a list of SMILES with matching
  lists of metadata gives a dictionary of per-molecule tensor lists.
- `predict_records` predicts one record. Its default `[M]+` is for already
  specified positive ionic states. Specify `[M+H]+` explicitly when reproducing
  a protonated neutral-molecule prediction.
- `predict_batch` accepts records and explicit equal-length metadata lists.
  `predict_tensors` accepts the dictionary from `build_batch`; both retain
  the upstream dictionary of `spec` and `frag` lists.
- `records_from_mol`, `records_from_smiles`, `build_batch`, `predict_records`,
  and `tensor_fingerprint` expose the original direct-input functions with their
  original signatures. Create the API first, or install the pinned upstream
  package, before using these helpers. `sha256` and `.checkpoint_identity`
  provide asset identities for caches and provenance.

## Exact extraction and changes

Source: `fragment_glacier_pretrained_adapter/src/runtime.py::load_model,sha256`
and `direct_input.py::records_from_mol,records_from_smiles,build_batch,
predict_records,tensor_fingerprint`, with the complete direct builder in
[`_direct_input.py`](_direct_input.py). Normal prediction is the same pinned
upstream method called by `src/run_equivalence.py::capture_normal`.
`microstates.py::FEATURE_CONFIG/_feature_hash` also live here unchanged, so
chemistry can request a feature identity without handling model tensors or
Graphormer configuration. The fixed CE=0/Orbitrap hash is for state deduplication;
prediction cache identity must instead cover the actual acquisition inputs.

The pinned commit is `67ba0f202305d4bed3134991dd4562fe1edb1763`; the checkpoint
SHA-256 is `5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11`.
The seed `20260919`, four torch threads, CPU loading, `.eval()`, `.freeze()`,
and `weights_only=False` match the prototype. No scientific behavior changed.

Operational changes: local defaults and optional explicit/environment paths
replace hard-coded prototype paths; asset
checks occur at construction and raise descriptive exceptions instead of
assertions; model dependencies/weights load lazily; an already imported foreign
`ms_pred` checkout is rejected; the prototype's global bytecode-writing switch
and report helpers are not copied. No files in the upstream checkout are edited.

## Assumptions and scientific pitfalls

Inputs must satisfy the same pretrained-model feature and instrument vocabulary.
The direct builder performs no ionic-state validation; use the fragment adapter
before passing ionic records. `spec` contains raw rows: coincident masses require
MAX pooling within a prediction, not addition. `frag` aligns with these raw rows.
Do not average raw fragment rows across states. Retain CPU and pinned library
versions when demanding bitwise equality; other architectures may alter floats.
Only load the verified public checkpoint because torch checkpoint loading can
execute serialized code. A different checkpoint is deliberately rejected.

`tests/test_glacier_api.py` compares primitive descriptors, every model input,
normal/direct outputs, fragment masks, and mixed padded batches against the
original implementation when present, and the frozen prototype oracle.
