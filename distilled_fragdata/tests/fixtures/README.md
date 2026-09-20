# Frozen prototype outputs

These are small synthetic chemistry/spectrum/metadata fixtures, produced by
calling the original prototype functions before comparison with the extraction.
They are not new MAGMa candidates or a rescored MS3 cohort. JSON files record
source hashes and inputs; `glacier_fixture.json` also records checkpoint, commit
and numerical environment. Keep these files when copying the folder.

| File | Oracle |
| --- | --- |
| `glacier_fixture.json`, `glacier_fixture.npz` | Original `runtime.load_model`, normal `predict_mol`, `direct_input`, and captured normal-route tensors; stereo/aromatic singles and mixed batch |
| `fragment_states.json` | Original `microstates` with explicit caps; retained records, tensors, unsupported/capped results and optional projection |
| `msnlib.json` | Original physical scan/canonical metadata and eight linkage cases, including complete matched peaks |
| `spectra.json` | Original sparse processing, microstate averaging and `ms3_analysis.score_bag` cosine/entropy/support |

Elapsed enumeration time is intentionally absent from equality comparisons.
Tests compare frozen results and additionally live prototype functions when
available. Do not overwrite fixtures with newly extracted outputs to make a
regression disappear; review any intended scientific change explicitly first.
