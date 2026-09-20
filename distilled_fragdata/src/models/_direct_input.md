# Exact primitive-record tensor builder

This private module is `fragment_glacier_pretrained_adapter/src/direct_input.py`
copied verbatim except for removing its `import runtime` line. The public
[`glacier_api.py`](glacier_api.py) selects the verified upstream checkout before
this module is imported. No function body, arithmetic, ordering, dtype, tensor
shape, tie-breaking, mass convention, or feature vocabulary was rewritten.
Its roughly 260-line size is intentional: splitting the audited arithmetic would
make this scientific equivalence less easy to check.

## Functions and data

- `records_from_mol(mol)` and `records_from_smiles(smiles)` return plain
  `{"atoms": [...], "bonds": [...]}` records, preserving atom order, graph,
  formal charge, and hydrogen counts. They use RDKit descriptors and pinned
  upstream constants, but no upstream graph builder or graph repair.
- `graph_tensors(records, pe_embed_k=..., embed_elem_group=...,
  multi_hop_max_dist=...)` builds exact graph features, weighted random-walk
  positional encodings, shortest paths, and adjacency.
- `build_batch(records_list, collision_energies, adducts, instruments, model)`
  returns padded Graphormer inputs and masses/formulas/metadata consumed by the
  pinned model. Metadata lengths, ordering, and vocabularies must match records.
- `predict_records(records, collision_energy, adduct, instrument, model)` returns
  one unmodified `spec`/`frag` prediction.
- `tensor_fingerprint(batch)` hashes sorted tensor names, dtypes, shapes and
  contents, and explicit `None` values. `_onehot` and `_binary` retain the
  prototype's feature encodings.

Example through the public interface:

```python
from src.models.glacier_api import GlacierAPI, records_from_smiles, build_batch

api = GlacierAPI("/path/to/ms-pred", "/path/to/best.ckpt")
records = records_from_smiles("C[N+](C)(C)C")
batch = build_batch([records], [20.0], ["[M]+"], ["Orbitrap"], api.model)
prediction = api.predict_tensors(batch)
```

## Assumptions and pitfalls

The heavy graph must be valid and nonempty. Explicit hydrogen graph nodes are
rejected; atom-local hydrogens are retained. Ionic graph/formula/charge/mass
validation belongs in the fragment adapter. Ring-edge feature ordering,
minimum-index predecessor ties, float32 positional encoding, float64 atomic
masses, padding values, and adduct shifts are scientifically consequential.
Gasteiger failures and nonfinite charges follow the original zero fallback.
Do not replace the mass hydrogen count with the feature hydrogen count: they
are deliberately extracted from different molecule representations.

There are no intentional scientific changes. Lazy importing is handled outside
this file. Regression tests compare this builder with the prototype builder
and tensors captured at the official normal prediction boundary.
