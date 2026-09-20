# Private bounded ionic-state search

**Purpose and API.** `enumerate_microstates(stripped, max_states=4, **caps)` returns
the prototype dictionary: `status`, `reason`, retained `states`, and `metadata`.
Its input is the exact stripped five-field record described in
[`fragment_adapter.md`](fragment_adapter.md). States carry ionic SMILES, complete
ionic counts, charge, m/z, `[M]+`, chemistry rank and native tensor fingerprint.
Metadata exposes search counts, caps, rejections and truncation.

```python
from src.chemistry.fragment_adapter import enumerate_microstates
result = enumerate_microstates(record, max_states=16, max_leaves=20000)
print(result["status"], result["metadata"]["stop_reason"])
```

**Source.** The complete `enumerate_microstates` body comes from
`fragment_glacier_pretrained_adapter/src/microstates.py`. Its nested suffix-DP,
depth-first search and final ranking remain intact to preserve visit order, cap
boundaries and deterministic metadata. The longer function is retained
deliberately: splitting its shared mutable search state would obscure the
equivalence review.

**Assumptions and hazards.** Supported ions are connected, charge +1 and closed
shell, with no graph repair. Local charge is -1/0/+1 and absolute-charge families
are 1/3/5/7. Search stops at the first family yielding chemical states. Default
caps remain 100 graph atoms, 20,000 leaves, 512 unique chemical states, 250,000 DP
states and 250,000 search nodes. Sanitization must preserve graph, aromaticity,
formula, total charge and searched charge family. Feature deduplication and
retention occur only after chemistry ranking. A successful but truncated result
is not an exhaustive ensemble. Targets and collision energy do not affect search.

**Intentional changes.** Only the requested default `max_states=4` replaces 16.
Set it to 16 to reproduce the original default. All same-setting results,
including cap metadata, are regression-checked; elapsed wall time is excluded.
