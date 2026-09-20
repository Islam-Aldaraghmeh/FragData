# Private fixed-graph chemistry

**Purpose.** This module holds the prototype's graph stripping, validation, local
valence domains and chemistry ranking. It delegates feature deduplication to the
model layer and keeps the
public adapter and the search engine separately readable.

**Functions and I/O.** `stripped_from_smiles(smiles, ionic_formula, charge,
precursor_mz)` returns exactly `atoms`, `bonds`, `ionic_formula`, `charge` and
`precursor_mz`. Atoms retain only element and aromaticity; bonds retain endpoints,
order and aromaticity. `_validate_stripped`, `_validate_state`, `_local_domains`
and `_chemistry_rank` are private enumeration helpers. Imported `_feature_hash` hashes all
actual checkpoint tensors at fixed CE zero, `[M]+` or `[M+H]+`, and Orbitrap.

```python
from src.chemistry.fragment_adapter import stripped_from_smiles
from src.chemistry.formula import ionic_mass
record = stripped_from_smiles("NCCO", "C2H8NO", 1, ionic_mass("C2H8NO"))
```

**Source.** Graph functions, chemistry constants and charge/valence templates
are copied from `fragment_glacier_pretrained_adapter/src/microstates.py`.
Its unchanged `_feature_hash` and `FEATURE_CONFIG` now live inside
`src.models.glacier_api`; formula helpers come from `src.chemistry.formula`.

**Assumptions and hazards.** A source SMILES supplies only the heavy graph. Its
hydrogens, local charge, atom ordering and stereochemistry are intentionally
discarded before canonical graph ordering, exactly as in the prototype. Isotopes,
disconnected graphs and explicit hydrogen nodes remaining after `RemoveHs` are
rejected. Pre-stripped graph records retain their supplied atom order. Fresh atom
construction prevents hidden local state from influencing canonicalization.
The chemistry rank is a weak declared site preference, not an energy calculation.
Tensor hashes require the pinned GLACIER featurizer and matching dependencies.

**Intentional changes.** Module boundaries and imports only. No graph repair,
altered ranking, additional chemistry domains or target-aware selection.
