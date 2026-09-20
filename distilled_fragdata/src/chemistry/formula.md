# Formula utilities

**Purpose and API.** `formula_counts(formula)` parses a formula or count dictionary;
`exact_mass(formula)` returns the neutral atomic mass sum; `ionic_mass(formula,
charge=1)` returns m/z after electron correction; `mass_matches(formula, charge,
precursor_mz)` applies the adapter's `max(0.005 Da, 20 ppm)` consistency check.
Counts are sorted by element. Formula text may have the prototype's terminal
charge notation, but the explicit `charge` argument determines ionic m/z.

```python
from src.chemistry.formula import formula_counts, ionic_mass
counts = formula_counts("C2H8NO+")
mz = ionic_mass(counts, charge=1)
```

**Extraction.** `formula_counts` and `ionic_mass` are copied from
`fragment_glacier_pretrained_adapter/src/microstates.py`, changing only references
to the mass constants. Constants are the exact values of `ms_pred.common` from
commit `67ba0f202305d4bed3134991dd4562fe1edb1763` in the working RDKit 2025.03.6
environment. This module imports neither RDKit nor `ms_pred`. The two small new
convenience functions use the same mass formula and tolerance.

**Assumptions and hazards.** These are elemental, most-common-isotope formulas;
parentheses, isotopic notation, repeated element symbols and unsupported elements
raise `ValueError`. The ionic formula already includes every hydrogen: do not add
a proton again. Electron mass is deliberately the prototype's `0.00054858`, not
a replacement constant with extra precision. The parser does not infer or verify
charge from a textual suffix; e.g. the prototype accepts `N+2` and strips `+2`.
`ionic_mass` preserves the old nonzero-charge behavior, including accepting
noninteger charge values; the fragment adapter separately supports only +1.

**Intentional changes.** No numerical changes; explicit standalone constants and
the two new convenience helpers remove the upstream dependency.
