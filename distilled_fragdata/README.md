# Distilled fragment data tools

Reusable code extracted from the working pretrained GLACIER prototype. The
chemistry, tensors, spectrum processing, links and scores retain the old
algorithms; small fixed fixtures compare both implementations. No model training
or MAGMa candidate generation is included.

```text
MSnLib → physical scan metadata → MS2/MS3 linkage → experimental MS3
                                                        ↓
existing MAGMa bags → FragmentCandidate                  metrics
                           ↓                              ↑
                 FragmentGlacierAdapter                  │
                           ↓                              │
                  pretrained GLACIER → candidate spectrum
                                                        ↓
                                                 candidate support
```

The default is **max_states=4**, as requested. The prototype used 16. Use the
same explicit cap and settings when comparing results; defaults can otherwise
produce a different ensemble. Ionic states use `[M]+` with the full ionic
formula. The neutral-reference projection is an optional diagnostic only.

## Setup

Keep this folder intact and put it on `PYTHONPATH` (or run from this folder).
`src` is a Python namespace package; there is no installer or configuration
framework. It has no runtime imports from the old prototype. Each Python file
has a neighboring Markdown explanation with its exact source and assumptions.

This repository contains the working environment and model assets. From the
repository root, run:

```sh
cd distilled_fragdata
export PYTHONPATH="$PWD"
../.venv/bin/python -m pytest tests -q
```

`GlacierAPI()` resolves `vendor/ms-pred/` and `checkpoints/best.ckpt` inside the
repository root by default, independently of the working directory. Explicit
paths and `GLACIER_UPSTREAM` / `GLACIER_CHECKPOINT` remain optional overrides.
The wrapper checks the commit, clean tracked source and checkpoint SHA256 before
use. The environment and large assets are local but Git-ignored; a fresh Git
clone needs them restored. See [requirements.md](requirements.md) for versions,
asset acquisition and Apple Silicon notes, and the [root README](../README.md)
for notebook setup. No sibling prototype checkout is required.

## Minimal fragment prediction

```python
from src.models.glacier_api import GlacierAPI
from src.chemistry.formula import ionic_mass
from src.chemistry.fragment_adapter import FragmentGlacierAdapter, stripped_from_smiles
from src.data.candidate_bags import FragmentCandidate

glacier = GlacierAPI()  # Uses the repository-local pinned source and checkpoint.
stripped = stripped_from_smiles("NCCO", "C2H8NO", 1, ionic_mass("C2H8NO"))
candidate = FragmentCandidate(
    "example", {k: stripped[k] for k in ("atoms", "bonds")},
    stripped["ionic_formula"], stripped["charge"], stripped["precursor_mz"],
)
adapter = FragmentGlacierAdapter(glacier, max_states=4)
prepared = adapter.prepare(candidate, collision_energy=20.0)
print(prepared["status"], prepared["metadata"])  # inspect unsupported/truncated cases
predicted = adapter.predict(candidate, collision_energy=20.0)
print(predicted.as_array())
```

`stripped_from_smiles` only supplies the heavy graph; it does not copy localized
hydrogens or charges into state selection. If your candidate already has the
heavy graph, pass it directly. Formula, total charge and precursor m/z must be
supplied from your candidate hypothesis. MSnLib parent annotations are not
isolated-fragment annotations. Initially, ions must be connected, closed-shell
and singly positive; unsupported/capped search is explicit.

## Existing bags and linked MS3

The loader accepts JSON containing one bag, a list of bags, or `{"bags": [...]}`;
JSONL contains one bag per line. Each bag has `bag_id` and `candidates`; each
candidate has `candidate_id` and either the prototype `stripped` dictionary or
`graph`, `ionic_formula`, `charge`, `precursor_mz`. Graphs contain `atoms` and
`bonds` as shown in [fragment_adapter.md](src/chemistry/fragment_adapter.md).
Other fields remain metadata. Nothing regenerates candidates.

```python
from src.data.candidate_bags import load_candidate_bag
from src.data.msnlib import iter_metadata, link_ms3, link_status, load_peaks, nce_to_msg_scalar
from src.experiments.ms3_candidate_support import candidate_support

# Small example export; for large files, retain only the relevant feature's rows.
rows = list(iter_metadata("your_export.json", include_ms2_peak_mzs=True))
parents = [r for r in rows if r["ms_level"] == 2]
child = next(r for r in rows if r["ms_level"] == 3)
link = link_ms3(child, parents)
if link_status(link) != "unique":
    raise ValueError(link)  # preserve ambiguity/conflicts for review

bag = load_candidate_bag("your_existing_bags.jsonl", bag_id="your-bag-id")
# Only if acquisition metadata establishes NCE: use this MS3 fragment's m/z.
model_ce = nce_to_msg_scalar(child["collision_energy"], child["isolated_mz"])
result = candidate_support(bag, adapter, load_peaks(child),
                           linkage=link, collision_energy=model_ce)
print(result["candidate_ids"], result["candidate_scores"])
```

The CE conversion `NCE × isolated_fragment_mz / 500` is the MassSpecGym
preprocessing convention, **not proven physical ID-X calibration**. The linker
preserves export-path/peak compatibility; it cannot attest a raw `spectrumRef`
relationship. It never chooses among ambiguous compatible peaks.

Single predictions max-pool coincident masses before filtering. State ensembles
sum separately normalized spectra and retain full support. Similarity support
weights are descriptive, not calibrated probabilities. Read
[spectrum.md](src/spectra/spectrum.md) and [metrics.md](src/spectra/metrics.md)
before changing these rules. The optional [disk cache](src/utils/prediction_cache.md)
has explicit input identities and no hidden resume behavior.

## Verification

The 17 focused tests cover actual old/new pretrained inference, every input
tensor, retained states at equal caps, invariants, frozen peaks, cosine/entropy,
conservative linkage, fragment-m/z CE conversion and cache identities. Frozen
reference outputs travel with the code. When old source is present, the same
tests also compare it live when explicitly configured. Without local model assets, model-dependent tests
explicitly skip; that is not a full equivalence verification.

See [DISTILLATION_REPORT.md](DISTILLATION_REPORT.md) for the source-to-module map,
intentional changes, test results and extraction limits.
