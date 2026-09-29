# %% [markdown]
# # Experiment 2 — learn intensities on MAGMa candidate peaks
# 
# This small proof of concept learns **intensity only** from ordinary MS2 spectra with known precursors. MAGMa supplies candidate fragments at observed peak masses. A fingerprint MLP scores each candidate; a peak receives the **maximum** score among its alternatives. Candidate identities remain latent.
# 
# No experimental intensity, MS3 observation, Glacier prediction, or ionic-state enumeration is a model input. Identical heavy graph + ionic formula + charge candidates are deduplicated within each peak. Empty bags remain zero in the reconstructed spectrum.
# 
# Select this repo's `.venv/bin/python` notebook kernel. Shared dataset, candidate and gallery code now lives in `magma_workflow/`; Experiment 1 imports it too.

# %%
from pathlib import Path
import json
import sys

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "intensity_model.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from IPython.display import display, Markdown, Image
from magma_workflow.dataset import Spectrum
from magma_workflow.candidates import get_candidate_bags
from intensity_model import predict_spectrum
from train_intensity import run_experiment, load_model, metrics_markdown, plot_comparison

torch.set_num_threads(2)
DATA = ROOT / "msnlib" / "20241003_enamdisc_pos_msn.json"
OUTPUT = ROOT / "results" / "experiment_2"

# %% [markdown]
# ## Small, auditable experiment
# 
# - Sample 300 precursor molecules, with up to three distinct collision energies each, using seed `20260922`.
# - Use `SINGLE_BEST_SCAN` MS2 with scalar energy, charge +1, supported adducts, and connected non-isotopic precursors of at most 35 heavy atoms.
# - Split canonical non-stereochemical precursor structures 70% / 15% / 15%. All energies and stereoisomers of one molecule stay together.
# - Concatenate 512-bit radius-2 Morgan fingerprints for precursor and fragment, plus energy, fragment mass, charge, total formula H, size, and MAGMa cut/depth descriptors.
# - Train a 128 → 64 → 1 MLP with Softplus outputs, max aggregation, and spectrum cosine loss. Select the checkpoint by validation cosine.
# 
# Fragment fingerprints describe the supplied heavy graph without sanitization or local ion-state assignment. Global formula H and charge are separate descriptors. We keep mixed-formula candidate bags here: they remain alternative explanations.
# 
# Implementation: [`intensity_model.py`](intensity_model.py), [`intensity_data.py`](intensity_data.py), [`train_intensity.py`](train_intensity.py).

# %%
# Saved results are loaded by default, so Run All does not retrain or load the full dataset.
# Set RETRAIN=True for the same small experiment. MAGMa features are cached locally.
RETRAIN = False
if RETRAIN or not (OUTPUT / "metrics.json").is_file():
    results = run_experiment(DATA, OUTPUT, molecules=300, seed=20260922,
                             epochs=40, patience=8)
else:
    results = json.loads((OUTPUT / "metrics.json").read_text())

print("Molecules:", results["molecule_counts"])
print("Spectra:", results["spectrum_counts"])
print(f"Selected epoch {results['best_epoch']} of {results['epochs_run']}")

# %% [markdown]
# ## Held-out evaluation
# 
# The following are the only performance metrics. Each spectrum has equal weight. Cosine uses **all observed experimental peak locations**, including zero predictions at unsupported peaks. Completely unsupported spectra have cosine zero. MAE independently base-peak normalizes each spectrum to `[0, 1]`, then averages per spectrum.
# 
# The baseline assigns equal intensity to every MAGMa-supported **peak**, regardless of candidate count. Test data are evaluated after validation checkpoint selection. These are conditional intensity scores at supplied peak locations, not an end-to-end peak-discovery test.

# %%
display(Markdown(metrics_markdown(results["metrics"])))

# %%
manifest = json.loads((OUTPUT / "manifest.json").read_text())
groups = {name: set(values) for name, values in manifest["splits"].items()}
assert not (groups["train"] & groups["validation"]
            or groups["train"] & groups["test"]
            or groups["validation"] & groups["test"])
print("Molecule groups are disjoint.")
print("Preparation audit:", manifest["counts"])
print("Recorded failures:", len(manifest["errors"]))

# %% [markdown]
# ## Experimental versus predicted spectra
# 
# The first three distinct test molecules are shown, selected without examining their scores. Predictions appear above zero and experimental intensities below on the same m/z axis. Each image also shows the equal-intensity baseline for that same spectrum.

# %%
for example in results["plots"]:
    display(Image(filename=str(OUTPUT / example["file"])))

# %% [markdown]
# ## Inference on one held-out example
# 
# `predict_spectrum(model, precursor_smiles, collision_energy, candidate_bags)` is the inference function. It returns an `N × 2` array of m/z and base-peak-normalized intensity in the supplied bag order. Optional MS3 entries are ignored. You can pass any already prepared MAGMa bags.
# 
# The example below reads a single saved test record and regenerates its candidates. It never loads the full library into the notebook. Targets are used only in the final display.

# %%
model = load_model(OUTPUT / "model.pt")
with DATA.open("rb") as stream:
    stream.seek(results["plots"][0]["source_offset"])
    spectrum = Spectrum.from_record(json.loads(stream.readline()))

bags = get_candidate_bags(spectrum, spectra=[])
predicted = predict_spectrum(model, spectrum.smiles,
                             spectrum.metadata["collision_energy"], bags)
assert np.array_equal(predicted[:, 0], spectrum.mz)
print(f"Predicted {np.count_nonzero(predicted[:, 1])} supported peaks of {len(predicted)} observed peaks")

# %%
example = {"smiles": spectrum.smiles,
           "collision_energy": spectrum.metadata["collision_energy"],
           "mz": spectrum.mz, "target": spectrum.intensity}
figure = plot_comparison(example, predicted[:, 1])
# Export through the same helper without accumulating live pyplot figures.
from io import BytesIO
preview = BytesIO()
figure.savefig(preview, format="png", dpi=130)
display(Image(data=preview.getvalue()))

# %% [markdown]
# ## Results and scope
# 
# Read [`results/experiment_2/SUMMARY.md`](results/experiment_2/SUMMARY.md) for the short result summary. The directory also contains the selected weights, metrics, plots, sample manifest and a local feature cache.
# 
# This is one bounded sample and one seed. A good spectrum score does not identify which latent fragment was correct. Observed peak locations are supplied to MAGMa, and unsupported peaks cannot be predicted. This experiment does not test unseen instruments, peak discovery, or performance without experimental mass locations.


