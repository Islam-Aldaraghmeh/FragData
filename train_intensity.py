"""Small CPU training/evaluation script for Experiment_2.ipynb.

Run: .venv/bin/python train_intensity.py
The first run generates MAGMa bags; later runs reuse the local data cache.
"""
import argparse
import copy
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.nn import functional as F

from intensity_model import (IntensityMLP, FEATURE_SIZE, EXTRA_FEATURES,
                             normalize_intensity, pool_candidate_scores,
                             spectrum_metrics)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "results" / "experiment_2"


def forward_batch(model, examples):
    """Batch candidates, max-pool by peak, then restore each spectrum's axis."""
    lengths = [len(example["target"]) for example in examples]
    offsets = np.cumsum([0] + lengths[:-1])
    features = np.concatenate([example["features"] for example in examples])
    indices = np.concatenate([example["peak_indices"] + offset
                              for example, offset in zip(examples, offsets)])
    scores = model(torch.as_tensor(features, dtype=torch.float32))
    peaks = pool_candidate_scores(scores, torch.as_tensor(indices), sum(lengths))
    return peaks.split(lengths)


@torch.no_grad()
def predict_examples(model, examples, batch_size=16):
    model.eval()
    predictions = []
    for start in range(0, len(examples), batch_size):
        predictions.extend(normalize_intensity(p.cpu().numpy()) for p in
                           forward_batch(model, examples[start:start + batch_size]))
    return predictions


def equal_peak_baseline(examples):
    """Equal score per supported PEAK, independent of its candidate count."""
    predictions = []
    for example in examples:
        values = np.zeros(len(example["target"]))
        values[example["peak_indices"]] = 1.
        predictions.append(values)
    return predictions


def fit_model(train, validation, *, seed=20260922, epochs=40, patience=8,
              batch_size=16, learning_rate=1e-3):
    """Cosine-only training; choose the checkpoint using validation spectra."""
    if not train or not validation or epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("Nonempty splits and positive epochs, patience and batch_size are required")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = IntensityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_score, best_epoch, best_state = -1., 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for start in range(0, len(train), batch_size):
            batch = [train[i] for i in order[start:start + batch_size]]
            predictions = forward_batch(model, batch)
            targets = [torch.as_tensor(example["target"], dtype=torch.float32) for example in batch]
            loss = torch.stack([1 - F.cosine_similarity(p, y, dim=0, eps=1e-8)
                                for p, y in zip(predictions, targets)]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.extend([float(loss.detach())] * len(batch))
        val_predictions = predict_examples(model, validation, batch_size)
        score = spectrum_metrics([e["target"] for e in validation], val_predictions)["mean_cosine"]
        history.append({"epoch": epoch, "train_cosine_loss": float(np.mean(losses)),
                        "validation_mean_cosine": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 5 == 0:
            print(f"epoch {epoch:02d}: loss={np.mean(losses):.4f}, validation cosine={score:.4f}", flush=True)
        if epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, history, best_epoch


def load_model(checkpoint_path=DEFAULT_OUTPUT / "model.pt"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["feature_size"] != FEATURE_SIZE:
        raise ValueError("Checkpoint features do not match intensity_model.py")
    model = IntensityMLP()
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def plot_comparison(example, prediction, baseline=None):
    """Predicted sticks above zero, experimental sticks below on identical axes."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    rows = 2 if baseline is not None else 1
    figure = Figure(figsize=(11, 3 * rows), layout="constrained")
    axes = np.atleast_1d(figure.subplots(rows, 1, sharex=True))
    series = [(prediction, "Fingerprint MLP")]
    if baseline is not None:
        series.append((baseline, "Equal supported-peak baseline"))
    for ax, (values, name) in zip(axes, series):
        score = spectrum_metrics([example["target"]], [values])["mean_cosine"]
        ax.vlines(example["mz"], 0, 100 * normalize_intensity(values),
                  color="#2764b7", linewidth=1.3, label=name)
        ax.vlines(example["mz"], 0, -100 * normalize_intensity(example["target"]),
                  color="#d77b32", linewidth=1.3, label="Experimental")
        ax.axhline(0, color="#555555", linewidth=.6)
        ax.set_ylim(-108, 108)
        ax.set_yticks([-100, -50, 0, 50, 100])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):g}"))
        ax.set_ylabel("Relative intensity (%)")
        ax.set_title(f"{name} · cosine {score:.3f} · CE {example['collision_energy']:g}")
        ax.legend(loc="upper right", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("m/z")
    figure.suptitle(f"Held-out test molecule: {example['smiles']}", fontsize=9, wrap=True)
    return figure


def metrics_markdown(metrics):
    lines = ["| Split | Predictor | Mean cosine | Median cosine | MAE | Cosine > 0.5 | Cosine > 0.7 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for split, predictors in metrics.items():
        for name, values in predictors.items():
            numbers = " | ".join(f"{value:.4f}" for value in values.values())
            lines.append(f"| {split} | {name} | {numbers} |")
    return "\n".join(lines)


def run_experiment(data_path=ROOT / "msnlib" / "20241003_enamdisc_pos_msn.json",
                   output_dir=DEFAULT_OUTPUT, *, molecules=300, seed=20260922,
                   epochs=40, patience=8):
    from intensity_data import prepare_dataset
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    examples, manifest = prepare_dataset(Path(data_path), output_dir,
                                         molecules=molecules, seed=seed)
    splits = {name: [e for e in examples if e["split"] == name]
              for name in ("train", "validation", "test")}
    if any(not group for group in splits.values()):
        raise ValueError("Need at least one valid spectrum in each split")
    print("Spectra:", {name: len(group) for name, group in splits.items()}, flush=True)
    training_started = time.perf_counter()
    model, history, best_epoch = fit_model(splits["train"], splits["validation"],
                                           seed=seed, epochs=epochs, patience=patience)
    training_seconds = time.perf_counter() - training_started
    # Test is first evaluated AFTER checkpoint selection, never used to tune.
    metrics, held_out_predictions = {}, {}
    for name in ("validation", "test"):
        group = splits[name]
        predictions = predict_examples(model, group)
        baseline = equal_peak_baseline(group)
        targets = [e["target"] for e in group]
        held_out_predictions[name] = (predictions, baseline)
        metrics[name] = {"fingerprint_mlp": spectrum_metrics(targets, predictions),
                         "equal_intensity": spectrum_metrics(targets, baseline)}
    torch.save({"state_dict": model.state_dict(), "feature_size": FEATURE_SIZE,
                "extra_features": list(EXTRA_FEATURES), "best_epoch": best_epoch,
                "seed": seed}, output_dir / "model.pt")
    # Three deterministic molecule-distinct test examples, selected without scores.
    first_by_molecule = {}
    for index, example in enumerate(splits["test"]):
        first_by_molecule.setdefault(example["molecule"], index)
    selected = list(first_by_molecule.values())[:3]
    predictions, baseline = held_out_predictions["test"]
    plot_records = []
    for number, index in enumerate(selected, 1):
        example = splits["test"][index]
        figure = plot_comparison(example, predictions[index], baseline[index])
        filename = f"test_example_{number}.png"
        figure.savefig(output_dir / filename, dpi=140)
        figure.savefig(output_dir / f"test_example_{number}.svg")
        plot_records.append({"file": filename, "test_index": index,
                             "source_offset": example["source_offset"], "smiles": example["smiles"]})
    results = {"seed": seed, "requested_molecules": molecules,
               "spectrum_counts": {k: len(v) for k, v in splits.items()},
               "molecule_counts": {k: len({e['molecule'] for e in v}) for k, v in splits.items()},
               "best_epoch": best_epoch, "epochs_run": len(history),
               "training_seconds": training_seconds,
               "total_seconds": time.perf_counter() - started,
               "metrics": metrics, "history": history, "plots": plot_records,
               "versions": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": np.__version__}}
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    delta = metrics["test"]["fingerprint_mlp"]["mean_cosine"] - metrics["test"]["equal_intensity"]["mean_cosine"]
    report = f"""# MAGMa intensity-only proof of concept

Seed {seed}; {results['molecule_counts']} molecules and {results['spectrum_counts']} spectra.
Source: `{Path(data_path).name}`. The manifest records selection, offsets, exclusions and generation failures.
Preparation recorded {len(manifest['errors'])} errors and retained {manifest['counts']['unsupported_spectra']} completely unsupported spectra.
Canonical non-stereochemical precursor SMILES define disjoint 70/15/15 molecule splits.
This bounded sample uses ordinary single-scan MS2, scalar collision energy, singly positive
supported adducts and connected precursors with at most 35 heavy atoms. Up to three energies per molecule.

The model concatenates radius-2 512-bit Morgan fingerprints for precursor and fragment
with eight explicit energy/structure descriptors. Its 128/64-unit MLP produces positive
candidate scores using Softplus. Exact heavy graph + formula + charge duplicates are removed
within a peak. **Peak intensity is the maximum candidate score**, keeping alternative
identities latent. Training uses only spectrum cosine loss (Adam, lr 0.001, batch 16).
No experimental intensity, quality score or peak rank is an input. MAGMa peak positions
come from observed masses; no Glacier, state enumeration or MS3 enters this experiment.

Fragment fingerprints use the supplied heavy graph with implicit H disabled and non-strict
property caching, without ionic sanitization or chemistry repair. Total H and charge are
separate descriptors. This can represent imperfect MAGMa graphs without claiming a valid ionic state.
Identical precursor/charge/mass records reuse MAGMa enumeration on their union of observed
mass locations; each spectrum retains its own original peak axis and separate target intensities.

## Held-out results

{metrics_markdown(metrics)}

Each spectrum has equal weight. Cosine uses every observed peak, assigning zero to
unsupported locations; completely unsupported spectra score zero. MAE uses independently
base-peak-normalized [0,1] vectors and is averaged per spectrum, then across spectra.
The baseline assigns one to every supported peak regardless of candidate count.

Best validation checkpoint: epoch {best_epoch} of {len(history)}; training {training_seconds:.1f}s on CPU (2 threads).
Later checkpoints did not exceed that validation score; the saved model is the selected checkpoint, not the last epoch.
Test mean-cosine difference from the equal-intensity baseline: {delta:+.4f}.
Test spectra were evaluated only after selection. This is a small, single-seed conditional
intensity experiment, not a measure of peak discovery or fragment identity accuracy.
No hyperparameter search or large experiment was run.

## Examples and reproduction

The first three distinct test molecules were chosen without examining scores.
Predictions are above zero and experimental peaks below; both sides use the same m/z axis.

""" + "\n\n".join(f"![Held-out example {i+1}]({p['file']})" for i, p in enumerate(plot_records)) + """

Open `../../Experiment_2.ipynb`, or run `.venv/bin/python train_intensity.py` from the repo root.
`dataset.pkl` is a generated local cache, `manifest.json` audits the sample, `model.pt`
contains the selected weights, and `metrics.json` contains metrics and training history.
Inference: `intensity_model.predict_spectrum(model, smiles, collision_energy, bags)`.
"""
    (output_dir / "SUMMARY.md").write_text(report)
    print(metrics_markdown(metrics), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "msnlib" / "20241003_enamdisc_pos_msn.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--molecules", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()
    run_experiment(args.data, args.output, molecules=args.molecules, seed=args.seed,
                   epochs=args.epochs, patience=args.patience)
