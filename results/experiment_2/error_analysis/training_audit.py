"""Bounded training audit; does not change baseline data, model, or metrics.

Run from the repository root with .venv/bin/python <this file>. Only the
original checkpoint is evaluated on test; diagnostic controls use train and
validation exclusively. Cached data contains no model/optimizer state.
"""
import copy
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intensity_model import IntensityMLP, FP_SIZE, spectrum_metrics
from train_intensity import forward_batch, predict_examples

OUTPUT = Path(__file__).resolve().parent
SEED = 20260922


def metrics(model, examples):
    return spectrum_metrics([e["target"] for e in examples], predict_examples(model, examples))


def masked_copy(examples, mask):
    result = []
    for example in examples:
        result.append(dict(example, features=example["features"].copy()))
        if mask == "no_precursor_fingerprint":
            result[-1]["features"][:, :FP_SIZE] = 0
        elif mask == "descriptors_only":
            result[-1]["features"][:, :2 * FP_SIZE] = 0
    return result


def fit_audit(train, validation, *, mask=None, learning_rate=1e-3, weight_decay=0.):
    """Match baseline update order, adding deterministic post-epoch scoring."""
    if mask:
        train, validation = masked_copy(train, mask), masked_copy(validation, mask)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = IntensityMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_score, best_epoch, best_state, history = -1., 0, None, []
    for epoch in range(1, 41):
        model.train()
        losses = []
        order = rng.permutation(len(train))
        for start in range(0, len(train), 16):
            batch = [train[i] for i in order[start:start + 16]]
            predictions = forward_batch(model, batch)
            targets = [torch.as_tensor(e["target"], dtype=torch.float32) for e in batch]
            loss = torch.stack([1 - F.cosine_similarity(p, y, dim=0, eps=1e-8)
                                for p, y in zip(predictions, targets)]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.extend([float(loss.detach())] * len(batch))
        train_metrics, val_metrics = metrics(model, train), metrics(model, validation)
        history.append({"epoch": epoch, "train_online_cosine_loss": float(np.mean(losses)),
                        "post_epoch_train": train_metrics, "post_epoch_validation": val_metrics})
        score = val_metrics["mean_cosine"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"mask": mask, "learning_rate": learning_rate, "weight_decay": weight_decay,
                   "best_epoch": best_epoch, "epochs_run": len(history),
                   "selected_train": metrics(model, train),
                   "selected_validation": metrics(model, validation), "history": history}


def gradient_audit(model, examples):
    """Count current max winners and actual score-level cosine gradients.

    With max pooling, only tied maxima in each peak can receive a gradient.
    A very small/zero gradient may also be due to cosine scale invariance.
    This counts candidate ROWS (including repeated precursor CE examples).
    """
    total_candidates = total_winners = ambiguous_candidates = ambiguous_winners = 0
    total_nonzero_gradients = ambiguous_nonzero_gradients = total_supported = 0
    model.eval()
    for example in examples:
        features = torch.as_tensor(example["features"], dtype=torch.float32)
        indices = torch.as_tensor(example["peak_indices"])
        with torch.no_grad():
            scores = model(features)
        scores.requires_grad_(True)
        prediction = scores.new_zeros(len(example["target"])).scatter_reduce(
            0, indices, scores, reduce="amax", include_self=True)
        target = torch.as_tensor(example["target"], dtype=torch.float32)
        loss = 1 - F.cosine_similarity(prediction, target, dim=0, eps=1e-8)
        gradients, = torch.autograd.grad(loss, scores)
        counts = torch.bincount(indices, minlength=len(example["target"]))
        winners = scores == prediction[indices]
        ambiguous = counts[indices] > 1
        nonzero_gradients = gradients.abs() > 1e-12
        total_candidates += len(scores)
        total_supported += int((counts > 0).sum())
        total_winners += int(winners.sum())
        total_nonzero_gradients += int(nonzero_gradients.sum())
        ambiguous_candidates += int(ambiguous.sum())
        ambiguous_winners += int((winners & ambiguous).sum())
        ambiguous_nonzero_gradients += int((nonzero_gradients & ambiguous).sum())
    return {"candidate_rows": total_candidates, "supported_peak_rows": total_supported,
            "max_winner_rows_including_ties": total_winners,
            "max_winner_fraction": total_winners / total_candidates,
            "score_gradient_nonzero_rows_at_abs_gt_1e-12": total_nonzero_gradients,
            "score_gradient_nonzero_fraction_at_abs_gt_1e-12": total_nonzero_gradients / total_candidates,
            "ambiguous_candidate_rows": ambiguous_candidates,
            "ambiguous_max_winner_fraction": ambiguous_winners / ambiguous_candidates,
            "ambiguous_score_gradient_nonzero_fraction_at_abs_gt_1e-12":
                ambiguous_nonzero_gradients / ambiguous_candidates}


def energy_response(model, examples, mask=None):
    """Validation counterfactual: hold candidate locations and graph fixed."""
    by_energy = {}
    for energy in (20., 60.):
        changed = masked_copy(examples, mask)
        for example in changed:
            example["features"][:, 2 * FP_SIZE] = energy / 100
        by_energy[energy] = predict_examples(model, changed)
    cosines, differences = [], []
    for low, high in zip(by_energy[20.], by_energy[60.]):
        denominator = np.linalg.norm(low) * np.linalg.norm(high)
        cosines.append(float(np.dot(low, high) / denominator) if denominator > 0 else 0.)
        differences.append(float(np.mean(np.abs(low - high))))
    return {"energies": [20., 60.], "axis": "full_observed_axis_same_candidates",
            "mean_prediction_cosine": float(np.mean(cosines)),
            "median_prediction_cosine": float(np.median(cosines)),
            "mean_absolute_base_peak_normalized_change": float(np.mean(differences)),
            "median_absolute_base_peak_normalized_change": float(np.median(differences))}


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    examples = pickle.loads((OUTPUT.parent / "dataset.pkl").read_bytes())
    splits = {name: [e for e in examples if e["split"] == name]
              for name in ("train", "validation", "test")}
    original = torch.load(OUTPUT.parent / "model.pt", map_location="cpu", weights_only=True)
    published = json.loads((OUTPUT.parent / "metrics.json").read_text())
    model, reproduction = fit_audit(splits["train"], splits["validation"])
    exact = all(torch.equal(value, original["state_dict"][name])
                for name, value in model.state_dict().items())
    max_difference = max(float((value - original["state_dict"][name]).abs().max())
                         for name, value in model.state_dict().items())
    reproduction["checkpoint_bitwise_equal"] = exact
    reproduction["checkpoint_max_absolute_parameter_difference"] = max_difference
    reproduction["published_validation_history_exact"] = all(
        row["post_epoch_validation"]["mean_cosine"] == previous["validation_mean_cosine"]
        and row["train_online_cosine_loss"] == previous["train_cosine_loss"]
        for row, previous in zip(reproduction["history"], published["history"]))
    reproduction["test_metrics_original_reproduction_only"] = metrics(model, splits["test"])
    reproduction["validation_energy_response"] = energy_response(model, splits["validation"])
    result = {"seed": SEED, "threads": 2,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "split_sizes": {name: {"spectra": len(rows),
                  "molecules": len({e['molecule'] for e in rows}),
                  "candidate_rows": sum(len(e['features']) for e in rows),
                  "observed_peak_rows": sum(len(e['target']) for e in rows)}
                  for name, rows in splits.items()},
              "baseline_reproduction": reproduction,
              "selected_checkpoint_gradient_audit": {
                  name: gradient_audit(model, splits[name]) for name in ("train", "validation")},
              "controls_validation_only": {}}
    for name, kwargs in (
        ("regularized_slower_optimization", {"learning_rate": 1e-4, "weight_decay": 1e-3}),
        ("no_precursor_fingerprint", {"mask": "no_precursor_fingerprint"}),
        ("descriptors_only", {"mask": "descriptors_only"}),
    ):
        control_model, control = fit_audit(splits["train"], splits["validation"], **kwargs)
        control["validation_energy_response"] = energy_response(
            control_model, splits["validation"], kwargs.get("mask"))
        result["controls_validation_only"][name] = control
        print(name, "epoch", control["best_epoch"], "train", control["selected_train"]["mean_cosine"],
              "validation", control["selected_validation"]["mean_cosine"], flush=True)
    (OUTPUT / "training_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Checkpoint exact:", exact, "max difference:", max_difference, flush=True)


if __name__ == "__main__":
    main()
