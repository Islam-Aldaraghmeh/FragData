"""Figures for the notebook (matplotlib, one consistent style).

Colors follow fixed roles: predicted / model = blue, observed = orange,
third series = aqua, everything neutral (unsupported peaks, references) = gray.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#9a9893", "#e6e5e1"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "regular", "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "text.color": INK, "axes.axisbelow": True,
    })


def model_comparison(summary, ci, metric="cos", reference=None, ax=None, title=None):
    """Point estimate + 95% cluster-bootstrap interval per model."""
    style()
    names = list(summary.index)
    y = np.arange(len(names))[::-1]
    ax = ax or plt.subplots(figsize=(7, 0.42 * len(names) + 1.2))[1]
    lo = np.array([ci[n][1] for n in names])
    hi = np.array([ci[n][2] for n in names])
    mid = summary[metric].values
    ax.hlines(y, lo, hi, color=BLUE, linewidth=2)
    ax.plot(mid, y, "o", color=BLUE, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    for yy, m in zip(y, mid):
        ax.text(m, yy + 0.22, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    if reference:
        for label, value in reference.items():
            ax.axvline(value, color=MUTED, linestyle="--", linewidth=1)
            ax.text(value, y.max() + 0.7, label, ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names)
    ax.set_ylim(-0.7, len(names) - 0.1 + (0.8 if reference else 0))
    ax.set_xlabel(f"mean {metric} (test, 95% molecule-bootstrap CI)")
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, loc="left")
    return ax


def training_curves(runs, ax_pair=None):
    """Train loss and validation cosine per epoch for several runs."""
    style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6)) if ax_pair is None else (None, ax_pair)
    colors = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#008300", "#6250d6", "#e34948"]
    for (name, run), c in zip(runs.items(), colors):
        h = run.history
        ep = [x["epoch"] for x in h]
        a.plot(ep, [x["train_loss"] for x in h], color=c, linewidth=2, label=name)
        b.plot(ep, [x["val_cos"] for x in h], color=c, linewidth=2, label=name)
        best = max(h, key=lambda x: x["val_cos"])
        b.plot(best["epoch"], best["val_cos"], "o", color=c, markersize=7, markeredgecolor="white")
    a.set(title="training loss", xlabel="epoch")
    b.set(title="validation mean cosine (dot = selected epoch)", xlabel="epoch")
    b.legend(fontsize=8.5, loc="lower right")
    return fig


def mirror(fs, peaks_by_id, spec, pred, title=None, ax=None, label="predicted"):
    """Measured spectrum (up) vs prediction (down), both scaled to their base peak.

    Gray peaks are measured peaks without any MAGMa candidate (cannot be predicted).
    """
    style()
    ax = ax or plt.subplots(figsize=(9, 3.8))[1]
    cid = fs.meta["cache_id"].values[spec]
    full = peaks_by_id[cid]
    a, b = fs.spec_bag_ptr[spec], fs.spec_bag_ptr[spec + 1]
    sup_idx = fs.bag_peak_index[a:b]
    supported = np.zeros(len(full), bool)
    supported[sup_idx] = True
    top = full[:, 1] / full[:, 1].max() * 100
    ax.vlines(full[~supported, 0], 0, top[~supported], color=MUTED, linewidth=1.2, label="measured, no candidate")
    ax.vlines(full[supported, 0], 0, top[supported], color=ORANGE, linewidth=1.6, label="measured")
    p = np.asarray(pred, float)
    ax.vlines(fs.bag_mz[a:b], 0, -p / max(p.max(), 1e-12) * 100, color=BLUE, linewidth=1.6, label=label)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_ylim(-108, 108)
    ax.set_yticks([-100, -50, 0, 50, 100], ["100", "50", "0", "50", "100"])
    ax.set_xlabel("m/z")
    ax.set_ylabel("relative intensity (%)")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    return ax


def bars(series, xlabel, title=None, ax=None, color=BLUE, fmt="{:.3f}"):
    """Horizontal bars with value labels (sorted as given)."""
    style()
    ax = ax or plt.subplots(figsize=(6.5, 0.36 * len(series) + 1))[1]
    y = np.arange(len(series))[::-1]
    ax.barh(y, series.values, height=0.6, color=color)
    for yy, v in zip(y, series.values):
        ax.text(v, yy, " " + fmt.format(v), va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, [str(i) for i in series.index])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, loc="left")
    return ax


def slices(slice_tables, baseline_tables=None, metric="cos", title="Where does the model fail?"):
    """Small multiples: mean metric per slice for the model (blue) and a baseline (gray)."""
    style()
    keys = list(slice_tables)
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        t = slice_tables[k]
        x = np.arange(len(t))
        w = 0.38 if baseline_tables else 0.6
        ax.bar(x - (w / 2 if baseline_tables else 0), t[metric].values, width=w, color=BLUE, label="model")
        if baseline_tables:
            ax.bar(x + w / 2, baseline_tables[k][metric].reindex(t.index).values, width=w, color=MUTED,
                   label="uniform")
        ax.set_xticks(x, [str(i).replace(".0", "") for i in t.index], rotation=0, fontsize=8.5)
        for xx, n in zip(x, t["n"].values):
            ax.text(xx, 0.02, f"n={n}", ha="center", fontsize=7, color="white", rotation=90, va="bottom")
        ax.set_title(k, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"mean {metric}")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    return fig


def scatter_xy(x, y, xlabel, ylabel, title=None, ax=None, diagonal=True):
    style()
    ax = ax or plt.subplots(figsize=(4.6, 4.2))[1]
    ax.scatter(x, y, s=10, color=BLUE, alpha=0.35, linewidths=0)
    if diagonal:
        ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    return ax


def lines(x, ys, xlabel, ylabel, title=None, ax=None, markers=True):
    """ys: {label: values}; first series blue, second orange, third aqua."""
    style()
    ax = ax or plt.subplots(figsize=(5.2, 3.6))[1]
    for (label, v), c in zip(ys.items(), [BLUE, ORANGE, AQUA, YELLOW]):
        ax.plot(x, v, color=c, linewidth=2, marker="o" if markers else None, markersize=7,
                markeredgecolor="white", label=label)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.legend(fontsize=8.5)
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    return ax


def histogram(values, xlabel, title=None, ax=None, bins=40, color=BLUE, label=None):
    style()
    ax = ax or plt.subplots(figsize=(5.2, 3.4))[1]
    ax.hist(values, bins=bins, color=color, alpha=0.85 if label is None else 0.55, label=label,
            edgecolor="white", linewidth=0.6)
    ax.set(xlabel=xlabel, ylabel="spectra")
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    return ax
