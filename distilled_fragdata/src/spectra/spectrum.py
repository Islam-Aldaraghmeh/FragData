"""Prototype sparse postprocessing, with a small Spectrum container."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Spectrum:
    """Positive finite peaks; operations return a new spectrum."""

    mz: np.ndarray
    intensity: np.ndarray

    def __post_init__(self):
        mz, intensity = np.asarray(self.mz), np.asarray(self.intensity)
        if mz.ndim != 1 or intensity.ndim != 1 or len(mz) != len(intensity):
            raise ValueError("m/z and intensity must be equal-length vectors")
        peaks = _array(np.column_stack([mz, intensity]))
        self.mz, self.intensity = peaks[:, 0], peaks[:, 1]

    @classmethod
    def from_peaks(cls, peaks):
        a = _array(peaks)
        return cls(a[:, 0], a[:, 1])

    def as_array(self):
        return np.column_stack([self.mz, self.intensity])

    def normalized(self):
        return Spectrum.from_peaks(normalize_spectrum(self.as_array()))

    def filtered(self, **settings):
        return Spectrum.from_peaks(prepare_spectrum(self.as_array(), **settings))

    def merged(self, **settings):
        return Spectrum.from_peaks(merge_peaks(self.as_array(), **settings))

    def match(self, other, *, atol=0.005, ppm=20.0):
        return match_peaks(self.as_array(), other.as_array(), atol=atol, ppm=ppm)

    @classmethod
    def average(cls, spectra, *, prepared=False, atol=0.005, ppm=20.0):
        arrays = [s.as_array() if isinstance(s, cls) else s for s in spectra]
        return cls.from_peaks(mean_normalized_spectrum(arrays, prepared=prepared, atol=atol, ppm=ppm))


def clean_raw(spec):
    """Prototype GLACIER-row filter, before strict sparse postprocessing."""
    spec = np.asarray(spec, dtype=float)
    return spec[np.isfinite(spec).all(axis=1) & (spec[:, 0] > 0) & (spec[:, 1] > 0)]


def _array(peaks) -> np.ndarray:
    a = np.asarray(peaks, dtype=float)
    if a.size == 0:
        return np.empty((0, 2), dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Expected N x 2 array of (m/z, intensity).")
    if not np.isfinite(a).all() or (a[:, 0] <= 0).any():
        raise ValueError("m/z must be positive and all spectrum values finite.")
    return a[a[:, 1] > 0].copy()


def mass_tolerance(a: float, b: float, *, atol: float = 0.005,
                   ppm: float = 20.0) -> float:
    return max(atol, max(a, b) * ppm * 1e-6)


def merge_peaks(peaks, *, atol: float = 0.005, ppm: float = 20.0,
                aggregation: str = "sum") -> np.ndarray:
    """Merge fixed-anchor groups; m/z is intensity-weighted before pooling.

    Sum conserves experimental/ensemble TIC; max follows GLACIER's native
    duplicate-output intensity pooling. The anchored sparse mass grouping is
    this evaluation's prescribed tolerance convention, not its native bins.
    """
    if aggregation not in ("sum", "max"):
        raise ValueError("aggregation must be 'sum' or 'max'.")
    a = _array(peaks)
    if len(a) == 0:
        return a
    a = a[np.argsort(a[:, 0], kind="stable")]
    groups = []
    start = 0
    for stop in range(1, len(a) + 1):
        if stop < len(a) and a[stop, 0] - a[start, 0] <= mass_tolerance(
                a[start, 0], a[stop, 0], atol=atol, ppm=ppm):
            continue
        group = a[start:stop]
        summed_intensity = float(group[:, 1].sum())
        intensity = summed_intensity if aggregation == "sum" else float(group[:, 1].max())
        groups.append((float(np.dot(group[:, 0], group[:, 1]) / summed_intensity), intensity))
        start = stop
    return np.asarray(groups, dtype=float).reshape(-1, 2)


def prepare_spectrum(peaks, *, relative_threshold: float = 1e-3,
                     max_peaks: int | None = 100, atol: float = 0.005,
                     ppm: float = 20.0, aggregation: str = "max") -> np.ndarray:
    """Prepare one predicted spectrum; max-pool BEFORE thresholding/top-k.

    Experimental targets should request aggregation='sum' to preserve TIC.
    """
    if relative_threshold < 0 or (max_peaks is not None and max_peaks < 1):
        raise ValueError("Filtering parameters must be nonnegative and top-k positive.")
    a = merge_peaks(peaks, atol=atol, ppm=ppm, aggregation=aggregation)
    if len(a) == 0:
        return a
    a = a[a[:, 1] >= relative_threshold * a[:, 1].max()]
    if max_peaks is not None and len(a) > max_peaks:
        # Intensity descending, mass ascending tie break.
        a = a[np.lexsort((a[:, 0], -a[:, 1]))[:max_peaks]]
    return a[np.argsort(a[:, 0], kind="stable")]


def normalize_spectrum(peaks) -> np.ndarray:
    a = _array(peaks)
    if len(a):
        a[:, 1] /= a[:, 1].sum()
    return a


def mean_normalized_spectrum(spectra: Iterable, *, prepared: bool = False,
                             atol: float = 0.005, ppm: float = 20.0) -> np.ndarray:
    """Equal prior weight per state; the aggregate retains its full support."""
    states = [normalize_spectrum(s if prepared else prepare_spectrum(s, atol=atol, ppm=ppm))
              for s in spectra]
    if not states:
        raise ValueError("At least one microstate spectrum is required.")
    nonempty = [s for s in states if len(s)]
    if not nonempty:
        return np.empty((0, 2), dtype=float)
    pooled = np.concatenate(nonempty)
    pooled[:, 1] /= len(states)
    return normalize_spectrum(merge_peaks(pooled, atol=atol, ppm=ppm))


def match_peaks(predicted, target, *, atol: float = 0.005,
                ppm: float = 20.0) -> list[tuple[int, int]]:
    """Greedily take globally closest admissible unused pairs, deterministically.

    This is global error ordering, not a maximum-cardinality assignment. Ties
    use predicted m/z, target m/z, then input indices; each peak is used once.
    Input spectra should already have been prepared/merged.
    """
    p, t = _array(predicted), _array(target)
    edges = []
    for i, (pm, _) in enumerate(p):
        for j, (tm, _) in enumerate(t):
            error = abs(pm - tm)
            if error <= mass_tolerance(pm, tm, atol=atol, ppm=ppm):
                edges.append((error, pm, tm, i, j))
    edges.sort()
    used_p, used_t, pairs = set(), set(), []
    for _, _, _, i, j in edges:
        if i not in used_p and j not in used_t:
            pairs.append((i, j))
            used_p.add(i)
            used_t.add(j)
    return pairs
