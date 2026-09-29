"""Fragment and spectrum galleries shared by experiment notebooks."""
from .candidates import _unpack_bag
from src.chemistry.formula import formula_counts
from helper import plot_smiles


def _fragment_graph_svg(fragment, size, show_atom_indices):
    """Draw a fresh heavy graph without sanitizing, kekulizing, or adding H."""
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    graph = fragment.graph
    if not graph.get("atoms"):
        raise ValueError("Fragment has no atoms")
    bond_types = {1.0: Chem.BondType.SINGLE, 1.5: Chem.BondType.AROMATIC,
                  2.0: Chem.BondType.DOUBLE, 3.0: Chem.BondType.TRIPLE}
    # Block RDKit log noise; a malformed record becomes a visible error card.
    with rdBase.BlockLogs():
        editable = Chem.RWMol()
        for record in graph["atoms"]:
            atom = Chem.Atom(record["element"])
            atom.SetNoImplicit(True)
            atom.SetNumExplicitHs(0)
            atom.SetNumRadicalElectrons(0)
            atom.SetIsAromatic(bool(record.get("aromatic", False)))
            editable.AddAtom(atom)
        for record in graph["bonds"]:
            order = float(record["order"])
            if order not in bond_types:
                raise ValueError(f"Unsupported bond order: {order}")
            i, j = int(record["i"]), int(record["j"])
            editable.AddBond(i, j, bond_types[order])
            editable.GetBondBetweenAtoms(i, j).SetIsAromatic(
                bool(record.get("aromatic", False)) or order == 1.5
            )
        molecule = editable.GetMol()
        molecule.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(molecule)
        rdDepictor.Compute2DCoords(molecule)

        drawer = rdMolDraw2D.MolDraw2DSVG(*size)
        options = drawer.drawOptions()
        options.prepareMolsBeforeDrawing = False
        options.includeRadicals = False
        options.addAtomIndices = show_atom_indices
        for atom in molecule.GetAtoms():
            if atom.GetDegree() == 0:
                options.atomLabels[atom.GetIdx()] = atom.GetSymbol()
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
    return svg[svg.index("<svg"):]


def plot_fragment_bags(candidate_bags, *, size=(260, 200), show_atom_indices=False):
    """Return notebook HTML: one row per (m/z, fragments, ms3_spectra) bag.

    All candidates retain input order; long rows scroll rather than wrap or
    truncate. Draw only heavy connectivity/bond orders, with formula/charge in
    captions. Imperfect valence/aromaticity does not require chemistry repair.
    Malformed graph records keep an error card. Source candidates are untouched.
    Pass either a list of bags or a single bag such as bags[i]. Use
    show_atom_indices=True to label the graph's local atom indices.
    """
    import math
    from numbers import Real
    from html import escape
    from IPython.display import HTML
    from rdkit import Chem  # Fail once if the notebook kernel lacks RDKit.

    if (len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                              for v in size)):
        raise ValueError("size must be (width, height) in positive integer pixels")
    width, height = size
    # Also accept the single (m/z, fragments) pair returned by bags[i].
    if (isinstance(candidate_bags, (tuple, list)) and len(candidate_bags) in (2, 3)
            and isinstance(candidate_bags[0], Real)):
        candidate_bags = [candidate_bags]
    sections = []
    for bag_index, bag in enumerate(candidate_bags):
        mz, fragments, _ = _unpack_bag(bag)
        mz = float(mz)
        if not math.isfinite(mz) or mz <= 0:
            raise ValueError(f"Bag {bag_index} has an invalid m/z")
        fragments = list(fragments)
        cards = []
        for fragment_index, fragment in enumerate(fragments):
            try:
                counts = formula_counts(fragment.ionic_formula)
                elements = ([e for e in ("C", "H") if e in counts]
                            if "C" in counts else [])
                elements += sorted(set(counts) - set(elements))
                formula = "".join(e + (str(counts[e]) if counts[e] != 1 else "")
                                  for e in elements)
            except (AttributeError, TypeError, ValueError):
                formula = str(getattr(fragment, "ionic_formula", "formula unavailable"))
            charge = getattr(fragment, "charge", "?")
            charge_text = f"{charge:+d}" if isinstance(charge, int) else str(charge)
            try:
                drawing = _fragment_graph_svg(fragment, (width, height), show_atom_indices)
            except (AttributeError, KeyError, IndexError, TypeError, ValueError,
                    RuntimeError, OverflowError) as error:
                # Keep this candidate's position without breaking the other drawings.
                reason = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
                drawing = (
                    f'<div style="height:{height}px;display:flex;align-items:center;'
                    'justify-content:center;padding:12px;box-sizing:border-box;color:#9b3535;'
                    'overflow:auto"><div><b>Graph unavailable</b><br>'
                    f'<small>{escape(reason)}</small></div></div>'
                )
            caption = (f'<b>Candidate {fragment_index}</b><br>{escape(formula)}'
                       f' &nbsp;|&nbsp; charge {escape(charge_text)}')
            cards.append(
                f'<div style="flex:0 0 {width}px;width:{width}px;border:1px solid #ddd;'
                'border-radius:6px;overflow:hidden;background:white;color:#222">'
                f'{drawing}<div style="padding:8px;text-align:center;font-size:13px">'
                f'{caption}</div></div>'
            )
        row = (''.join(cards) if cards else
               '<div style="padding:16px;color:#666">No candidate fragments</div>')
        sections.append(
            '<section style="margin:0 0 18px">'
            f'<div style="margin-bottom:8px"><b>Bag {bag_index} · m/z {mz:.5f}</b>'
            f' &nbsp;({len(fragments)} candidates)</div>'
            '<div style="display:flex;flex-wrap:nowrap;gap:10px;overflow-x:auto;'
            f'padding-bottom:8px">{row}</div></section>'
        )
    if not sections:
        return HTML('<p>No candidate bags to display.</p>')
    return HTML(
        '<div style="max-width:100%;font-family:system-ui,sans-serif">'
        '<p style="font-size:13px">Heavy-atom sketches; hydrogen and charge positions '
        'are not assigned. Scroll sideways to see every candidate.</p>'
        + ''.join(sections) + '</div>'
    )

def _gallery_bag_list(bags, name):
    """Normalize pairs/triples to (m/z, items, ms3_spectra), preserving order."""
    import math
    from numbers import Real

    if (isinstance(bags, (tuple, list)) and len(bags) in (2, 3)
            and isinstance(bags[0], Real)):
        bags = [bags]
    result = []
    for i, bag in enumerate(bags):
        mz, items, ms3_spectra = _unpack_bag(bag)
        mz = float(mz)
        if not math.isfinite(mz) or mz <= 0:
            raise ValueError(f"{name}: bag {i} has an invalid m/z")
        result.append((mz, list(items), ms3_spectra))
    return result


def _gallery_peaks(spectrum, bag_index, candidate_index):
    """Accept prediction arrays or either repo Spectrum class, without mutation."""
    import numpy as np

    if spectrum is None:
        return None
    if hasattr(spectrum, "as_array"):
        spectrum = spectrum.as_array()
    elif hasattr(spectrum, "peaks"):
        spectrum = spectrum.peaks
    peaks = np.asarray(spectrum, dtype=float)
    if peaks.size == 0:
        return np.empty((0, 2), dtype=float)
    if (peaks.ndim != 2 or peaks.shape[1] != 2 or not np.isfinite(peaks).all()
            or (peaks[:, 0] <= 0).any() or (peaks[:, 1] < 0).any()):
        raise ValueError(
            f"Bag {bag_index}, candidate {candidate_index}: expected finite N x 2 "
            "peaks with positive m/z and nonnegative intensities"
        )
    return peaks.copy()


def _gallery_overlap_colors(prediction, observed, *, color_matches=True, strict=None):
    """Color raw visible peaks by m/z presence, without changing either spectrum.

    With strict=None, any positive counterpart within max(0.005 Da, 20 ppm)
    is a match. Otherwise the SAME pair must also differ by at most strict
    percentage points of base-peak-normalized intensity (0..100 per spectrum).
    color_matches=False disables highlighting. Neither option affects cosine.
    A missing spectrum has no comparison; an empty spectrum has no matches.
    """
    import numpy as np

    if not color_matches or prediction is None or observed is None:
        return None, None
    prediction_colors = np.full(len(prediction), "#dc2626", dtype="<U7")
    observed_colors = np.full(len(observed), "#dc2626", dtype="<U7")
    pi = np.flatnonzero(prediction[:, 1] > 0)
    oi = np.flatnonzero(observed[:, 1] > 0)
    pm, om = prediction[pi, 0, None], observed[oi, 0][None, :]
    tolerance = np.maximum(0.005, np.maximum(pm, om) * 20e-6)
    matches = np.abs(pm - om) <= tolerance
    if strict is not None and len(pi) and len(oi):
        predicted_relative = 100 * (prediction[pi, 1] / prediction[pi, 1].max())
        observed_relative = 100 * (observed[oi, 1] / observed[oi, 1].max())
        difference = np.abs(predicted_relative[:, None] - observed_relative[None, :])
        matches &= difference <= strict + 1e-12  # Include floating-point boundary ties.
    prediction_colors[pi[matches.any(axis=1)]] = "#16a34a"
    observed_colors[oi[matches.any(axis=0)]] = "#16a34a"
    return prediction_colors.tolist(), observed_colors.tolist()


def _gallery_spectrum_svg(peaks, size, x_max, y_max, normalize, color="#2563eb",
                          peak_colors=None):
    """Fixed axes geometry keeps baselines aligned even for missing predictions."""
    from io import StringIO
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    width, height = size
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    # Fixed pixel padding, not tight_layout/bbox_inches='tight' per spectrum.
    axes = figure.add_axes((58 / width, 45 / height,
                            (width - 74) / width, (height - 67) / height))
    if peaks is not None and len(peaks) and peaks[:, 1].max() > 0:
        intensities = peaks[:, 1].copy()
        if normalize:
            intensities = 100 * intensities / intensities.max()
        axes.vlines(peaks[:, 0], 0, intensities,
                    color=peak_colors if peak_colors is not None else color, linewidth=1.4)
    else:
        message = "Prediction unavailable" if peaks is None else "No nonzero peaks"
        axes.text(0.5, 0.5, message, transform=axes.transAxes,
                  ha="center", va="center", color="#64748b", fontsize=9)
    axes.set_xlim(0, x_max)
    axes.set_ylim(0, y_max)
    axes.set_xlabel("m/z", fontsize=9, labelpad=5)
    axes.set_ylabel("Relative intensity (%)" if normalize else "Intensity",
                    fontsize=9, labelpad=5)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if normalize:
        axes.set_yticks([0, 50, 100])
    else:
        axes.yaxis.set_major_locator(MaxNLocator(nbins=3))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        axes.yaxis.set_major_formatter(formatter)
        axes.yaxis.get_offset_text().set_fontsize(8)
    axes.tick_params(labelsize=8, colors="#475569", length=3)
    axes.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#cbd5e1")
    stream = StringIO()
    figure.savefig(stream, format="svg", facecolor="white")
    # Figure is constructed directly: no pyplot figure/window is registered.
    svg = stream.getvalue()
    return svg[svg.index("<svg"):]


def _gallery_mirror_svg(prediction, observed, size, x_max, y_max, normalize, *,
                        prediction_colors=None, observed_colors=None,
                        prediction_label="Prediction"):
    """Compare positive prediction and negative MS3 on one symmetric axis."""
    from io import StringIO
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    width, height = size
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    axes = figure.add_axes((58 / width, 45 / height,
                            (width - 74) / width, (height - 67) / height))
    for peaks, sign, color, colors, label in (
            (prediction, 1, "#2563eb", prediction_colors, prediction_label),
            (observed, -1, "#0f766e", observed_colors, "MS3")):
        if peaks is not None and len(peaks) and peaks[:, 1].max() > 0:
            intensities = peaks[:, 1].copy()
            if normalize:
                intensities = 100 * (intensities / intensities.max())
            axes.vlines(peaks[:, 0], 0, sign * intensities,
                        color=colors if colors is not None else color, linewidth=1.4)
        else:
            message = f"{label} unavailable" if peaks is None else f"{label}: no nonzero peaks"
            axes.text(0.5, 0.75 if sign > 0 else 0.25, message,
                      transform=axes.transAxes, ha="center", va="center",
                      color="#64748b", fontsize=9)
    axes.axhline(0, color="#475569", linewidth=0.9)
    axes.set_xlim(0, x_max)
    axes.set_ylim(-y_max, y_max)
    axes.set_xlabel("m/z", fontsize=9, labelpad=5)
    axes.set_ylabel("Relative intensity (%)" if normalize else "Intensity",
                    fontsize=9, labelpad=5)
    axes.set_title(f"{prediction_label} (+) / MS3 (−)", fontsize=9, color="#334155", pad=7)
    axes.xaxis.set_major_locator(MaxNLocator(nbins=4))
    if normalize:
        axes.set_yticks([-100, -50, 0, 50, 100])
    else:
        axes.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 3))
        axes.yaxis.set_major_formatter(formatter)
        axes.yaxis.get_offset_text().set_fontsize(8)
    axes.tick_params(labelsize=8, colors="#475569", length=3)
    axes.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#cbd5e1")
    stream = StringIO()
    figure.savefig(stream, format="svg", facecolor="white")
    svg = stream.getvalue()
    return svg[svg.index("<svg"):]


def _gallery_svg_image(svg, size, description):
    """Separate SVG image documents avoid repeated SVG ID/CSS collisions."""
    import base64
    from html import escape

    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{encoded}" '
            f'width="{size[0]}" height="{size[1]}" alt="{escape(description, quote=True)}" '
            'style="display:block;max-width:none">')


def _gallery_ms3_energy_values(spectrum):
    """Use the same recorded collision energies for filtering and captions."""
    metadata = spectrum.metadata
    energy = metadata.get("collision_energy")
    if energy is None:
        energy_path = metadata.get("msn_collision_energies")
        energy = energy_path[-1] if energy_path is not None and len(energy_path) else None
    return list(energy) if isinstance(energy, (list, tuple)) else [energy]


def _gallery_matches_ms3_energy(spectrum, requested):
    """A merged observation matches any listed CE; its peaks are never split."""
    import math

    for value in _gallery_ms3_energy_values(spectrum):
        if isinstance(value, bool):
            continue
        try:
            energy = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(energy) and any(
                math.isclose(energy, wanted, rel_tol=0, abs_tol=1e-6)
                for wanted in requested):
            return True
    return False


def _gallery_ms3_label(spectrum):
    """Describe the same observation in its plot and cosine-match caption."""
    values = _gallery_ms3_energy_values(spectrum)
    energy = ", ".join("unknown" if value is None else
                       f"{value:g}" if isinstance(value, (int, float)) else str(value)
                       for value in values) or "unknown"
    scan = spectrum.metadata.get("scan_number", "?")
    if isinstance(scan, (list, tuple)):
        scan = ", ".join(str(value) for value in scan)
    return energy, str(scan)


def _gallery_best_ms3_cosines(predictions, observations):
    """Return (best cosine, displayed MS3 index) per original candidate.

    Match the repo's MS3 candidate-support scoring: predictions are already
    prepared Glacier ensembles; sum-merge experimental peaks without threshold
    or top-k truncation. Neither input is changed. Empty targets cannot rank
    candidates; empty predictions score zero against a usable target.
    """
    from src.spectra.metrics import cosine_similarity
    from src.spectra.spectrum import normalize_spectrum, prepare_spectrum

    targets = []
    for index, peaks in enumerate(observations):
        if peaks is None:
            continue
        target = normalize_spectrum(prepare_spectrum(
            peaks, relative_threshold=0, max_peaks=None, aggregation="sum"))
        if len(target):
            targets.append((index, target))
    scores = []
    for peaks in predictions:
        if peaks is None or not targets:
            scores.append((None, None))
            continue
        matches = [(cosine_similarity(peaks, target,
                                      prepared_predicted=True, prepared_target=True), index)
                   for index, target in targets]
        # max is stable: tied observations retain their displayed order.
        scores.append(max(matches, key=lambda match: match[0]))
    return scores


def _gallery_mixture_weight(fit, candidate_index, ms3_index):
    """Keep each fitted coefficient next to its original candidate."""
    from html import escape

    if fit["status"] != "ok":
        label, detail, attribute = "Mix weight n/a", fit["reason"], ""
    elif candidate_index not in fit["usable_indices"]:
        label, detail, attribute = "Mix weight n/a", "No usable predicted peaks", ""
    else:
        weight = float(fit["weights"][candidate_index])
        label = f"Mix weight {weight:.3f}"
        detail = f"Coefficient {weight:.8g}; fitted weights sum to 1, not a probability"
        attribute = f' data-weight="{weight:.17g}"'
    return (f'<div class="mixture-weight"{attribute} title="{escape(detail, quote=True)}">'
            f'MS3 #{ms3_index} · {label}</div>')


def _gallery_mixture_card(fit, ms3_index, target, size, x_max, y_max, normalize,
                          *, color_matches=True, strict=None, mirror=False):
    """Compact fitted spectrum for the visible result preview."""
    from html import escape

    valid = fit["status"] == "ok"
    score = f'Fit cosine {fit["cosine"]:.3f}' if valid else "Fit unavailable"
    baseline = (f'Best single (aligned): {fit["best_single_cosine"]:.3f}'
                if valid else fit["reason"])
    header = ('<div class="candidate-heading"><b>Mixed spectrum</b>'
              f'<strong>{score}</strong></div>')
    stripe = (f'<div class="mixture-weight" title="{escape(baseline, quote=True)}">'
              f'{escape(baseline)}</div>')
    peaks = fit["spectrum"] if valid else None
    colors, target_colors = _gallery_overlap_colors(
        peaks, target, color_matches=color_matches, strict=strict)
    if mirror:
        size = (size[0], 2 * size[1])
        svg = _gallery_mirror_svg(peaks, target, size, x_max, y_max, normalize,
                                  prediction_colors=colors, observed_colors=target_colors,
                                  prediction_label="Mixed spectrum")
        drawing = _gallery_svg_image(svg, size, f"Fitted mixture mirrored against MS3 {ms3_index}")
    elif valid:
        svg = _gallery_spectrum_svg(peaks, size, x_max, y_max, normalize, peak_colors=colors)
        drawing = _gallery_svg_image(svg, size, f"Fitted mixture for MS3 {ms3_index}")
    else:
        drawing = f'<div class="graph-placeholder">{escape(fit["reason"])}</div>'
    mirror_attribute = f' data-mirror-ms3-index="{ms3_index}"' if mirror else ''
    panel = "mirror-panel" if mirror else "spectrum-panel"
    return (f'<article class="mixture-card" data-mixture-ms3-index="{ms3_index}"{mirror_attribute}>'
            f'{header}{stripe}<div class="{panel}">{drawing}</div></article>')


def _render_spectrum_gallery(spectrum_bags, size, normalize,
                             candidate_bags=None, fragment_height=210,
                             show_atom_indices=False, ms3_collision_energies=None,
                             only_with_ms3=False, fit_mixture=False,
                             mixture_min_candidates=2, color_matches=True, strict=None,
                             mirror=False):
    from html import escape
    from IPython.display import HTML
    from numbers import Real
    import math

    if not isinstance(mirror, bool):
        raise ValueError("mirror must be a boolean")
    if not isinstance(color_matches, bool):
        raise ValueError("color_matches must be a boolean")
    if strict is not None:
        if (isinstance(strict, bool) or not isinstance(strict, Real)
                or not math.isfinite(strict) or strict < 0):
            raise ValueError("strict must be None or a finite nonnegative intensity tolerance in percentage points")
        strict = float(strict)
    if not isinstance(fit_mixture, bool):
        raise ValueError("fit_mixture must be a boolean")
    if (isinstance(mixture_min_candidates, bool) or not isinstance(mixture_min_candidates, int)
            or mixture_min_candidates < 2):
        raise ValueError("mixture_min_candidates must be an integer of at least 2")
    if not isinstance(only_with_ms3, bool):
        raise ValueError("only_with_ms3 must be a boolean")
    requested = () if ms3_collision_energies is None else ms3_collision_energies
    if not isinstance(requested, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 for value in requested):
        raise ValueError("ms3_collision_energies must be a list of finite, nonnegative numbers")
    requested = tuple(float(value) for value in requested)

    if (len(size) != 2 or any(isinstance(v, bool) or not isinstance(v, int) for v in size)
            or size[0] < 240 or size[1] < 170):
        raise ValueError("size must be integer (width, height), at least (240, 170) pixels")
    if (isinstance(fragment_height, bool) or not isinstance(fragment_height, int)
            or fragment_height < 100):
        raise ValueError("fragment_height must be an integer of at least 100 pixels")
    width, height = size
    mirror_size = (width, 2 * height)
    combined = candidate_bags is not None
    sections = []
    for bag_index, (mz, spectra, _) in enumerate(spectrum_bags):
        all_observed = candidate_bags[bag_index][2] if combined else []
        observed = ([child for child in all_observed
                     if _gallery_matches_ms3_energy(child, requested)]
                    if requested else all_observed)
        mirrored = bool(mirror and combined and observed)
        if combined and only_with_ms3 and not observed:
            continue
        peaks_list = [_gallery_peaks(s, bag_index, j) for j, s in enumerate(spectra)]
        observed_peaks = [_gallery_peaks(s, bag_index, j) for j, s in enumerate(observed)]
        mixtures = []
        if combined and fit_mixture and observed and len(peaks_list) >= mixture_min_candidates:
            from src.spectra.mixture import fit_spectral_mixture
            mixtures = [fit_spectral_mixture(peaks_list, target,
                                             min_candidates=mixture_min_candidates)
                        for target in observed_peaks]
        comparison_peaks = peaks_list + observed_peaks + [fit["spectrum"] for fit in mixtures]
        # Compare candidates on the same scales within this bag, including peaks
        # above the precursor if present, rather than silently clipping them.
        x_max = 1.05 * max([mz] + [float(p[:, 0].max()) for p in comparison_peaks
                                  if p is not None and len(p)])
        y_max = (105.0 if normalize else
                 1.05 * max([1e-12] + [float(p[:, 1].max()) for p in comparison_peaks
                                      if p is not None and len(p)]))
        if not normalize and y_max <= 1.05e-12:
            y_max = 1.0
        scores = (_gallery_best_ms3_cosines(peaks_list, observed_peaks)
                  if combined else [(None, None)] * len(peaks_list))
        candidate_order = sorted(
            range(len(peaks_list)),
            key=lambda index: -(scores[index][0] if scores[index][0] is not None else -1),
        )
        # Colors are specific to a candidate AND an observation. A shared MS3
        # may have different green/red peaks below different predictions.
        overlap_colors = [[_gallery_overlap_colors(peaks, target,
                                                    color_matches=color_matches, strict=strict)
                           for peaks in peaks_list]
                          for target in observed_peaks] if combined else []
        cards = []
        for candidate_index in candidate_order:
            peaks = peaks_list[candidate_index]
            count = 0 if peaks is None else len(peaks)
            status = "unavailable" if peaks is None else f"{count} peaks"
            header = (f'<div class="candidate-heading"><b>Candidate {candidate_index}</b>'
                      f'<span>{status}</span></div>')
            molecule_panel = ""
            if combined:
                cosine, ms3_index = scores[candidate_index]
                score_label = "Cosine n/a" if cosine is None else f"Best cosine {cosine:.3f}"
                if ms3_index is not None:
                    energy, scan = _gallery_ms3_label(observed[ms3_index])
                    reference = f"MS3 #{ms3_index} · CE {energy} · Scan {scan}"
                else:
                    reference = ("Prediction unavailable" if peaks is None
                                 else "No MS3 with nonzero peaks at displayed energies")
                detail = f"{status} · {reference}"
                header = (
                    '<div class="candidate-heading candidate-heading-scored">'
                    f'<b>Candidate {candidate_index}</b><strong>{score_label}</strong>'
                    f'<span class="cosine-reference" title="{escape(detail, quote=True)}">'
                    f'{escape(detail)}</span></div>'
                )
                fragment = candidate_bags[bag_index][1][candidate_index]
                try:
                    graph = _fragment_graph_svg(fragment, (width, fragment_height), show_atom_indices)
                    drawing = _gallery_svg_image(graph, (width, fragment_height),
                                                 f"Fragment {candidate_index}")
                except (AttributeError, KeyError, IndexError, TypeError, ValueError,
                        RuntimeError, OverflowError) as error:
                    reason = str(error).strip().splitlines()[0] if str(error).strip() else type(error).__name__
                    drawing = ('<div class="graph-placeholder"><b>Graph unavailable</b>'
                               f'<small>{escape(reason)}</small></div>')
                try:
                    counts = formula_counts(fragment.ionic_formula)
                    elements = ([e for e in ("C", "H") if e in counts]
                                if "C" in counts else [])
                    elements += sorted(set(counts) - set(elements))
                    formula = "".join(e + (str(counts[e]) if counts[e] != 1 else "")
                                      for e in elements)
                except (AttributeError, TypeError, ValueError):
                    formula = str(getattr(fragment, "ionic_formula", "formula unavailable"))
                charge = getattr(fragment, "charge", "?")
                charge_text = f"{charge:+d}" if isinstance(charge, int) else str(charge)
                caption = f"{formula}  |  charge {charge_text}"
                molecule_panel = (
                    f'<div class="fragment-panel">{drawing}</div>'
                    f'<div class="formula-caption" title="{escape(caption, quote=True)}">'
                    f'{escape(caption)}</div>'
                )
            peak_colors = overlap_colors[0][candidate_index][0] if overlap_colors else None
            if mirrored:
                spectrum_svg = _gallery_mirror_svg(
                    peaks, observed_peaks[0], mirror_size, x_max, y_max, normalize,
                    prediction_colors=peak_colors,
                    observed_colors=overlap_colors[0][candidate_index][1])
            else:
                spectrum_svg = _gallery_spectrum_svg(peaks, size, x_max, y_max, normalize,
                                                     peak_colors=peak_colors)
            spectrum_panel = _gallery_svg_image(
                spectrum_svg, mirror_size if mirrored else size,
                f"Spectrum {candidate_index}" + (" mirrored against MS3 0" if mirrored else ""))
            weight = _gallery_mixture_weight(mixtures[0], candidate_index, 0) if mixtures else ""
            mirror_attribute = ' data-mirror-ms3-index="0"' if mirrored else ''
            panel = "mirror-panel" if mirrored else "spectrum-panel"
            cards.append(
                f'<article class="candidate-card" data-candidate-index="{candidate_index}"{mirror_attribute}>'
                f'{header}{molecule_panel}{weight}<div class="{panel}">{spectrum_panel}</div></article>'
            )
        if combined:
            columns = max(len(cards), 1)
            grid = [f'<div class="bag-grid" style="grid-template-columns:repeat({columns}, {width + 2}px)">']
            if observed:
                energy, scan = _gallery_ms3_label(observed[0])
                grid.append('<div class="ms3-row-heading">'
                            f'Comparison with MS3 #0 · CE {escape(energy)} · Scan {escape(scan)}</div>')
            for column in range(columns):
                if column < len(cards):
                    grid.append(cards[column])
                else:
                    grid.append('<div class="candidate-padding">No candidates in this bag</div>')
            for ms3_index, child in enumerate(observed):
                label, scan = _gallery_ms3_label(child)
                if ms3_index > 0:
                    grid.append('<div class="ms3-row-heading">'
                                f'Comparison with MS3 #{ms3_index} · CE {escape(label)} · Scan {escape(scan)}</div>')
                    # Repeat predictions for each additional observation so a
                    # green/red stick always refers to the MS3 directly below.
                    for candidate_index in candidate_order:
                        predicted_colors, measured_colors = overlap_colors[ms3_index][candidate_index]
                        if mirrored:
                            svg = _gallery_mirror_svg(
                                peaks_list[candidate_index], observed_peaks[ms3_index],
                                mirror_size, x_max, y_max, normalize,
                                prediction_colors=predicted_colors, observed_colors=measured_colors)
                        else:
                            svg = _gallery_spectrum_svg(
                                peaks_list[candidate_index], size, x_max, y_max, normalize,
                                peak_colors=predicted_colors)
                        drawing = _gallery_svg_image(svg, mirror_size if mirrored else size,
                            f"Spectrum {candidate_index}" +
                            (f" mirrored against MS3 {ms3_index}" if mirrored else ""))
                        weight = (_gallery_mixture_weight(mixtures[ms3_index], candidate_index, ms3_index)
                                  if mixtures else "")
                        mirror_attribute = f' data-mirror-ms3-index="{ms3_index}"' if mirrored else ''
                        panel = "mirror-panel" if mirrored else "spectrum-panel"
                        comparison_label = "Prediction / MS3" if mirrored else "Prediction"
                        grid.append(
                            f'<article class="comparison-card" data-candidate-index="{candidate_index}" '
                            f'data-comparison-ms3-index="{ms3_index}"{mirror_attribute}>'
                            f'<div class="candidate-heading"><b>Candidate {candidate_index}</b>'
                            f'<span>{comparison_label}</span></div>'
                            f'{weight}<div class="{panel}">{drawing}</div></article>')
                    if not cards:
                        grid.append('<div class="ms3-padding">No predictions in this bag</div>')
                if mirrored:
                    if not cards:
                        svg = _gallery_mirror_svg(None, observed_peaks[ms3_index], mirror_size,
                                                  x_max, y_max, normalize)
                        drawing = _gallery_svg_image(svg, mirror_size, f"Measured MS3 {ms3_index}")
                        grid.append(f'<div class="mirror-panel" data-mirror-ms3-index="{ms3_index}">'
                                    f'{drawing}</div>')
                    continue
                grid.append('<div class="ms3-row-heading">'
                            f'Measured MS3 #{ms3_index} · compared with each prediction above</div>')
                for candidate_index in candidate_order or [None]:
                    peak_colors = (overlap_colors[ms3_index][candidate_index][1]
                                   if candidate_index is not None else None)
                    svg = _gallery_spectrum_svg(observed_peaks[ms3_index], size, x_max, y_max,
                                                normalize, color="#0f766e", peak_colors=peak_colors)
                    drawing = _gallery_svg_image(svg, size, f"Measured MS3, CE {label}, scan {scan}")
                    candidate_attribute = (f' data-candidate-index="{candidate_index}"'
                                           if candidate_index is not None else '')
                    grid.append(
                        f'<article class="ms3-card" data-ms3-index="{ms3_index}"{candidate_attribute}>'
                        f'<div class="ms3-heading"><b>MS3 #{ms3_index} · CE {escape(label)}</b>'
                        f'<span>Scan {escape(scan)}</span></div>'
                        f'<div class="spectrum-panel">{drawing}</div></article>')
            if not observed:
                grid.append('<div class="ms3-row-heading">Measured MS3</div>')
                message = ("No MS3 spectra at selected collision energies"
                           if all_observed and requested else "No linked MS3 spectra")
                for column in range(columns):
                    grid.append(f'<div class="ms3-padding">{message if column == 0 else ""}</div>')
            grid.append('</div>')
            row = '<div class="bag-scroll">' + ''.join(grid) + '</div>'
        else:
            content = ''.join(cards) if cards else '<div class="empty-bag">No candidates in this bag</div>'
            row = f'<div class="bag-row">{content}</div>'
        # Results live outside the horizontally scrolling candidate grid.
        # Even large bags show the mixed spectrum without scrolling sideways.
        previews = []
        for ms3_index, fit in enumerate(mixtures):
            label, scan = _gallery_ms3_label(observed[ms3_index])
            reference = ""
            if not mirrored:
                mixed = fit["spectrum"] if fit["status"] == "ok" else None
                _, colors = _gallery_overlap_colors(mixed, observed_peaks[ms3_index],
                                                    color_matches=color_matches, strict=strict)
                svg = _gallery_spectrum_svg(observed_peaks[ms3_index], size, x_max, y_max,
                                            normalize, color="#0f766e", peak_colors=colors)
                drawing = _gallery_svg_image(svg, size, f"MS3 {ms3_index} compared with fitted mixture")
                reference = (
                    f'<article class="ms3-card" data-ms3-index="{ms3_index}" data-mixture-reference="true">'
                    f'<div class="ms3-heading"><b>MS3 #{ms3_index} · CE {escape(label)}</b>'
                    f'<span>Scan {escape(scan)}</span></div>'
                    '<div class="mixture-weight">Measured reference</div>'
                    f'<div class="spectrum-panel">{drawing}</div></article>')
            result = _gallery_mixture_card(fit, ms3_index, observed_peaks[ms3_index],
                                           size, x_max, y_max, normalize,
                                           color_matches=color_matches, strict=strict, mirror=mirror)
            previews.append(
                f'<div class="mixture-preview" data-mixture-preview-index="{ms3_index}">'
                f'<div class="mixture-preview-heading"><b>Mixed spectrum · MS3 #{ms3_index}'
                f' · CE {escape(label)} · Scan {escape(scan)}</b></div>'
                f'<div class="mixture-preview-body">{result}{reference}</div></div>')
        if combined and fit_mixture and observed and len(peaks_list) < mixture_min_candidates:
            previews.append('<p class="mixture-preview-heading">'
                            f'Mixture skipped: requires at least {mixture_min_candidates} candidates.</p>')
        preview = ''.join(previews)
        sections.append(
            f'<section class="bag-section" data-bag-index="{bag_index}">'
            f'<div class="bag-heading"><b>Bag {bag_index} · m/z {mz:.5f}</b>'
            f'<span>{len(spectra)} candidates'
            + ((f' · {len(observed)} of {len(all_observed)} measured MS3' if requested
                else f' · {len(observed)} measured MS3') if combined else '')
            + f'</span></div>{preview}{row}</section>'
        )
    if not sections:
        if combined and only_with_ms3:
            message = ("No bags with MS3 spectra at selected collision energies."
                       if requested else "No bags with linked MS3 spectra.")
            return HTML(f'<p>{message}</p>')
        return HTML('<p>No spectrum bags to display.</p>')
    style = """
    <style>
    .magma-spectrum-gallery {font-family:system-ui,sans-serif;max-width:100%;color:#172033}
    .magma-spectrum-gallery .bag-section {margin:0 0 22px;padding:12px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px}
    .magma-spectrum-gallery .bag-heading {display:flex;gap:16px;align-items:baseline;margin:0 0 10px;font-size:14px}
    .magma-spectrum-gallery .bag-heading span {color:#64748b;font-size:12px}
    .magma-spectrum-gallery .bag-row {display:flex;flex-wrap:nowrap;gap:12px;overflow-x:auto;align-items:flex-start;padding:2px 2px 12px}
    .magma-spectrum-gallery .bag-scroll {overflow-x:auto;padding:2px 2px 12px}
    .magma-spectrum-gallery .bag-grid {display:grid;gap:12px;width:max-content;align-items:start}
    .magma-spectrum-gallery .ms3-row-heading {grid-column:1 / -1;font-size:12px;color:#0f766e;font-weight:600;padding:6px 0 0}
    .magma-spectrum-gallery .ms3-card {box-sizing:content-box;width:var(--panel-width);overflow:hidden;border:1px solid #99d5c8;border-radius:9px;background:white}
    .magma-spectrum-gallery .mixture-card {box-sizing:content-box;width:var(--panel-width);overflow:hidden;border:1px solid #c4b5fd;border-radius:9px;background:white}
    .magma-spectrum-gallery .mixture-card .candidate-heading {background:#f5f3ff;color:#5b21b6}
    .magma-spectrum-gallery .mixture-card .candidate-heading strong {font-size:12px;white-space:nowrap}
    .magma-spectrum-gallery .mixture-weight {box-sizing:border-box;height:32px;line-height:32px;padding:0 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center;font-size:12px;color:#5b21b6;background:#f5f3ff}
    .magma-spectrum-gallery .mixture-preview {max-width:100%;margin:0 0 20px}
    .magma-spectrum-gallery .mixture-preview-heading {font-size:13px;color:#5b21b6;margin:0 0 8px;overflow-wrap:anywhere}
    .magma-spectrum-gallery .mixture-preview-body {display:flex;flex-wrap:wrap;gap:12px;align-items:start}
    .magma-spectrum-gallery .mixture-preview-body > article {flex:0 0 var(--panel-width)}
    .magma-spectrum-gallery .comparison-card {box-sizing:content-box;width:var(--panel-width);overflow:hidden;border:1px solid #dbe3ef;border-radius:9px;background:white}
    .magma-spectrum-gallery .ms3-heading {box-sizing:border-box;height:42px;padding:0 12px;display:flex;gap:8px;align-items:center;justify-content:space-between;font-size:13px;background:#e8f7f2;color:#115e59;overflow:auto}
    .magma-spectrum-gallery .ms3-heading span {font-size:11px;white-space:nowrap}
    .magma-spectrum-gallery .candidate-padding {box-sizing:border-box;width:calc(var(--panel-width) + 2px);height:calc(var(--fragment-height) + var(--spectrum-height) + 113px);display:flex;align-items:center;justify-content:center;color:#64748b;font-size:13px}
    .magma-spectrum-gallery .ms3-padding {box-sizing:border-box;width:calc(var(--panel-width) + 2px);height:calc(var(--spectrum-height) + 45px);display:flex;align-items:center;justify-content:center;color:#64748b;font-size:13px}
    .magma-spectrum-gallery .candidate-card {box-sizing:content-box;flex:0 0 var(--panel-width);width:var(--panel-width);overflow:hidden;background:white;border:1px solid #dbe3ef;border-radius:9px;box-shadow:0 2px 5px #0f172a08}
    .magma-spectrum-gallery .candidate-heading {box-sizing:border-box;height:42px;padding:0 12px;display:flex;align-items:center;justify-content:space-between;font-size:13px;background:#eff6ff;border-bottom:1px solid #dbeafe}
    .magma-spectrum-gallery .candidate-heading span {font-size:11px;color:#64748b}
    .magma-spectrum-gallery .candidate-heading-scored {height:64px;display:grid;grid-template-columns:minmax(0,1fr) auto;grid-template-rows:22px 18px;gap:0 8px;padding:11px 12px}
    .magma-spectrum-gallery .candidate-heading-scored b {min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .magma-spectrum-gallery .candidate-heading-scored strong {font-size:12px;color:#115e59;white-space:nowrap}
    .magma-spectrum-gallery .cosine-reference {grid-column:1 / -1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .magma-spectrum-gallery .fragment-panel {box-sizing:content-box;width:var(--panel-width);height:var(--fragment-height);overflow:hidden;background:white}
    .magma-spectrum-gallery .formula-caption {box-sizing:border-box;height:46px;line-height:46px;padding:0 10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:center;font-size:13px;color:#334155;background:white}
    .magma-spectrum-gallery .spectrum-panel {box-sizing:content-box;width:var(--panel-width);height:var(--spectrum-height);border-top:1px solid #e2e8f0;overflow:hidden;background:white}
    .magma-spectrum-gallery .mirror-panel {box-sizing:content-box;width:var(--panel-width);height:var(--mirror-height);border-top:1px solid #e2e8f0;overflow:hidden;background:white}
    .magma-spectrum-gallery .graph-placeholder {box-sizing:border-box;height:100%;padding:20px;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:8px;text-align:center;color:#9b3535;overflow:auto}
    .magma-spectrum-gallery .graph-placeholder small {overflow-wrap:anywhere}
    .magma-spectrum-gallery .empty-bag {padding:18px;color:#64748b;font-size:13px}
    .magma-spectrum-gallery .gallery-note {font-size:12px;color:#64748b;margin:0 0 12px}
    </style>
    """
    note = ("Each fragment sits above the spectrum with the same candidate index. " if combined else "")
    note += ("Intensities are scaled to each spectrum's highest peak (100%). " if normalize
             else "Original intensities are shown on a shared scale within each bag. ")
    if combined:
        note += ("Candidates are ordered left to right by best cosine against the displayed MS3 spectra; "
                 "original candidate numbers are retained. Ties keep their original order; unavailable "
                 "predictions sort last when MS3 is usable. Without usable MS3, the original order is kept. "
                 "Each selected MS3 has its own comparison row beneath the same candidate columns. "
                 "The score caption identifies the best-matching MS3; CE units are as recorded. ")
        if mirror:
            note += ("Mirror view: prediction above zero, measured MS3 below zero on one shared m/z axis "
                     "with equal positive/negative intensity scales. A matching pattern reflects across zero. ")
        if color_matches:
            green_label = "shared m/z" if strict is None else "m/z and relative intensity match"
            note += (f'<span style="color:#16a34a;font-weight:600">Green: {green_label}.</span> '
                     '<span style="color:#dc2626;font-weight:600">Red: no qualifying counterpart.</span> '
                     'Mass tolerance: max(0.005 Da, 20 ppm) on nonzero peaks. ')
            if strict is not None:
                note += (f"Intensity tolerance: ±{strict:g} percentage points, with each spectrum's "
                         'strongest peak scaled to 100, even when plotting original intensities. ')
            note += 'Missing predictions leave the MS3 colors neutral. '
        else:
            note += 'Peak matching colors are off. '
        if requested:
            note += "MS3 collision-energy filter: " + ", ".join(f"{value:g}" for value in requested) + ". "
        if only_with_ms3:
            note += "Only bags with displayed MS3 spectra are shown; original bag indices are retained. "
    if combined and fit_mixture:
        note += (f"Mixture experiment: at least {mixture_min_candidates} usable predictions; "
                 "weights in [0, 1] sum to 1 and are fitted separately to each MS3. "
                 "Mixed spectra are shown above the candidate rows. "
                 "Fit cosine uses a fixed aligned mass axis; weights are not probabilities. ")
    note += "Scroll each bag sideways to see every candidate and observation."
    return HTML(
        style + f'<div class="magma-spectrum-gallery" style="--panel-width:{width}px;'
        f'--fragment-height:{fragment_height}px;--spectrum-height:{height}px;--mirror-height:{2 * height}px">'
        f'<p class="gallery-note">{note}</p>' + ''.join(sections) + '</div>'
    )


def plot_spectrum_bags(spectrum_bags, *, size=(320, 230), normalize=True):
    """Plot (m/z, predicted_spectra, ms3_spectra) bags as horizontal rows.

    Accept one bag or a list of bags. Each spectrum may be an N x 2 peak array,
    a repo Spectrum, or None. Missing/empty spectra retain their full-size slot.
    normalize=True displays base-peak percentages on a copy; False displays the
    original intensities. Axes are shared within each bag. Returns notebook HTML.
    """
    bags = _gallery_bag_list(spectrum_bags, "spectrum_bags")
    return _render_spectrum_gallery(bags, size, normalize)


def plot_fragments_with_spectra(candidate_bags, spectrum_bags, *, size=(320, 230),
                               fragment_height=210, normalize=True,
                               show_atom_indices=False, ms3_collision_energies=None,
                               only_with_ms3=False, fit_mixture=False,
                               mixture_min_candidates=2, color_matches=True, strict=None,
                               mirror=False):
    """Plot fragments above predictions, with a measured MS3 row below each bag.

    Supply exactly the bags used to generate spectrum_bags, in the same order.
    Accept either two single bags or two bag lists. Reject different bag counts,
    m/z values, or candidate counts instead of truncating or guessing a pairing.
    Retain None slots in the supplied data. Within-bag reordering cannot be
    detected from peak arrays, so do not sort input fragments/spectra separately.

    size sets (column width, spectrum height); fragment_height reserves equal
    space for every molecule. Fixed caption/plot padding keeps baselines aligned,
    including malformed graphs and long formulas. Each selected candidate_bag[2]
    observation gets a comparison row, labeled by collision energy and scan.
    mirror=True puts prediction above zero and MS3 below zero in one plot for
    every selected observation, including fitted mixtures. Shared m/z and
    symmetric y limits make matching peaks reflect across zero. The mirror
    image uses twice size[1] in height. normalize=True scales each half's
    strongest peak independently to 100%; False retains original intensities
    on the same symmetric scale. Missing predictions leave the upper half
    labeled unavailable. Bags without MS3 keep their prediction-only plots.
    mirror=False (default) uses the original separate plots.
    Repeat its MS3 below every candidate; repeat predictions for additional MS3
    rows. color_matches=True (default) colors qualifying pairs green and the
    remaining peaks red; False uses uniform prediction/MS3 colors throughout,
    including fitted mixtures. The m/z tolerance is max(0.005 Da, 20 ppm).
    strict=None (default) disables the intensity check. strict=x also requires
    the SAME peak pair's base-peak-normalized intensities (0..100 independently
    per spectrum) to differ by no more than x percentage points, either way.
    For example, strict=10 accepts 50 vs 40..60; strict=0 requires equality.
    This check uses relative intensities even with normalize=False, and accepts
    any qualifying counterpart. Both options affect colors only: cosine scores,
    candidate order and mixture fitting are unchanged. Missing spectra stay
    neutral. Source spectra remain unchanged.
    ms3_collision_energies is a list of recorded CE values; [] or None shows all,
    including unknown energies. Nonempty lists match within 1e-6 without unit
    conversion. A multi-energy observation matches any listed energy and stays
    intact. only_with_ms3=True skips bags with no observations AFTER that filter;
    original bag indices and candidate/prediction pairing are preserved.

    Within each displayed bag, pairs are sorted by their BEST cosine against
    the displayed MS3 spectra. Titles show the score and its MS3 reference.
    Reuse the repo's raw-intensity cosine and one-to-one tolerance matching
    (max(0.005 Da, 20 ppm)). Predictions are scored as supplied (Glacier already
    prepares its ensembles); observed peaks are sum-merged with no threshold or
    top-k cutoff. Ties retain input order. None predictions show n/a and sort last
    when there is a usable MS3; empty predictions score zero. Without any MS3
    containing positive peaks, scores are n/a and the original order is kept.
    Candidate numbers always refer to the original input positions.
    Comparison rows share candidate columns, axes and one horizontal scroll area.
    Sorting/filtering only affect this view; source bags and spectra never change.

    fit_mixture=True shows each fitted spectrum and its measured MS3 above the
    scrolling candidate rows, plus weights on the candidate cards. Require mixture_min_candidates usable
    predictions (default 2). Each prediction is normalized to unit total intensity;
    nonnegative least squares maximizes cosine on a fixed tolerance-aligned axis.
    Weights are rescaled to sum to 1, satisfying [0, 1] without restricting cosine.
    Fit cosine and the best single aligned score use that experimental axis;
    existing individual sparse cosine scores and sorting are unchanged. No peaks
    are truncated. Missing/empty predictions are excluded; empty/no-overlap
    targets and failed fits show a reason instead of a fabricated mixture.
    Weights describe this fit, not molecular probabilities or measured abundances.
    Returns notebook HTML.
    """
    import math

    fragments = _gallery_bag_list(candidate_bags, "candidate_bags")
    spectra = _gallery_bag_list(spectrum_bags, "spectrum_bags")
    if len(fragments) != len(spectra):
        raise ValueError(f"Bag counts differ: {len(fragments)} fragments vs {len(spectra)} spectra")
    for i, ((fragment_mz, candidates, _), (spectrum_mz, predictions, _)) in enumerate(zip(fragments, spectra)):
        if not math.isclose(fragment_mz, spectrum_mz, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f"Bag {i}: m/z differs ({fragment_mz} vs {spectrum_mz}); use matching bag order")
        if len(candidates) != len(predictions):
            raise ValueError(
                f"Bag {i}: {len(candidates)} fragments vs {len(predictions)} spectra; "
                "retain None for missing predictions"
            )
    return _render_spectrum_gallery(spectra, size, normalize, fragments,
                                     fragment_height, show_atom_indices,
                                     ms3_collision_energies=ms3_collision_energies,
                                     only_with_ms3=only_with_ms3,
                                     fit_mixture=fit_mixture,
                                     mixture_min_candidates=mixture_min_candidates,
                                     color_matches=color_matches, strict=strict, mirror=mirror)

