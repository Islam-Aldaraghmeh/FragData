"""Exercise shared notebook imports without executing data/model/demo cells."""
import importlib.util
import itertools
import json
import base64
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFINITION_CELLS = (
    "candidate-setup", "ms3-linkage", "magma-graph", "formula-filter",
    "glacier-prediction", "fragment-gallery-code", "spectrum-gallery-code",
)


@pytest.fixture
def notebook(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    from Spectrum_Classes import Spectrum, SpectrumForest, parse_spectrum_forest

    namespace = dict(json=json, Path=Path, np=np, Spectrum=Spectrum,
                     SpectrumForest=SpectrumForest,
                     parse_spectrum_forest=parse_spectrum_forest)
    cells = {cell["id"]: cell for cell in json.loads(
        (REPO_ROOT / "Experiment_1.ipynb").read_text())["cells"]}
    for cell_id in DEFINITION_CELLS:
        source = "".join(cells[cell_id]["source"])
        exec(compile(source, f"Experiment_1.ipynb:{cell_id}", "exec"), namespace)
    from magma_workflow import candidates, plotting
    monkeypatch.setattr(candidates, "_source_spectra", None)
    monkeypatch.setattr(candidates, "_source_by_smiles", None)
    # Gallery functions now resolve helpers in their importable module. Keep
    # patches pointed at those real globals, rather than a copied namespace.
    for name, value in namespace.items():
        if name != "__builtins__":
            monkeypatch.setitem(plotting.__dict__, name, value)
    monkeypatch.setitem(plotting.__dict__, "_asset_path", candidates._asset_path)
    return plotting.__dict__


def make_spectrum(notebook, scan, level=2, *, feature="feature-A", ce=40,
                  parent_ce=40, isolated=100.007, peaks=None):
    metadata = {
        "feature_id": feature, "scan_number": scan,
        "usi": [f"mzspec:fixture:{feature}:{scan}"],
        "merge_type": "SINGLE_BEST_SCAN", "collision_energy": ce,
        "charge": 1, "adduct": "[M+H]+", "dataset_id": "fixture",
    }
    if level >= 3:
        metadata["msn_precursor_mzs"] = [200.0, isolated]
        metadata["msn_collision_energies"] = [parent_ce, ce]
    if level == 4:
        metadata["msn_precursor_mzs"] += [50.0]
        metadata["msn_collision_energies"] = [parent_ce, 20, ce]
    if peaks is None:
        peaks = [[100.0, 20.0], [100.008, 30.0], [150.0, 10.0]] if level == 2 else [[50.0, 5.0]]
    # Real MSnLib uses the original root precursor here even on MS3 records.
    return notebook["Spectrum"]("CC", 200.0, level,
                                np.asarray(peaks, dtype=float).reshape(-1, 2), metadata)


@pytest.fixture
def acquisition(notebook):
    root = make_spectrum(notebook, 10)
    children = [make_spectrum(notebook, 20, 3, ce=20),
                make_spectrum(notebook, 21, 3, ce=20),
                make_spectrum(notebook, 22, 3, ce=60, isolated=150.0)]
    unrelated = [
        make_spectrum(notebook, 11, ce=30),
        make_spectrum(notebook, 23, 3, ce=20, parent_ce=30),
        make_spectrum(notebook, 10, feature="feature-B"),
        make_spectrum(notebook, 20, 3, feature="feature-B", ce=20),
        make_spectrum(notebook, 5, 3, ce=20),
        make_spectrum(notebook, 30, 4, ce=60),
        make_spectrum(notebook, 25, 3, ce=20, isolated=170),
    ]
    return root, children, [root, *children, *unrelated]


def make_candidate(notebook, candidate_id, mz, *, formula="CH3", peak_index=None):
    metadata = {} if peak_index is None else {"peak_index": peak_index}
    return notebook["FragmentCandidate"](
        candidate_id, {"atoms": [{"element": "C", "aromatic": False}], "bonds": []},
        formula, 1, mz, metadata=metadata)


def scans(spectra):
    return [spectrum.metadata["scan_number"] for spectrum in spectra]


def test_candidate_observations_are_independent_and_excluded_from_model_identity(notebook):
    first = make_candidate(notebook, "same", 100.0)
    second = make_candidate(notebook, "same", 100.0)
    original_repr, original_input = repr(first), first.stripped
    assert first.ms3_spectra is not second.ms3_spectra
    first.ms3_spectra.append(make_spectrum(notebook, 20, 3, ce=20))
    assert second.ms3_spectra == []
    # Comparing these distinct Spectrum instances directly would compare NumPy
    # arrays and raise; runtime observations must not enter candidate equality.
    second.ms3_spectra.append(make_spectrum(notebook, 20, 3, ce=20))
    assert first == second
    assert repr(first) == original_repr
    assert first.stripped == original_input
    assert set(first.stripped) == {"atoms", "bonds", "ionic_formula", "charge", "precursor_mz"}


def test_direct_children_obey_ancestry_and_nearest_peak(notebook, acquisition):
    root, children, source = acquisition
    result = notebook["find_ms3_children"](root, source)
    assert not result.get(0)
    assert set(scans(result[1])) == {20, 21}
    assert scans(result[2]) == [22]
    assert sum(map(len, result.values())) == 3
    assert all(child.precursor_mz == root.precursor_mz for child in children)
    assert all(child.ms_level == 3 for values in result.values() for child in values)


def test_prebuilt_forest_and_default_notebook_lookup(notebook, acquisition):
    root, _, source = acquisition
    forest = notebook["parse_spectrum_forest"](source)
    explicit = notebook["find_ms3_children"](root, forest=forest)
    assert set(scans(explicit[1])) == {20, 21}
    from magma_workflow.candidates import set_spectrum_source
    set_spectrum_source(spectra_by_smiles={root.smiles: source})
    implicit = notebook["find_ms3_children"](root)
    assert {i: scans(values) for i, values in implicit.items()} == {
        i: scans(values) for i, values in explicit.items()}
    set_spectrum_source(source)
    fallback = notebook["find_ms3_children"](root)
    assert sum(map(len, fallback.values())) == 3


def test_missing_source_is_distinct_from_no_children(notebook, acquisition):
    root, _, _ = acquisition
    with pytest.raises(ValueError, match="(?i)(spectra|source|forest|dataset)"):
        notebook["find_ms3_children"](root)
    assert not any(notebook["find_ms3_children"](root, []).values())
    assert not any(notebook["find_ms3_children"](root, [root]).values())
    empty = make_spectrum(notebook, 10, peaks=[])
    assert not any(notebook["find_ms3_children"](empty, [empty]).values())


def test_attach_preserves_reordered_bag_identity_and_fragment_links(notebook, acquisition):
    root, _, source = acquisition
    candidates = [make_candidate(notebook, "first", 100.008, peak_index=1),
                  make_candidate(notebook, "second", 100.008, peak_index=1)]
    high = make_candidate(notebook, "high", 150.0, peak_index=2)
    original = [(150.0, [high]), (100.008, candidates), (100.0, [])]
    attached = notebook["attach_ms3_spectra"](root, original, source)
    assert [bag[0] for bag in attached] == [150.0, 100.008, 100.0]
    assert scans(attached[0][2]) == [22]
    assert set(scans(attached[1][2])) == {20, 21}
    assert attached[2][2] == []
    for _, fragments, observations in attached:
        for fragment in fragments:
            assert fragment.ms3_spectra is observations
    assert all(len(bag) == 2 for bag in original)
    filtered = notebook["filter_same_formula_bags"](attached)
    assert [bag[0] for bag in filtered] == [150.0, 100.008]
    assert filtered[0][2] is attached[0][2]
    assert filtered[1][2] is attached[1][2]


def test_formula_filter_excludes_mixed_bag_without_dropping_observations(notebook):
    first = make_candidate(notebook, "first", 100.0, formula="CH3")
    same = make_candidate(notebook, "same", 100.0, formula={"H": 3, "C": 1})
    other = make_candidate(notebook, "other", 100.0, formula="CH4")
    observations = [make_spectrum(notebook, 20, 3, ce=20)]
    result = notebook["filter_same_formula_bags"](
        [(100.0, [first, same], observations),
         (101.0, [first, other], []), (102.0, [], [])])
    assert len(result) == 1
    assert result[0][1] == [first, same]
    assert result[0][2] is observations


def test_attach_rejects_conflicting_candidate_peak_indices(notebook, acquisition):
    root, _, source = acquisition
    conflicting = [make_candidate(notebook, "first", 100.008, peak_index=0),
                   make_candidate(notebook, "second", 100.008, peak_index=1)]
    with pytest.raises(ValueError, match="(?i)(conflict|peak_index|indices)"):
        notebook["attach_ms3_spectra"](root, [(100.008, conflicting)], source)


def test_prediction_preserves_triples_and_unsupported_slots(notebook, monkeypatch):
    pytest.importorskip("rdkit")
    from src.chemistry import fragment_adapter

    class PredictableAdapter:
        def __init__(self, glacier, max_states):
            self.glacier = glacier

        def predict(self, candidate, **kwargs):
            if candidate.candidate_id == "unsupported":
                raise ValueError("No represented states: regression fixture")
            peaks = np.array([[25.0, 0.5], [50.0, 1.0]])
            return SimpleNamespace(as_array=lambda: peaks)

    monkeypatch.setattr(fragment_adapter, "FragmentGlacierAdapter", PredictableAdapter)
    fragments = [make_candidate(notebook, "ok", 100.0),
                 make_candidate(notebook, "unsupported", 100.0)]
    observations = [make_spectrum(notebook, 20, 3, ce=20)]
    with pytest.warns(RuntimeWarning, match="candidate 1"):
        predicted = notebook["predict_candidate_bags"](
            object(), [(100.0, fragments, observations), (120.0, [], [])],
            collision_energy=30)
    assert [bag[0] for bag in predicted] == [100.0, 120.0]
    assert predicted[0][1][0].shape == (2, 2)
    assert predicted[0][1][1] is None
    assert predicted[0][2] is observations
    assert predicted[1][1] == predicted[1][2] == []


def test_combined_gallery_accepts_triples_and_shows_observed_energies(notebook):
    pytest.importorskip("rdkit")
    pytest.importorskip("matplotlib")
    pytest.importorskip("IPython")
    fragments = [make_candidate(notebook, "ok", 100.0),
                 make_candidate(notebook, "missing", 100.0)]
    observations = [make_spectrum(notebook, 20, 3, ce=20),
                    make_spectrum(notebook, 21, 3, ce=40),
                    make_spectrum(notebook, 22, 3, ce=60)]
    unknown_energy = make_spectrum(notebook, 23, 3)
    unknown_energy.metadata.pop("collision_energy")
    unknown_energy.metadata["msn_collision_energies"] = []
    observations.append(unknown_energy)
    before = [observation.peaks.copy() for observation in observations]
    peaks = np.array([[25.0, 1.0], [50.0, 0.2]])
    bag = (100.0, fragments, observations)
    predicted = (100.0, [peaks, None], observations)
    html = notebook["plot_fragments_with_spectra"](bag, predicted).data
    visible = re.sub(r"<[^>]*>", " ", html)
    assert "MS3" in visible or "MS³" in visible
    for energy in (20, 40, 60):
        assert re.search(rf"CE\s*(?:=|:)?\s*{energy}\b", visible)
    assert "CE unknown" in visible
    assert "Candidate 0" in html and "Candidate 1" in html
    assert "data:image/svg+xml;base64," in html
    for observation, original in zip(observations, before):
        np.testing.assert_array_equal(observation.peaks, original)
    assert notebook["plot_fragment_bags"](bag).data
    assert notebook["plot_spectrum_bags"](predicted).data


def test_legacy_pairs_and_missing_ms3_still_render(notebook):
    pytest.importorskip("rdkit")
    pytest.importorskip("matplotlib")
    pytest.importorskip("IPython")
    fragment = make_candidate(notebook, "one", 100.0)
    pair = (100.0, [fragment])
    predicted = (100.0, [None])
    assert notebook["plot_fragment_bags"](pair).data
    assert notebook["plot_spectrum_bags"](predicted).data
    assert notebook["plot_fragments_with_spectra"](pair, predicted).data
    assert notebook["plot_fragments_with_spectra"](
        (*pair, []), (*predicted, [])).data
    with pytest.raises(ValueError, match="(?i)(differ|matching|order)"):
        notebook["plot_fragments_with_spectra"](pair, (101.0, [None]))


@pytest.fixture
def fast_cosine_gallery(notebook, monkeypatch):
    """Keep real scoring/layout, replacing only expensive SVG drawing."""
    monkeypatch.setitem(
        notebook, "_fragment_graph_svg",
        lambda fragment, *_: f"<svg>{fragment.candidate_id}</svg>")
    monkeypatch.setitem(
        notebook, "_gallery_spectrum_svg",
        lambda peaks, *_, **__: f"<svg>{None if peaks is None else peaks.tolist()}</svg>")
    return notebook["plot_fragments_with_spectra"]


def cosine_cards(html):
    return [(int(index), contents) for index, contents in re.findall(
        r'<article class="candidate-card" data-candidate-index="(\d+)"[^>]*>(.*?)</article>',
        html, re.DOTALL)]


def test_gallery_cosine_sort_preserves_pairs_ties_and_inputs(notebook, fast_cosine_gallery):
    fragments = [make_candidate(notebook, f"fragment-{i}", 100.0) for i in range(6)]
    predictions = [np.array([[70.0, 1.0]]), np.array([[50.0, 1.0]]),
                   np.array([[50.0, 3.0], [60.0, 4.0]]), None,
                   np.empty((0, 2)), np.array([[50.0, 6.0], [60.0, 8.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=40,
                                 peaks=[[50.0, 3.0], [60.0, 4.0]])]
    original_predictions = [None if peaks is None else peaks.copy() for peaks in predictions]
    original_observed = observations[0].peaks.copy()
    fragment_ids = [id(fragment) for fragment in fragments]
    prediction_ids = [id(peaks) for peaks in predictions]
    candidate_bag, predicted_bag = (100.0, fragments, observations), (100.0, predictions, observations)

    cards = cosine_cards(fast_cosine_gallery(candidate_bag, predicted_bag).data)
    assert [index for index, _ in cards] == [2, 5, 1, 0, 4, 3]
    expected = {0: "0.000", 1: "0.600", 2: "1.000", 4: "0.000", 5: "1.000"}
    for index, card in cards:
        if index == 3:
            assert "Cosine n/a" in card
        else:
            assert f"Best cosine {expected[index]}" in card
        # The SVG payloads identify the original graph AND prediction at every
        # sorted position, catching either half being reordered independently.
        images = re.findall(r'data:image/svg\+xml;base64,([^"\s]+)', card)
        assert base64.b64decode(images[0]).decode() == f"<svg>fragment-{index}</svg>"
        peaks = original_predictions[index]
        assert base64.b64decode(images[1]).decode() == (
            f"<svg>{None if peaks is None else peaks.tolist()}</svg>")
    assert [id(fragment) for fragment in fragments] == fragment_ids
    assert [id(peaks) for peaks in predictions] == prediction_ids
    for actual, original in zip(predictions, original_predictions):
        if original is not None:
            np.testing.assert_array_equal(actual, original)
    np.testing.assert_array_equal(observations[0].peaks, original_observed)
    # Standalone spectrum plotting continues to use input order.
    standalone = notebook["plot_spectrum_bags"](predicted_bag).data
    assert [index for index, _ in cosine_cards(standalone)] == list(range(6))


def test_gallery_cosine_uses_best_displayed_ms3_and_full_caption(notebook, fast_cosine_gallery):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(3)]
    predictions = [np.array([[50.0, 1.0], [60.0, 2.0]]),
                   np.array([[50.0, 1.0]]), np.array([[70.0, 1.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=20, peaks=[[50.0, 1.0]]),
                    make_spectrum(notebook, 40, 3, ce=40, peaks=[[60.0, 1.0]]),
                    make_spectrum(notebook, [70, 71], 3, ce=[20, 60], peaks=[[70.0, 1.0]])]
    candidates, spectra = (100.0, fragments, observations), (100.0, predictions, observations)
    cards = cosine_cards(fast_cosine_gallery(candidates, spectra).data)
    assert [index for index, _ in cards] == [1, 2, 0]
    by_index = dict(cards)
    assert "Best cosine 0.894" in by_index[0]
    assert "MS3 #1 · CE 40 · Scan 40" in by_index[0]
    assert "MS3 #0 · CE 20 · Scan 20" in by_index[1]
    assert "MS3 #2 · CE 20, 60 · Scan 70, 71" in by_index[2]

    filtered = fast_cosine_gallery(candidates, spectra, ms3_collision_energies=[40]).data
    cards = cosine_cards(filtered)
    assert [index for index, _ in cards] == [0, 1, 2]
    assert "Best cosine 0.894" in cards[0][1]
    # Reference is the index within the displayed MS3 row after CE filtering.
    assert "MS3 #0 · CE 40 · Scan 40" in cards[0][1]
    assert all("Best cosine 0.000" in card for _, card in cards[1:])
    # One selected observation repeats under all three candidate columns.
    assert filtered.count('class="ms3-card"') == len(fragments)
    assert re.findall(r'data-ms3-index="(\d+)"', filtered) == ["0"] * len(fragments)
    merged_only = cosine_cards(fast_cosine_gallery(
        candidates, spectra, ms3_collision_energies=[60]).data)
    assert [index for index, _ in merged_only] == [2, 0, 1]
    assert "MS3 #0 · CE 20, 60 · Scan 70, 71" in merged_only[0][1]
    assert scans(observations) == [20, 40, [70, 71]]


@pytest.mark.parametrize("target_peaks", [None, [], [[50.0, 0.0]]])
def test_gallery_cosine_keeps_order_without_usable_ms3(
        notebook, fast_cosine_gallery, target_peaks):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(3)]
    predictions = [None, np.array([[50.0, 1.0]]), np.empty((0, 2))]
    observations = ([] if target_peaks is None else
                    [make_spectrum(notebook, 20, 3, peaks=target_peaks)])
    cards = cosine_cards(fast_cosine_gallery(
        (100.0, fragments, observations), (100.0, predictions, observations)).data)
    assert [index for index, _ in cards] == [0, 1, 2]
    assert all("Cosine n/a" in card for _, card in cards)


@pytest.mark.parametrize("target_peaks, expected", [
    ([[50.004, 3.0], [600.010, 4.0]], "1.000"),
    ([[50.006, 3.0], [600.013, 4.0]], "0.000"),
])
def test_gallery_cosine_uses_absolute_and_ppm_tolerance(
        notebook, fast_cosine_gallery, target_peaks, expected):
    fragment = make_candidate(notebook, "tolerance", 700.0)
    observations = [make_spectrum(notebook, 20, 3, peaks=target_peaks)]
    html = fast_cosine_gallery(
        (700.0, [fragment], observations),
        (700.0, [np.array([[50.0, 3.0], [600.0, 4.0]])], observations)).data
    assert f"Best cosine {expected}" in cosine_cards(html)[0][1]


def test_gallery_cosine_sums_close_target_peaks(notebook, fast_cosine_gallery):
    fragment = make_candidate(notebook, "target-summing", 100.0)
    observations = [make_spectrum(notebook, 20, 3,
                                 peaks=[[50.0, 3.0], [50.001, 4.0], [60.0, 7.0]])]
    html = fast_cosine_gallery(
        (100.0, [fragment], observations),
        (100.0, [np.array([[50.0, 1.0], [60.0, 1.0]])], observations)).data
    assert "Best cosine 1.000" in cosine_cards(html)[0][1]


def test_gallery_cosine_retains_full_prediction_and_target_support(notebook, fast_cosine_gallery):
    fragment = make_candidate(notebook, "full-support", 250.0)
    masses = np.arange(50.0, 200.0)
    target = np.column_stack([masses, np.ones(150)])
    prediction = np.column_stack([masses, np.r_[np.ones(100), np.full(50, 2.0)]])
    observations = [make_spectrum(notebook, 20, 3, peaks=target)]
    html = fast_cosine_gallery(
        (250.0, [fragment], observations), (250.0, [prediction], observations)).data
    # All 150 peaks contribute: dot=200, ||p||²=300, ||t||²=150.
    expected = 200 / np.sqrt(300 * 150)
    assert f"Best cosine {expected:.3f}" in cosine_cards(html)[0][1]


def test_gallery_overlap_colors_match_raw_positive_peaks_without_mutation(notebook):
    green, red = "#16a34a", "#dc2626"
    prediction = np.array([[600.0, 4.0], [40.0, 0.0], [50.0, 3.0],
                           [50.002, 2.0], [100.0, 1.0], [80.0, 2.0]])
    observed = np.array([[80.0, 0.0], [600.011, 2.0], [50.004, 2.0],
                         [100.006, 3.0], [40.0, 1.0]])
    originals = prediction.copy(), observed.copy()
    colors = notebook["_gallery_overlap_colors"](prediction, observed)
    # Match 600 within 20 ppm and both nearby 50 peaks within 0.005 Da.
    # Zero-intensity rows keep their positions but cannot create matches.
    assert list(colors[0]) == [green, red, green, green, red, red]
    assert list(colors[1]) == [red, green, green, red, red]
    for actual, original in zip((prediction, observed), originals):
        np.testing.assert_array_equal(actual, original)

    assert notebook["_gallery_overlap_colors"](None, observed) == (None, None)
    assert notebook["_gallery_overlap_colors"](prediction, None) == (None, None)
    pred_colors, observed_colors = notebook["_gallery_overlap_colors"](
        np.empty((0, 2)), observed)
    assert list(pred_colors) == []
    assert list(observed_colors) == [red] * len(observed)


def test_gallery_strict_colors_use_inclusive_normalized_intensity_bounds(notebook):
    green, red = "#16a34a", "#dc2626"
    prediction = np.array([[40, 100], [50, 50], [60, 50], [70, 50], [80, 50], [90, 10]],
                          dtype=float)
    observed = np.array([[40, 20], [50, 12], [60, 8], [70, 12.00002],
                         [80, 7.99998], [90.006, 2]], dtype=float)
    originals = prediction.copy(), observed.copy()
    colors = notebook["_gallery_overlap_colors"]

    # Independently normalize to 100: +/-10 percentage points passes, but a
    # slightly larger difference fails in either direction. Equal intensity
    # cannot rescue the final peak, whose mass lies outside tolerance.
    expected = [green, green, green, red, red, red]
    assert colors(prediction, observed, strict=10) == (expected, expected)
    loose = [green, green, green, green, green, red]
    assert colors(prediction, observed, strict=None) == (loose, loose)
    exact = [green, red, red, red, red, red]
    assert colors(prediction, observed, strict=0) == (exact, exact)
    for actual, original in zip((prediction, observed), originals):
        np.testing.assert_array_equal(actual, original)


def test_gallery_strict_colors_require_one_counterpart_and_ignore_scale_and_zeros(notebook):
    green, red = "#16a34a", "#dc2626"
    prediction = np.array([[100, 5], [200, 10], [300, 0]], dtype=float)
    observed = np.array([[100.004, 9], [100.006, 5], [200, 10], [300, 5]], dtype=float)
    colors = notebook["_gallery_overlap_colors"]
    # Neither observation near 100 passes both conditions: the first has the
    # right mass, the second has the right intensity. They must not be combined.
    expected = ([red, green, red], [red, red, green, red])
    assert colors(prediction, observed, strict=1) == expected
    assert colors(prediction * [1, 0.0001], observed * [1, 3000], strict=1) == expected
    assert colors(np.array([[100, 0.0]]), np.array([[100, 0.0]]), strict=0) == ([red], [red])
    assert colors(np.empty((0, 2)), observed, strict=0) == ([], [red] * len(observed))
    assert colors(np.array([[50, 0.3], [100, 1.0]]),
                  np.array([[50, 3.0], [100, 10.0]]), strict=0) == ([green, green], [green, green])
    assert colors(prediction, observed, color_matches=False, strict=0) == (None, None)


def test_gallery_color_options_cover_all_ms3_and_mixture_plots_without_changing_fit(
        notebook, fast_cosine_gallery, monkeypatch):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(2)]
    predictions = [np.array([[50.0, 1.0]]), np.array([[60.0, 1.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=20,
                                 peaks=[[50.0, 0.25], [60.0, 0.75]]),
                    make_spectrum(notebook, 40, 3, ce=40,
                                 peaks=[[50.0, 0.8], [60.0, 0.2]])]
    inputs = [*predictions, *(item.peaks for item in observations)]
    originals = [peaks.copy() for peaks in inputs]
    rendered = []

    def capture(peaks, size, x_max, y_max, normalize, *args, peak_colors=None, **kwargs):
        rendered.append((peaks.copy(), normalize, peak_colors))
        return "<svg></svg>"

    monkeypatch.setitem(notebook, "_gallery_spectrum_svg", capture)
    bags = (100.0, fragments, observations), (100.0, predictions, observations)

    def draw(**kwargs):
        rendered.clear()
        html = fast_cosine_gallery(*bags, fit_mixture=True, **kwargs).data
        # Two candidates x two observations x both spectra, plus two mixed
        # spectra and their measured references: every path must honor kwargs.
        assert len(rendered) == 12
        assert html.count('class="mixture-preview"') == 2
        return html, list(rendered)

    baseline_html, baseline = draw()
    strict_html, strict = draw(strict=10, normalize=False)
    _, normalized_strict = draw(strict=10, normalize=True)
    disabled_html, disabled = draw(color_matches=False, strict=10)
    assert all(colors is None for _, _, colors in disabled)
    assert any(a[2] != b[2] for a, b in zip(baseline, strict))
    assert [item[2] for item in strict] == [item[2] for item in normalized_strict]
    assert all(not normalize for _, normalize, _ in strict)

    # Color/intensity settings change neither candidate order, cosine labels,
    # fitted weights, nor any plotted mass/intensity values.
    def scores_and_weights(html):
        return ([index for index, _ in cosine_cards(html)],
                re.findall(r'(?:Best cosine|Cosine) [0-9.]+', html),
                re.findall(r'data-weight=["\']([^"\']+)', html))

    assert scores_and_weights(baseline_html) == scores_and_weights(strict_html)
    assert scores_and_weights(baseline_html) == scores_and_weights(disabled_html)
    for before, strict_after, disabled_after in zip(baseline, strict, disabled):
        np.testing.assert_array_equal(before[0], strict_after[0])
        np.testing.assert_array_equal(before[0], disabled_after[0])
    for actual, original in zip(inputs, originals):
        np.testing.assert_array_equal(actual, original)


def test_gallery_color_option_validation(notebook, fast_cosine_gallery):
    candidate = make_candidate(notebook, "validation", 100.0)
    bags = (100.0, [candidate], []), (100.0, [np.array([[50.0, 1.0]])], [])
    for invalid in (True, "10", -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="strict"):
            fast_cosine_gallery(*bags, strict=invalid)
    for invalid in (0, 1, "false", None):
        with pytest.raises(ValueError, match="color_matches"):
            fast_cosine_gallery(*bags, color_matches=invalid)


def test_gallery_spectrum_svg_draws_requested_peak_colors(notebook):
    pytest.importorskip("matplotlib")
    svg = notebook["_gallery_spectrum_svg"](
        np.array([[25.0, 1.0], [50.0, 2.0]]), (320, 230), 105, 105, True,
        peak_colors=["#16a34a", "#dc2626"])
    assert "stroke: #16a34a" in svg
    assert "stroke: #dc2626" in svg


def test_gallery_repeated_ms3_is_colored_for_each_candidate_without_changing_scores(
        notebook, fast_cosine_gallery, monkeypatch):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(3)]
    predictions = [np.array([[50.0, 1.0]]), np.array([[60.0, 1.0]]), None]
    observed = make_spectrum(notebook, 20, 3, ce=40,
                             peaks=[[50.0, 3.0], [60.0, 4.0]])
    original = observed.peaks.copy()
    expected_scores = notebook["_gallery_best_ms3_cosines"](predictions, [observed.peaks])
    rendered = []

    def capture(peaks, *args, peak_colors=None, **kwargs):
        colors = None if peak_colors is None else list(peak_colors)
        rendered.append((None if peaks is None else peaks.copy(), colors))
        return "<svg></svg>"

    monkeypatch.setitem(notebook, "_gallery_spectrum_svg", capture)
    html = fast_cosine_gallery(
        (100.0, fragments, [observed]), (100.0, predictions, [observed])).data
    cards = cosine_cards(html)
    assert [index for index, _ in cards] == [1, 0, 2]
    assert "Best cosine 0.800" in cards[0][1]
    assert "Best cosine 0.600" in cards[1][1]
    assert "Cosine n/a" in cards[2][1]
    assert html.count('class="ms3-card"') == 3

    # The observation is drawn once per sorted column: its overlap colors must
    # differ even though every copy uses exactly the same measured peak data.
    ms3_colors = [colors for peaks, colors in rendered
                  if peaks is not None and peaks.shape == original.shape
                  and np.array_equal(peaks, original)]
    assert ms3_colors == [["#dc2626", "#16a34a"],
                          ["#16a34a", "#dc2626"], None]
    prediction_colors = [colors for peaks, colors in rendered
                         if peaks is not None and len(peaks) == 1]
    assert prediction_colors == [["#16a34a"], ["#16a34a"]]
    assert notebook["_gallery_best_ms3_cosines"](predictions, [observed.peaks]) == expected_scores
    np.testing.assert_array_equal(observed.peaks, original)


def test_gallery_compares_each_selected_ms3_independently(
        notebook, fast_cosine_gallery, monkeypatch):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(2)]
    predictions = [np.array([[50.0, 1.0], [70.0, 1.0]]),
                   np.array([[60.0, 1.0], [80.0, 1.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=20,
                                 peaks=[[50.0, 2.0], [60.0, 1.0]]),
                    make_spectrum(notebook, 60, 3, ce=60,
                                 peaks=[[70.0, 1.0], [80.0, 2.0]])]
    rendered = []

    def capture(peaks, *args, peak_colors=None, **kwargs):
        rendered.append((peaks.copy(), None if peak_colors is None else list(peak_colors)))
        return "<svg></svg>"

    monkeypatch.setitem(notebook, "_gallery_spectrum_svg", capture)
    candidates, spectra = (100.0, fragments, observations), (100.0, predictions, observations)
    html = fast_cosine_gallery(candidates, spectra).data
    assert [index for index, _ in cosine_cards(html)] == [0, 1]
    assert html.count('class="ms3-card"') == 4
    assert re.findall(r'data-ms3-index="(\d+)"', html) == ["0", "0", "1", "1"]
    for original in [*predictions, *(observation.peaks for observation in observations)]:
        colors = [colors for peaks, colors in rendered if np.array_equal(peaks, original)]
        assert colors == [["#16a34a", "#dc2626"], ["#dc2626", "#16a34a"]]

    rendered.clear()
    filtered = fast_cosine_gallery(candidates, spectra, ms3_collision_energies=[60]).data
    assert [index for index, _ in cosine_cards(filtered)] == [1, 0]
    assert filtered.count('class="ms3-card"') == 2
    assert not any(np.array_equal(peaks, observations[0].peaks) for peaks, _ in rendered)


def test_gallery_mixture_is_opt_in(notebook, fast_cosine_gallery, monkeypatch):
    from src.spectra import mixture

    def unexpected_fit(*args, **kwargs):
        pytest.fail("The default gallery must not run the experimental mixture fit")

    monkeypatch.setattr(mixture, "fit_spectral_mixture", unexpected_fit)
    if "fit_spectral_mixture" in notebook:
        monkeypatch.setitem(notebook, "fit_spectral_mixture", unexpected_fit)
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(2)]
    predictions = [np.array([[50.0, 1.0]]), np.array([[60.0, 1.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=20,
                                 peaks=[[50.0, 0.25], [60.0, 0.75]])]
    candidates, spectra = (100.0, fragments, observations), (100.0, predictions, observations)
    default = fast_cosine_gallery(candidates, spectra).data
    explicit = fast_cosine_gallery(candidates, spectra, fit_mixture=False).data
    assert default == explicit
    assert [index for index, _ in cosine_cards(default)] == [1, 0]
    assert 'class="mixture-card"' not in default
    assert 'class="mixture-weight"' not in default


def test_gallery_mixture_shows_each_ms3_fit_without_changing_candidates(
        notebook, fast_cosine_gallery):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(2)]
    predictions = [np.array([[50.0, 1.0]]), np.array([[60.0, 1.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=20,
                                 peaks=[[50.0, 0.25], [60.0, 0.75]]),
                    make_spectrum(notebook, 40, 3, ce=40,
                                 peaks=[[50.0, 0.8], [60.0, 0.2]])]
    originals = [peaks.copy() for peaks in predictions]
    observed_originals = [spectrum.peaks.copy() for spectrum in observations]
    candidates, spectra = (100.0, fragments, observations), (100.0, predictions, observations)
    original_cards = cosine_cards(fast_cosine_gallery(candidates, spectra).data)
    html = fast_cosine_gallery(candidates, spectra, fit_mixture=True).data
    fitted_cards = cosine_cards(html)
    assert [index for index, _ in fitted_cards] == [index for index, _ in original_cards]

    # Coefficients stay attached to input candidate indices despite cosine sorting.
    for index, contents in fitted_cards:
        weight = float(re.search(r'data-weight=["\']([^"\']+)', contents).group(1))
        assert weight == pytest.approx([0.25, 0.75][index], abs=1e-6)
    second_row = re.findall(
        r'<article class="comparison-card"[^>]*data-candidate-index="(\d+)"'
        r'[^>]*data-comparison-ms3-index="1"[^>]*>(.*?)</article>', html, re.DOTALL)
    assert len(second_row) == 2
    for index, contents in second_row:
        weight = float(re.search(r'data-weight=["\']([^"\']+)', contents).group(1))
        assert weight == pytest.approx([0.8, 0.2][int(index)], abs=1e-6)

    mixtures = re.findall(r'<article class="mixture-card"([^>]*)>(.*?)</article>',
                          html, re.DOTALL)
    assert len(mixtures) == 2
    for attributes, contents in mixtures:
        assert "data-candidate-index" not in attributes
        assert "1.000" in contents
        assert "data:image/svg+xml;base64," in contents
    # Fitted spectra are visible before the horizontally scrolling candidates.
    assert html.index('class="mixture-preview"') < html.index('class="bag-scroll"')
    assert html.count('class="mixture-preview"') == 2
    assert "grid-template-columns:repeat(2," in html
    assert html.count('data-mixture-reference="true"') == 2
    assert html.count('class="ms3-card"') == 6
    for current, original in zip(predictions, originals):
        np.testing.assert_array_equal(current, original)
    for current, original in zip(observations, observed_originals):
        np.testing.assert_array_equal(current.peaks, original)


def test_gallery_mixture_respects_energy_filter_and_candidate_threshold(
        notebook, fast_cosine_gallery, monkeypatch):
    from src.spectra import mixture

    original_fit, calls = mixture.fit_spectral_mixture, []

    def track_fit(predictions, target, **kwargs):
        calls.append(np.array(target, copy=True))
        return original_fit(predictions, target, **kwargs)

    monkeypatch.setattr(mixture, "fit_spectral_mixture", track_fit)
    if "fit_spectral_mixture" in notebook:
        monkeypatch.setitem(notebook, "fit_spectral_mixture", track_fit)
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(2)]
    predictions = [np.array([[50.0, 1.0]]), np.array([[60.0, 1.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=20,
                                 peaks=[[50.0, 0.25], [60.0, 0.75]]),
                    make_spectrum(notebook, 40, 3, ce=40,
                                 peaks=[[50.0, 0.8], [60.0, 0.2]])]
    candidates, spectra = (100.0, fragments, observations), (100.0, predictions, observations)
    filtered = fast_cosine_gallery(
        candidates, spectra, fit_mixture=True, ms3_collision_energies=[40]).data
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], observations[1].peaks)
    assert filtered.count('class="mixture-card"') == 1
    for index, contents in cosine_cards(filtered):
        weight = float(re.search(r'data-weight=["\']([^"\']+)', contents).group(1))
        assert weight == pytest.approx([0.8, 0.2][index], abs=1e-6)

    calls.clear()
    skipped = fast_cosine_gallery(
        candidates, spectra, fit_mixture=True, mixture_min_candidates=3).data
    assert not calls
    assert 'class="mixture-card"' not in skipped
    assert 'class="mixture-weight"' not in skipped


def test_gallery_mirror_is_opt_in_and_keeps_prediction_only_bags(
        notebook, fast_cosine_gallery, monkeypatch):
    candidate = make_candidate(notebook, "one", 100.0)
    peaks = np.array([[50.0, 1.0]])
    observed = make_spectrum(notebook, 20, 3, ce=40, peaks=[[50.0, 2.0]])
    inputs = (100.0, [candidate], [observed]), (100.0, [peaks], [observed])

    def unexpected_mirror(*args, **kwargs):
        pytest.fail("Mirror rendering should only run for an opted-in comparison")

    monkeypatch.setitem(notebook, "_gallery_mirror_svg", unexpected_mirror)
    assert fast_cosine_gallery(*inputs).data == fast_cosine_gallery(*inputs, mirror=False).data
    without_ms3 = (100.0, [candidate], []), (100.0, [peaks], [])
    default = fast_cosine_gallery(*without_ms3).data
    mirrored = fast_cosine_gallery(*without_ms3, mirror=True).data
    assert cosine_cards(default) == cosine_cards(mirrored)
    assert 'class="mirror-panel"' not in mirrored
    for invalid in (0, 1, "false", None):
        with pytest.raises(ValueError, match="mirror"):
            fast_cosine_gallery(*inputs, mirror=invalid)


def test_gallery_mirror_pairs_every_selected_ms3_and_preserves_sorting(
        notebook, fast_cosine_gallery, monkeypatch):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(3)]
    predictions = [np.array([[50.0, 1.0]]), np.array([[60.0, 1.0]]), None]
    observations = [make_spectrum(notebook, 20, 3, ce=20, peaks=[[50.0, 2.0]]),
                    make_spectrum(notebook, 60, 3, ce=60, peaks=[[60.0, 3.0]])]
    inputs = (100.0, fragments, observations), (100.0, predictions, observations)
    originals = [None if peaks is None else peaks.copy() for peaks in predictions]
    observed_originals = [observed.peaks.copy() for observed in observations]
    rendered = []

    def capture(prediction, observed, size, *args, **kwargs):
        rendered.append((prediction, observed, size))
        return "<svg>mirrored comparison</svg>"

    monkeypatch.setitem(notebook, "_gallery_mirror_svg", capture)
    baseline = fast_cosine_gallery(*inputs).data
    html = fast_cosine_gallery(*inputs, mirror=True).data
    assert [index for index, _ in cosine_cards(html)] == [
        index for index, _ in cosine_cards(baseline)] == [0, 1, 2]
    assert len(rendered) == 6
    assert html.count('class="mirror-panel"') == 6
    assert 'class="ms3-card"' not in html
    assert re.findall(r'data-mirror-ms3-index="(\d+)"', html) == ["0"] * 3 + ["1"] * 3
    for target in observations:
        compared = [prediction for prediction, observed, _ in rendered
                    if np.array_equal(observed, target.peaks)]
        assert len(compared) == 3
        assert sum(prediction is None for prediction in compared) == 1
        for expected in predictions[:2]:
            assert any(prediction is not None and np.array_equal(prediction, expected)
                       for prediction in compared)
    assert "CE 20" in html and "CE 60" in html
    assert re.findall(r'(?:Best cosine|Cosine) [0-9.]+', html) == re.findall(
        r'(?:Best cosine|Cosine) [0-9.]+', baseline)

    rendered.clear()
    filtered = fast_cosine_gallery(*inputs, mirror=True, ms3_collision_energies=[60]).data
    assert [index for index, _ in cosine_cards(filtered)] == [1, 0, 2]
    assert len(rendered) == 3
    assert all(np.array_equal(observed, observations[1].peaks)
               for _, observed, _ in rendered)
    for current, original in zip(predictions, originals):
        if original is not None:
            np.testing.assert_array_equal(current, original)
    for current, original in zip(observations, observed_originals):
        np.testing.assert_array_equal(current.peaks, original)


def test_gallery_mirror_colors_and_mixtures_keep_original_scores_and_weights(
        notebook, fast_cosine_gallery, monkeypatch):
    fragments = [make_candidate(notebook, str(i), 100.0) for i in range(2)]
    predictions = [np.array([[50.0, 1.0]]), np.array([[60.0, 1.0]])]
    observations = [make_spectrum(notebook, 20, 3, ce=20,
                                 peaks=[[50.0, 0.25], [60.0, 0.75]]),
                    make_spectrum(notebook, 40, 3, ce=40,
                                 peaks=[[50.0, 0.8], [60.0, 0.2]])]
    inputs = (100.0, fragments, observations), (100.0, predictions, observations)
    rendered = []

    def capture(prediction, observed, size, x_max, y_max, normalize, **kwargs):
        rendered.append((prediction, observed, normalize, kwargs))
        return "<svg>mirrored comparison</svg>"

    monkeypatch.setitem(notebook, "_gallery_mirror_svg", capture)
    baseline = fast_cosine_gallery(*inputs, fit_mixture=True).data

    def scores_and_weights(html):
        return ([index for index, _ in cosine_cards(html)],
                re.findall(r'(?:Best cosine|Cosine) [0-9.]+', html),
                re.findall(r'data-weight=["\']([^"\']+)', html))

    for color_matches, strict, normalize in ((True, None, True), (True, 10, False),
                                               (False, 10, True)):
        rendered.clear()
        html = fast_cosine_gallery(*inputs, mirror=True, fit_mixture=True,
                                  color_matches=color_matches, strict=strict,
                                  normalize=normalize).data
        assert len(rendered) == 6  # 2 references x (2 candidates + 1 mixture)
        assert html.count('class="mirror-panel"') == 6
        assert html.count('class="mixture-card"') == 2
        assert 'data-mixture-reference="true"' not in html
        assert 'class="ms3-card"' not in html
        assert html.index('class="mixture-preview"') < html.index('class="bag-scroll"')
        assert scores_and_weights(html) == scores_and_weights(baseline)
        for prediction, observed, actual_normalize, kwargs in rendered:
            expected = (notebook["_gallery_overlap_colors"](prediction, observed, strict=strict)
                        if color_matches else (None, None))
            assert kwargs.get("prediction_colors") == expected[0]
            assert kwargs.get("observed_colors") == expected[1]
            assert actual_normalize == normalize


@pytest.mark.parametrize("normalize, y_max, expected_prediction, expected_observed", [
    (True, 105, [100.0, 50.0], [-50.0, -100.0]),
    (False, 11, [8.0, 4.0], [-5.0, -10.0]),
])
def test_gallery_mirror_svg_has_shared_mass_axis_and_symmetric_intensity_axis(
        notebook, monkeypatch, normalize, y_max, expected_prediction, expected_observed):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    from matplotlib.colors import to_rgba

    prediction = np.array([[25.0, 8.0], [50.0, 4.0]])
    observed = np.array([[25.0, 5.0], [75.0, 10.0]])
    originals = prediction.copy(), observed.copy()
    figures = []

    def capture(figure, stream, **kwargs):
        figures.append(figure)
        stream.write("<svg></svg>")

    monkeypatch.setattr(Figure, "savefig", capture)
    notebook["_gallery_mirror_svg"](
        prediction, observed, (320, 460), 105, y_max, normalize,
        prediction_colors=["#16a34a", "#dc2626"],
        observed_colors=["#dc2626", "#16a34a"])
    assert len(figures) == 1
    figure = figures[0]
    np.testing.assert_allclose(figure.get_size_inches() * figure.dpi, [320, 460])
    assert len(figure.axes) == 1
    axes = figure.axes[0]
    assert axes.get_xlim() == (0, 105)
    assert axes.get_ylim() == (-y_max, y_max)
    assert len(axes.collections) == 2
    positive, negative = axes.collections
    np.testing.assert_allclose(positive.get_segments(), [
        [[25.0, 0], [25.0, expected_prediction[0]]],
        [[50.0, 0], [50.0, expected_prediction[1]]]])
    np.testing.assert_allclose(negative.get_segments(), [
        [[25.0, 0], [25.0, expected_observed[0]]],
        [[75.0, 0], [75.0, expected_observed[1]]]])
    np.testing.assert_allclose(positive.get_colors(), [to_rgba("#16a34a"), to_rgba("#dc2626")])
    np.testing.assert_allclose(negative.get_colors(), [to_rgba("#dc2626"), to_rgba("#16a34a")])
    np.testing.assert_array_equal(prediction, originals[0])
    np.testing.assert_array_equal(observed, originals[1])


@pytest.mark.parametrize("prediction", [None, np.empty((0, 2)), np.array([[50.0, 0.0]])])
def test_gallery_mirror_svg_keeps_ms3_when_prediction_is_unavailable_or_empty(
        notebook, monkeypatch, prediction):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    observed = np.array([[50.0, 5.0]])
    figures = []

    def capture(figure, stream, **kwargs):
        figures.append(figure)
        stream.write("<svg></svg>")

    monkeypatch.setattr(Figure, "savefig", capture)
    notebook["_gallery_mirror_svg"](prediction, observed, (320, 460), 105, 105, True)
    axes = figures[0].axes[0]
    assert axes.get_ylim() == (-105, 105)
    assert len(axes.collections) == 1
    np.testing.assert_allclose(axes.collections[0].get_segments(), [[[50.0, 0], [50.0, -100.0]]])
    labels = " ".join([axes.get_title(), *(text.get_text() for text in axes.texts)])
    expected = "prediction unavailable" if prediction is None else "no nonzero peaks"
    assert expected in labels.lower()
    assert "MS3" in labels
    np.testing.assert_array_equal(observed, [[50.0, 5.0]])


@pytest.fixture
def real_feature(notebook):
    path = REPO_ROOT / "msnlib/20241003_enamdisc_pos_msn.json"
    if not path.exists():
        pytest.skip("Local MSnLib dataset is not installed")
    feature = "20240502_pluskal_enamine_5007_C8_id_MSn_positive.mzML msn trees:3"
    with path.open() as handle:
        records = [json.loads(line) for line in itertools.islice(handle, 99)]
    source = [notebook["Spectrum"].from_record(record) for record in records
              if record.get("feature_id") == feature
              and record.get("merge_type") == "SINGLE_BEST_SCAN"]
    root = next(spectrum for spectrum in source
                if spectrum.ms_level == 2 and spectrum.metadata["scan_number"] == 186)
    return root, source


def test_real_feature_resolves_all_sixteen_ms3_scans(notebook, real_feature):
    root, source = real_feature
    result = notebook["find_ms3_children"](root, source)
    assert {index: len(values) for index, values in result.items() if values} == {
        22: 3, 34: 3, 49: 3, 55: 4, 58: 3}
    assert set(scans(result[55])) == {192, 193, 745, 746}
    assert sorted(child.metadata["collision_energy"] for child in result[55]) == [40, 40, 60, 60]


def test_live_magma_attaches_real_ms3_without_model_load(notebook, real_feature):
    for module in ("rdkit", "torch", "torch_scatter", "dgl", "numba", "pytorch_lightning"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"Optional live MAGMa dependency is unavailable: {module}")
    try:
        notebook["_asset_path"](None, "GLACIER_UPSTREAM", "vendor/ms-pred")
    except FileNotFoundError:
        pytest.skip("Optional pinned ms-pred checkout is not installed")
    root, source = real_feature
    bags = notebook["get_candidate_bags"](root, source)
    assert len(bags) == len(root.peaks) == 62
    assert sum(len(bag[2]) for bag in bags) == 16
    assert len(bags[55][2]) == 4
    assert any(bag[1] for bag in bags if bag[2])
    for _, fragments, observations in bags:
        for fragment in fragments:
            assert fragment.ms3_spectra is observations
