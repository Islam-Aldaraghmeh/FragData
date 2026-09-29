"""Notebook variable previews must not format entire datasets or peak arrays."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Spectrum_Classes import (
    SpectraBySmiles, Spectrum, SpectrumCollection, SpectrumForest,
)


class NoPreview:
    def __repr__(self):
        raise AssertionError("A summary preview visited a contained object")

    __str__ = __repr__


class NoArrayPreview(np.ndarray):
    def __repr__(self):
        raise AssertionError("A spectrum preview rendered its peak array")

    __str__ = __repr__


class NoMappingPreview(dict):
    def __repr__(self):
        raise AssertionError("A spectrum preview rendered its metadata mapping")

    __str__ = __repr__


@pytest.mark.parametrize("formatter", [repr, str])
def test_collection_previews_never_render_elements_or_mapping_keys(formatter):
    element, key = NoPreview(), NoPreview()
    spectra = SpectrumCollection([element] * 50_000)
    by_smiles = SpectraBySmiles({key: spectra})

    assert len(formatter(spectra)) < 256
    assert len(formatter(by_smiles)) < 256


@pytest.mark.parametrize("formatter", [repr, str])
def test_spectrum_preview_never_renders_peaks_or_metadata(formatter):
    peaks = np.array([[100.0, 20.0], [150.0, 10.0]]).view(NoArrayPreview)
    metadata = NoMappingPreview(
        scan_number=[42], collision_energy=40, feature_id="fixture",
        unrelated=NoPreview(),
    )
    spectrum = Spectrum("CC", 200.0, 2, peaks, metadata)

    preview = formatter(spectrum)

    assert "Spectrum" in preview
    assert len(preview) < 512
    assert spectrum.peaks is peaks
    assert spectrum.metadata is metadata
    np.testing.assert_array_equal(spectrum.mz, [100.0, 150.0])


@pytest.mark.parametrize("formatter", [repr, str])
def test_forest_preview_never_renders_nested_collections(formatter):
    key, value = NoPreview(), NoPreview()
    forest = SpectrumForest(
        nodes={key: value}, parent_of={key: value},
        parent_peak_mz={key: value}, orphans=[value],
    )

    assert len(formatter(forest)) < 512


def test_summary_containers_preserve_list_and_dictionary_behavior():
    first, second, third = object(), object(), object()
    spectra = SpectrumCollection([first, second])
    assert isinstance(spectra, list)
    assert spectra[0] is first
    assert list(spectra) == [first, second]
    spectra.append(third)
    assert spectra[1:] == [second, third]
    assert spectra.pop() is third
    assert len(spectra) == 2

    by_smiles = SpectraBySmiles({"CC": spectra})
    assert isinstance(by_smiles, dict)
    assert by_smiles.get("CC") is spectra
    assert by_smiles.setdefault("CC", []) is spectra
    by_smiles["O"] = SpectrumCollection([third])
    assert list(by_smiles) == ["CC", "O"]
    assert by_smiles["O"][0] is third
    del by_smiles["O"]
    assert dict(by_smiles) == {"CC": spectra}


def test_ipython_pretty_uses_bounded_collection_summary():
    from IPython.lib.pretty import pretty

    spectra = SpectrumCollection([NoPreview()] * 50_000)
    by_smiles = SpectraBySmiles({NoPreview(): spectra})

    assert pretty(spectra) == repr(spectra)
    assert pretty(by_smiles) == repr(by_smiles)


@pytest.fixture
def notebook_loader():
    notebook_path = Path(__file__).resolve().parents[1] / "Experiment_1.ipynb"
    notebook = json.loads(notebook_path.read_text())
    cell = next(cell for cell in notebook["cells"] if cell.get("id") == "96ffee3e")
    namespace = dict(
        json=json, Path=Path, Spectrum=Spectrum,
        SpectrumCollection=SpectrumCollection, SpectraBySmiles=SpectraBySmiles,
    )
    exec(compile("".join(cell["source"]), "Experiment_1.ipynb:loader", "exec"), namespace)
    return namespace


def record(scan, *, smiles="CC", level=2):
    return dict(
        smiles=smiles, precursor_mz=200.0, ms_level=level,
        peaks=[[100.0, 20.0], [150.0, 10.0]],
        merge_type="SINGLE_BEST_SCAN", scan_number=[scan],
        feature_id="fixture", collision_energy=40, charge=1, adduct="[M+H]+",
        usi=[f"mzspec:fixture:{scan}"],
    )


def test_loader_filters_before_construction_and_preserves_records(
        notebook_loader, tmp_path, capsys):
    records = [record(1), record(2, level=3), record(3, smiles="O")]
    path = tmp_path / "library.json"
    # These merged records cannot become Spectrum objects: they lack required
    # fields. The loader must exclude them before from_record() is called.
    merged_record = {"merge_type": "SAME_ENERGY", "peaks": "not an array"}
    path.write_text("\n" + "\n".join(
        json.dumps(value) for value in [merged_record, *records]) + "\n")

    spectra, grouped = notebook_loader["parse_dataset"](path, verbose=True)

    assert isinstance(spectra, SpectrumCollection)
    assert isinstance(grouped, SpectraBySmiles)
    assert len(spectra) == len(records)
    assert "2 unique SMILES" in capsys.readouterr().out
    for actual, original in zip(spectra, records):
        expected = Spectrum.from_record(original)
        assert actual.smiles == expected.smiles
        assert actual.precursor_mz == expected.precursor_mz
        assert actual.ms_level == expected.ms_level
        assert actual.metadata == expected.metadata
        np.testing.assert_array_equal(actual.peaks, expected.peaks)
    assert isinstance(grouped["CC"], SpectrumCollection)
    assert grouped["CC"][0] is spectra[0]
    assert grouped["CC"][1] is spectra[1]
    assert grouped["O"][0] is spectra[2]


def test_compatibility_reads_only_first_matching_anchor(
        notebook_loader, tmp_path, capsys):
    first = record(1)
    anchor = tmp_path / "anchor.json"
    anchor.write_text(
        '\n{"merge_type": "SAME_ENERGY"}\n'
        + json.dumps(first) + "\nTHIS MUST NEVER BE PARSED\n"
    )

    notebook_loader["check_compatability"]([Spectrum.from_record(first)], anchor)

    assert capsys.readouterr().out.strip() == "Dataset compatible with anchor"


def test_compatibility_still_detects_different_metadata_keys(
        notebook_loader, tmp_path, capsys):
    first = record(1)
    anchor = tmp_path / "anchor.json"
    anchor.write_text(json.dumps({**first, "additional_metadata": 42}) + "\n")

    notebook_loader["check_compatability"]([Spectrum.from_record(first)], anchor)

    assert "NOT COMPATIBLE" in capsys.readouterr().out
