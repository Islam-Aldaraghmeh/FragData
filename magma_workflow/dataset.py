"""MSnLib loading with compact, notebook-safe spectrum containers."""
import json
from pathlib import Path

from Spectrum_Classes import (
    Spectrum, SpectrumForest, SpectrumCollection, SpectraBySmiles, parse_spectrum_forest,
)


def parse_dataset(dataset_path, verbose=False):
    # Bounded previews prevent IDE inspectors from formatting every spectrum.
    spectra_by_smiles = SpectraBySmiles()
    spectra = SpectrumCollection()
    with Path(dataset_path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("merge_type") != "SINGLE_BEST_SCAN":
                continue
            spectrum = Spectrum.from_record(record)
            spectra_by_smiles.setdefault(spectrum.smiles, SpectrumCollection()).append(spectrum)
            spectra.append(spectrum)

    if verbose:
        print(f"{len(spectra_by_smiles):,} unique SMILES")
    return spectra, spectra_by_smiles


def check_compatability(spectra, data_path_anchor):
    if not spectra:
        raise ValueError("The selected dataset contains no SINGLE_BEST_SCAN spectra")
    # Only the first matching record is needed; do not load the whole anchor.
    with Path(data_path_anchor).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("merge_type") == "SINGLE_BEST_SCAN":
                anchor_keys = set(Spectrum.from_record(record).metadata)
                break
        else:
            raise ValueError("The anchor contains no SINGLE_BEST_SCAN spectra")
    if anchor_keys == set(spectra[0].metadata):
        print("Dataset compatible with anchor")
    else:
        print("ERROR DATASET NOT COMPATIBLE WITH ANCHOR")

