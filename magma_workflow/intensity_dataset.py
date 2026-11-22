from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from torch.utils.data import Dataset

from pathlib import Path

import torch
from tqdm.auto import tqdm

from .candidates import get_candidate_bags


from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib


@dataclass(slots=True)
class MolecularGraph:
    """Chemical graph before conversion to model tensors."""

    atoms: list[dict]
    bonds: list[dict]


@dataclass(slots=True)
class IntensityCandidate:
    """One structural hypothesis for a fragment peak."""

    candidate_id: str

    # Fragment itself
    graph: MolecularGraph
    ionic_formula: dict[str, int]
    charge: int
    theoretical_mz: float

    # Indices into the root MolecularGraph.
    # This is how the fragment is connected to its precursor.
    parent_atom_indices: list[int]


@dataclass(slots=True)
class IntensityBag:
    """One observed m/z and all candidate fragments explaining it."""

    peak_index: int
    peak_mz: float

    # Original measured intensity; normalization happens during training.
    intensity: float

    candidates: list[IntensityCandidate]
    

@dataclass(slots=True)
class IntensitySpectrum:
    """One training example: root molecule + all representable fragment peaks."""

    root_smiles: str
    root_graph: MolecularGraph

    precursor_mz: float
    adduct: str
    collision_energy: float

    # Only bags with >= 1 candidate.
    bags: list[IntensityBag]
    
class IntensityDataset(Dataset):
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)

        self.files = sorted(
            (self.cache_dir / "examples").glob("*.pt")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return torch.load(
            self.files[index],
            weights_only=False,
        )   


# ============================================================


def _root_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    atoms = [
        {
            "element": atom.GetSymbol(),
            "formal_charge": atom.GetFormalCharge(),
            "aromatic": atom.GetIsAromatic(),
            "degree": atom.GetDegree(),
            "total_h": atom.GetTotalNumHs(),
            "hybridization": str(atom.GetHybridization()),
            "in_ring": atom.IsInRing(),
        }
        for atom in mol.GetAtoms()
    ]

    bonds = [
        {
            "i": bond.GetBeginAtomIdx(),
            "j": bond.GetEndAtomIdx(),
            "order": bond.GetBondTypeAsDouble(),
            "aromatic": bond.GetIsAromatic(),
            "conjugated": bond.GetIsConjugated(),
            "in_ring": bond.IsInRing(),
        }
        for bond in mol.GetBonds()
    ]

    return {
        "atoms": atoms,
        "bonds": bonds,
    }


# ============================================================


def _to_intensity_candidate(candidate):
    origins = candidate.metadata.get("origins", [])

    if not origins:
        raise ValueError(
            f"{candidate.candidate_id} has no MAGMa origin"
        )

    parent_atom_indices = origins[0]["parent_atom_indices"]

    return IntensityCandidate(
        candidate_id=candidate.candidate_id,
        graph=candidate.graph,
        ionic_formula=dict(candidate.ionic_formula),
        charge=int(candidate.charge),
        theoretical_mz=float(candidate.metadata["theoretical_mz"]),
        parent_atom_indices=list(parent_atom_indices),
    )


# ============================================================


def _spectrum_cache_id(spectrum):
    usi = spectrum.metadata.get("usi")

    if isinstance(usi, (list, tuple)) and usi:
        identity = str(usi[0])
    else:
        identity = (
            f"{spectrum.smiles}|"
            f"{spectrum.precursor_mz}|"
            f"{spectrum.metadata.get('feature_id')}|"
            f"{spectrum.metadata.get('scan_number')}|"
            f"{spectrum.metadata.get('collision_energy')}"
        )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:24]
    

# ============================================================
    
    
def build_intensity_spectrum(spectrum, candidate_bags):
    if spectrum.ms_level != 2:
        raise ValueError("IntensitySpectrum requires an MS2 spectrum")

    bags = []

    for peak_mz, candidates, _ms3 in candidate_bags:
        # We only train on representable peaks.
        if not candidates:
            continue

        peak_index = candidates[0].metadata["peak_index"]

        if not np.isclose(
            spectrum.mz[peak_index],
            peak_mz,
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError("Candidate bag does not match spectrum peak")

        bags.append(
            IntensityBag(
                peak_index=peak_index,
                peak_mz=float(peak_mz),
                intensity=float(spectrum.intensity[peak_index]),
                candidates=[
                    _to_intensity_candidate(candidate)
                    for candidate in candidates
                ],
            )
        )

    if not bags:
        return None

    return IntensitySpectrum(
        root_smiles=spectrum.smiles,
        root_graph=_root_graph(spectrum.smiles),
        precursor_mz=float(spectrum.precursor_mz),
        adduct=spectrum.metadata["adduct"],
        collision_energy=float(
            spectrum.metadata["collision_energy"]
        ),
        bags=bags,
    )


# ===========================================================


def _build_one_intensity_spectrum(
    spectrum,
    mz_tolerance,
    ppm_tolerance,
    max_tree_depth,
    max_broken_bonds,
    max_boundary_cuts,
    peak_assignment,
):
    """
    Process one MS2 spectrum inside a worker process.

    MAGMa runs here. No MS3 lookup is performed because the
    intensity dataset does not use MS3 information.
    """

    candidate_bags = get_candidate_bags(
        spectrum,
        spectra=None,
        mz_tolerance=mz_tolerance,
        ppm_tolerance=ppm_tolerance,
        max_tree_depth=max_tree_depth,
        max_broken_bonds=max_broken_bonds,
        max_boundary_cuts=max_boundary_cuts,
        peak_assignment=peak_assignment,
        include_ms3=False,
    )

    return build_intensity_spectrum(
        spectrum,
        candidate_bags,
    )

# ============================================================


SUPPORTED_ADDUCTS = {"[M+H]+", "[M]+"}


def build_intensity_dataset(
    spectra,
    cache_dir="data/intensity_dataset",
    *,
    mz_tolerance=0.005,
    ppm_tolerance=20.0,
    max_tree_depth=3,
    max_broken_bonds=6,
    max_boundary_cuts=6,
    peak_assignment="nearest",
    num_workers=4,
    force_rebuild=False,
):
    """
    Build the cached intensity dataset in parallel.

    Each MS2 spectrum is processed independently in a worker process.
    Completed examples are written immediately to disk.

    Existing cached examples are skipped unless force_rebuild=True.
    """

    cache_dir = Path(cache_dir)

    examples_dir = cache_dir / "examples"
    empty_dir = cache_dir / "empty"
    failed_dir = cache_dir / "failed"

    examples_dir.mkdir(parents=True, exist_ok=True)
    empty_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    spectra = list(spectra)

    # ---------------------------------------------------------
    # Filter to MS2 spectra supported by candidate generation
    # ---------------------------------------------------------

    ms2_spectra = [
        spectrum
        for spectrum in spectra
        if spectrum.ms_level == 2
    ]

    supported = [
        spectrum
        for spectrum in ms2_spectra
        if (
            spectrum.metadata.get("adduct") in SUPPORTED_ADDUCTS
            and spectrum.metadata.get("charge") == 1
        )
    ]

    print(f"MS2 spectra:  {len(ms2_spectra)}")
    print(f"Supported:    {len(supported)}")
    print(f"Unsupported:  {len(ms2_spectra) - len(supported)}")
    print(f"Workers:      {num_workers}")

    # ---------------------------------------------------------
    # Determine what still needs processing
    # ---------------------------------------------------------

    jobs = []

    n_cached = 0
    n_known_empty = 0
    n_known_failed = 0

    for spectrum in supported:
        cache_id = _spectrum_cache_id(spectrum)

        example_path = examples_dir / f"{cache_id}.pt"
        empty_path = empty_dir / f"{cache_id}.txt"
        failed_path = failed_dir / f"{cache_id}.txt"

        if not force_rebuild:
            if example_path.exists():
                n_cached += 1
                continue

            if empty_path.exists():
                n_known_empty += 1
                continue

            if failed_path.exists():
                n_known_failed += 1
                continue

        jobs.append((cache_id, spectrum))

    print(f"Cached:       {n_cached}")
    print(f"Known empty:  {n_known_empty}")
    print(f"Known failed: {n_known_failed}")
    print(f"Remaining:    {len(jobs)}")

    if not jobs:
        return IntensityDataset(cache_dir)

    # ---------------------------------------------------------
    # Run MAGMa in parallel
    # ---------------------------------------------------------

    n_built = 0
    n_empty = 0
    n_failed = 0

    with ProcessPoolExecutor(
        max_workers=num_workers
    ) as executor:

        future_to_id = {}

        for cache_id, spectrum in jobs:
            future = executor.submit(
                _build_one_intensity_spectrum,
                spectrum,
                mz_tolerance,
                ppm_tolerance,
                max_tree_depth,
                max_broken_bonds,
                max_boundary_cuts,
                peak_assignment,
            )

            future_to_id[future] = cache_id

        progress = tqdm(
            as_completed(future_to_id),
            total=len(future_to_id),
            desc="Generating MAGMa candidates",
            unit="spectrum",
        )

        for future in progress:
            cache_id = future_to_id[future]

            example_path = examples_dir / f"{cache_id}.pt"
            empty_path = empty_dir / f"{cache_id}.txt"
            failed_path = failed_dir / f"{cache_id}.txt"

            try:
                example = future.result()

                if example is None:
                    empty_path.touch()
                    n_empty += 1

                else:
                    torch.save(
                        example,
                        example_path,
                    )
                    n_built += 1

            except Exception as exc:
                message = (
                    f"{type(exc).__name__}: {exc}"
                )

                failed_path.write_text(message)

                n_failed += 1

                tqdm.write(
                    f"FAILED {cache_id}: {message}"
                )

            progress.set_postfix(
                built=n_built,
                empty=n_empty,
                failed=n_failed,
            )

    print()
    print("Finished")
    print(f"  built:  {n_built}")
    print(f"  empty:  {n_empty}")
    print(f"  failed: {n_failed}")

    return IntensityDataset(cache_dir)