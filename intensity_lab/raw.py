"""Load cached MAGMa intensity examples without depending on their class schema.

``magma_workflow.intensity_dataset.IntensityDataset`` unpickles slotted
dataclasses. Examples written before ``root_smiles`` was added (the complete
``data/intensity_parallel_test`` cache) then load as objects with missing
attributes. Here pickles are read into plain dicts instead, so every cache
version works, and each example is joined with its MSnLib record through the
same USI hash that ``build_intensity_dataset`` used as the file name. The join
adds the InChIKey (for leakage-safe splits), the SMILES, and the *full*
measured peak list (for evaluating against all peaks, not only supported ones).
"""
from __future__ import annotations

import hashlib
import io
import json
import pickle
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import CACHE_DIR, REPO_ROOT

DEFAULT_EXAMPLE_DIRS = (
    REPO_ROOT / "data" / "intensity_dataset_v1",
    REPO_ROOT / "data" / "intensity_parallel_test",
)
DEFAULT_MSNLIB = REPO_ROOT / "msnlib" / "20241003_enamdisc_pos_msn.json"
RAW_VERSION = "raw-v2"


class _Record:
    """Stand-in for any dataclass found in the pickle; keeps only its fields."""

    def __setstate__(self, state):
        if isinstance(state, tuple):  # (dict_state, slots_state) for slotted classes
            merged = {}
            for part in state:
                if part:
                    merged.update(part)
            state = merged
        self.__dict__.update(state)


class _SchemaFreeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("magma_workflow", "src.")):
            return _Record
        return super().find_class(module, name)


def _read_torch_pickle(path):
    """Read a torch.save() zip file whose payload contains only plain Python."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
        return _SchemaFreeUnpickler(io.BytesIO(archive.read(name))).load()


def _compact_example(path):
    """Keep only what featurization needs; drop per-candidate fragment graphs."""
    ex = _read_torch_pickle(path)
    atoms = ex.root_graph["atoms"]
    bonds = ex.root_graph["bonds"]
    bags = []
    for bag in ex.bags:
        bags.append((
            int(bag.peak_index),
            float(bag.peak_mz),
            float(bag.intensity),
            [
                (dict(c.ionic_formula), int(c.charge), float(c.theoretical_mz),
                 np.asarray(c.parent_atom_indices, dtype=np.int16))
                for c in bag.candidates
            ],
        ))
    return {
        "cache_id": path.stem,
        "atoms": atoms,
        "bonds": bonds,
        "precursor_mz": float(ex.precursor_mz),
        "adduct": ex.adduct,
        "collision_energy": float(ex.collision_energy),
        "bags": bags,
    }


def _usi_cache_id(usi):
    return hashlib.sha256(str(usi).encode("utf-8")).hexdigest()[:24]


def read_msnlib_ms2_index(path=DEFAULT_MSNLIB):
    """Map cache_id -> metadata + full peaks for SINGLE_BEST_SCAN MS2 records."""
    index = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("merge_type") != "SINGLE_BEST_SCAN" or int(record.get("ms_level")) != 2:
                continue
            usi = record.get("usi")
            if not isinstance(usi, list) or not usi:
                continue
            peaks = np.asarray(record["peaks"], dtype=np.float64).reshape(-1, 2)
            # One scan can be annotated with several co-isolated structures, so
            # keep every record and resolve against the cached root graph later.
            index.setdefault(_usi_cache_id(usi[0]), []).append({
                "usi": usi[0],
                "smiles": record["smiles"],
                "inchikey": record.get("inchikey") or "",
                "compound_name": record.get("compound_name"),
                "feature_id": record.get("feature_id"),
                "precursor_purity": float(record.get("precursor_purity") or np.nan),
                "quality_chimeric": record.get("quality_chimeric"),
                "msnlib_collision_energy": float(record["collision_energy"]),
                "peaks": peaks.astype(np.float32),
            })
    return index


def _graph_signature(smiles):
    """Element sequence + bond list exactly as ``_root_graph`` would emit them."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return ([a.GetSymbol() for a in mol.GetAtoms()],
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()])


def _example_signature(ex):
    return ([a["element"] for a in ex["atoms"]], [(int(b["i"]), int(b["j"])) for b in ex["bonds"]])


def _example_files(example_dirs):
    """First directory wins for duplicate cache ids (newest schema listed first)."""
    files = {}
    for directory in example_dirs:
        directory = Path(directory) / "examples"
        if not directory.exists():
            continue
        for path in directory.glob("*.pt"):
            files.setdefault(path.stem, path)
    return dict(sorted(files.items()))


def load_raw_examples(example_dirs=DEFAULT_EXAMPLE_DIRS, msnlib_path=DEFAULT_MSNLIB,
                      cache_dir=CACHE_DIR, num_workers=8, force=False, verbose=True):
    """Return a list of compact example dicts, joined with MSnLib metadata.

    The result is cached as one pickle. The cache key covers the set of example
    files and the MSnLib file size, so adding new MAGMa examples rebuilds it.
    """
    files = _example_files(example_dirs)
    msnlib_path = Path(msnlib_path)
    key = hashlib.sha256(json.dumps({
        "version": RAW_VERSION,
        "ids": list(files),
        "msnlib": [msnlib_path.name, msnlib_path.stat().st_size],
    }).encode()).hexdigest()[:16]
    cache_path = Path(cache_dir) / f"raw_examples_{key}.pkl"
    if cache_path.exists() and not force:
        start = time.time()
        with cache_path.open("rb") as handle:
            examples = pickle.load(handle)
        if verbose:
            print(f"Loaded {len(examples):,} cached raw examples in {time.time() - start:.1f}s "
                  f"({cache_path.name})")
        return examples

    start = time.time()
    paths = list(files.values())
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        examples = list(pool.map(_compact_example, paths, chunksize=64))
    if verbose:
        print(f"Read {len(examples):,} MAGMa example pickles in {time.time() - start:.1f}s")

    start = time.time()
    meta = read_msnlib_ms2_index(msnlib_path)
    joined, missing, unresolved, shared = [], 0, 0, 0
    for ex in examples:
        records = meta.get(ex["cache_id"])
        if not records:
            missing += 1
            continue
        target = _example_signature(ex)
        matches = [r for r in records if _graph_signature(r["smiles"]) == target]
        if len({r["smiles"] for r in matches}) != 1:
            unresolved += 1
            continue
        info = dict(matches[0])
        info["usi_n_structures"] = len({r["inchikey"][:14] for r in records})
        shared += info["usi_n_structures"] > 1
        if abs(info["msnlib_collision_energy"] - ex["collision_energy"]) > 1e-6:
            raise ValueError(f"{ex['cache_id']}: collision energy mismatch with MSnLib")
        ex.update(info)
        joined.append(ex)
    if verbose:
        print(f"Joined with MSnLib metadata in {time.time() - start:.1f}s: "
              f"{len(joined):,} kept, {missing} without a record, {unresolved} unresolved, "
              f"{shared} scans annotated with more than one structure (flagged)")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(joined, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return joined


def find_spectrum(usi, path=DEFAULT_MSNLIB):
    """Stream MSnLib and return the SINGLE_BEST_SCAN MS2 ``Spectrum`` with this USI."""
    from magma_workflow.dataset import Spectrum

    needle = f'"{usi}"'
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if needle not in line:
                continue
            record = json.loads(line)
            if record.get("merge_type") == "SINGLE_BEST_SCAN" and int(record["ms_level"]) == 2 \
                    and record["usi"][0] == usi:
                record["ms_level"] = int(record["ms_level"])
                record["charge"] = int(record["charge"])
                record["collision_energy"] = float(record["collision_energy"])
                record["precursor_mz"] = float(record["precursor_mz"])
                return Spectrum.from_record(record)
    raise KeyError(usi)
