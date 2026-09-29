"""Small, reproducible molecule-disjoint MAGMa intensity dataset builder.

Selection reads acquisition/structure metadata, never intensity or quality.
MAGMa sees union m/z locations with zero placeholders, not experimental labels.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from numbers import Real
from pathlib import Path
import pickle
import random
import time

import numpy as np
from rdkit import Chem, rdBase

from intensity_model import FEATURE_SIZE, candidate_features, normalize_intensity
from magma_workflow.candidates import get_candidate_bags
from magma_workflow.dataset import Spectrum
from src.models.assets import COMMIT

ROOT = Path(__file__).resolve().parent
GENERATION = dict(max_tree_depth=3, max_broken_bonds=6, max_boundary_cuts=6,
                  mz_tolerance=0.005, ppm_tolerance=20.0)


def split_molecules(group_keys, seed):
    """Deterministic 70/15/15 group split, with no group crossing boundaries."""
    keys = sorted(set(group_keys))
    if len(keys) < 3:
        raise ValueError("At least three molecule groups are required")
    random.Random(seed).shuffle(keys)
    first = min(len(keys)-2, max(1, int(.70 * len(keys))))
    second = min(len(keys)-1, max(first+1, int(.85 * len(keys))))
    return dict(train=keys[:first], validation=keys[first:second], test=keys[second:])


@lru_cache(maxsize=32768)
def _molecule(smiles):
    if not isinstance(smiles, str) or not smiles:
        return None, "missing_smiles"
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "invalid_smiles"
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "disconnected_precursor"
    if any(a.GetIsotope() or a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        return None, "isotope_or_explicit_hydrogen"
    if not 1 <= mol.GetNumHeavyAtoms() <= 35:
        return None, "heavy_atom_limit"
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False), None


def _index(path):
    groups, exclusions, digest = defaultdict(list), Counter(), hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            offset, line = stream.tell(), stream.readline()
            if not line:
                break
            digest.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("merge_type") != "SINGLE_BEST_SCAN" or record.get("ms_level") != 2:
                exclusions["not_ordinary_single_scan_ms2"] += 1
                continue
            ce, mz = record.get("collision_energy"), record.get("precursor_mz")
            if isinstance(ce, bool) or not isinstance(ce, Real) or not np.isfinite(ce) or ce < 0:
                exclusions["non_scalar_or_invalid_collision_energy"] += 1
                continue
            if record.get("charge") != 1 or record.get("adduct") not in {"[M+H]+", "[M]+"}:
                exclusions["unsupported_charge_or_adduct"] += 1
                continue
            if not isinstance(mz, Real) or not np.isfinite(mz) or mz <= 0:
                exclusions["invalid_precursor_mz"] += 1
                continue
            smiles = record.get("smiles")
            molecule, reason = _molecule(smiles if isinstance(smiles, str) else "")
            if reason:
                exclusions[reason] += 1
            else:
                groups[molecule].append((offset, float(ce)))
    return groups, dict(exclusions), digest.hexdigest()


def prepare_dataset(data_path, output_dir, *, molecules=300, seed=20260922):
    """Return (examples, manifest), caching features and full provenance locally.

    At most three distinct collision energies are sampled per molecule. Invalid
    targets are reported and excluded; valid targets with generation failures or
    no supported peaks stay in evaluation with an all-zero predicted spectrum.
    Cache reuse checks source path/size/mtime plus settings and feature code.
    """
    if isinstance(molecules, bool) or not isinstance(molecules, int) or molecules < 3:
        raise ValueError("molecules must be an integer >= 3")
    path, output = Path(data_path).resolve(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stat = path.stat()
    files = ("intensity_data.py", "intensity_model.py", "magma_workflow/candidates.py",
             "distilled_fragdata/src/chemistry/heavy_graph.py")
    cache_key = dict(schema=1, source=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                     molecules=molecules, seed=seed, max_collision_energies=3,
                     max_heavy_atoms=35, magma_commit=COMMIT, generation=GENERATION,
                     code_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    cache, manifest_path = output/"dataset.pkl", output/"manifest.json"
    if cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("cache_key") == cache_key:
            with cache.open("rb") as stream:
                return pickle.load(stream), manifest
    started = time.perf_counter()
    groups, exclusions, source_hash = _index(path)
    rng, chosen = random.Random(seed), sorted(groups)
    rng.shuffle(chosen)
    chosen = chosen[:molecules]
    if len(chosen) < 3:
        raise ValueError("At least three eligible precursor molecule groups are required")
    splits = split_molecules(chosen, seed)
    split_of = {mol: split for split, keys in splits.items() for mol in keys}
    manifest = dict(cache_key=cache_key, source_sha256=source_hash, exclusions=exclusions,
                    eligible_molecules=len(groups), eligible_records=sum(map(len, groups.values())),
                    splits=splits, selected_records=[], errors=[], generation_calls=0,
                    timings=dict(index_seconds=time.perf_counter()-started))
    examples, generation_seconds, feature_seconds = [], 0., 0.
    with path.open("rb") as stream:
        for number, molecule in enumerate(chosen, 1):
            by_ce = defaultdict(list)
            for offset, ce in groups[molecule]:
                by_ce[ce].append(offset)
            energies = sorted(by_ce)
            rng.shuffle(energies)
            batches = defaultdict(list)
            for ce in energies[:3]:
                offset = rng.choice(by_ce[ce])
                stream.seek(offset)
                record = json.loads(stream.readline())
                audit = dict(molecule=molecule, split=split_of[molecule], source_offset=offset,
                             collision_energy=ce, status="selected")
                manifest["selected_records"].append(audit)
                spectrum = Spectrum.from_record(record)
                if (not len(spectrum.peaks) or not np.isfinite(spectrum.peaks).all()
                        or (spectrum.mz <= 0).any() or (spectrum.intensity < 0).any()
                        or spectrum.intensity.max() <= 0):
                    audit["status"] = "invalid_target_excluded"
                    manifest["errors"].append(dict(audit, reason="Invalid or all-zero experimental peaks"))
                    continue
                key = (spectrum.smiles, spectrum.metadata["adduct"], spectrum.precursor_mz)
                batches[key].append((spectrum, audit))
            for batch in batches.values():
                first = batch[0][0]
                axis = np.unique(np.concatenate([s.mz for s, _ in batch]))
                placeholder = Spectrum(first.smiles, first.precursor_mz, 2,
                                       np.column_stack((axis, np.zeros(len(axis)))), first.metadata.copy())
                tick = time.perf_counter()
                manifest["generation_calls"] += 1
                error = None
                try:
                    union_bags = get_candidate_bags(placeholder, spectra=[], **GENERATION)
                except (ValueError, RuntimeError) as caught:
                    error = f"{type(caught).__name__}: {caught}"
                    union_bags = [(float(mz), [], []) for mz in axis]
                generation_seconds += time.perf_counter()-tick
                for spectrum, audit in batch:
                    bags = [union_bags[int(i)] for i in np.searchsorted(axis, spectrum.mz)]
                    tick = time.perf_counter()
                    features, indices = candidate_features(spectrum.smiles, audit["collision_energy"], bags)
                    feature_seconds += time.perf_counter()-tick
                    audit.update(status="generation_error" if error else "prepared",
                                 peaks=len(bags), supported_peaks=len(set(indices.tolist())),
                                 candidates=len(features))
                    if error:
                        manifest["errors"].append(dict(audit, reason=error))
                    examples.append(dict(split=audit["split"], molecule=molecule, smiles=spectrum.smiles,
                        collision_energy=audit["collision_energy"], mz=spectrum.mz.astype(np.float64).copy(),
                        target=normalize_intensity(spectrum.intensity).astype(np.float32),
                        features=features.astype(np.float32).reshape(-1, FEATURE_SIZE),
                        peak_indices=indices.astype(np.int64), source_offset=audit["source_offset"]))
            if number % 10 == 0 or number == len(chosen):
                print(f"Prepared {number}/{len(chosen)} molecules, {len(examples)} spectra", flush=True)
    manifest["timings"].update(generation_seconds=generation_seconds, feature_seconds=feature_seconds,
                               total_seconds=time.perf_counter()-started)
    manifest["counts"] = dict(selected_molecules=len(chosen), selected_spectra=len(manifest["selected_records"]),
        examples=len(examples), by_split=dict(Counter(e["split"] for e in examples)),
        unsupported_spectra=sum(len(e["features"]) == 0 for e in examples),
        errors=len(manifest["errors"]), candidate_rows=sum(len(e["features"]) for e in examples))
    with cache.open("wb") as stream:
        pickle.dump(examples, stream, protocol=pickle.HIGHEST_PROTOCOL)
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    return examples, manifest
