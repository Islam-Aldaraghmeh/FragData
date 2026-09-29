"""Expanded molecule-disjoint data for the small CE-conditioned intensity model.

Reuse MAGMa once per precursor/union mass axis, then assign candidates to the
nearest centroid separately in each scan. All original 300 molecules are
excluded, including their validation/test groups, so comparison with the old
model uses new molecules. Intensities are labels only, never MAGMa inputs.
"""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import pickle
import random
import time

import numpy as np
from rdkit import Chem, rdBase

from intensity_data import GENERATION, ROOT, _index, split_molecules
from intensity_model import candidate_features as old_candidate_features, normalize_intensity
from magma_workflow.candidates import get_candidate_bags, assign_candidates_to_nearest_peak
from magma_workflow.dataset import Spectrum
from src.models.assets import COMMIT


def _worker_init():
    import torch
    torch.set_num_threads(1)


def _connectivity_key(smiles):
    """Also group tautomer/protonation variants that normalize to one InChI."""
    with rdBase.BlockLogs():
        key = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    if not key:
        raise ValueError(f"Cannot establish molecule identity: {smiles}")
    return key.split("-")[0]


def _prepare_molecule(job):
    """A worker sees at most three selected records, not the whole source file."""
    from intensity_model_v2 import FEATURE_SIZE, candidate_features

    molecule, split, records = job
    batches, examples, audits, errors = defaultdict(list), [], [], []
    generation_seconds = feature_seconds = 0.
    for offset, record in records:
        spectrum = Spectrum.from_record(record)
        audit = dict(molecule=molecule, split=split, source_offset=offset,
                     collision_energy=float(record["collision_energy"]), status="selected")
        audits.append(audit)
        if (not len(spectrum.peaks) or not np.isfinite(spectrum.peaks).all()
                or (spectrum.mz <= 0).any() or (spectrum.intensity < 0).any()
                or spectrum.intensity.max() <= 0):
            audit["status"] = "invalid_target_excluded"
            errors.append(dict(audit, reason="Invalid or all-zero experimental peaks"))
            continue
        batches[(spectrum.smiles, spectrum.metadata["adduct"], spectrum.precursor_mz)].append(
            (spectrum, audit))
    for batch in batches.values():
        first = batch[0][0]
        axis = np.unique(np.concatenate([s.mz for s, _ in batch]))
        placeholder = Spectrum(first.smiles, first.precursor_mz, 2,
                               np.column_stack((axis, np.zeros(len(axis)))), first.metadata.copy())
        tick, error = time.perf_counter(), None
        try:
            union_bags = get_candidate_bags(placeholder, spectra=[], peak_assignment="all", **GENERATION)
        except (ValueError, RuntimeError) as caught:
            error = f"{type(caught).__name__}: {caught}"
            union_bags = [(float(mz), [], []) for mz in axis]
        generation_seconds += time.perf_counter() - tick
        for spectrum, audit in batch:
            # Assignment on the union would let a different CE's centroid steal
            # a candidate. Subset first, then choose the nearest mass in this scan.
            all_bags = [union_bags[int(i)] for i in np.searchsorted(axis, spectrum.mz)]
            nearest_bags = assign_candidates_to_nearest_peak(all_bags)
            for peak_index, (_, candidates, _) in enumerate(nearest_bags):
                for candidate in candidates:
                    candidate.metadata["peak_index"] = peak_index
            tick = time.perf_counter()
            features, indices = candidate_features(spectrum.smiles, audit["collision_energy"], nearest_bags)
            example = dict(split=split, molecule=molecule, smiles=spectrum.smiles,
                collision_energy=audit["collision_energy"], mz=spectrum.mz.astype(np.float64).copy(),
                target=normalize_intensity(spectrum.intensity).astype(np.float32),
                features=features.astype(np.float32).reshape(-1, FEATURE_SIZE),
                peak_indices=indices.astype(np.int64), source_offset=audit["source_offset"])
            if split != "train":
                for prefix, bags in (("old", all_bags), ("old_nearest", nearest_bags)):
                    old_features, old_indices = old_candidate_features(spectrum.smiles, audit["collision_energy"], bags)
                    example[f"{prefix}_features"] = old_features
                    example[f"{prefix}_peak_indices"] = old_indices
            feature_seconds += time.perf_counter() - tick
            audit.update(status="generation_error" if error else "prepared", peaks=len(nearest_bags),
                         supported_peaks=len(set(indices.tolist())), candidates=len(features),
                         original_supported_peaks=sum(bool(bag[1]) for bag in all_bags))
            if error:
                errors.append(dict(audit, reason=error))
            examples.append(example)
    return examples, audits, errors, len(batches), generation_seconds, feature_seconds


def prepare_dataset(data_path, output_dir, *, molecules=1500, seed=20260923, workers=2):
    """Return examples/manifest; cache, provenance, and split audit live in output_dir.

    Sample up to three distinct CE values per canonical non-stereochemical
    molecule. Keep valid spectra with no support as all-zero predictions during
    evaluation; record generation failures rather than selecting easy spectra.
    `workers=1` provides a simple sequential/debugging execution path.
    """
    if isinstance(molecules, bool) or not isinstance(molecules, int) or molecules < 3:
        raise ValueError("molecules must be an integer >= 3")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    path, output = Path(data_path).resolve(), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    original_path = ROOT / "results/experiment_2/manifest.json"
    original = json.loads(original_path.read_text())
    excluded = set().union(*map(set, original["splits"].values()))
    excluded_connectivity = {_connectivity_key(mol) for mol in excluded}
    stat = path.stat()
    files = ("intensity_data.py", "intensity_data_v2.py", "intensity_model.py", "intensity_model_v2.py",
             "magma_workflow/candidates.py", "distilled_fragdata/src/chemistry/heavy_graph.py")
    cache_key = dict(schema=2, source=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        molecules=molecules, seed=seed, max_collision_energies=3, max_heavy_atoms=35,
        generation=GENERATION, peak_assignment="nearest_per_scan", magma_commit=COMMIT,
        grouping="canonical_nonstereo_smiles_plus_inchikey_connectivity_guard",
        original_manifest_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        numpy_version=np.__version__, rdkit_version=rdBase.rdkitVersion,
        code_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files})
    cache, manifest_path = output / "dataset.pkl", output / "manifest.json"
    if cache.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("cache_key") == cache_key:
            with cache.open("rb") as stream:
                return pickle.load(stream), manifest
    started = time.perf_counter()
    groups, exclusions, source_hash = _index(path)
    rng, eligible = random.Random(seed), sorted(set(groups) - excluded)
    rng.shuffle(eligible)
    chosen, connectivity_of, seen = [], {}, set(excluded_connectivity)
    for molecule in eligible:
        connectivity = _connectivity_key(molecule)
        if connectivity not in seen:
            chosen.append(molecule)
            connectivity_of[molecule] = connectivity
            seen.add(connectivity)
        if len(chosen) == molecules:
            break
    if len(chosen) != molecules:
        raise ValueError(f"Requested {molecules} new groups; only {len(chosen)} are eligible")
    splits = split_molecules(chosen, seed)
    split_of = {mol: split for split, keys in splits.items() for mol in keys}
    assert len(split_of) == sum(map(len, splits.values())) == len(chosen)
    assert not set(split_of).intersection(excluded)
    jobs = []
    with path.open("rb") as stream:
        for molecule in chosen:
            by_ce = defaultdict(list)
            for offset, ce in groups[molecule]:
                by_ce[ce].append(offset)
            energies, records = sorted(by_ce), []
            rng.shuffle(energies)
            for ce in energies[:3]:
                offset = rng.choice(by_ce[ce])
                stream.seek(offset)
                records.append((offset, json.loads(stream.readline())))
            jobs.append((molecule, split_of[molecule], records))
    manifest = dict(cache_key=cache_key, source_sha256=source_hash, exclusions=exclusions,
        eligible_molecules=len(groups), eligible_records=sum(map(len, groups.values())),
        excluded_original_groups=sorted(excluded), splits=splits,
        excluded_original_connectivity_keys=sorted(excluded_connectivity),
        split_connectivity_keys={split: [connectivity_of[mol] for mol in keys]
                                 for split, keys in splits.items()},
        selected_records=[], errors=[], generation_calls=0, workers=workers,
        timings=dict(index_and_selection_seconds=time.perf_counter() - started))
    examples, generation_seconds, feature_seconds = [], 0., 0.
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                initializer=_worker_init) if workers > 1 else None)
    try:
        results = pool.map(_prepare_molecule, jobs) if pool else map(_prepare_molecule, jobs)
        for number, result in enumerate(results, 1):
            rows, audits, errors, calls, generation, features = result
            examples.extend(rows)
            manifest["selected_records"].extend(audits)
            manifest["errors"].extend(errors)
            manifest["generation_calls"] += calls
            generation_seconds += generation
            feature_seconds += features
            if number % 25 == 0 or number == len(chosen):
                print(f"Prepared {number}/{len(chosen)} new molecules, {len(examples)} spectra", flush=True)
    finally:
        if pool:
            pool.shutdown(cancel_futures=True)
    manifest["timings"].update(generation_worker_seconds=generation_seconds,
        feature_worker_seconds=feature_seconds, total_seconds=time.perf_counter() - started)
    manifest["counts"] = dict(selected_molecules=len(chosen), selected_spectra=len(manifest["selected_records"]),
        examples=len(examples), molecules_by_split={k: len(v) for k, v in splits.items()},
        by_split=dict(Counter(e["split"] for e in examples)),
        unsupported_spectra=sum(not len(e["features"]) for e in examples),
        errors=len(manifest["errors"]), candidate_rows=sum(len(e["features"]) for e in examples))
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(examples, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(cache)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return examples, manifest
