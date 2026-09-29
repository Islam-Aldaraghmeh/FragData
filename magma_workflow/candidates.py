"""MAGMa candidate bags and measured MS3 linkage; no Glacier or microstates."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from copy import copy, deepcopy

import numpy as np

from .dataset import parse_spectrum_forest
from src.data.candidate_bags import FragmentCandidate
from src.chemistry.formula import formula_counts, ionic_mass, exact_mass
from src.models.assets import COMMIT, asset_path as _asset_path


def _select_magma_upstream(upstream_path=None):
    """Select the pinned MAGMa source checkout, without loading model code."""
    upstream = _asset_path(upstream_path, "GLACIER_UPSTREAM", "vendor/ms-pred")
    command = ["git", "-C", str(upstream)]
    actual = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT or subprocess.check_output(command + ["diff", "HEAD"], text=True):
        raise ValueError(f"MAGMa requires the clean ms-pred checkout at {COMMIT}")
    source = upstream / "src"
    imported = sys.modules.get("ms_pred")
    if imported is not None and not Path(imported.__file__).resolve().is_relative_to(source):
        raise RuntimeError("Another ms_pred checkout is imported; restart the kernel.")
    os.environ.setdefault("DGLBACKEND", "pytorch")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _candidate_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

_source_spectra = None
_source_by_smiles = None


def set_spectrum_source(spectra=None, spectra_by_smiles=None):
    """Configure optional MS3 lookup context for interactive notebooks.

    Explicit spectra= or forest= arguments always take precedence. Calling
    without arguments clears the default source. This stores references only.
    """
    global _source_spectra, _source_by_smiles
    _source_spectra = spectra
    _source_by_smiles = spectra_by_smiles


def _unpack_bag(bag):
    """Read a new triple or a legacy pair; preserve the shared MS3 list."""
    if len(bag) not in (2, 3):
        raise ValueError("Expected (m/z, items, ms3_spectra) or legacy (m/z, items)")
    mz, items = bag[:2]
    children = bag[2] if len(bag) == 3 else []
    return mz, list(items), children if isinstance(children, list) else list(children)


def find_ms3_children(spectrum, spectra=None, *, forest=None, mz_tolerance=0.01):
    """Return {original_MS2_peak_index: [direct_MS3_Spectrum, ...]}.

    Reuse parse_spectrum_forest/children and the nearest-parent-peak convention
    from Explore_dataset.ipynb. Scope the forest to the selected feature and
    keep separate scans even when their collision energies are the same.
    Child.precursor_mz may describe the root molecule: the isolated fragment m/z
    comes from forest.parent_peak_mz (the final msn_precursor_mzs entry).

    Supply related spectra or a prebuilt forest. Omitted spectra use the source
    configured by set_spectrum_source, preferring its SMILES grouping.
    An explicit empty list means no available observations; missing context is
    an error rather than an unperformed search reported as zero children.
    """
    if spectrum.ms_level != 2:
        raise ValueError("MS3 child lookup requires an MS2 spectrum")
    if not np.isfinite(mz_tolerance) or mz_tolerance < 0:
        raise ValueError("mz_tolerance must be finite and nonnegative")
    if forest is not None and spectra is not None:
        raise ValueError("Supply either spectra or forest, not both")
    if forest is None:
        if spectra is None:
            spectra = (_source_by_smiles or {}).get(spectrum.smiles)
            if spectra is None:
                spectra = _source_spectra
        if spectra is None:
            raise ValueError("Pass spectra=related_scans or forest=... to find MS3 children")
        spectra = list(spectra)
        if not spectra:
            return {}
        feature_id = spectrum.metadata.get("feature_id")
        if not feature_id:
            raise ValueError("The MS2 spectrum needs feature_id metadata for child linkage")
        related = [s for s in spectra if s.ms_level in (2, 3)
                   and s.metadata.get("feature_id") == feature_id]
        parent_usi = spectrum.metadata.get("usi")
        if not parent_usi or not isinstance(parent_usi, (list, tuple)):
            raise ValueError("The MS2 spectrum needs its library USI list for child linkage")
        if not any(s.metadata.get("usi", [None])[0] == parent_usi[0] for s in related):
            related.append(spectrum)
        forest = parse_spectrum_forest(related, mz_tol=mz_tolerance)
    parent_usi = spectrum.metadata.get("usi")
    if not parent_usi or not isinstance(parent_usi, (list, tuple)):
        raise ValueError("The MS2 spectrum needs its library USI list for child linkage")
    parent_id = parent_usi[0]
    if parent_id not in forest.nodes:
        raise ValueError("The selected MS2 scan is not in the forest; use SINGLE_BEST_SCAN spectra")
    parent = forest.nodes[parent_id]
    if (parent.ms_level != 2 or parent.metadata.get("feature_id") != spectrum.metadata.get("feature_id")
            or not np.array_equal(parent.mz, spectrum.mz)):
        raise ValueError("The supplied forest does not match the selected MS2 spectrum")
    by_peak = {}
    if not len(spectrum.mz):
        return by_peak
    for child in forest.children(parent_id):
        if child.ms_level != 3:
            continue
        isolated_mz = forest.parent_peak_mz[child.metadata["usi"][0]]
        errors = np.abs(spectrum.mz - isolated_mz)
        peak_index = int(np.argmin(errors))
        if errors[peak_index] <= mz_tolerance:
            by_peak.setdefault(peak_index, []).append(child)
    return by_peak


def attach_ms3_spectra(spectrum, candidate_bags, spectra=None, *, forest=None,
                       mz_tolerance=0.01):
    """Add/refresh bag[2] and fragment.ms3_spectra using this MS2 scan's children.

    Each candidate references the same measured observations as its bag. These
    spectra support the peak, not a particular structural hypothesis. They are
    never passed to Glacier or used to select candidates.
    Accept full, filtered, or reordered bags; generated peak_index metadata
    preserves the original MS2 peak identity after filtering or sorting.
    """
    children_by_peak = find_ms3_children(spectrum, spectra, forest=forest,
                                         mz_tolerance=mz_tolerance)
    bags = [_unpack_bag(bag) for bag in candidate_bags]
    original_order = len(bags) == len(spectrum.mz) and all(
        np.isclose(float(bag[0]), spectrum.mz[i], atol=1e-8, rtol=0)
        for i, bag in enumerate(bags)
    )
    result = []
    for i, (mz, fragments, _) in enumerate(bags):
        indices = {c.metadata.get("peak_index") for c in fragments}
        if len(indices) > 1:
            raise ValueError(f"Bag {i}: fragments disagree on their original peak_index")
        if fragments and len(indices) == 1 and None not in indices:
            peak_index = indices.pop()
        elif original_order:
            peak_index = i
        else:
            matches = np.flatnonzero(np.isclose(spectrum.mz, float(mz), atol=1e-8, rtol=0))
            if len(matches) != 1:
                raise ValueError(f"Bag {i}: cannot uniquely identify its original MS2 peak")
            peak_index = int(matches[0])
        if (not isinstance(peak_index, (int, np.integer)) or not 0 <= peak_index < len(spectrum.mz)
                or not np.isclose(float(mz), spectrum.mz[peak_index], atol=1e-8, rtol=0)):
            raise ValueError(f"Bag {i}: peak_index does not match this MS2 spectrum")
        children = list(children_by_peak.get(peak_index, []))
        for fragment in fragments:
            fragment.ms3_spectra = children
        result.append((mz, fragments, children))
    return result


def random_ms2_with_ms3(spectra=None, *, seed=None, mz_tolerance=0.01):
    """Return a random MS2 Spectrum with at least one linked MS3 child.

    Pass all related scans (including MS3), or omit spectra to use set_spectrum_source context.
    Uses SINGLE_BEST_SCAN records and the same linkage as find_ms3_children.
    Selection is uniform over eligible MS2 records; seed makes it repeatable.
    Raises ValueError when no eligible MS2 exists.
    """
    import random

    if not np.isfinite(mz_tolerance) or mz_tolerance < 0:
        raise ValueError("mz_tolerance must be finite and nonnegative")
    if spectra is None:
        spectra = _source_spectra
    if spectra is None:
        raise ValueError("Call set_spectrum_source or pass spectra including MS2 and MS3 scans")

    by_feature = {}
    candidates = []
    for scan in spectra:
        if scan.ms_level not in (2, 3) or scan.metadata.get("merge_type") != "SINGLE_BEST_SCAN":
            continue
        key = (scan.smiles, scan.metadata.get("feature_id"))
        by_feature.setdefault(key, []).append(scan)
        if scan.ms_level == 2:
            candidates.append(scan)

    random.Random(seed).shuffle(candidates)
    forests = {}
    for candidate in candidates:
        key = (candidate.smiles, candidate.metadata.get("feature_id"))
        if key not in forests:
            forests[key] = parse_spectrum_forest(by_feature[key], mz_tol=mz_tolerance)
        if find_ms3_children(candidate, forest=forests[key], mz_tolerance=mz_tolerance):
            return candidate
    raise ValueError("No MS2 spectrum with linked MS3 children was found")


def _magma_fragment_graph(engine, fragment_mask):
    """Induce the heavy graph; restore aromaticity only on intact parent rings."""
    from rdkit import Chem
    from src.chemistry.heavy_graph import _canonical_graph

    parent = Chem.MolFromSmiles(engine.smiles)
    if parent is None or parent.GetNumAtoms() != engine.mol.GetNumAtoms():
        raise ValueError("MAGMa parent atom mapping failed")
    if [a.GetSymbol() for a in parent.GetAtoms()] != list(engine.atom_symbols):
        raise ValueError("MAGMa parent element mapping failed")
    edges = lambda mol: {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
    }
    if edges(parent) != edges(engine.mol):
        raise ValueError("MAGMa parent bond mapping failed")

    keep = [i for i in range(engine.natoms) if int(fragment_mask) & (1 << i)]
    indices = {old: new for new, old in enumerate(keep)}
    restored = set()
    for ring in parent.GetRingInfo().BondRings():
        bonds = [parent.GetBondWithIdx(i) for i in ring]
        if all(b.GetIsAromatic() and b.GetBeginAtomIdx() in indices
               and b.GetEndAtomIdx() in indices for b in bonds):
            restored.update(tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                            for b in bonds)
    aromatic_atoms = {i for edge in restored for i in edge}
    atoms = [{"element": engine.atom_symbols[i], "aromatic": i in aromatic_atoms}
             for i in keep]
    bonds = []
    for bond in engine.mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in indices and b in indices:
            aromatic = tuple(sorted((a, b))) in restored
            bonds.append({"i": min(indices[a], indices[b]),
                          "j": max(indices[a], indices[b]),
                          "order": 1.5 if aromatic else bond.GetBondTypeAsDouble(),
                          "aromatic": aromatic})
    return _canonical_graph({"atoms": atoms, "bonds": bonds}), keep


def assign_candidates_to_nearest_peak(candidate_bags):
    """Assign each exact fragment identity to its nearest already-matched peak.

    Identity means canonical heavy graph + ionic formula + charge, not source
    ID. Ties prefer lower m/z, then earlier bag order. Do not read intensities,
    change peak locations, or discard empty bags. Different structures remain
    latent alternatives at their winning peak. Candidate objects/metadata are
    copied; the measured MS3 list stays associated with its original bag.

    Apply this *after* selecting one scan's peak axis when MAGMa was run once
    on the union of several scans. Existing tolerance matching defines the
    eligible bags; this helper never introduces an additional match.
    """
    from src.chemistry.heavy_graph import _canonical_graph

    bags = [_unpack_bag(bag) for bag in candidate_bags]
    identities, winners = {}, {}
    for index, (mz, fragments, _) in enumerate(bags):
        mz = float(mz)
        if not np.isfinite(mz) or mz <= 0:
            raise ValueError("Bag m/z values must be finite and positive")
        for position, fragment in enumerate(fragments):
            formula = formula_counts(fragment.ionic_formula)
            raw = json.dumps({"graph": fragment.graph, "formula": formula,
                              "charge": fragment.charge}, sort_keys=True)
            if raw not in identities:
                identities[raw] = _candidate_digest({
                    "graph": _canonical_graph(fragment.graph),
                    "formula": formula, "charge": fragment.charge,
                })
            identity = identities[raw]
            theoretical_mz = ionic_mass(formula, fragment.charge)
            rank = (abs(mz - theoretical_mz), mz, index, position)
            if identity not in winners or rank < winners[identity][0]:
                winners[identity] = (rank, fragment, theoretical_mz)

    result = [(mz, [], children) for mz, _, children in bags]
    # Iterate by original position so surviving alternatives keep their order.
    for rank, fragment, theoretical_mz in sorted(winners.values(), key=lambda item: item[0][2:]):
        index = rank[2]
        mz, fragments, children = result[index]
        chosen = copy(fragment)
        chosen.metadata = dict(fragment.metadata)
        chosen.precursor_mz = float(mz)
        chosen.metadata.update(
            theoretical_mz=theoretical_mz,
            mass_error_da=float(mz) - theoretical_mz,
            mass_error_ppm=(float(mz) - theoretical_mz) / theoretical_mz * 1e6,
        )
        chosen.ms3_spectra = children
        fragments.append(chosen)
    return result


def get_candidate_bags(spectrum, spectra=None, *, forest=None,
                       upstream_path=None, mz_tolerance=0.005,
                       ppm_tolerance=20.0, max_tree_depth=3, max_broken_bonds=6,
                       max_boundary_cuts=6, ms3_mz_tolerance=0.01,
                       peak_assignment="all"):
    """Run MAGMa and return [(peak_mz, [FragmentCandidate], [MS3 Spectrum])].

    Related scans come from spectra/forest, or set_spectrum_source context by default.
    Each candidate.ms3_spectra references its bag's measured MS3 list.
    Preserve peak order, including duplicate m/z values and unmatched peaks.
    Match every compatible native fragment/H-shift/mass-channel hypothesis;
    deduplicate only equal heavy graph + full ionic formula + charge, retaining
    all origins. Candidate precursor_mz is the observed fragment m/z.
    Intensities do not select or rank candidates. No model weights are needed.
    Wider matching tolerances are allowed for exploration, but the Glacier
    adapter still requires formula/mass agreement within max(0.005 Da, 20 ppm).
    peak_assignment="nearest" assigns each identity to only its closest
    eligible centroid; "all" preserves the original multi-match behavior.
    """
    if peak_assignment not in {"all", "nearest"}:
        raise ValueError("peak_assignment must be 'all' or 'nearest'")
    if spectrum.ms_level != 2:
        raise ValueError("Use an MS2 spectrum: MS3+ SMILES identify the original parent.")
    adduct = spectrum.metadata.get("adduct")
    if adduct not in {"[M+H]+", "[M]+"} or spectrum.metadata.get("charge") != 1:
        raise ValueError("Supported precursors have charge +1 and adduct [M+H]+ or [M]+.")
    for name, value in (("mz_tolerance", mz_tolerance), ("ppm_tolerance", ppm_tolerance)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (("max_tree_depth", max_tree_depth),
                        ("max_broken_bonds", max_broken_bonds),
                        ("max_boundary_cuts", max_boundary_cuts)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    mzs = np.asarray(spectrum.mz, dtype=float)
    parent_mz = float(spectrum.precursor_mz)
    if (mzs.ndim != 1 or not np.isfinite(mzs).all() or (mzs <= 0).any()
            or not np.isfinite(parent_mz) or parent_mz <= 0):
        raise ValueError("Spectrum m/z values must be finite and positive")

    _select_magma_upstream(upstream_path)
    from rdkit import Chem
    from ms_pred import common
    from ms_pred.magma.fragmentation import FragmentEngine

    parent = Chem.MolFromSmiles(spectrum.smiles)
    if parent is None:
        raise ValueError("Invalid precursor SMILES")
    if len(Chem.GetMolFrags(parent)) != 1:
        raise ValueError("Disconnected precursor graphs are unsupported")
    if any(a.GetIsotope() or a.GetAtomicNum() == 1 for a in parent.GetAtoms()):
        raise ValueError("Isotope labels and explicit hydrogen nodes are unsupported")

    engine = FragmentEngine(spectrum.smiles, max_tree_depth=max_tree_depth,
                            max_broken_bonds=max_broken_bonds)
    tolerance = lambda mz: max(mz_tolerance, ppm_tolerance * mz * 1e-6)
    if abs(engine.full_weight + common.ion2mass[adduct] - parent_mz) > tolerance(parent_mz):
        raise ValueError("The canonical precursor structure/adduct does not match precursor_mz")
    if not len(mzs):
        return []
    engine.generate_fragments()
    tolerances = np.maximum(mz_tolerance, ppm_tolerance * mzs * 1e-6)
    per_peak = [{} for _ in mzs]
    graph_cache = {}
    added_proton = int(adduct == "[M+H]+")
    inventory = Counter(engine.atom_symbols)
    inventory["H"] = int(engine.total_hs) + added_proton
    channels = (("source_adduct_retaining", added_proton),
                ("positive_charge_transfer", 0))

    for frag_hash, mask, shift_index, mass, score in zip(*engine.get_frag_masses()):
        boundary_cuts = int(engine.score_fragment(int(mask))[0])
        if boundary_cuts > max_boundary_cuts:
            continue
        keep = [i for i in range(engine.natoms) if int(mask) & (1 << i)]
        signed_shift = int(engine.shift_buckets[int(shift_index)])
        neutral = Counter(engine.atom_symbols[i] for i in keep)
        neutral["H"] = int(sum(engine.atom_hs[i] for i in keep)) + signed_shift
        if neutral["H"] < 0 or abs(exact_mass(dict(neutral)) - float(mass)) > 1e-7:
            raise ValueError("MAGMa native mass/formula disagreement")

        for channel, added_h in channels:
            counts = dict(neutral)
            counts["H"] += added_h
            counts = formula_counts(counts)
            if any(n > inventory[element] for element, n in counts.items()):
                raise ValueError("Candidate exceeds the precursor/adduct elemental inventory")
            theoretical_mz = ionic_mass(counts, charge=1)
            matching = np.flatnonzero(np.abs(mzs - theoretical_mz) <= tolerances)
            if not len(matching):
                continue
            if frag_hash not in graph_cache:
                graph_cache[frag_hash] = _magma_fragment_graph(engine, mask)
            graph, atom_indices = graph_cache[frag_hash]
            identity = {**graph, "ionic_formula": counts, "charge": 1}
            candidate_id = "candidate_" + _candidate_digest(identity)
            entry = engine.frag_to_entry[frag_hash]
            origin = {
                "fragment_hash": str(frag_hash), "fragment_mask": str(int(mask)),
                "parent_atom_indices": atom_indices, "base_formula": entry["form"],
                "h_shift_bucket_index": int(shift_index), "h_shift_signed": signed_shift,
                "mass_channel": channel, "engine_mass": float(mass),
                "boundary_cut_count": boundary_cuts, "bond_score": float(score),
                "tree_depth": int(entry["tree_depth"]),
                "engine_max_broken": int(entry["max_broken"]),
            }
            for index in matching:
                if candidate_id not in per_peak[index]:
                    per_peak[index][candidate_id] = FragmentCandidate(
                        candidate_id=candidate_id, graph=deepcopy(graph),
                        ionic_formula=dict(counts), charge=1, precursor_mz=float(mzs[index]),
                        metadata={
                            "peak_index": int(index), "theoretical_mz": theoretical_mz,
                            "source_parent_smiles": spectrum.smiles,
                            "engine_parent_smiles": engine.smiles, "parent_adduct": adduct,
                            "charge_provenance": "assumed singly positive fragment",
                            "origins": [],
                        },
                    )
                per_peak[index][candidate_id].metadata["origins"].append(deepcopy(origin))
    bags = [(float(mz), [candidates[key] for key in sorted(candidates)])
            for mz, candidates in zip(mzs, per_peak)]
    if peak_assignment == "nearest":
        bags = assign_candidates_to_nearest_peak(bags)
    return attach_ms3_spectra(spectrum, bags, spectra, forest=forest,
                               mz_tolerance=ms3_mz_tolerance)

def filter_same_formula_bags(candidate_bags):
    """Keep nonempty bags with exactly one ionic formula/charge combination.

    Formula strings and element-count dictionaries are compared by composition.
    Preserve bag order and fragment order, without merging bags or mutating input.
    Singleton bags qualify; empty and mixed-formula bags do not.
    The third field and candidate MS3 references are preserved.
    """
    result = []
    for bag in candidate_bags:
        mz, fragments, ms3_spectra = _unpack_bag(bag)
        formulas = {
            (tuple(formula_counts(fragment.ionic_formula).items()), fragment.charge)
            for fragment in fragments
        }
        if len(formulas) == 1:
            result.append((mz, list(fragments), ms3_spectra))
    return result
