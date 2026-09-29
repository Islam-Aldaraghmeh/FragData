"""Frozen scientific notebook definitions with repository-local asset discovery; no demo cells execute."""
import json
from pathlib import Path
import numpy as np
from Spectrum_Classes import Spectrum, SpectrumForest, parse_spectrum_forest


# Notebook cell: candidate-setup
import hashlib
import os
import subprocess
import sys
import warnings
from collections import Counter
from copy import deepcopy

# Run the notebook from the repository, or one of its subdirectories.
REPO_ROOT = next(
    p for p in (Path.cwd(), *Path.cwd().parents)
    if (p / "distilled_fragdata").is_dir() and (p / "Spectrum_Classes.py").is_file()
)
if str(REPO_ROOT / "distilled_fragdata") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "distilled_fragdata"))

from src.data.candidate_bags import FragmentCandidate
from src.chemistry.formula import formula_counts, ionic_mass, exact_mass
from src.models.glacier_api import GlacierAPI, COMMIT

# Asset discovery is repository-local; scientific definitions remain frozen.
from src.models.assets import asset_path as _asset_path


def _select_magma_upstream(upstream_path=None):
    """Use the same pinned checkout as Glacier, without requiring weights."""
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

# Notebook cell: ms3-linkage
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

    Supply related spectra or a prebuilt forest. In this notebook, omitted
    spectra use spectra_by_smiles[spectrum.smiles], then raw_spectra as fallback.
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
            spectra = globals().get("spectra_by_smiles", {}).get(spectrum.smiles)
            if spectra is None:
                spectra = globals().get("raw_spectra")
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


# Notebook cell: magma-graph
def _magma_fragment_graph(engine, fragment_mask):
    """Induce the heavy graph; restore aromaticity only on intact parent rings."""
    from rdkit import Chem
    from src.chemistry._state_graph import _canonical_graph

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


def get_candidate_bags(spectrum, spectra=None, *, forest=None,
                       upstream_path=None, mz_tolerance=0.005,
                       ppm_tolerance=20.0, max_tree_depth=3, max_broken_bonds=6,
                       max_boundary_cuts=6, ms3_mz_tolerance=0.01):
    """Run MAGMa and return [(peak_mz, [FragmentCandidate], [MS3 Spectrum])].

    Related scans come from spectra/forest, or the notebook dataset by default.
    Each candidate.ms3_spectra references its bag's measured MS3 list.
    Preserve peak order, including duplicate m/z values and unmatched peaks.
    Match every compatible native fragment/H-shift/mass-channel hypothesis;
    deduplicate only equal heavy graph + full ionic formula + charge, retaining
    all origins. Candidate precursor_mz is the observed fragment m/z.
    Intensities do not select or rank candidates. No model weights are needed.
    Wider matching tolerances are allowed for exploration, but the Glacier
    adapter still requires formula/mass agreement within max(0.005 Da, 20 ppm).
    """
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
    return attach_ms3_spectra(spectrum, bags, spectra, forest=forest,
                               mz_tolerance=ms3_mz_tolerance)

# Notebook cell: formula-filter
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

# Notebook cell: glacier-prediction
def load_glacier_model(upstream_path=None, checkpoint_path=None):
    """Load and return the repo's verified, frozen CPU GlacierAPI model wrapper.

    Explicit paths override GLACIER_UPSTREAM / GLACIER_CHECKPOINT and repository-local
    defaults. The existing API checks the source commit and checkpoint
    SHA256. Loading happens here, so repeated prediction calls reuse the weights.
    """
    upstream = _asset_path(upstream_path, "GLACIER_UPSTREAM", "vendor/ms-pred")
    checkpoint = _asset_path(checkpoint_path, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt")
    glacier = GlacierAPI(upstream, checkpoint)
    _ = glacier.model
    return glacier


def predict_candidate_bags(glacier, candidate_bags, *, collision_energy,
                           instrument="Orbitrap", max_states=4,
                           on_unsupported="none"):
    """Return [(bag_mz, [Nx2 spectrum or None], ms3_spectra)] at the same indices.

    collision_energy is an explicit nonnegative scalar in the model's input
    convention, shared by these predictions. It is not inferred from MSnLib
    metadata. The adapter predicts closed-shell +1 ionic states with [M]+ and
    averages their individually normalized spectra (max_states=4 by default).
    Only unsupported ionic-state hypotheses may become None. Other ValueErrors
    include the bag/candidate index; other exception types propagate unchanged.
    Use on_unsupported='raise' for strict use.
    """
    from src.chemistry.fragment_adapter import FragmentGlacierAdapter

    ce = float(collision_energy)
    if not np.isfinite(ce) or ce < 0:
        raise ValueError("collision_energy must be a finite, nonnegative scalar")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    if on_unsupported not in {"none", "raise"}:
        raise ValueError("on_unsupported must be 'none' or 'raise'")
    adapter = FragmentGlacierAdapter(glacier, max_states=max_states)
    predictions = []
    for bag_index, bag in enumerate(candidate_bags):
        mz, fragments, ms3_spectra = _unpack_bag(bag)
        spectra = []
        for fragment_index, fragment in enumerate(fragments):
            try:
                predicted = adapter.predict(fragment, collision_energy=ce, instrument=instrument)
            except ValueError as error:
                context = (f"bag {bag_index}, candidate {fragment_index} "
                           f"({fragment.candidate_id}, m/z {mz}): {error}")
                # This is the adapter's explicit no-representable-state failure.
                if on_unsupported == "none" and str(error).startswith("No represented states:"):
                    warnings.warn(context, RuntimeWarning, stacklevel=2)
                    spectra.append(None)
                    continue
                raise ValueError(context) from error
            spectra.append(predicted.as_array())
        predictions.append((mz, spectra, ms3_spectra))
    return predictions