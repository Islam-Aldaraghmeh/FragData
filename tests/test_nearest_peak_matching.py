"""Nearest-centroid matching is structural, scan-local, and intensity blind."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magma_workflow import candidates
from magma_workflow.dataset import Spectrum
from src.chemistry.formula import ionic_mass


def fragment(name="candidate", *, branch=False, formula="C4H5"):
    edges = [(0, 1), (0, 2), (0, 3)] if branch else [(0, 1), (1, 2), (2, 3)]
    graph = {
        "atoms": [{"element": "C", "aromatic": False} for _ in range(4)],
        "bonds": [{"i": i, "j": j, "order": 1., "aromatic": False}
                  for i, j in edges],
    }
    return candidates.FragmentCandidate(name, graph, formula, 1,
                                         ionic_mass(formula), metadata={})


def matched_bags(masses, item=None):
    item = item or fragment()
    result = []
    for index, mz in enumerate(masses):
        candidate = deepcopy(item)
        candidate.metadata["peak_index"] = index
        candidate.precursor_mz = float(mz)
        result.append((mz, [candidate], []))
    return result


def occupied(bags):
    return [mz for mz, fragments, _ in bags if fragments]


def test_nearby_centroids_have_only_one_owner_and_signed_mass_error():
    masses = [53.035224, 53.038414, 53.0415, 53.042976]
    bags = matched_bags(masses)
    actual = candidates.assign_candidates_to_nearest_peak(bags)
    assert [bag[0] for bag in actual] == masses
    assert occupied(actual) == [53.038414]
    assert sum(len(bag[1]) for bag in actual) == 1
    chosen = actual[1][1][0]
    theoretical = ionic_mass("C4H5", 1)
    assert theoretical == pytest.approx(53.03857658)
    assert chosen.metadata["mass_error_da"] == pytest.approx(53.038414 - theoretical)
    assert chosen.metadata["mass_error_ppm"] == pytest.approx(
        (53.038414 - theoretical) / theoretical * 1e6)
    assert chosen.precursor_mz == 53.038414


def test_tie_prefers_lower_mass_independent_of_bag_order():
    mass = ionic_mass("C4H5")
    delta = 2 ** -10
    bags = matched_bags([mass + delta, mass - delta])
    assert occupied(candidates.assign_candidates_to_nearest_peak(bags)) == [mass - delta]
    assert occupied(candidates.assign_candidates_to_nearest_peak(bags[::-1])) == [mass - delta]
    duplicate_mass = matched_bags([mass, mass])
    actual = candidates.assign_candidates_to_nearest_peak(duplicate_mass)
    assert len(actual[0][1]) == 1 and actual[1][1] == []


def test_exact_identity_ignores_source_id_and_atom_order_but_preserves_isomers():
    linear = fragment("source-a")
    linear.graph["atoms"][0]["element"] = "N"
    linear.ionic_formula = "C3H5N"
    duplicate = deepcopy(linear)
    duplicate.candidate_id = "source-b"
    duplicate.graph["atoms"].reverse()
    duplicate.graph["bonds"] = [
        {**bond, "i": 3 - bond["j"], "j": 3 - bond["i"]}
        for bond in linear.graph["bonds"]
    ]
    branched = fragment("source-a", branch=True, formula="C3H5N")
    branched.graph["atoms"][0]["element"] = "N"
    mass = ionic_mass("C3H5N")
    bags = [(mass - .001, [linear, duplicate, branched], []),
            (mass + .004, [duplicate, branched], [])]
    actual = candidates.assign_candidates_to_nearest_peak(bags)
    assert [len(bag[1]) for bag in actual] == [2, 0]
    assert [f.graph for f in actual[0][1]] == [linear.graph, branched.graph]


def test_formula_and_charge_are_part_of_identity():
    first = fragment()
    second = fragment(formula="C4H7")
    doubly_charged = fragment()
    doubly_charged.charge = 2
    actual = candidates.assign_candidates_to_nearest_peak(
        [(53., [first, second, doubly_charged], [])])
    assert len(actual[0][1]) == 3


def test_no_input_mutation_and_ms3_stays_with_original_bag():
    bags = matched_bags([53.035224, 53.038414])
    observations = [object()]
    bags[1][2].extend(observations)
    original = deepcopy(bags[1][1][0])
    actual = candidates.assign_candidates_to_nearest_peak(bags)
    assert all(len(bag[1]) == 1 for bag in bags)
    assert bags[1][1][0] == original
    chosen = actual[1][1][0]
    assert chosen is not bags[1][1][0]
    assert chosen.metadata is not bags[1][1][0].metadata
    assert actual[1][2] is bags[1][2]
    assert chosen.ms3_spectra is actual[1][2]
    assert bags[1][1][0].ms3_spectra == []


def test_assignment_ignores_experimental_intensity_metadata():
    bags = matched_bags([53.035224, 53.038414])
    bags[0][1][0].metadata.update(intensity=1e100, normalized_intensity=1., peak_rank=1)
    bags[1][1][0].metadata.update(intensity=0., normalized_intensity=0., peak_rank=100)
    actual = candidates.assign_candidates_to_nearest_peak(bags)
    assert occupied(actual) == [53.038414]
    bags[0][1][0].metadata["intensity"] = 0.
    bags[1][1][0].metadata["intensity"] = 1e100
    assert occupied(candidates.assign_candidates_to_nearest_peak(bags)) == [53.038414]


def test_nearest_is_applied_after_selecting_each_collision_energy_axis():
    # The high-energy centroid is slightly farther away. Assigning ownership on
    # the union would wrongly strip support from its otherwise valid scan.
    union_bags = matched_bags([53.038414, 53.039])
    first_scan = candidates.assign_candidates_to_nearest_peak([union_bags[0]])
    second_scan = candidates.assign_candidates_to_nearest_peak([union_bags[1]])
    assert len(first_scan[0][1]) == len(second_scan[0][1]) == 1
    assert candidates.assign_candidates_to_nearest_peak(union_bags)[1][1] == []


def test_empty_and_legacy_bags_preserve_axis():
    assert candidates.assign_candidates_to_nearest_peak([]) == []
    assert candidates.assign_candidates_to_nearest_peak([(1., []), (2., [], [])]) == [
        (1., [], []), (2., [], [])]


@pytest.mark.parametrize("mz", [np.nan, np.inf, 0., -1.])
def test_invalid_peak_locations_raise(mz):
    with pytest.raises(ValueError, match="finite and positive"):
        candidates.assign_candidates_to_nearest_peak([(mz, [])])


def test_generator_default_all_matching_is_unchanged_and_nearest_is_opt_in():
    if not (ROOT / "vendor/ms-pred").is_dir():
        pytest.skip("Pinned MAGMa checkout is not installed")

    class IntensityBlindSpectrum(Spectrum):
        @property
        def intensity(self):
            raise AssertionError("MAGMa assignment read experimental intensities")

    mass = ionic_mass("CH3")
    spectrum = IntensityBlindSpectrum(
        "CC", ionic_mass("C2H7"), 2,
        np.asarray([[mass - .002, 1e6], [mass + .001, 0.]]),
        {"adduct": "[M+H]+", "charge": 1},
    )
    original = candidates.get_candidate_bags(spectrum, spectra=[])
    explicit_all = candidates.get_candidate_bags(spectrum, spectra=[], peak_assignment="all")
    assert original == explicit_all
    assert all(bag[1] for bag in original)
    nearest = candidates.get_candidate_bags(spectrum, spectra=[], peak_assignment="nearest")
    assert nearest == candidates.assign_candidates_to_nearest_peak(original)
    assert nearest[0][1] == [] and nearest[1][1]


def test_invalid_assignment_option_fails_before_loading_magma():
    with pytest.raises(ValueError, match="peak_assignment"):
        candidates.get_candidate_bags(None, peak_assignment="highest_intensity")
