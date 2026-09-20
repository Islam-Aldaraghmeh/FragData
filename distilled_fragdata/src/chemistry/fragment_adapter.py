"""Prepare bounded ionic states for the official pretrained GLACIER model."""
from collections.abc import Mapping
from copy import deepcopy
import math

import numpy as np
from rdkit import Chem, rdBase

from src.chemistry.formula import formula_counts, ionic_mass
from src.chemistry._enumeration import enumerate_microstates
from src.chemistry._state_graph import (
    stripped_from_smiles, _counts_from_mol, _graph_record,
    _validate_state, _feature_hash,
)
from src.models.glacier_api import records_from_mol


def project_neutral_references(ionic_smiles):
    """All valid fixed-graph f-H+ references; complete reference+adduct is f.

    This is a representation control, never an alteration of the ionic state.
    Returned references are deduplicated by every actual fixed-context tensor.
    """
    ion = Chem.MolFromSmiles(ionic_smiles)
    if ion is None or any(a.GetIsotope() for a in ion.GetAtoms()):
        return []
    if len(Chem.GetMolFrags(ion)) != 1 or sum(a.GetFormalCharge() for a in ion.GetAtoms()) != 1:
        return []
    if any(a.GetNumRadicalElectrons() for a in ion.GetAtoms()):
        return []
    counts = _counts_from_mol(ion)
    if counts.get("H", 0) == 0:
        return []
    reference_counts = dict(counts)
    reference_counts["H"] -= 1
    reference_counts = {s: n for s, n in reference_counts.items() if n}
    graph = _graph_record(ion)
    candidates = {}
    for site, atom in enumerate(ion.GetAtoms()):
        if atom.GetTotalNumHs() == 0:
            continue
        reference = Chem.Mol(ion)
        for a in reference.GetAtoms():
            a.SetNumExplicitHs(a.GetTotalNumHs())
            a.SetNoImplicit(True)
        target = reference.GetAtomWithIdx(site)
        target.SetNumExplicitHs(target.GetNumExplicitHs()-1)
        target.SetFormalCharge(target.GetFormalCharge()-1)
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(reference)
            if _validate_state(reference, graph, reference_counts, 0):
                continue
            smiles = Chem.MolToSmiles(reference, canonical=True, isomericSmiles=False)
            candidates.setdefault(smiles, []).append(site)
        except (ValueError, RuntimeError):
            continue
    results, seen = [], set()
    for smiles in sorted(candidates):
        try:
            signature = _feature_hash(Chem.MolFromSmiles(smiles), "[M+H]+")
        except (ValueError, RuntimeError):
            continue
        if signature in seen:
            continue
        seen.add(signature)
        results.append({"reference_smiles": smiles, "ionic_smiles": ionic_smiles,
                        "ionic_formula": counts, "reference_formula": reference_counts,
                        "charge": 1, "formula_mz": ionic_mass(counts),
                        "removed_proton_sites": candidates[smiles],
                        "adduct": "[M+H]+", "feature_hash": signature,
                        "total_abs_charge": sum(abs(a.GetFormalCharge()) for a in Chem.MolFromSmiles(smiles).GetAtoms()),
                        "representation": "neutral_reference_plus_proton"})
    return results


class FragmentGlacierAdapter:
    """Keep enumeration target-independent and average normalized state spectra.

    ``glacier`` is a GlacierAPI instance, required only for prediction. Construct
    one first to configure the pinned featurizer when ms_pred is not installed.
    """

    def __init__(self, glacier=None, max_states=4, **enumeration_caps):
        self.glacier = glacier
        self.max_states = max_states
        self.enumeration_caps = enumeration_caps

    def prepare(self, candidate, collision_energy, instrument="Orbitrap"):
        """Return the complete enumeration audit and one GLACIER input per state.

        Accept a FragmentCandidate, a prototype candidate with ``stripped``, or
        the exact five-field stripped record. No source local H/Q enters search.
        """
        if hasattr(candidate, "stripped"):
            stripped = candidate.stripped
        elif isinstance(candidate, Mapping):
            stripped = candidate.get("stripped", candidate)
        else:
            raise TypeError("Expected FragmentCandidate or stripped record")
        ce = float(collision_energy)
        if not math.isfinite(ce):
            raise ValueError("Collision energy must be finite")
        result = enumerate_microstates(deepcopy(stripped), self.max_states,
                                       **self.enumeration_caps)
        inputs = [{"records": records_from_mol(Chem.MolFromSmiles(state["ionic_smiles"])),
                   "collision_energy": ce, "adduct": "[M]+", "instrument": instrument}
                  for state in result["states"]]
        return {**result, "inputs": inputs}

    def predict(self, candidate, collision_energy, instrument="Orbitrap"):
        """Return a Spectrum using the prototype's equal-weight microstate mean."""
        from src.spectra.spectrum import (
            Spectrum, clean_raw, prepare_spectrum, normalize_spectrum,
            mean_normalized_spectrum,
        )
        if self.glacier is None:
            raise ValueError("Prediction requires a GlacierAPI instance")
        prepared = self.prepare(candidate, collision_energy, instrument)
        if not prepared["states"]:
            raise ValueError(f"No represented states: {prepared['reason']}")
        spectra = []
        for model_input in prepared["inputs"]:
            raw = self.glacier.predict_records(**model_input)["spec"].detach().cpu().numpy()
            assert np.isfinite(raw).all()
            spectra.append(normalize_spectrum(prepare_spectrum(clean_raw(raw))))
        return Spectrum.from_peaks(mean_normalized_spectrum(spectra, prepared=True))
