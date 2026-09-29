"""Standalone opt-in radical experiment using the shared exact-graph enumerator."""
from copy import deepcopy
import math
from rdkit import Chem
from . import baseline_snapshot
from .experiment import predict_prepared
from src.chemistry.fragment_adapter import FragmentGlacierAdapter
from src.models.glacier_api import records_from_mol
from src.chemistry._radical_enumeration import (
    ROUTE, PARITY_FAILURE, enumerate_radical_states, _radical_domains, _validate_radical,
)


def prepare_radical_candidate(candidate, collision_energy, *, model=None,
                              allow_radical=False, max_states=4, instrument="Orbitrap"):
    """Baseline first; only its explicit odd-electron rejection permits fallback."""
    ce = float(collision_energy)
    if not math.isfinite(ce):
        raise ValueError("Collision energy must be finite")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 1:
        raise ValueError("max_states must be a positive integer")
    original = FragmentGlacierAdapter(model, max_states=max_states).prepare_baseline(candidate, ce, instrument)
    baseline = {**original, "route": "baseline", "experimental": False,
                "baseline_reason": original["reason"]}
    if original["states"] or not allow_radical or original["reason"] != PARITY_FAILURE:
        return baseline
    stripped = candidate.stripped if hasattr(candidate, "stripped") else candidate.get("stripped", candidate)
    result = enumerate_radical_states(deepcopy(stripped), max_states=max_states)
    inputs = [dict(records=records_from_mol(Chem.MolFromSmiles(s["ionic_smiles"])),
                   collision_energy=ce, instrument=instrument, adduct="[M]+")
              for s in result["states"]]
    return {**result, "inputs": inputs, "route": ROUTE, "experimental": True,
            "baseline_reason": original["reason"]}


def predict_radical_candidate(model, candidate, collision_energy, *,
                              allow_radical=False, max_states=4, instrument="Orbitrap"):
    prepared = prepare_radical_candidate(candidate, collision_energy, model=model,
        allow_radical=allow_radical, max_states=max_states, instrument=instrument)
    return {**prepared, "spectrum": predict_prepared(model, prepared)}
