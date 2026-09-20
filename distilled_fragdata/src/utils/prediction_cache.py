"""Small atomic disk cache of raw predicted spectra, never target spectra."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


class PredictionCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(candidate_id, state_id, formula, charge, collision_energy, instrument,
            checkpoint_id, *, input_fingerprint, settings=None):
        """Include exact model-input hash and runtime/policy settings if used."""
        payload = {"schema": 1, "candidate": candidate_id, "state": state_id,
                   "formula": formula, "charge": charge, "ce": collision_energy,
                   "instrument": instrument, "checkpoint": checkpoint_id,
                   "inputs": input_fingerprint, "settings": settings or {}}
        data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(data.encode()).hexdigest()

    def _path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Expected a SHA256 cache key")
        return self.directory / (key + ".npz")

    def get(self, key):
        path = self._path(key)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as saved:
            return saved["spec"].copy()

    def put(self, key, peaks):
        path = self._path(key)
        spec = np.asarray(peaks)
        if spec.ndim != 2 or spec.shape[1] != 2 or not np.isfinite(spec).all():
            raise ValueError("Expected finite N x 2 raw prediction")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, spec=spec)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
