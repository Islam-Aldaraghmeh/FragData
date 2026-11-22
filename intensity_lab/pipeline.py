"""One-call setup shared by the notebook and command-line runs."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .features import build_feature_store
from .raw import load_raw_examples
from .splits import make_splits


@dataclass
class Data:
    examples: list
    fs: object
    splits: dict

    def data_key(self, split="molecule"):
        digest = hashlib.sha256(json.dumps(
            {k: v.tolist() for k, v in self.splits[split].items()}).encode()).hexdigest()[:12]
        return f"{self.fs.key}-{split}-{digest}"

    @property
    def peaks(self):
        return {e["cache_id"]: e["peaks"] for e in self.examples}


def load_data(num_workers=12, verbose=True, split_seed=0):
    examples = load_raw_examples(num_workers=num_workers, verbose=verbose)
    fs = build_feature_store(examples, num_workers=num_workers, verbose=verbose)
    splits = make_splits(fs.meta, seed=split_seed)
    return Data(examples, fs, splits)


def set_threads(n=None):
    import os

    import torch

    n = n or max(1, min(10, (os.cpu_count() or 4) - 2))
    torch.set_num_threads(n)
    return n
