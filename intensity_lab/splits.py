"""Leakage-safe splits.

A *molecule group* joins spectra that share either the first InChIKey block
(2D connectivity) or MAGMa's canonical SMILES. Every collision energy and every
replicate of one molecule therefore lands in the same split.

* ``molecule`` split: random 80/10/10 over molecule groups (the main benchmark).
* ``scaffold`` split: whole Bemis-Murcko scaffolds per split, largest scaffolds
  in train, so test chemotypes are unseen (a harder generalization test).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import CACHE_DIR


def molecule_groups(meta):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip("ik:" + meta["mol_id"], "smi:" + meta["engine_smiles"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = ["ik:" + m for m in meta["mol_id"]]
    labels = pd.factorize(pd.Series([find(r) for r in roots]))[0]
    return labels


def _scaffold(smiles):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    mol = Chem.MolFromSmiles(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol) if mol is not None else ""
    return scaffold or "acyclic"


def make_splits(meta, seed=0, fractions=(0.8, 0.1, 0.1), cache_dir=CACHE_DIR):
    """Return {'molecule': {'train','val','test'}, 'scaffold': {...}} of spectrum indices."""
    digest = hashlib.sha256(json.dumps([seed, list(fractions), list(meta["cache_id"])]).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"splits_{digest}.json"
    if path.exists():
        raw = json.loads(path.read_text())
        return {k: {s: np.array(v[s], dtype=np.int64) for s in v} for k, v in raw.items()}

    groups = molecule_groups(meta)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_train = int(fractions[0] * len(unique))
    n_val = int(fractions[1] * len(unique))
    assign = {g: "train" for g in unique[:n_train]}
    assign.update({g: "val" for g in unique[n_train:n_train + n_val]})
    assign.update({g: "test" for g in unique[n_train + n_val:]})
    mol_split = np.array([assign[g] for g in groups])

    # Scaffold split: all spectra of a scaffold together; big scaffolds -> train.
    group_smiles = pd.Series(meta["engine_smiles"].values).groupby(groups).first()
    group_scaffold = {g: _scaffold(s) for g, s in group_smiles.items()}
    scaffold_of_spec = np.array([group_scaffold[g] for g in groups])
    sizes = pd.Series(groups).groupby(scaffold_of_spec).nunique().sort_values(ascending=False)
    order = list(sizes.index)
    # Shuffle within equal sizes so ties are not alphabetical.
    order = sorted(order, key=lambda s: (-sizes[s], rng.random()))
    total = len(unique)
    scaffold_assign, counts = {}, {"train": 0, "val": 0}
    for scaffold in order:
        n = sizes[scaffold]
        if counts["train"] + n <= fractions[0] * total:
            split = "train"
        elif counts["val"] + n <= fractions[1] * total:
            split = "val"
        else:
            split = "test"
        counts[split] = counts.get(split, 0) + n
        scaffold_assign[scaffold] = split
    scaf_split = np.array([scaffold_assign[s] for s in scaffold_of_spec])

    out = {}
    for name, labels in (("molecule", mol_split), ("scaffold", scaf_split)):
        out[name] = {s: np.flatnonzero(labels == s) for s in ("train", "val", "test")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: {s: v[s].tolist() for s in v} for k, v in out.items()}))
    return out


def audit_split(meta, split):
    """Print sizes and verify there is no molecule or SMILES shared across parts."""
    groups = molecule_groups(meta)
    rows = []
    sets = {}
    for part, idx in split.items():
        sets[part] = {
            "group": set(groups[idx]), "ik": set(meta["mol_id"].values[idx]),
            "smiles": set(meta["engine_smiles"].values[idx]),
        }
        rows.append({"part": part, "spectra": len(idx), "molecules": len(sets[part]["group"]),
                     **{f"CE{int(c)}": int((meta["collision_energy"].values[idx] == c).sum())
                        for c in sorted(meta["collision_energy"].unique())}})
    leaks = {}
    parts = list(split)
    for i, a in enumerate(parts):
        for b in parts[i + 1:]:
            leaks[f"{a}/{b}"] = {k: len(sets[a][k] & sets[b][k]) for k in ("group", "ik", "smiles")}
    return pd.DataFrame(rows), leaks


def max_train_similarity(meta, split, radius=2, n_bits=2048):
    """Max Tanimoto (Morgan) of each val/test molecule to any training molecule."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    smiles = meta["engine_smiles"].values
    cache = {}

    def fp(s):
        if s not in cache:
            cache[s] = gen.GetFingerprint(Chem.MolFromSmiles(s))
        return cache[s]

    train = sorted(set(smiles[split["train"]]))
    train_fps = [fp(s) for s in train]
    out = np.full(len(meta), np.nan)
    for part in ("val", "test"):
        uniq = sorted(set(smiles[split[part]]))
        sims = {s: max(DataStructs.BulkTanimotoSimilarity(fp(s), train_fps)) for s in uniq}
        for i in split[part]:
            out[i] = sims[smiles[i]]
    return out
