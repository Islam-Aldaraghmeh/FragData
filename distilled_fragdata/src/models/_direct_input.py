"""Independent primitive-record -> exact pretrained GLACIER tensors.

RDKit is used only by ``records_from_mol`` to obtain ordinary chemistry
descriptors. ``build_batch`` accepts plain atom/bond records, and never invokes
an upstream graph builder, FragmentEngine, or chemistry repair routine.
"""
from collections import deque
import hashlib
import json

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdPartialCharges

from ms_pred import common


def records_from_mol(mol):
    """Extract descriptors without canonicalizing, changing bonds, or neutralizing.

The caller must validate ionic state constraints before this function. Explicit
H graph nodes are unsupported here; use atom-local explicit H counts instead.
    """
    if mol is None or any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError("A valid graph with heavy-atom nodes is required")
    feature_mol = Chem.RemoveHs(Chem.Mol(mol))
    mass_mol = Chem.Mol(mol)
    Chem.Kekulize(mass_mol, clearAromaticFlags=True)
    try:
        rdPartialCharges.ComputeGasteigerCharges(feature_mol)
    except Exception:
        pass
    rings = [list(r) for r in Chem.GetSymmSSSR(feature_mol)]
    atom_ring_counts = [sum(i in ring for ring in rings) for i in range(feature_mol.GetNumAtoms())]
    ring_edges = {}
    for ring in rings:
        for i, j in zip(ring, ring[1:] + ring[:1]):
            key = tuple(sorted((i, j)))
            ring_edges[key] = ring_edges.get(key, 0) + 1
    atoms = []
    for a, mass_atom in zip(feature_mol.GetAtoms(), mass_mol.GetAtoms()):
        charge = float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.
        if not np.isfinite(charge):
            charge = 0.
        symbol = a.GetSymbol()
        if symbol not in common.ELEMENT_TO_MASS:
            raise ValueError(f"Unsupported element {symbol}")
        atom_mass = (common.P_TBL.GetMassForIsotope(symbol, mass_atom.GetIsotope())
                     if mass_atom.GetIsotope() else common.ELEMENT_TO_MASS[symbol])
        atoms.append({
            "symbol": symbol, "atomic_num": a.GetAtomicNum(),
            "degree": a.GetTotalDegree() - a.GetTotalNumHs(),
            "formal_charge": a.GetFormalCharge(), "hybridization": int(a.GetHybridization()),
            "aromatic": a.GetIsAromatic(), "hydrogens": a.GetTotalNumHs(),
            "mass_hydrogens": mass_atom.GetNumImplicitHs() + mass_atom.GetNumExplicitHs(),
            "in_ring": a.IsInRing(), "ring_sizes": [n for n in common.COMMON_RING_SIZES if a.IsInRingSize(n)],
            "fused_ring": atom_ring_counts[a.GetIdx()] >= 2,
            "chirality": int(a.GetChiralTag()), "gasteiger_charge": charge,
            "atomic_mass": float(atom_mass),
        })
    bonds = []
    for b in feature_mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds.append({
            "i": i, "j": j, "type": int(b.GetBondType()), "order": b.GetBondTypeAsDouble(),
            "conjugated": b.GetIsConjugated(), "in_ring": b.IsInRing(),
            "ring_sizes": [n for n in common.COMMON_RING_SIZES if b.IsInRingSize(n)],
            "fused_ring": ring_edges.get(tuple(sorted((i, j))), 0) >= 2,
            "stereo": int(b.GetStereo()),
        })
    return {"atoms": atoms, "bonds": bonds}


def records_from_smiles(smiles):
    return records_from_mol(Chem.MolFromSmiles(smiles))


def _onehot(value, options, other=False):
    options = list(options)
    result = [float(value == x) for x in options]
    return result + ([float(value not in options)] if other else [])


def _binary(value):
    return [float(not value), float(value)]


def graph_tensors(records, *, pe_embed_k, embed_elem_group, multi_hop_max_dist):
    atoms, bonds = records["atoms"], records["bonds"]
    n = len(atoms)
    if n == 0:
        raise ValueError("Empty heavy graph")
    adj = np.zeros((n, n), dtype=np.float32)
    edge = np.zeros((n, n, 28), dtype=np.float32)
    for b in bonds:
        i, j = b["i"], b["j"]
        if i == j or not (0 <= i < n and 0 <= j < n) or adj[i, j]:
            raise ValueError("Invalid or duplicate edge")
        adj[i, j] = adj[j, i] = b["order"]
        f = _onehot(b["type"], map(int, common.COMMON_BOND_TYPES), True)
        f += _binary(b["conjugated"]) + _binary(b["in_ring"])
        for size in common.COMMON_RING_SIZES:
            f += [float(size in b["ring_sizes"]), float(size not in b["ring_sizes"])]
        f += _binary(b["fused_ring"])
        f += _onehot(b["stereo"], map(int, common.VALID_BOND_STEREO), True)
        edge[i, j] = edge[j, i] = f
    features = []
    for a in atoms:
        f = _onehot(a["atomic_num"], common.VALID_ATOM_NUM, True)
        f += _onehot(a["degree"], range(common.MAX_COMMON_DEGREE + 1), True)
        f += _onehot(a["formal_charge"], range(-common.MAX_ABS_FORMAL_CHARGE, common.MAX_ABS_FORMAL_CHARGE + 1), True)
        f += _onehot(a["hybridization"], map(int, common.COMMON_HYBRIDIZATION), True)
        f += [float(a["aromatic"])]
        f += _onehot(a["hydrogens"], range(common.COMMON_MAX_HYDROGEN_COUNTS + 1), True)
        f += _binary(a["in_ring"])
        for size in common.COMMON_RING_SIZES:
            f += [float(size in a["ring_sizes"]), float(size not in a["ring_sizes"])]
        f += _binary(a["fused_ring"])
        f += _onehot(a["chirality"], map(int, common.COMMON_CHIRALITY))
        if embed_elem_group:
            f += common.element_to_group[a["symbol"]].tolist()
        f += [a["gasteiger_charge"]]
        features.append(f)
    x = torch.tensor(features, dtype=torch.float32)
    if pe_embed_k:
        # Row-major edges and float32 accumulation match the official weighted
        # random walk, without depending on DGL or an upstream PE helper.
        src, dst = np.nonzero(adj)
        weights = torch.from_numpy(adj[src, dst].copy())
        degree_weights = torch.zeros(n, dtype=torch.float32)
        degree_weights.index_add_(0, torch.from_numpy(src), weights)
        transition = torch.zeros((n, n), dtype=torch.float32)
        transition[src, dst] = weights / (degree_weights[src] + 1e-30)
        power = transition
        diagonal = [torch.diagonal(power)]
        for _ in range(pe_embed_k - 1):
            power = power @ transition
            diagonal.append(torch.diagonal(power))
        x = torch.cat([x, torch.stack(diagonal, dim=1)], dim=-1)
    neighbors = [np.flatnonzero(adj[i]).tolist() for i in range(n)]
    distances = np.full((n, n), 1023, dtype=np.int64)
    paths = np.zeros((n, n, multi_hop_max_dist, 28), dtype=np.int64)
    for source in range(n):
        distances[source, source] = 0
        q = deque([source])
        while q:
            vertex = q.popleft()
            for neighbor in neighbors[vertex]:
                if distances[source, neighbor] == 1023:
                    distances[source, neighbor] = distances[source, vertex] + 1
                    q.append(neighbor)
        for target in range(n):
            if target == source or distances[source, target] == 1023:
                continue
            reverse_path = [target]
            while reverse_path[-1] != source:
                current = reverse_path[-1]
                # Upstream frontier BFS uses argmax over atom indices, not queue
                # discovery order, to break equal-length predecessor ties.
                parent = min(v for v in neighbors[current]
                             if distances[source, v] == distances[source, current] - 1)
                reverse_path.append(parent)
            path = reverse_path[::-1]
            for hop, (i, j) in enumerate(zip(path, path[1:])):
                if hop == multi_hop_max_dist:
                    break
                paths[source, target, hop] = edge[i, j].astype(np.int64)
    return {
        "x": x, "attn_bias": torch.zeros((n + 1, n + 1), dtype=torch.float32),
        "attn_edge_type": torch.from_numpy(edge), "spatial_pos": torch.from_numpy(distances),
        "degree": torch.tensor([a["degree"] for a in atoms], dtype=torch.long),
        "edge_input": torch.from_numpy(paths), "num_atoms": n,
        "adj_matrix": torch.from_numpy(adj),
    }


def build_batch(records_list, collision_energies, adducts, instruments, model):
    """Build tensors from primitive records, preserving predict_mol dtypes."""
    b = len(records_list)
    if not (len(collision_energies) == len(adducts) == b):
        raise ValueError("Batch metadata lengths differ")
    graphs = [graph_tensors(r, pe_embed_k=model.pe_embed_k,
                           embed_elem_group=model.embed_elem_group,
                           multi_hop_max_dist=model.multi_hop_max_dist) for r in records_list]
    sizes = [g["num_atoms"] for g in graphs]
    n = max(sizes)
    graph_batch = {
        "x": torch.zeros((b, n, model.node_feats), dtype=torch.float32),
        "attn_bias": torch.full((b, n + 1, n + 1), -99999., dtype=torch.float32),
        "attn_edge_type": torch.zeros((b, n, n, model.edge_feats), dtype=torch.float32),
        "spatial_pos": torch.zeros((b, n, n), dtype=torch.long),
        "degree": torch.zeros((b, n), dtype=torch.long),
        "edge_input": torch.zeros((b, n, n, model.multi_hop_max_dist, model.edge_feats), dtype=torch.float32),
    }
    adjacency = torch.zeros((b, n, n), dtype=torch.float32)
    masses = torch.zeros((b, n), dtype=torch.float64)
    hydrogens = torch.zeros((b, n), dtype=torch.long)
    root_forms, atom_forms = [], []
    for i, (records, g, size) in enumerate(zip(records_list, graphs, sizes)):
        for key in ["x", "degree"]:
            graph_batch[key][i, :size] = g[key]
        graph_batch["attn_bias"][i, :size + 1, :size + 1] = g["attn_bias"]
        for key in ["attn_edge_type", "spatial_pos", "edge_input"]:
            graph_batch[key][i, :size, :size] = g[key]
        adjacency[i, :size, :size] = g["adj_matrix"]
        local_forms = np.zeros((size, len(common.VALID_ELEMENTS)), dtype=np.float64)
        for j, a in enumerate(records["atoms"]):
            h = a["mass_hydrogens"]
            hydrogens[i, j] = h
            masses[i, j] = a["atomic_mass"] + h * common.ELEMENT_TO_MASS["H"]
            local_forms[j, common.element_to_ind[a["symbol"]]] += 1
            local_forms[j, common.element_to_ind["H"]] += h
        atom_forms.append(torch.from_numpy(local_forms))
        root_forms.append(torch.from_numpy(local_forms.sum(axis=0)))
    shift = torch.tensor([[common.ion2mass[a], -common.ELECTRON_MASS if common.is_positive_adduct(a)
                           else common.ELECTRON_MASS] for a in adducts])
    return {
        "graphormer_input": graph_batch, "num_atoms": torch.tensor(sizes, dtype=torch.long),
        "adducts": torch.tensor([common.ion2onehot_pos[a] for a in adducts], dtype=torch.float32),
        "collision_engs": torch.tensor(collision_energies, dtype=torch.float32),
        "root_form_vecs": torch.stack(root_forms), "masses": masses,
        "adduct_mass_shifts": shift, "atom_form_vecs": torch.cat(atom_forms),
        "adj_matrices": adjacency, "atom_hs": hydrogens,
        "total_hs": hydrogens.sum(dim=1),
        "instruments": torch.tensor([common.instrument2onehot_pos[s] for s in instruments], dtype=torch.float32)
                       if model.embed_instrument else None,
    }


def predict_records(records, collision_energy, adduct, instrument, model):
    batch = build_batch([records], [collision_energy], [adduct], [instrument], model)
    result = model.predict_inten_frag_batch(batch)
    return {key: value[0] for key, value in result.items()}


def tensor_fingerprint(batch):
    """Hash all actual model inputs including shape/dtype, excluding labels."""
    h = hashlib.sha256()
    def update(prefix, value):
        if isinstance(value, dict):
            for key in sorted(value):
                update(prefix + "/" + key, value[key])
        elif isinstance(value, torch.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            h.update(json.dumps([prefix, str(array.dtype), list(array.shape)]).encode())
            h.update(array.tobytes())
        elif value is None:
            h.update((prefix + "=None").encode())
        else:
            raise TypeError(type(value))
    update("", batch)
    return h.hexdigest()
