"""Small CPU interface to the exact pinned, public pretrained GLACIER model."""
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

COMMIT = "67ba0f202305d4bed3134991dd4562fe1edb1763"
CHECKPOINT_SHA256 = "5a47cecca707d3abd5a49c7dbac99d100aa2a586d5d4f848f1e8e35140d7db11"
# Exact checkpoint featurizer configuration; no weights are loaded here.
FEATURE_CONFIG = SimpleNamespace(pe_embed_k=14, embed_elem_group=True,
                                 multi_hop_max_dist=10, node_feats=88,
                                 edge_feats=28, embed_instrument=True)


def sha256(path):
    """Stream the same SHA-256 used by the prototype's checkpoint check."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _direct():
    # Keep torch, RDKit, DGL and ms_pred out of imports of the public module.
    return importlib.import_module("._direct_input", __package__)


def records_from_mol(mol):
    return _direct().records_from_mol(mol)


def records_from_smiles(smiles):
    return _direct().records_from_smiles(smiles)


def build_batch(records_list, collision_energies, adducts, instruments, model):
    return _direct().build_batch(records_list, collision_energies, adducts, instruments, model)


def tensor_fingerprint(batch):
    return _direct().tensor_fingerprint(batch)


def predict_records(records, collision_energy, adduct, instrument, model):
    return _direct().predict_records(records, collision_energy, adduct, instrument, model)


def _feature_hash(mol, adduct):
    batch = build_batch([records_from_mol(mol)], [0.], [adduct], ["Orbitrap"], FEATURE_CONFIG)
    return tensor_fingerprint(batch)


class GlacierAPI:
    """Verify external assets, then lazily load frozen GLACIER on CPU.

    Paths may also come from GLACIER_UPSTREAM and GLACIER_CHECKPOINT.
    Outputs are unmodified upstream ``spec`` rows and ``frag`` masks. A row is
    not a postprocessed peak: coincident masses still need MAX pooling.
    """

    def __init__(self, upstream_path=None, checkpoint_path=None):
        upstream_path = upstream_path or os.environ.get("GLACIER_UPSTREAM")
        checkpoint_path = checkpoint_path or os.environ.get("GLACIER_CHECKPOINT")
        if upstream_path is None or checkpoint_path is None:
            raise ValueError("Provide upstream_path and checkpoint_path (or GLACIER_UPSTREAM / GLACIER_CHECKPOINT)")
        self.upstream_path = Path(upstream_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self._model = None
        self._verify_assets()
        self._select_upstream()

    def _verify_assets(self):
        command = ["git", "-C", str(self.upstream_path)]
        actual = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
        if actual != COMMIT:
            raise ValueError(f"Expected ms-pred commit {COMMIT}, found {actual}")
        if subprocess.check_output(command + ["diff", "HEAD"], text=True):
            raise ValueError("The pinned ms-pred checkout has tracked modifications")
        self.checkpoint_sha256 = sha256(self.checkpoint_path)
        if self.checkpoint_sha256 != CHECKPOINT_SHA256:
            raise ValueError("The GLACIER checkpoint SHA-256 differs from the public checkpoint")

    def _select_upstream(self):
        source = self.upstream_path / "src"
        imported = sys.modules.get("ms_pred")
        if imported is not None and not Path(imported.__file__).resolve().is_relative_to(source):
            raise RuntimeError("Another ms_pred checkout is already imported; start a fresh Python process")
        os.environ.setdefault("DGLBACKEND", "pytorch")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))

    @property
    def model(self):
        """Load with the prototype's seed, thread count and frozen eval settings."""
        if self._model is None:
            import torch
            from ms_pred.glacier.joint_model import JointModel

            torch.manual_seed(20260919)
            torch.set_num_threads(4)
            self._model = JointModel.load_from_checkpoint(
                str(self.checkpoint_path), map_location="cpu", weights_only=False)
            self._model.eval()
            self._model.freeze()
            if any(parameter.requires_grad for parameter in self._model.parameters()):
                raise RuntimeError("GLACIER parameters were not frozen")
        return self._model

    @property
    def checkpoint_identity(self):
        return {"ms_pred_commit": COMMIT, "checkpoint_sha256": self.checkpoint_sha256}

    def predict_molecule(self, smiles, collision_energy, adduct="[M+H]+", instrument="Orbitrap"):
        """Call upstream predict_mol directly; list inputs require list metadata."""
        return self.model.predict_mol(smiles, collision_energy, adduct, "cpu", instrument=instrument)

    def predict_records(self, records, collision_energy, adduct="[M]+", instrument="Orbitrap"):
        """Predict one primitive heavy-atom record using the exact direct route."""
        return predict_records(records, collision_energy, adduct, instrument, self.model)

    def predict_tensors(self, batch):
        """Consume an already built CPU batch; retain upstream per-item lists."""
        return self.model.predict_inten_frag_batch(batch)

    def predict_batch(self, records_list, collision_energies, adducts, instruments):
        """Pad a list of primitive records with explicit per-item metadata."""
        batch = build_batch(records_list, collision_energies, adducts, instruments, self.model)
        return self.predict_tensors(batch)
