"""Default asset resolution must work without a sibling checkout or shell setup."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "distilled_fragdata"))

from src.models import assets
from src.models.glacier_api import CHECKPOINT_SHA256, COMMIT


@pytest.fixture(autouse=True)
def clear_asset_overrides(monkeypatch):
    monkeypatch.delenv("GLACIER_UPSTREAM", raising=False)
    monkeypatch.delenv("GLACIER_CHECKPOINT", raising=False)


@pytest.mark.parametrize("environment_key,relative_path", [
    ("GLACIER_UPSTREAM", "vendor/ms-pred"),
    ("GLACIER_CHECKPOINT", "checkpoints/best.ckpt"),
])
def test_defaults_are_inside_repository_independent_of_cwd(
        monkeypatch, tmp_path, environment_key, relative_path):
    monkeypatch.chdir(tmp_path)
    assert assets.REPO_ROOT == REPO_ROOT
    actual = assets.asset_path(None, environment_key, relative_path)
    assert actual == (REPO_ROOT / relative_path).resolve()
    assert actual.is_relative_to(REPO_ROOT)
    assert actual.exists()


def test_explicit_path_then_environment_then_local_default(monkeypatch, tmp_path):
    local_root = tmp_path / "project"
    local = local_root / "checkpoints/best.ckpt"
    local.parent.mkdir(parents=True)
    local.touch()
    environment = tmp_path / "environment.ckpt"
    environment.touch()
    explicit = tmp_path / "explicit.ckpt"
    explicit.touch()
    monkeypatch.setattr(assets, "REPO_ROOT", local_root)
    resolve = lambda path: assets.asset_path(path, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt")

    assert resolve(None) == local.resolve()
    monkeypatch.setenv("GLACIER_CHECKPOINT", str(environment))
    assert resolve(None) == environment.resolve()
    assert resolve(explicit) == explicit.resolve()


def test_missing_local_asset_never_searches_other_checkouts(monkeypatch, tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for other_root in (project / "fragment_glacier_pretrained_adapter",
                       tmp_path / "external-repository/fragment_glacier_pretrained_adapter"):
        checkpoint = other_root / "checkpoints/best.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
    monkeypatch.setattr(assets, "REPO_ROOT", project)
    with pytest.raises(FileNotFoundError) as caught:
        assets.asset_path(None, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt")
    assert str(project / "checkpoints/best.ckpt") in str(caught.value)


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_requested_override_does_not_fall_back(monkeypatch, tmp_path, explicit):
    valid_local = tmp_path / "project/checkpoints/best.ckpt"
    valid_local.parent.mkdir(parents=True)
    valid_local.touch()
    missing = tmp_path / "missing.ckpt"
    monkeypatch.setattr(assets, "REPO_ROOT", valid_local.parents[1])
    monkeypatch.setenv("GLACIER_CHECKPOINT", str(valid_local if explicit else missing))
    with pytest.raises(FileNotFoundError) as caught:
        assets.asset_path(missing if explicit else None,
                          "GLACIER_CHECKPOINT", "checkpoints/best.ckpt")
    assert str(missing) in str(caught.value)


def test_fresh_api_uses_verified_local_assets_without_loading_weights(tmp_path):
    environment = os.environ.copy()
    environment.pop("GLACIER_UPSTREAM", None)
    environment.pop("GLACIER_CHECKPOINT", None)
    environment.pop("DISTILLED_PROTOTYPE_SRC", None)
    environment["PYTHONPATH"] = str(REPO_ROOT / "distilled_fragdata")
    script = """
import json
import sys
from src.models.glacier_api import GlacierAPI
api = GlacierAPI(aromatic_fix=True, radical_fix=True)
assert api._model is None
assert 'torch' not in sys.modules
print(json.dumps({
    'upstream': str(api.upstream_path),
    'checkpoint': str(api.checkpoint_path),
    'identity': api.checkpoint_identity,
    'settings': api.recovery_settings,
}))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            env=environment, text=True, capture_output=True,
                            check=True, timeout=60)
    actual = json.loads(result.stdout)
    assert actual == {
        "upstream": str(REPO_ROOT / "vendor/ms-pred"),
        "checkpoint": str(REPO_ROOT / "checkpoints/best.ckpt"),
        "identity": {"ms_pred_commit": COMMIT, "checkpoint_sha256": CHECKPOINT_SHA256},
        "settings": {"aromatic_fix": True, "radical_fix": True},
    }
