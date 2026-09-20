"""Local prototype is an optional live oracle; frozen fixtures travel with us."""
import os
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PROTOTYPE = PROJECT.parent / "fragment_glacier_pretrained_adapter"
OLD_SRC = Path(os.environ.get("DISTILLED_PROTOTYPE_SRC", PROTOTYPE / "src"))
sys.path.insert(0, str(PROJECT))
if OLD_SRC.is_dir():
    sys.path.insert(0, str(OLD_SRC))


@pytest.fixture(scope="session")
def prototype_src():
    return OLD_SRC if OLD_SRC.is_dir() else None


@pytest.fixture(scope="session")
def glacier_paths():
    upstream = Path(os.environ.get("GLACIER_UPSTREAM", PROTOTYPE / "vendor/ms-pred"))
    checkpoint = Path(os.environ.get("GLACIER_CHECKPOINT", PROTOTYPE / "checkpoints/best.ckpt"))
    if not (upstream.is_dir() and checkpoint.is_file()):
        pytest.skip("Set GLACIER_UPSTREAM and GLACIER_CHECKPOINT for real checkpoint regression")
    return upstream, checkpoint


@pytest.fixture(scope="session")
def model(glacier_paths, prototype_src):
    if prototype_src is not None and glacier_paths[0].resolve() == (PROTOTYPE / "vendor/ms-pred").resolve():
        from runtime import load_model
        return load_model()
    from src.models.glacier_api import GlacierAPI
    return GlacierAPI(*glacier_paths).model


@pytest.fixture(scope="session")
def glacier(glacier_paths, model):
    from src.models.glacier_api import GlacierAPI
    api = GlacierAPI(*glacier_paths)
    api._model = model
    return api
