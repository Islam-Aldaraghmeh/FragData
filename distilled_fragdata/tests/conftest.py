"""Repository-local model assets; a live prototype oracle is explicit opt-in."""
import os
from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
configured_prototype = os.environ.get("DISTILLED_PROTOTYPE_SRC")
OLD_SRC = Path(configured_prototype).expanduser().resolve() if configured_prototype else None
sys.path.insert(0, str(PROJECT))
if OLD_SRC is not None and OLD_SRC.is_dir():
    sys.path.insert(0, str(OLD_SRC))


@pytest.fixture(scope="session")
def prototype_src():
    return OLD_SRC if OLD_SRC is not None and OLD_SRC.is_dir() else None


@pytest.fixture(scope="session")
def glacier_paths():
    from src.models.assets import asset_path

    return (asset_path(None, "GLACIER_UPSTREAM", "vendor/ms-pred"),
            asset_path(None, "GLACIER_CHECKPOINT", "checkpoints/best.ckpt"))


@pytest.fixture(scope="session")
def model(glacier_paths):
    from src.models.glacier_api import GlacierAPI
    return GlacierAPI(*glacier_paths).model


@pytest.fixture(scope="session")
def glacier(glacier_paths, model):
    from src.models.glacier_api import GlacierAPI
    api = GlacierAPI(*glacier_paths)
    api._model = model
    return api
