"""Resolve model assets inside this repository unless explicitly overridden."""
import os
from pathlib import Path

COMMIT = "67ba0f202305d4bed3134991dd4562fe1edb1763"

REPO_ROOT = Path(__file__).resolve().parents[3]


def asset_path(explicit, environment_key, relative_path):
    """Explicit path, then environment override, then the repository-local asset.

    Defaults are based on this file's location, not the working directory.
    Missing assets fail locally; sibling repositories are never searched.
    """
    configured = explicit if explicit is not None else os.environ.get(environment_key)
    path = (Path(configured).expanduser() if configured is not None
            else REPO_ROOT / relative_path).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{environment_key} asset does not exist: {path}. "
            f"Place it at {REPO_ROOT / relative_path}, or provide an explicit path."
        )
    return path
