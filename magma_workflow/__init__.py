"""Shared notebook utilities; model code is imported only on request.

Use magma_workflow.dataset, .candidates, .plotting, or .glacier explicitly.
The package itself does not import Torch, Glacier, or ionic-state enumeration.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
# The existing distilled package uses the namespace `src` internally.
_distilled = str(REPO_ROOT / "distilled_fragdata")
if _distilled not in sys.path:
    sys.path.insert(0, _distilled)
