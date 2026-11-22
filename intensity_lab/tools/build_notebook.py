"""Build Experiment_1_agentic.ipynb from Experiment_1.ipynb (cells 0-3) + notebook_cells.py.

usage: .venv/bin/python intensity_lab/tools/build_notebook.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from notebook_cells import CELLS  # noqa: E402

src = json.loads((REPO / "Experiment_1.ipynb").read_text())
orig = src["cells"]


def cell(kind, text):
    lines = text.strip("\n").splitlines(keepends=True)
    c = {"cell_type": kind, "metadata": {}, "source": lines}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


cells = []
for kind, text in CELLS:
    if kind == "orig":  # copy a cell of the original notebook verbatim (without outputs)
        c = dict(orig[text])
        c = {k: v for k, v in c.items() if k != "id"}
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        cells.append(c)
    else:
        cells.append(cell(kind, text))

nb = {"cells": cells, "metadata": src["metadata"], "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"agentic-{i:03d}"
out = REPO / "Experiment_1_agentic.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"wrote {out} with {len(cells)} cells")
