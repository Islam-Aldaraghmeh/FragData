"""Dependency boundaries and explicit dataset context for shared notebooks."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_candidate_import_does_not_load_predictor_or_enumerator():
    subprocess.run([sys.executable, "-c", """
import sys
from magma_workflow.candidates import get_candidate_bags
from magma_workflow.dataset import Spectrum
from src.chemistry.heavy_graph import _canonical_graph
for name in ('torch', 'src.models.glacier_api', 'src.chemistry._state_graph',
             'src.chemistry.fragment_adapter', 'src.chemistry._enumeration'):
    assert name not in sys.modules, name
"""], cwd=ROOT, check=True, capture_output=True, text=True)


def test_explicit_empty_source_overrides_configured_notebook_source(monkeypatch):
    import numpy as np
    from magma_workflow import candidates
    from magma_workflow.dataset import Spectrum

    class UnreadableSource:
        def __iter__(self):
            raise AssertionError("An explicit empty source read the notebook dataset")

    monkeypatch.setattr(candidates, "_source_spectra", None)
    monkeypatch.setattr(candidates, "_source_by_smiles", None)
    candidates.set_spectrum_source(UnreadableSource())
    spectrum = Spectrum("CC", 31.0, 2, np.array([[15.0, 1.0]]), {})
    assert candidates.find_ms3_children(spectrum, spectra=[]) == {}
    candidates.set_spectrum_source()
    assert candidates._source_spectra is None

