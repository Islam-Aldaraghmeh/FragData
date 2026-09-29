"""Apply compact previews to an existing notebook without reloading its data.

This script does not interrupt or restart a kernel. Pass its connection file;
interrupt an already diagnosed stuck preview separately before running it.
"""
import argparse
import json
from pathlib import Path
import time

from jupyter_client import BlockingKernelClient

ROOT = Path(__file__).resolve().parents[2]
CODE = '''
def _magma_apply_preview_fix():
    import importlib
    import json
    from pathlib import Path
    import Spectrum_Classes

    ns = globals()
    model_id = id(ns.get("model"))
    updated = importlib.reload(Spectrum_Classes)
    for name in ("Spectrum", "SpectrumForest"):
        previous = ns.get(name)
        if isinstance(previous, type):
            previous.__repr__ = getattr(updated, name).__repr__
    ns["SpectrumCollection"] = updated.SpectrumCollection
    ns["SpectraBySmiles"] = updated.SpectraBySmiles
    replacements = {}
    counts = {}
    for name in ("raw_spectra", "spectra"):
        old = ns.get(name)
        if isinstance(old, list):
            if id(old) not in replacements:
                replacements[id(old)] = updated.SpectrumCollection(old)
            new = replacements[id(old)]
            assert len(new) == len(old)
            assert all(a is b for a, b in zip(new, old))
            ns[name] = new
            counts[name] = len(new)
    old_groups = ns.get("spectra_by_smiles")
    if isinstance(old_groups, dict):
        ns["spectra_by_smiles"] = updated.SpectraBySmiles(
            (key, updated.SpectrumCollection(scans)) for key, scans in old_groups.items())
        counts["smiles_groups"] = len(old_groups)
    notebook = json.loads((Path(__REPO_ROOT__) / "Experiment_1.ipynb").read_text())
    loader = next(c for c in notebook["cells"] if c.get("id") == "96ffee3e")
    exec(compile("".join(loader["source"]), "Experiment_1.ipynb:loader", "exec"), ns)
    assert id(ns.get("model")) == model_id
    return dict(counts=counts, model_preserved=True,
                dataset_preview=repr(ns.get("raw_spectra")),
                spectra_alias_preserved=ns.get("raw_spectra") is ns.get("spectra"))

_magma_preview_fix_result = _magma_apply_preview_fix()
del _magma_apply_preview_fix
'''.replace('__REPO_ROOT__', repr(str(ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('connection_file')
    args = parser.parse_args()
    client = BlockingKernelClient(connection_file=args.connection_file)
    client.load_connection_file()
    client.start_channels(iopub=False, stdin=False, hb=False, control=False)
    started = time.perf_counter()
    try:
        msg_id = client.execute(CODE, silent=True, store_history=False,
                                allow_stdin=False, stop_on_error=False,
                                user_expressions={'result': '_magma_preview_fix_result'})
        while True:
            reply = client.get_shell_msg(timeout=45)
            if reply.get('parent_header', {}).get('msg_id') == msg_id:
                break
        content = reply['content']
        result = dict(seconds=time.perf_counter()-started, status=content['status'],
                      result=content.get('user_expressions', {}).get('result'),
                      error=content.get('evalue'))
        (ROOT/'recovery_lab/notebook_lag/live_fix.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
        if content['status'] != 'ok':
            raise RuntimeError('Kernel rejected preview fix: '+str(content.get('evalue')))
    finally:
        client.stop_channels()


if __name__ == '__main__':
    main()
