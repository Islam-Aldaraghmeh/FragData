"""Replay one censored preparation, then profile it without changing policy."""
import cProfile
import io
import json
from pathlib import Path
import pstats
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from recovery_lab.stress_pipeline import OUT, budget, namespace, write_json


def main():
    ns = namespace()
    model = ns['load_glacier_model'](aromatic_fix=True, radical_fix=True)
    from src.chemistry.fragment_adapter import FragmentGlacierAdapter
    adapter = FragmentGlacierAdapter(model, max_states=4)
    source = json.loads((OUT / 'spectrum_03.json').read_text())
    row = next(r for r in source['candidate_rows'] if r['outcome'] == 'timeout')
    report = {k: row[k] for k in ('candidate_id', 'mz', 'stripped', 'has_ms3')}
    report['original_prepare_seconds'] = row['prepare_seconds']
    runs = []
    for profiled in (False, True):
        profile = cProfile.Profile()
        started = time.perf_counter()
        result = {'profiled': profiled}
        try:
            with budget(60):
                if profiled:
                    profile.enable()
                prepared = adapter.prepare(row['stripped'], collision_energy=60)
                result.update({k: prepared.get(k) for k in
                               ('status', 'reason', 'route', 'metadata',
                                'baseline_metadata', 'states')})
        except Exception as error:
            result.update(error_type=type(error).__name__, error=str(error))
        finally:
            profile.disable()
        result['seconds'] = time.perf_counter() - started
        if profiled:
            profile.dump_stats(str(OUT / 'slow_case.prof'))
            stream = io.StringIO()
            pstats.Stats(profile, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(25)
            (OUT / 'slow_case_profile.txt').write_text(stream.getvalue())
        runs.append(result)
        write_json(OUT / 'slow_case_replay.json', {**report, 'runs': runs})
        print(json.dumps({k: v for k, v in result.items() if k not in
                          ('states', 'metadata', 'baseline_metadata')}), flush=True)


if __name__ == '__main__':
    main()
