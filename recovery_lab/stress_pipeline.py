"""Reproducible, isolated notebook pipeline probe; no production behavior changes.

Run from the repository with .venv/bin/python recovery_lab/stress_pipeline.py.
Samples distinct feature/SMILES groups across two libraries, confirms actual
MS3 linkage, then times MAGMa, ionic-state preparation, and model inference.
Timeouts are diagnostic censoring, not claims of chemistry impossibility.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'distilled_fragdata'))
from Spectrum_Classes import Spectrum, SpectrumForest, parse_spectrum_forest

OUT = ROOT / 'recovery_lab/stress_20260922'
def namespace():
    """Import the shared workflow without executing notebook cells."""
    from magma_workflow import candidates, glacier
    ns = dict(json=json, Path=Path, np=np, Spectrum=Spectrum, SpectrumForest=SpectrumForest,
              parse_spectrum_forest=parse_spectrum_forest)
    ns.update({name: getattr(candidates, name) for name in
               ('get_candidate_bags', 'filter_same_formula_bags', 'find_ms3_children')})
    ns.update({name: getattr(glacier, name) for name in
               ('load_glacier_model', 'predict_candidate_bags')})
    return ns


class ProbeTimeout(TimeoutError):
    pass


@contextmanager
def budget(seconds):
    def expired(signum, frame):
        raise ProbeTimeout('Diagnostic deadline; stack: ' + ''.join(traceback.format_stack(frame)[-6:]))
    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.01, seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=lambda v: v.tolist() if hasattr(v,'tolist') else str(v))+'\n')


def select_cohort(ns, name, count, seed):
    path = ROOT/'msnlib'/name
    offsets, roots, has_ms3 = defaultdict(list), [], set()
    started = time.perf_counter()
    with path.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line: break
            record = json.loads(line)
            if record.get('merge_type') != 'SINGLE_BEST_SCAN' or record.get('ms_level') not in (2,3):
                continue
            key = (record['smiles'], record['feature_id'])
            offsets[key].append(offset)
            if record['ms_level'] == 3:
                has_ms3.add(key)
            else:
                roots.append((key, offset))
    eligible = [(key, offset) for key, offset in roots if key in has_ms3]
    random.Random(seed).shuffle(eligible)
    selected, used = [], set()
    with path.open('rb') as stream:
        for key, root_offset in eligible:
            if key in used: continue
            scans, root = [], None
            for offset in offsets[key]:
                stream.seek(offset); record=json.loads(stream.readline())
                item=Spectrum.from_record(record);scans.append(item)
                if offset==root_offset:root=item
            children=ns['find_ms3_children'](root, scans)
            if not children:continue
            selected.append(dict(dataset=name,root_offset=root_offset,offsets=offsets[key],
                feature_id=key[1],smiles=key[0],scan=root.metadata.get('scan_number'),
                precursor_mz=root.precursor_mz,adduct=root.metadata.get('adduct'),
                ms3_count=sum(map(len,children.values()))))
            used.add(key)
            if len(selected)==count:break
    print(f'Selected {len(selected)} from {name}: {len(eligible)} MS2 records in MS3-containing groups; indexing {time.perf_counter()-started:.1f}s',flush=True)
    return selected


def load_scans(entry):
    scans, root=[],None
    with (ROOT/'msnlib'/entry['dataset']).open('rb') as stream:
        for offset in entry['offsets']:
            stream.seek(offset);s=Spectrum.from_record(json.loads(stream.readline()));scans.append(s)
            if offset==entry['root_offset']:root=s
    return root,scans


def compact_prepared(result):
    return {key:result.get(key) for key in ('status','reason','route','baseline_reason','metadata',
            'baseline_metadata','audit','states') if key in result}


def run_one(ns, model, entry, number, limit):
    from src.chemistry.fragment_adapter import FragmentGlacierAdapter
    from recovery_lab.experiment import predict_prepared
    started=time.perf_counter();deadline=started+limit
    root, scans=load_scans(entry)
    report=dict(index=number,**entry,status='started',candidate_rows=[])
    target=OUT/f'spectrum_{number:02d}.json'
    phase='candidate_generation'
    tick=time.perf_counter()
    try:
        with budget(min(45,deadline-tick)):
            bags=ns['get_candidate_bags'](root, scans)
        report['generation_seconds']=time.perf_counter()-tick
        filtered=ns['filter_same_formula_bags'](bags)
        report.update(total_bags=len(bags),homogeneous_bags=len(filtered),
            raw_candidates=sum(len(b[1]) for b in bags),
            candidates=sum(len(b[1]) for b in filtered),
            linked_ms3_bags=sum(bool(b[2]) for b in filtered),
            linked_ms3_candidates=sum(len(b[1]) for b in filtered if b[2]),
            preparation_seconds=0.,prediction_seconds=0.,model_state_calls=0)
        adapter=FragmentGlacierAdapter(model,max_states=4)
        for bi,(mz,candidates,children) in enumerate(filtered):
            for ci,candidate in enumerate(candidates):
                if time.perf_counter()>=deadline:
                    raise ProbeTimeout('Whole spectrum time budget reached between candidates')
                row=dict(bag_index=bi,candidate_index=ci,candidate_id=candidate.candidate_id,
                    mz=mz,has_ms3=bool(children),stripped=candidate.stripped,
                    metadata=candidate.metadata)
                phase='state_preparation';tick=time.perf_counter()
                try:
                    with budget(min(12,deadline-tick)):
                        prepared=adapter.prepare(candidate,collision_energy=60.,instrument='Orbitrap')
                    row.update(compact_prepared(prepared));row['prepare_seconds']=time.perf_counter()-tick
                    report['preparation_seconds']+=row['prepare_seconds']
                    if prepared['states']:
                        phase='model_prediction';tick=time.perf_counter()
                        with budget(min(12,deadline-tick)):
                            spectrum=predict_prepared(model,prepared).as_array()
                        row['prediction_seconds']=time.perf_counter()-tick
                        report['prediction_seconds']+=row['prediction_seconds']
                        report['model_state_calls']+=len(prepared['inputs'])
                        row['prediction_peaks']=len(spectrum)
                        row['prediction_valid']=bool(len(spectrum) and np.isfinite(spectrum).all()
                            and np.isclose(spectrum[:,1].sum(),1))
                        row['prediction_sha256']=hashlib.sha256(spectrum.tobytes()).hexdigest()
                        row['outcome']='predicted' if row['prediction_valid'] else 'invalid_output'
                    else:row['outcome']='unavailable'
                except Exception as error:
                    row.update(outcome='timeout' if isinstance(error,ProbeTimeout) else 'exception',
                               error_type=type(error).__name__,error=str(error),error_phase=phase,
                               phase_seconds=time.perf_counter()-tick)
                    if phase=='state_preparation':
                        row['prepare_seconds']=row['phase_seconds'];report['preparation_seconds']+=row['phase_seconds']
                    else:
                        row['prediction_seconds']=row['phase_seconds'];report['prediction_seconds']+=row['phase_seconds']
                report['candidate_rows'].append(row)
                if len(report['candidate_rows'])%25==0:
                    write_json(target,report)
                    print(f'  spectrum {number}: {len(report["candidate_rows"])}/{report["candidates"]} candidates, {time.perf_counter()-started:.1f}s',flush=True)
        report['status']='complete'
    except Exception as error:
        report.update(status='timeout' if isinstance(error,ProbeTimeout) else 'exception',
                      error=str(error),error_type=type(error).__name__,error_phase=phase)
    report['elapsed_seconds']=time.perf_counter()-started
    report['outcomes']=dict(Counter(r['outcome'] for r in report['candidate_rows']))
    report['reasons']=dict(Counter(r.get('reason',r.get('error_type')) for r in report['candidate_rows'] if r['outcome']!='predicted'))
    write_json(target,report)
    print(f'{number}: {report["status"]}, candidates {len(report["candidate_rows"])}/{report.get("candidates","?")}, '
          f'{report["elapsed_seconds"]:.1f}s, {report["outcomes"]}, reasons={report["reasons"]}',flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--count',type=int,default=30)
    parser.add_argument('--seconds',type=float,default=90);parser.add_argument('--seed',type=int,default=20260922)
    parser.add_argument('--select-only',action='store_true');parser.add_argument('--start',type=int,default=0)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    ns=namespace();cohort_path=OUT/'cohort.json'
    if cohort_path.exists():cohort=json.loads(cohort_path.read_text())['spectra']
    else:
        first=2*args.count//3
        cohort=select_cohort(ns,'20241003_enamdisc_pos_msn.json',first,args.seed)
        cohort+=select_cohort(ns,'20241003_mcedrug_pos_msn.json',args.count-first,args.seed+1)
        write_json(cohort_path,dict(seed=args.seed,selection='shuffle eligible MS2, one per feature/SMILES group',spectra=cohort))
    if args.select_only:return
    tick=time.perf_counter();model=ns['load_glacier_model'](aromatic_fix=True,radical_fix=True)
    write_json(OUT/'run_settings.json',dict(seed=args.seed,collision_energy=60.,max_states=4,
        aromatic_fix=True,radical_fix=True,spectrum_time_limit=args.seconds,candidate_phase_limit=12,
        model_load_seconds=time.perf_counter()-tick,checkpoint=model.checkpoint_identity,
        python=sys.version,notebook_sha256=hashlib.sha256((ROOT/'Experiment_1.ipynb').read_bytes()).hexdigest()))
    print(f'Model loaded in {time.perf_counter()-tick:.2f}s',flush=True)
    for i,entry in enumerate(cohort[:args.count]):
        if i<args.start or (OUT/f'spectrum_{i:02d}.json').exists():continue
        run_one(ns,model,entry,i,args.seconds)

if __name__=='__main__':main()
