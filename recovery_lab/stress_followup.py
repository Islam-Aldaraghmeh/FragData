"""Small exact-output check of filtering/caching and display cost after stress run.

No application caching or chemistry policy is changed by this demonstration.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import time
import warnings

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from recovery_lab.stress_pipeline import namespace,load_scans,write_json,OUT,budget


def main():
    ns=namespace()
    from magma_workflow.plotting import plot_fragments_with_spectra
    ns['plot_fragments_with_spectra'] = plot_fragments_with_spectra
    model=ns['load_glacier_model'](aromatic_fix=True,radical_fix=True)
    settings=json.loads((OUT/'run_settings.json').read_text())
    assert settings['checkpoint']==model.checkpoint_identity
    assert {key:settings[key] for key in model.recovery_settings}==model.recovery_settings
    ce,max_states=settings['collision_energy'],settings['max_states']
    context=dict(checkpoint=model.checkpoint_identity,settings=model.recovery_settings,
                 collision_energy=ce,max_states=max_states,instrument='Orbitrap')

    def cache_key(candidate):
        return json.dumps(dict(stripped=candidate.stripped,context=context),sort_keys=True)

    results=[]
    for index in (0,3):
        original=json.loads((OUT/f'spectrum_{index:02d}.json').read_text())
        assert original['status']!='started','Wait for the spectrum probe to finish'
        root,scans=load_scans(original)
        t=time.perf_counter()
        bags=ns['filter_same_formula_bags'](ns['get_candidate_bags'](root,scans))
        generation=time.perf_counter()-t
        indexed=[(bi,b) for bi,b in enumerate(bags) if b[2]]
        chosen=[b for _,b in indexed]
        assert sum(len(b[1]) for b in bags)==original['candidates']
        t=time.perf_counter()
        with budget(90),warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            spectra=ns['predict_candidate_bags'](model,chosen,collision_energy=ce,max_states=max_states)
        subset_seconds=time.perf_counter()-t
        expected={(r['bag_index'],r['candidate_index']):r for r in original['candidate_rows']}
        assert len(expected)==len(original['candidate_rows'])
        assert len(spectra)==len(chosen)
        checks=[];cache={};unavailable_checks=0;uncompared=Counter()
        for (bi,(mz,candidates,observations)),(predicted_mz,predictions,_) in zip(indexed,spectra):
            assert predicted_mz==mz and len(predictions)==len(candidates)
            for ci,(candidate,prediction) in enumerate(zip(candidates,predictions)):
                old=expected.get((bi,ci))
                if old:
                    assert (old['candidate_id'],old['mz'],old['stripped'])==(candidate.candidate_id,mz,candidate.stripped)
                if old and old['outcome']=='predicted':
                    exact=prediction is not None and hashlib.sha256(prediction.tobytes()).hexdigest()==old['prediction_sha256']
                    assert exact,'Subset changed a prediction'
                    checks.append(exact)
                elif old and old['outcome']=='unavailable':
                    assert prediction is None
                    unavailable_checks+=1
                else:
                    uncompared['original_'+old['outcome'] if old else 'original_untested']+=1
                cache[cache_key(candidate)]=None if prediction is None else prediction.copy()
        # Measures only lookup/copy of stored outputs, not a fresh model run.
        t=time.perf_counter();cached=[]
        for mz,candidates,observations in chosen:
            values=[cache[cache_key(c)] for c in candidates]
            cached.append((mz,[None if v is None else v.copy() for v in values],observations))
        cache_seconds=time.perf_counter()-t
        for (_,reference,_),(_,actual,_) in zip(spectra,cached):
            for left,right in zip(reference,actual):
                if left is None:assert right is None
                else:
                    assert left.dtype==right.dtype and left.shape==right.shape
                    assert left.tobytes()==right.tobytes()
        plots=[]
        for mixture in (False,True):
            t=time.perf_counter()
            with budget(90):
                html=ns['plot_fragments_with_spectra'](chosen,spectra,ms3_collision_energies=[60],
                    only_with_ms3=True,fit_mixture=mixture).data
            plots.append(dict(fit_mixture=mixture,seconds=time.perf_counter()-t,
                html_bytes=len(html.encode()),svg_images=html.count('data:image/svg+xml;base64,'),
                mixture_cards=html.count('class="mixture-card"')))
        row=dict(spectrum_index=index,all_candidates=sum(len(b[1])for b in bags),
            ms3_candidates=sum(len(b[1])for b in chosen),generation_seconds=generation,
            ms3_only_prediction_seconds=subset_seconds,exact_prediction_checks=len(checks),
            exact_unavailable_checks=unavailable_checks,uncompared_originals=dict(uncompared),
            all_exact=bool(checks or unavailable_checks) and not uncompared,
            same_model_repeat_cache_seconds=cache_seconds,plots=plots)
        results.append(row);print(json.dumps(row),flush=True)
    write_json(OUT/'quick_wins.json',dict(scope='isolated proof of concept; no production cache installed',
        context=context,cache_timing_scope='lookup and copy only, after populating an in-memory cache',results=results))

if __name__=='__main__':main()
