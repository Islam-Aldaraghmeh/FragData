"""Diagnose the frozen Experiment 2 model; do not train or replace its outputs."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from intensity_model import spectrum_metrics
from train_intensity import load_model, predict_examples, equal_peak_baseline, plot_comparison

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator else 0.


def effective_peaks(values):
    """Participation ratio: 1 for a single peak, N for N equal peaks."""
    return float(values.sum() ** 2 / np.dot(values, values)) if np.dot(values, values) else 0.


def aggregate(rows):
    columns = ('cosine', 'baseline_cosine', 'support_ceiling', 'supported_cosine',
               'missing_support_gap', 'supported_shape_gap', 'supported_peak_fraction',
               'supported_tic_fraction', 'base_peak_supported', 'top1_correct',
               'true_base_peak_rank', 'target_effective_peaks', 'predicted_effective_peaks',
               'target_tic_in_top3_predicted', 'mean_candidates_per_supported_peak')
    result = {'spectra': len(rows), 'molecules': len({r['molecule'] for r in rows})}
    if rows:
        result.update({column: float(np.mean([row[column] for row in rows])) for column in columns})
        result['median_cosine'] = float(np.median([r['cosine'] for r in rows]))
        result['median_sharpness_ratio'] = float(np.median([
            r['predicted_effective_peaks'] / r['target_effective_peaks'] for r in rows]))
    return result


def main():
    torch.set_num_threads(2)
    with (SOURCE / 'dataset.pkl').open('rb') as stream:
        examples = pickle.load(stream)
    model = load_model(SOURCE / 'model.pt')
    predictions = predict_examples(model, examples)
    baseline = equal_peak_baseline(examples)
    rows = []
    for index, (example, predicted, equal) in enumerate(zip(examples, predictions, baseline)):
        target = example['target'].astype(float)
        support = equal.astype(bool)
        counts = np.bincount(example['peak_indices'], minlength=len(target))
        ceiling = float(np.linalg.norm(target[support]) / np.linalg.norm(target))
        score = cosine(target, predicted)
        supported_score = cosine(target[support], predicted[support])
        assert np.isclose(score, ceiling * supported_score), (index, score, ceiling, supported_score)
        # Every distinct possible peak maximum contributes, not each candidate.
        order = np.argsort(-predicted, kind='stable')
        true_top = int(np.argmax(target))
        supported_counts = counts[support]
        row = dict(index=index, source_offset=example['source_offset'], split=example['split'],
            molecule=example['molecule'], smiles=example['smiles'], collision_energy=example['collision_energy'],
            peaks=len(target), supported_peaks=int(support.sum()), candidates=len(example['features']),
            cosine=score, baseline_cosine=cosine(target, equal), support_ceiling=ceiling,
            supported_cosine=supported_score, missing_support_gap=1-ceiling, supported_shape_gap=ceiling-score,
            supported_peak_fraction=float(support.mean()),
            supported_tic_fraction=float(target[support].sum()/target.sum()),
            base_peak_supported=bool(support[true_top]), top1_correct=bool(target[order[0]]==target[true_top]),
            true_base_peak_rank=int(np.flatnonzero(order==true_top)[0])+1,
            target_effective_peaks=effective_peaks(target), predicted_effective_peaks=effective_peaks(predicted),
            target_tic_in_top3_predicted=float(target[order[:3]].sum()/target.sum()),
            mean_candidates_per_supported_peak=float(np.mean(supported_counts)) if len(supported_counts) else 0.,
            max_candidates_per_peak=int(counts.max(initial=0)),
            ambiguous_peak_fraction=float(np.mean(supported_counts>1)) if len(supported_counts) else 0.)
        rows.append(row)
    with (OUT / 'per_spectrum.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summaries = {split: aggregate([r for r in rows if r['split']==split])
                 for split in ('train', 'validation', 'test')}
    by_ce = {split: {str(ce): aggregate([r for r in rows if r['split']==split and r['collision_energy']==ce])
                    for ce in sorted({r['collision_energy'] for r in rows if r['split']==split})}
             for split in ('train', 'validation', 'test')}
    slices = {}
    for split in ('validation','test'):
        sample = [r for r in rows if r['split']==split]
        predicates = {
            'ceiling_ge_0.99': lambda r:r['support_ceiling']>=.99,
            'ceiling_below_0.9': lambda r:r['support_ceiling']<.9,
            'base_peak_unsupported': lambda r:not r['base_peak_supported'],
            'at_most_3_effective_target_peaks':lambda r:r['target_effective_peaks']<=3,
            'more_than_3_effective_target_peaks':lambda r:r['target_effective_peaks']>3,
            'mean_candidates_le_2':lambda r:r['mean_candidates_per_supported_peak']<=2,
            'mean_candidates_gt_2':lambda r:r['mean_candidates_per_supported_peak']>2,
        }
        slices[split] = {name:aggregate([r for r in sample if predicate(r)])
                         for name,predicate in predicates.items()}
    # Cluster bootstrap treats a molecule's repeated collision energies together.
    rng = np.random.default_rng(20260922)
    held_out = [r for r in rows if r['split']=='test']
    groups = defaultdict(list)
    for row in held_out:
        groups[row['molecule']].append(row)
    group_rows = list(groups.values())
    bootstrap=[]
    for _ in range(2000):
        sampled=[r for i in rng.integers(0,len(group_rows),len(group_rows)) for r in group_rows[i]]
        bootstrap.append([np.mean([r['cosine'] for r in sampled]),
                          np.mean([r['cosine']-r['baseline_cosine'] for r in sampled])])
    bootstrap=np.asarray(bootstrap)
    intervals=dict(test_mean_cosine_95pct=np.quantile(bootstrap[:,0],[.025,.975]).tolist(),
                   test_delta_vs_equal_95pct=np.quantile(bootstrap[:,1],[.025,.975]).tolist())
    # Distinct error types selected transparently after inspecting frozen results.
    unique=[]
    selectors=[('worst_supported',lambda r:r['support_ceiling']>=.99,lambda r:r['cosine']),
               ('largest_support_loss',lambda r:True,lambda r:r['support_ceiling']),
               ('typical_error',lambda r:r['support_ceiling']>=.99,lambda r:abs(r['cosine']-summaries['test']['median_cosine'])),
               ('best_prediction',lambda r:True,lambda r:-r['cosine'])]
    for label,predicate,order in selectors:
        row=min((r for r in held_out if predicate(r) and r['index'] not in [x['index'] for x in unique]),key=order)
        example,prediction=examples[row['index']],predictions[row['index']]
        figure=plot_comparison(example,prediction,baseline[row['index']])
        figure.suptitle(f"{label.replace('_',' ')} | support ceiling {row['support_ceiling']:.3f}\n{example['smiles']}",fontsize=10)
        figure.savefig(OUT/f'{label}.png',dpi=140)
        unique.append(dict(row,label=label,plot=f'{label}.png'))
    figure=Figure(figsize=(12,8),layout='constrained')
    axes=figure.subplots(2,2)
    sample=sorted(held_out,key=lambda r:r['cosine'])
    axes[0,0].plot([r['support_ceiling'] for r in sample],label='Perfect intensities on MAGMa support',color='#48844f')
    axes[0,0].plot([r['cosine'] for r in sample],label='Frozen MLP',color='#2764b7')
    axes[0,0].set(xlabel='Test spectra, sorted by MLP cosine',ylabel='Cosine',ylim=(0,1.03))
    axes[0,0].legend(fontsize=8,loc='lower right')
    energies=list(by_ce['test'])
    xs=np.arange(len(energies))
    for shift,key,label,color in [(-.25,'cosine','MLP','#2764b7'),(0,'baseline_cosine','Equal','#999999'),(.25,'support_ceiling','Support ceiling','#48844f')]:
        axes[0,1].bar(xs+shift,[by_ce['test'][ce][key] for ce in energies],width=.25,label=label,color=color)
    axes[0,1].set(xticks=xs,xticklabels=[f"{float(ce):g}\nn={by_ce['test'][ce]['spectra']}" for ce in energies],
                  xlabel='Collision energy',ylabel='Mean cosine',ylim=(0,1.05))
    axes[0,1].legend(fontsize=8)
    axes[1,0].scatter([r['target_effective_peaks'] for r in sample],[r['predicted_effective_peaks'] for r in sample],s=16,alpha=.7)
    maximum=max(max(r['target_effective_peaks'],r['predicted_effective_peaks']) for r in sample)
    axes[1,0].plot([0,maximum],[0,maximum],'--',color='#666666')
    axes[1,0].set(xlabel='Experimental effective peak count',ylabel='Predicted effective peak count',title='Above diagonal = predictions too diffuse')
    ranks=np.asarray([r['true_base_peak_rank'] for r in sample])
    bounds=[(1,1),(2,3),(4,5),(6,10),(11,20),(21,np.inf)]
    axes[1,1].bar(['1','2–3','4–5','6–10','11–20','>20'],
                  [int(np.sum((ranks>=low)&(ranks<=high))) for low,high in bounds],color='#2764b7')
    axes[1,1].set(xlabel='Predicted rank of the experimental base peak',ylabel='Test spectra',title='Rank 1 is correct')
    figure.savefig(OUT/'overview.png',dpi=140)
    result=dict(checkpoint_sha256=hashlib.sha256((SOURCE/'model.pt').read_bytes()).hexdigest(),
                summaries=summaries,by_collision_energy=by_ce,slices=slices,
                molecule_bootstrap=intervals,examples=unique,
                explanation='support_ceiling = ||target_supported||2 / ||target_all||2; exact per-spectrum cosine = support_ceiling * supported_cosine')
    (OUT/'diagnostics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summaries':summaries,'test_by_ce':by_ce['test'],'test_slices':slices['test'],'intervals':intervals},indent=2))


if __name__=='__main__':
    main()
