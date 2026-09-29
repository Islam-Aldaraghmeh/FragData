"""Train two small intensity MLPs; compare to the frozen original on fresh molecules.

Run .venv/bin/python train_intensity_v2.py. MAGMa preparation uses a local cache.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from intensity_model import spectrum_metrics
from intensity_model_v2 import (EnergyConditionedMLP, FEATURE_SIZE, SCHEMA,
                                with_collision_energy)
from train_intensity import (ROOT, forward_batch, predict_examples, equal_peak_baseline,
                             load_model as load_old_model, plot_comparison)

OUTPUT = ROOT / "results" / "experiment_2_v2"


def fit_model(train, validation, *, use_fingerprints=True, seed=20260923,
              epochs=100, patience=15, batch_size=32, learning_rate=0.002,
              weight_decay=0.01):
    """Cosine loss, AdamW and validation-only checkpoint selection."""
    if not train or not validation or min(epochs,patience,batch_size)<1:
        raise ValueError("Nonempty splits and positive training limits are required")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    model = EnergyConditionedMLP(use_fingerprints=use_fingerprints)
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    best, best_epoch, best_state, history = -1., 0, None, []
    for epoch in range(1,epochs+1):
        model.train()
        losses=[]
        order=rng.permutation(len(train))
        for start in range(0,len(train),batch_size):
            batch=[train[i] for i in order[start:start+batch_size]]
            predictions=forward_batch(model,batch)
            loss=torch.stack([1-F.cosine_similarity(prediction,torch.as_tensor(e["target"]),dim=0,eps=1e-8)
                              for prediction,e in zip(predictions,batch)]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.extend([float(loss.detach())]*len(batch))
        prediction=predict_examples(model,validation)
        score=spectrum_metrics([e["target"] for e in validation],prediction)["mean_cosine"]
        history.append(dict(epoch=epoch,train_cosine_loss=float(np.mean(losses)),validation_mean_cosine=score))
        if score>best:
            best,best_epoch,best_state=score,epoch,copy.deepcopy(model.state_dict())
        if epoch==1 or epoch%10==0:
            print(f"{'fingerprint' if use_fingerprints else 'descriptor'} epoch {epoch:03d}: "
                  f"loss={np.mean(losses):.4f}, validation={score:.4f}",flush=True)
        if epoch-best_epoch>=patience:
            break
    model.load_state_dict(best_state)
    return model.eval(),history,best_epoch


def compare_metrics(examples,predictions):
    """Full-axis metrics; tied predicted maxima receive fractional top-one credit."""
    targets=[e["target"] for e in examples]
    result=spectrum_metrics(targets,predictions)
    def effective(values):
        return float(values.sum()**2/np.dot(values,values)) if np.dot(values,values)>0 else 0.
    accuracy=[]
    for y,p in zip(targets,predictions):
        winners=p==p.max(initial=0)
        accuracy.append(float(np.mean(y[winners]==y.max())) if p.max(initial=0)>0 else 0.)
    result.update(predicted_effective_peak_count=float(np.mean([effective(p) for p in predictions])),
                  experimental_effective_peak_count=float(np.mean([effective(y) for y in targets])),
                  strongest_peak_accuracy=float(np.mean(accuracy)))
    return result


def legacy_examples(examples, nearest=False):
    prefix="old_nearest" if nearest else "old"
    return [dict(e,features=e[f"{prefix}_features"],peak_indices=e[f"{prefix}_peak_indices"])
            for e in examples]


def energy_response(model,examples,*,old=False,equal=False):
    """Change only CE on fixed candidate support; recompute all head interactions."""
    if equal:
        predictions=[equal_peak_baseline(examples)]*2
    else:
        predictions=[]
        for energy in (20.,60.):
            changed=[]
            for example in examples:
                if old:
                    features=example["features"].copy()
                    features[:,1024]=energy/100
                else:
                    features=with_collision_energy(example["features"],energy)
                changed.append(dict(example,features=features))
            predictions.append(predict_examples(model,changed))
    eligible=[i for i,e in enumerate(examples) if len(np.unique(e["peak_indices"]))>=2]
    cosines,changes=[],[]
    for index in eligible:
        a,b=predictions[0][index],predictions[1][index]
        denominator=np.linalg.norm(a)*np.linalg.norm(b)
        cosines.append(float(np.dot(a,b)/denominator) if denominator else 0.)
        changes.append(float(np.mean(np.abs(a-b))))
    return dict(energies=[20,60],spectra_with_at_least_two_supported_peaks=len(eligible),
                excluded_zero_or_single_supported_peak_spectra=len(examples)-len(eligible),
                mean_prediction_cosine=float(np.mean(cosines)) if cosines else None,
                mean_absolute_normalized_intensity_change=float(np.mean(changes)) if changes else None)


def metrics_markdown(metrics):
    lines=["| Model / support | Mean cosine | Median | >0.5 | >0.7 | Effective peaks | Strongest peak |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for name,m in metrics.items():
        lines.append(f"| {name} | {m['mean_cosine']:.4f} | {m['median_cosine']:.4f} | "
                     f"{m['fraction_cosine_gt_0.5']:.1%} | {m['fraction_cosine_gt_0.7']:.1%} | "
                     f"{m['predicted_effective_peak_count']:.2f} | {m['strongest_peak_accuracy']:.1%} |")
    return "\n".join(lines)


def plot_energy_example(model, examples, output_dir):
    """First test molecule with CE20/60, picked without inspecting its scores."""
    from matplotlib.figure import Figure
    paired={}
    for example in examples:
        paired.setdefault(example['molecule'],{})[example['collision_energy']]=example
    group=next((g for g in paired.values() if 20. in g and 60. in g),None)
    if group is None:
        return None
    figure=Figure(figsize=(11,6),layout="constrained")
    axes=figure.subplots(2,2,sharex=True,sharey=True)
    for row,energy in enumerate((20.,60.)):
        example=group[energy]
        prediction=predict_examples(model,[example])[0]
        for column,(values,title,color) in enumerate(((example['target'],'Experimental','#d77b32'),
                                                      (prediction,'Energy-conditioned MLP','#2764b7'))):
            axes[row,column].vlines(example['mz'],0,100*values,color=color,linewidth=1.3)
            axes[row,column].set(title=f"{title} · CE {energy:g}",ylim=(0,108))
            axes[row,column].spines[['top','right']].set_visible(False)
        axes[row,0].set_ylabel('Relative intensity (%)')
    for ax in axes[-1]:ax.set_xlabel('m/z')
    figure.suptitle(group[20.]['smiles'],fontsize=10)
    name='collision_energy_example.png'
    figure.savefig(output_dir/name,dpi=140)
    return name


def run_experiment(data_path=ROOT/"msnlib/20241003_enamdisc_pos_msn.json",output_dir=OUTPUT,
                   *,molecules=1500,seed=20260923,workers=2,epochs=100,patience=15):
    from intensity_data_v2 import prepare_dataset
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    examples,manifest=prepare_dataset(data_path,output_dir,molecules=molecules,seed=seed,workers=workers)
    splits={name:[e for e in examples if e["split"]==name] for name in ("train","validation","test")}
    models,histories,selected={}, {}, {}
    for name,use_fp in (("energy_mlp",True),("descriptor_only",False)):
        model,history,best_epoch=fit_model(splits["train"],splits["validation"],
            use_fingerprints=use_fp,seed=seed,epochs=epochs,patience=patience)
        models[name],histories[name],selected[name]=model,history,best_epoch
        torch.save(dict(schema=SCHEMA,feature_size=FEATURE_SIZE,config=model.config,
                        state_dict=model.state_dict(),seed=seed,best_epoch=best_epoch),
                   output_dir/("model.pt" if use_fp else "descriptor_model.pt"))
    # All new-model fitting/selection is finished before this test evaluation.
    old_path=ROOT/"results/experiment_2/model.pt"
    old_model=load_old_model(old_path)
    metrics,responses,predictions_by_split,support={}, {}, {}, {}
    for split in ("validation","test"):
        group=splits[split]
        old_all,old_nearest=legacy_examples(group),legacy_examples(group,nearest=True)
        predictions={name:predict_examples(model,group) for name,model in models.items()}
        predictions.update(old_original_support=predict_examples(old_model,old_all),
                           old_nearest_support=predict_examples(old_model,old_nearest),
                           equal_nearest_support=equal_peak_baseline(group),
                           equal_original_support=equal_peak_baseline(old_all))
        metrics[split]={name:compare_metrics(group,pred) for name,pred in predictions.items()}
        responses[split]={name:energy_response(model,group) for name,model in models.items()}
        responses[split].update(old_original_support=energy_response(old_model,old_all,old=True),
                               old_nearest_support=energy_response(old_model,old_nearest,old=True),
                               equal_nearest_support=energy_response(None,group,equal=True))
        support[split]={}
        for name,records in (("original",old_all),("nearest",group)):
            ceiling=[]
            for e in records:
                mask=np.zeros(len(e["target"]),dtype=bool)
                mask[e["peak_indices"]]=True
                ceiling.append(float(np.linalg.norm(e["target"][mask])/np.linalg.norm(e["target"])))
            support[split][name]=dict(mean_oracle_cosine=float(np.mean(ceiling)),
                unsupported_spectra=sum(len(e["peak_indices"])==0 for e in records))
        predictions_by_split[split]=predictions
    unique={}
    for index,e in enumerate(splits["test"]):unique.setdefault(e["molecule"],index)
    plots=[]
    for number,index in enumerate(list(unique.values())[:3],1):
        example=splits["test"][index]
        figure=plot_comparison(example,predictions_by_split["test"]["energy_mlp"][index],
                                predictions_by_split["test"]["equal_nearest_support"][index])
        filename=f"test_example_{number}.png"
        figure.savefig(output_dir/filename,dpi=140)
        plots.append(dict(file=filename,test_index=index,source_offset=example["source_offset"]))
    energy_plot=plot_energy_example(models['energy_mlp'],splits['test'],output_dir)
    results=dict(seed=seed,molecule_counts={k:len({e['molecule'] for e in v}) for k,v in splits.items()},
        spectrum_counts={k:len(v) for k,v in splits.items()},best_epochs=selected,
        parameter_counts={name:sum(p.numel() for p in m.parameters()) for name,m in models.items()},
        training_settings=dict(max_epochs=epochs,patience=patience,batch_size=32,lr=.002,weight_decay=.01),
        metrics=metrics,ce_response=responses,support=support,history=histories,plots=plots,energy_plot=energy_plot,
        old_checkpoint_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-started)
    (output_dir/"metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    report=f"""# Improved MAGMa intensity-only model

On the same fresh test cohort, the small fingerprint model reaches
**{metrics['test']['energy_mlp']['mean_cosine']:.4f} mean cosine**, versus
**{metrics['test']['old_original_support']['mean_cosine']:.4f}** for the frozen old model.
The descriptor-only baseline reaches **{metrics['test']['descriptor_only']['mean_cosine']:.4f}**;
its weights are available separately as `descriptor_model.pt`. Both checkpoints were
selected using validation. No test-driven hyperparameter changes were made.

The same latent max aggregation and full-spectrum cosine loss are retained.
The sample contains {results['molecule_counts']} molecules and {results['spectrum_counts']} spectra.
All 300 previously examined molecules and equivalent InChI connectivity identities are excluded.
Train/validation/test remain molecule-disjoint; the frozen old model is evaluated on the **same fresh test spectra**.

Nearest assignment uses mass alone: a canonical fragment graph/formula/charge goes to one
closest observed centroid within its existing MAGMa matches. Ties prefer lower m/z then
original order. Assignment occurs separately for each scan after union generation, so CE
scans cannot steal one another's candidates. Empty bags remain zero predictions.

The model has {results['parameter_counts']['energy_mlp']:,} parameters versus 140,545 previously.
It uses 20 structural/mass-error descriptors, a downweighted 16-unit fingerprint projection
(128 bits each for precursor and fragment), and a 48/24-unit MLP. Its three outputs are
fragment-specific coefficients of `a + b E + c E²`, with `E=(CE−40)/20`; Softplus converts
this to positive candidate intensity. Thus energy interacts explicitly with fragment features.
The FP branch uses dropout0.15; AdamW uses weight decay0.01 and lr0.002. No intensity labels
enter features. No Glacier, microstate enumeration, GNN or new framework is used.

Both models were fixed before test evaluation. Validation selected epochs {selected}.
The descriptor-only baseline retains the same conditioning but omits the fingerprint branch.

## Fresh held-out test results

{metrics_markdown(metrics['test'])}

Experimental effective peak count: **{metrics['test']['energy_mlp']['experimental_effective_peak_count']:.2f}**.
Effective count is `(sum I)² / sum(I²)`, averaged per spectrum. Strongest-peak accuracy
uses fractional credit among exactly tied predicted maxima (uniform tie breaking); a zero
prediction gets zero credit. This matters particularly for the equal-intensity baseline.
All observed peaks contribute, with zero predictions for unsupported locations. Metrics
macro-average across spectra. Normalized MAE is also retained in `metrics.json`.

`old_original_support` is the old checkpoint and old matching. `old_nearest_support`
isolates the matching change with its weights frozen. Both equal baselines show support effects.
These fresh-cohort numbers should be compared within this table, not directly against the
earlier 45-molecule test cohort. Nearest/support oracle cosines are
{support['test']['nearest']['mean_oracle_cosine']:.4f}/{support['test']['original']['mean_oracle_cosine']:.4f}.

## Collision-energy response

Hold each candidate set fixed and change only CE20→CE60. Energy interactions are recomputed.
Zero/single-supported-peak spectra are excluded because their normalized shape cannot change.
Smaller prediction cosine indicates more shape change, not necessarily better chemistry.

| Model | CE20/60 prediction cosine | Mean absolute intensity change (percentage points) |
|---|---:|---:|
"""
    for name,r in responses["test"].items():
        similarity=r['mean_prediction_cosine']
        change=r['mean_absolute_normalized_intensity_change']
        similarity_text="n/a" if similarity is None else f"{similarity:.6f}"
        change_text="n/a" if change is None else f"{100*change:.3f}"
        report+=f"| {name} | {similarity_text} | {change_text} |\n"
    if energy_plot:
        report+=f"\nFirst test molecule with both energies, selected without scores:\n\n![Energy example]({energy_plot})\n"
    report+=f"""
## Validation and examples

{metrics_markdown(metrics['validation'])}

The first three distinct test molecules were chosen without examining scores.
"""+"\n\n".join(f"![Test example {i+1}]({p['file']})" for i,p in enumerate(plots))+"""

Run `.venv/bin/python train_intensity_v2.py` or open `../../Experiment_2.ipynb`.
`intensity_model_v2.predict_spectrum(model, smiles, collision_energy, bags)` performs
nearest assignment and prediction. The original checkpoint and published old-cohort
artifacts remain available under `../experiment_2/`.

The improvements bundle more training molecules, nearest matching, explicit energy
conditioning and regularization; this run does not isolate every change's contribution.
The frozen-old-model nearest-support row isolates matching alone. Predictions remain
too diffuse compared with experimental spectra, and the descriptor baseline remains
competitive with the fingerprint branch. This is still an intensity experiment
conditional on observed MAGMa-supported peak locations, not peak discovery.
"""
    (output_dir/"SUMMARY.md").write_text(report)
    print(metrics_markdown(metrics["test"]),flush=True)
    return results


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"msnlib/20241003_enamdisc_pos_msn.json")
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--molecules",type=int,default=1500)
    parser.add_argument("--seed",type=int,default=20260923)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--patience",type=int,default=15)
    args=parser.parse_args()
    run_experiment(args.data,args.output,molecules=args.molecules,seed=args.seed,
                   workers=args.workers,epochs=args.epochs,patience=args.patience)
