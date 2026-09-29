"""Render the saved diagnostic results without retraining or loading the library."""
import json
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure

OUT=Path(__file__).resolve().parent
feature=json.loads((OUT/'feature_audit.json').read_text())
training=json.loads((OUT/'training_audit.json').read_text())
case=feature['ce_examples'][0]
figure=Figure(figsize=(12,7),layout='constrained')
axes=figure.subplots(2,2,sharex=True,sharey=True)
for row,ce in enumerate((20,60)):
    mz=np.asarray(case[f'mz{ce}'])
    for col,key in enumerate(('target','prediction')):
        values=100*np.asarray(case[f'{key}{ce}'])
        ax=axes[row,col]
        ax.vlines(mz,0,values,color='#d77b32' if col==0 else '#2764b7',lw=2)
        ax.set(xlim=(65,265),ylim=(0,115),title=f"{'Experimental' if col==0 else 'MLP prediction'} · CE {ce}",ylabel='Relative intensity (%)')
        for i in (0,len(mz)-1):
            ax.annotate(f'{mz[i]:.3f}\n{values[i]:.2f}%',(mz[i],values[i]),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
for ax in axes[1]:ax.set_xlabel('m/z')
figure.suptitle('The observed base peak switches with energy; the model prediction barely changes\n'
                 +case['record']['molecule'],fontsize=11)
figure.savefig(OUT/'collision_energy_failure.png',dpi=150)

figure=Figure(figsize=(11,4),layout='constrained')
axes=figure.subplots(1,2)
history=training['baseline_reproduction']['history']
for name,label,color in [('train','Train','#2764b7'),('validation','Validation','#d77b32')]:
    axes[0].plot([x['epoch'] for x in history],[x[f'post_epoch_{name}']['mean_cosine'] for x in history],marker='o',label=label,color=color)
axes[0].axvline(1,color='#666666',ls='--',lw=1)
axes[0].set(xlabel='Epoch',ylabel='Mean cosine',title='Extra training fit does not transfer to new molecules')
axes[0].legend()
controls=training['controls_validation_only']
values=[training['baseline_reproduction']['selected_validation']['mean_cosine']]+[controls[k]['selected_validation']['mean_cosine'] for k in ['regularized_slower_optimization','no_precursor_fingerprint','descriptors_only']]
axes[1].barh(['Original','Slower + regularized','No precursor FP','Descriptors only'],values,color=['#2764b7','#999999','#999999','#48844f'])
axes[1].invert_yaxis()
axes[1].set(xlim=(0,.65),xlabel='Validation mean cosine',title='Three diagnostic controls; no test selection')
for i,value in enumerate(values):axes[1].text(value+.01,i,f'{value:.3f}',va='center')
figure.savefig(OUT/'training_diagnostics.png',dpi=150)
