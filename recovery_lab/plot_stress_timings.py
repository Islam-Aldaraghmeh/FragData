"""Export measured pipeline stage timings; mark diagnostically censored runs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = Path(__file__).resolve().parent / 'stress_20260922'
    rows = [json.loads(p.read_text()) for p in sorted(out.glob('spectrum_*.json'))]
    rows = [r for r in rows if r['status'] != 'started']
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11, max(5, .29 * len(rows) + 2)))
    left = np.zeros(len(rows))
    stages = [('generation_seconds', 'MAGMa', '#9aa4b2'),
              ('preparation_seconds', 'Ionic-state preparation', '#dd9742'),
              ('prediction_seconds', 'Glacier inference', '#387fb2')]
    for key, label, color in stages:
        values = np.array([r.get(key, 0) for r in rows])
        ax.barh(y, values, left=left, height=.72, color=color, label=label)
        left += values
    labels = []
    for i, r in enumerate(rows):
        censored = r['status'] == 'timeout' or r['outcomes'].get('timeout', 0) > 0
        dataset = 'Enamdisc' if 'enamdisc' in r['dataset'] else 'MCE drug'
        labels.append(f"{r['index']:02d}  {dataset}" + (' *' if censored else ''))
        count = len(r['candidate_rows'])
        detail = ('Precursor rejected' if r['status'] == 'exception'
                  else f"{count}/{r.get('candidates', '?')}")
        ax.text(left[i] + .7, i, detail,
                va='center', fontsize=8, color='#374151')
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(60, color='#9c3b3b', linestyle='--', linewidth=1)
    ax.set_xlim(0, max(100, left.max() + 16))
    ax.set_xlabel('Measured seconds (model loading excluded)')
    ax.set_title(f'{len(rows)} sampled MS2 spectra with linked MS3', loc='left', pad=30)
    ax.text(0, 1.01, '* Censored by diagnostic budget. Row-end numbers: attempted/total candidates.',
            transform=ax.transAxes, fontsize=9, color='#4b5563')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.065), ncol=3, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(out / f'timings.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
