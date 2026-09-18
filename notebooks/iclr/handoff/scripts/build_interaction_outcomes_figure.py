"""Plot all frozen held-out task outcomes, paired by training seed."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import mean

os.environ.setdefault('MPLCONFIGDIR', '/tmp/iclr-interaction-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 123)
ARMS = ('P1_UNREG', 'P1_MIX', 'P1_NORM')


def render():
    path = ROOT/'data/locked_evaluation_20260916/held_aside_results.csv'
    manifest = json.loads(path.with_name('evaluation_manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['results_csv_sha256']
    with path.open() as f:
        rows = list(csv.DictReader(f))
    lookup = {(r['task'], r['condition'], int(r['seed'])): r for r in rows}
    expected = {(t, a, s) for t in ('rte', 'mrpc') for a in ARMS for s in SEEDS}
    assert len(rows) == len(lookup) == 18 and set(lookup) == expected
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.labelsize': 8, 'axes.titlesize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 3.65))
    markers = ('o', 's', '^')
    for col, task in enumerate(('rte', 'mrpc')):
        for row, metric in enumerate(('held_aside_loss', 'held_aside_accuracy')):
            ax = axes[row, col]
            scale = 1 if row == 0 else 100
            by_seed = [[scale*float(lookup[task, a, s][metric]) for a in ARMS] for s in SEEDS]
            for vals, marker in zip(by_seed, markers):
                ax.plot(range(3), vals, color='0.65', linewidth=0.65,
                        marker=marker, markersize=4, markerfacecolor='white',
                        markeredgecolor='0.25', markeredgewidth=0.8)
            ax.plot(range(3), [mean(v[i] for v in by_seed) for i in range(3)],
                    linestyle='none', marker='_', color='black', markersize=16,
                    markeredgewidth=1.6, zorder=5)
            ax.set_xticks(range(3), ('UNREG', 'MIX', 'NORM'))
            ax.set_xlim(-0.3, 2.3)
            if row == 0:
                ax.set_title(task.upper(), fontweight='bold')
                ax.set_ylabel('Prediction loss (nats)')
                ax.set_ylim(0, 3 if task == 'rte' else 1)
            else:
                ax.set_ylabel('Accuracy (%)')
                ax.set_ylim(65, 90)
                ax.set_yticks((65, 70, 75, 80, 85, 90))
            ax.grid(axis='y', color='0.9', linewidth=0.5)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2)
    handles = [Line2D([], [], linestyle='none', marker=m, markersize=4,
                      markerfacecolor='white', markeredgecolor='0.25',
                      label=f'Seed {s}') for s,m in zip(SEEDS, markers)]
    handles.append(Line2D([], [], linestyle='none', marker='_', color='black',
                          markersize=14, markeredgewidth=1.6, label='Mean'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               bbox_to_anchor=(0.52, -0.005))
    fig.subplots_adjust(left=0.095, right=0.98, top=0.92, bottom=0.16,
                        hspace=0.37, wspace=0.30)
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf', metadata={'CreationDate': None, 'ModDate': None,
                                          'Creator': 'Validated frozen-endpoint analysis'})
    plt.close(fig)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--write', action='store_true')
    group.add_argument('--check', action='store_true')
    args = parser.parse_args()
    pdf = render()
    path = ROOT/'figures/interaction_task_outcomes.pdf'
    if args.write:
        path.write_bytes(pdf)
    else:
        assert path.read_bytes() == pdf, 'Outcome figure differs from frozen data/rendering.'
    print('PASS: all 18 hash-bound held-out rows plotted, paired by seed; no imputation.')


if __name__ == '__main__':
    main()
