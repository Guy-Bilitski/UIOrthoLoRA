"""Reconcile archived module diagnostics and derive normalized block energies.

No model training or synthetic experimental data. The script checks every run
against the separately archived summary before constructing figure/table data.
"""
from collections import defaultdict
import csv
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
TASKS = [('rte_lin', 'RTE'), ('mrpc_lin', 'MRPC'), ('cola_lin', 'CoLA'),
         ('stsb_lin', 'STS-B'), ('sst2_lin', 'SST-2')]
CONDS = [('A_no_reg', 'A'), ('B_muE_only', 'B'), ('C_muE_nuD', 'C')]


def records(folder):
    for path in sorted((ROOT / 'data' / folder).glob('*.csv')):
        with path.open(newline='') as handle:
            yield from csv.DictReader(handle)


def key(row):
    task = row['task'].replace('sts-b', 'stsb').replace('sts_b', 'stsb')
    condition = next(c for c, _ in CONDS if row['run_name'].startswith(c))
    return task, condition, int(row['seed'])


def analyze(include_runs=False):
    grouped = defaultdict(list)
    seen = set()
    max_frame_discrepancy = 0.0
    for row in records('legacy_layers'):
        identifier = (*key(row), row['layer_name'])
        assert identifier not in seen, identifier
        seen.add(identifier)
        # Leak*_F divides by ||Delta||; OffTailRatio_F divides by ||C||.
        # Recover normalized allocation in the recorded coordinate frame.
        squared = [float(row[name])**2 for name in ('Leak11_F', 'Leak12_F', 'Leak21_F')]
        off = float(row['OffTailRatio_F'])**2
        assert 0 < off <= 1 and sum(squared) > 0
        max_frame_discrepancy = max(max_frame_discrepancy, abs(sum(squared) / off - 1))
        for name, value in zip(('pLL', 'pLT', 'pTL'), squared):
            row[name] = off * value / sum(squared)
        row['pCross'] = row['pLT'] + row['pTL']
        row['pTT'] = 1 - off
        assert abs(row['pLL'] + row['pCross'] + row['pTT'] - 1) < 1e-12
        grouped[key(row)].append(row)
    assert len(seen) == 1296 and len(grouped) == 27
    expected_modules = {
        f'base_model.model.roberta.encoder.layer.{layer}.attention.{suffix}'
        for layer in range(12)
        for suffix in ('self.query', 'self.key', 'self.value', 'output.dense')
    }
    for run, rows in grouped.items():
        assert len(rows) == 48 and {r['layer_name'] for r in rows} == expected_modules, run

    summaries = {key(row): row for row in records('legacy_mixing')}
    assert set(grouped) == set(summaries)
    max_summary_error = 0.0
    for run, rows in grouped.items():
        summary = summaries[run]
        fields = [f for f in summary if f.startswith('mean_') and f[5:] in rows[0]]
        assert len(fields) >= 10
        for field in fields:
            computed = mean(float(r[field[5:]]) for r in rows)
            reported = float(summary[field])
            error = abs(computed - reported)
            max_summary_error = max(max_summary_error, error)
            assert error <= 1e-10 * max(1, abs(reported)), (run, field, computed, reported)

    task_means = {}
    for task, _ in TASKS:
        expected_seeds = {42} if task == 'sst2_lin' else {17, 42}
        for condition, _ in CONDS:
            runs = [rows for (t, c, _), rows in grouped.items() if (t, c) == (task, condition)]
            assert {seed for t, c, seed in grouped if (t, c) == (task, condition)} == expected_seeds
            task_means[task, condition] = {
                metric: mean(mean(float(r[metric]) for r in rows) for rows in runs)
                for metric in ('pLL', 'pLT', 'pTL', 'pCross', 'pTT')
            }

    pairs, cross_reduced_runs = [], 0
    run_reductions = defaultdict(int)
    for (task, condition, seed), arows in grouped.items():
        if condition != 'A_no_reg':
            continue
        crows = grouped[task, 'C_muE_nuD', seed]
        bmap = {r['layer_name']: r for r in crows}
        pairs.extend((a, bmap[a['layer_name']]) for a in arows)
        cross_reduced_runs += mean(r['pCross'] for r in crows) < mean(r['pCross'] for r in arows)
        for field in ('RelPert_F', 'OffTailRatio_F', 'Drift_U', 'Drift_V'):
            run_reductions[field] += mean(float(r[field]) for r in crows) < mean(float(r[field]) for r in arows)
    assert len(pairs) == 432
    report = {
        'module_rows': len(seen), 'runs': len(grouped), 'paired_modules': len(pairs),
        'cross_reduced_runs': cross_reduced_runs,
        'run_mean_reductions': dict(run_reductions),
        'module_reductions': {
            metric: sum(float(c[metric]) < float(a[metric]) for a, c in pairs)
            for metric in ('pLL', 'pCross', 'OffTailRatio_F', 'RelPert_F', 'Drift_U', 'Drift_V')
        },
        'max_frame_energy_discrepancy': max_frame_discrepancy,
        'max_summary_absolute_error': max_summary_error,
    }
    if include_runs:
        return task_means, report, {
            run: {metric: mean(float(r[metric]) for r in rows)
                  for metric in ('pLL', 'pLT', 'pTL', 'pCross', 'pTT')}
            for run, rows in grouped.items()
        }
    return task_means, report


def latex():
    values, _, runs = analyze(include_runs=True)
    panels, rows = [], []
    for task, label in TASKS:
        n = 1 if task == 'sst2_lin' else 2
        panel = [r'\begin{minipage}[t]{0.19\linewidth}', r'\centering\small ' + label + f' ($n={n}$)' + r'\par\smallskip']
        for condition, short in CONDS:
            record = values[task, condition]
            parts = ''.join('{' + f'{record[f]:.7f}' + '}' for f in ('pLL', 'pCross', 'pTT'))
            panel.append(r'\allocationbar{' + short + '}' + parts + r'\par\smallskip')
            for seed in (42,) if n == 1 else (17, 42):
                run = runs[task, condition, seed]
                rows.append(' & '.join([label, short, str(seed)] +
                                      [f'{100*run[f]:.2f}' for f in ('pLL', 'pLT', 'pTL', 'pTT')]) + r' \\')
        panel.append(r'\allocationbar{N}{0.4444444}{0.4444444}{0.1111112}\par\smallskip')
        panel.append(r'\end{minipage}')
        panels.append('\n'.join(panel))
        if task != TASKS[-1][0]:
            rows.append(r'\midrule')
    return {'BLOCK FIGURE': '\n\\hfill\n'.join(panels) + '\n',
            'BLOCK TABLE': '\n'.join(rows) + '\n'}


if __name__ == '__main__':
    import json
    values, report = analyze()
    print(json.dumps(report, indent=2))
    for (task, condition), value in values.items():
        print(task, condition, ' '.join(f'{name}={value[name]:.6f}' for name in ('pLL', 'pCross', 'pTT')))
