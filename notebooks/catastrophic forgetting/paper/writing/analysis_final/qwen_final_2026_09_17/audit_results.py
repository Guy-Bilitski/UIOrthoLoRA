"""Recompute final manuscript contrasts from the pinned, unmodified JSON summaries."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / 'evidence/results'
LLAMA = [
    ('LoRA+wd', 'tia1_frc_lorawd_wd0p3_lr5e4_s43'),
    ('CLoRA', 'tia1_frc_clora_k1024_lr3e4_s44'),
    ('MiLoRA', 'tia1_frc_milora_lr3e4_s43'),
    ('LoRA', 'tia1_frc_lora_r16_lr3e4_s43'),
    ('LoRA-Null', 'tia1_frc_loranull_r16_lr5e4_s43'),
]
QWEN = [
    ('LoRA', 'tia1_qwsw_lora_r16_lr5e5_s43'),
    ('MiLoRA', 'tia1_qwsw_milora_lr1e4_s43'),
    ('LoRA+wd', 'tia1_qwsw_lorawd_wd0p3_lr1e4_s43'),
    ('CLoRA 2e-4', 'tia1_qwsw_clora_k1024_lr2e4_s43'),
    ('CLoRA 3e-4', 'tia1_qwsw_clora_k1024_lr3e4_s43'),
    ('LoRA-Null', 'tia1_qwsw_loranull_r16_lr2e4_s43'),
]
# Recorded construction infeasibility: two Llama E arms and two Qwen E arms.
INFEASIBLE_E = {LLAMA[3][1], LLAMA[4][1], QWEN[3][1], QWEN[5][1]}

def headline(run, arm):
    suffix = '__rl50' if arm == 'A' else '__k10allabl' + arm
    return json.loads((RESULTS / (run + suffix) / 'summary.json').read_text())['headline']

def write_csv(name, rows):
    with (ROOT / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

def main():
    rows = []
    for arch, configs in [('Llama', LLAMA), ('Qwen', QWEN)]:
        for name, run in configs:
            A, B, C, D = [headline(run, x) for x in 'ABCD']
            E = headline(run, 'E') if run not in INFEASIBLE_E else None
            row = dict(architecture=arch, design=name, run=run)
            for metric in ['retention_mean', 'cs_avg', 'fdelta', 'bbh', 'mmlu_pro']:
                for contrast, left, right in [('C-B', C, B), ('D-A', D, A), ('B-A', B, A), ('C-A', C, A)]:
                    row[contrast + '_' + metric] = round(left[metric] - right[metric], 8)
            row['E-B_retention_mean'] = round(E['retention_mean'] - B['retention_mean'], 8) if E else None
            rows.append(row)
    write_csv('contrasts.csv', rows)
    for key in ['C-B_retention_mean', 'C-B_cs_avg', 'D-A_retention_mean', 'B-A_retention_mean', 'C-B_fdelta', 'D-A_fdelta', 'E-B_retention_mean']:
        vals = [r[key] for r in rows if r[key] is not None]
        print(key, 'positive', sum(x > 0 for x in vals), 'negative', sum(x < 0 for x in vals), 'n', len(vals))
    print('B>A gains removed by D:', sum(r['B-A_retention_mean'] > 0 and r['D-A_retention_mean'] < 0 for r in rows), '/', sum(r['B-A_retention_mean'] > 0 for r in rows))
    print('C>B on both components:', sum(r['C-B_bbh'] > 0 and r['C-B_mmlu_pro'] > 0 for r in rows))
    print(json.dumps(rows, indent=2))

    run = QWEN[4][1]
    ray = []
    for suffix in ['__rayf100', '__rayf150', '__k10allablC', '__rayf246', '__rl50', '__rayf330', '__rayrand1f246', '__rayrand2f246', '__rayrand3f246', '__k10allablB', '__k10allablD', '__k10allablE']:
        h = json.loads((RESULTS / (run + suffix) / 'summary.json').read_text())['headline']
        ray.append(dict(variant=suffix, **{k: h[k] for k in ['fdelta', 'retention_mean', 'cs_avg']}))
    write_csv('qwen_ray.csv', ray)
    x = np.array([r['fdelta'] for r in ray[:6]])
    fits = {}
    for metric in ['retention_mean', 'cs_avg']:
        y = np.array([r[metric] for r in ray[:6]])
        for form, degree, tx in [('linear', 1, x), ('quadratic', 2, x), ('log', 1, np.log(x))]:
            coefficients = np.polyfit(tx, y, degree)
            predicted = np.polyval(coefficients, tx)
            r2 = 1 - np.sum((y-predicted)**2)/np.sum((y-y.mean())**2)
            residuals = {r['variant']: float(r[metric] - np.polyval(coefficients, np.log(r['fdelta']) if form == 'log' else r['fdelta'])) for r in ray[6:]}
            fits[metric + '_' + form] = dict(coefficients=coefficients.tolist(), r2=float(r2), residuals=residuals)
    (ROOT / 'qwen_ray_fits.json').write_text(json.dumps(fits, indent=2)+'\n')
    print('RAY', json.dumps(ray, indent=2))
    print('FITS', json.dumps(fits, indent=2))
    near = [ray[i] for i in [3, 6, 7, 8, 9]]
    print('Near-magnitude spans', {k: max(r[k] for r in near)-min(r[k] for r in near) for k in ['fdelta','retention_mean','cs_avg']})
    sc = [headline('tia1_qwsw_sclora_lr2e5_s43', a) for a in 'ABCDE']
    print('SC A-E ranges', {k: [min(r[k] for r in sc),max(r[k] for r in sc)] for k in ['fdelta','retention_mean','cs_avg']})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    parser.add_argument('--output', type=Path, default=ROOT)
    args = parser.parse_args()
    RESULTS = args.results.resolve()
    ROOT = args.output.resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    main()
