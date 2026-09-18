"""Reproduce behavior-first results and the complete available strict-band snapshot."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (17, 42, 123)
ARMS = ('P1_UNREG', 'P1_MIX', 'P1_NORM')


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def summarize(values):
    values = list(values)
    assert len(values) == 3
    return dict(values=values, mean=mean(values), sample_sd=stdev(values))


def cell(summary, digits):
    return f'${summary["mean"]:.{digits}f} \\pm {summary["sample_sd"]:.{digits}f}$'


def analyze():
    from analyze_final_evidence import analyze as verify_final
    from analyze_review_v2 import analyze as verify_locked
    final, _, _ = verify_final()
    verify_locked()
    focused_path = ROOT/'data/focused_norm/runs.csv'
    held_path = ROOT/'data/locked_evaluation_20260916/held_aside_results.csv'
    band_path = ROOT/'data/final_evidence_20260916/band_results.csv'
    coverage_path = ROOT/'data/final_evidence_20260916/band_coverage.json'
    focused = {r['run_id']: r for r in read_csv(focused_path) if r['stage']=='confirmation'}
    held = read_csv(held_path)
    assert {r['run_id'] for r in held} == set(focused) and len(held)==18
    lookup = {(r['task'], r['condition'], int(r['seed'])): r for r in held}
    summaries, lines = {}, []
    for task in ('rte', 'mrpc'):
        for arm in ARMS:
            rows = [lookup[task, arm, seed] for seed in SEEDS]
            summary = dict(
                accuracy_percent=summarize(100*float(r['held_aside_accuracy']) for r in rows),
                nll=summarize(float(r['held_aside_loss']) for r in rows),
                cross_percent=summarize(100*(float(focused[r['run_id']]['pooled_LT'])+float(focused[r['run_id']]['pooled_TL'])) for r in rows),
                rho_pooled=summarize(float(focused[r['run_id']]['rho_pooled']) for r in rows))
            summaries[task+'/'+arm] = summary
            lines.append(' & '.join([task.upper(),arm.removeprefix('P1_'),cell(summary['accuracy_percent'],2),cell(summary['nll'],3),cell(summary['cross_percent'],1),cell(summary['rho_pooled'],4)]) + r' \\')
        if task=='rte': lines.append(r'\midrule')
    main_table = r'''\begin{table}[t]
\centering
\caption{Interaction control, size control and unregularized training at the same fixed endpoints. All conditions use the practical leading-plus-tail adapter without rotations. Entries are mean$\pm$sample SD over three seeds. Accuracy and prediction loss use held-out task examples; Cross is the pooled percentage of update energy connecting leading and tail directions, and $\rho_F$ is relative update size. Table~\ref{tab:matchedconfirmation} gives per-seed size differences after calibration.}
\label{tab:learningandmixing}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{llcccc}
\toprule
Task & Condition & Accuracy (\%) & Task loss (nat) & Cross (\%) & $\rho_F$\\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}'''
    band = read_csv(band_path)
    band_order = {'BAND_LEAD_DIAG':0,'BAND_LEAD_ROT64':1,'BAND_MID_DIAG':2,'BAND_MID_ROT64':3,'BAND_TAIL_DIAG':4,'BAND_TAIL_ROT64':5,'P1_HEAD_BASE':6}
    band = sorted(band, key=lambda r: (r['purpose']!='confirmation', r['task'], band_order[r['condition']], int(r['seed'])))
    lines = []
    previous_purpose = None
    for r in band:
        if r['purpose'] != previous_purpose:
            if previous_purpose is not None: lines.append(r'\midrule')
            label = 'All 21 confirmation endpoints' if r['purpose']=='confirmation' else 'Timing pilots (excluded from confirmations)'
            lines.append(r'\multicolumn{7}{l}{\textit{'+label+r'}} \\')
        previous_purpose = r['purpose']
        if r['condition']=='P1_HEAD_BASE':
            band_name, core = 'Head only', '---'
        else:
            _, band_name, flexibility = r['condition'].split('_')
            band_name = band_name.title()
            core = 'Diagonal' if flexibility=='DIAG' else 'Rotated 64'
        lines.append(' & '.join([r['task'].upper(),band_name,core,r['seed'],f'{100*float(r["fixed_accuracy"]):.2f}',f'{float(r["fixed_task_loss"]):.3f}',f'{int(r["trainable_adapter_params"]):,}'])+r' \\')
    band_table = r'''\begin{table}[t]
\centering
\caption{All 21 strict-band/head confirmations and two timing pilots. Scores use fixed endpoints on the inner-selection split; each row is one run. Pilots are excluded from every confirmation summary. Adapter counts exclude the common 592,130-parameter trained head. Band updates remain inside their selected pretrained span by construction.}
\label{tab:bandavailable}
\begin{tabular}{llllrrr}
\toprule
Task & Band & Core & Seed & Acc. (\%) & Task loss & Adapter params\\
\midrule
''' + '\n'.join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}'''
    arms = final['band']['arms']
    lines = []
    for arm, label in [
            ('P1_HEAD_BASE','Classifier only'),
            ('BAND_LEAD_DIAG','Leading coefficients'),
            ('BAND_LEAD_ROT64','Leading coefficients + rotations'),
            ('BAND_MID_DIAG','Middle coefficients'),
            ('BAND_MID_ROT64','Middle coefficients + rotations'),
            ('BAND_TAIL_DIAG','Tail coefficients'),
            ('BAND_TAIL_ROT64','Tail coefficients + rotations')]:
        s = arms['rte/'+arm]
        cells = [label,cell(s['accuracy'],2),cell(s['task_loss'],3),f'{s["adapter_entries"]:,}']
        lines.append(' & '.join(cells)+r' \\')
    tail_table = r'''\begin{table}[t]
\centering
\caption{RTE adaptation in leading, middle and tail directions. Accuracy and cross-entropy (lower is better) are fixed-endpoint inner-selection results, mean$\pm$sample SD over three paired seeds. Every condition trains the same 592,130-parameter classifier; adapter counts exclude it.}
\label{tab:tailcapacity}
\begin{tabular}{lccc}
\toprule
Trainable backbone update & Accuracy (\%) & Task loss (nat) & Adapter params\\
\midrule
'''+'\n'.join(lines)+r'''
\bottomrule
\end{tabular}
\end{table}'''
    lines = []
    from analyze_final_evidence import BAND_ARMS, LABELS
    for arm in BAND_ARMS:
        s = arms['rte/'+arm]
        off = cell(s['within_band_offdiagonal_percent'],2) if s['within_band_offdiagonal_percent']['n'] else '---'
        rho = cell(s['rho'],4) if s['rho']['n'] else '$0$'
        lines.append(' & '.join([LABELS[arm],cell(s['accuracy'],2),cell(s['task_loss'],3),off,rho])+r' \\')
    summary_table = r'''\begin{table}[t]
\centering
\caption{Complete RTE band comparison, mean$\pm$sample SD, three seeds per row. Accuracy and loss use the inner-selection split. Off-diagonal energy is a percentage within the selected band, computed directly from the compact trained cores; diagonal residuals are float32 roundoff. The head-only backbone update is zero, so its energy fraction is undefined. All nonzero updates have structural off-band zero.}
\label{tab:bandcomplete}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccc}
\toprule
Condition & Accuracy (\%) & Task loss (nat) & Within-band off-diag. (\%) & $\rho_F$\\
\midrule
'''+'\n'.join(lines)+r'''
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}'''
    def interval(s, digits):
        lo,hi = s['nominal_t95']
        return f'$[{lo:.{digits}f},{hi:.{digits}f}]$'
    lines = []
    for key, s in final['band']['contrasts'].items():
        _,label = key.split('/',1)
        for arm in BAND_ARMS:
            label=label.replace(arm,LABELS[arm])
        label=label.replace(' minus ',' -- ').replace('rotation gain: ','Rotation gain: ').replace('ROT64','rotation').replace('DIAG','diagonal')
        lines.append(' & '.join([label,cell(s['accuracy_pp'],2),interval(s['accuracy_pp'],2),cell(s['task_loss'],3),interval(s['task_loss'],3)])+r' \\')
    contrasts_table = r'''\begin{table}[t]
\centering
\caption{All registered paired RTE contrasts at the fixed endpoint on inner-selection examples. Entries are mean$\pm$sample SD of the three seed differences and nominal 95\% paired $t$ intervals (two degrees of freedom). Rotation-gain differences compare rotation-minus-diagonal changes between bands. Intervals are descriptive and unadjusted for multiple comparisons.}
\label{tab:bandpaired}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{lcccc}
\toprule
Contrast & Accuracy (pp) & Interval & Task loss (nat) & Interval\\
\midrule
'''+'\n'.join(lines)+r'''
\bottomrule
\end{tabular}
\end{adjustbox}
\end{table}'''
    out = dict(scope='Behavior-first view of practical held-aside confirmations and all 21 strict-band/head inner-selection confirmations. Two pilots remain separate. No new model evaluation or training.',
               seed_order=list(SEEDS), practical_held_aside=summaries,
               strict_band_snapshot=[{k:r[k] for k in ('task','condition','seed','purpose','run_id','fixed_steps','fixed_accuracy','fixed_task_loss','trainable_adapter_params','validation_sha256')} for r in band],
               source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (focused_path,held_path,band_path,coverage_path)})
    return out, {'STORY MAIN RESULTS':main_table,'STORY TAIL RESULTS':tail_table,
                 'STORY BAND SUMMARY':summary_table,'STORY BAND CONTRASTS':contrasts_table,
                 'STORY BAND SNAPSHOT':band_table}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write',action='store_true');mode.add_argument('--check',action='store_true')
    args=parser.parse_args()
    out,regions=analyze()
    tex_path=ROOT/'neurips_2026.tex'; tex=tex_path.read_text()
    for name,body in regions.items():
        begin=f'% BEGIN GENERATED {name}';end=f'% END GENERATED {name}'
        assert tex.count(begin)==tex.count(end)==1,name
        replacement=begin+'\n% Generated by scripts/build_tail_story_tables.py.\n'+body+'\n'+end
        updated=re.sub(re.escape(begin)+r'.*?'+re.escape(end),lambda m:replacement,tex,flags=re.S)
        if args.check: assert updated==tex, 'Stale '+name
        tex=updated
    path=ROOT/'data/tail_story_analysis.json';serialized=json.dumps(out,indent=2,sort_keys=True)+'\n'
    if args.write: tex_path.write_text(tex);path.write_text(serialized)
    if args.check: assert path.read_text()==serialized, 'Stale story analysis'
    print('PASS: all 18 practical endpoints joined; held-aside behavior and geometry reproduced; all available band outcomes kept separate from pilots.')


if __name__=='__main__':
    main()
