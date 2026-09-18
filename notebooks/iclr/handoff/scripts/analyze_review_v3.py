"""Descriptive round-3 analyses: loss contrasts, frontier sensitivity and power assumptions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import nct, norm, t

from analyze_final_evidence import summarize, cell
from analyze_review_v2 import analyze as verify_locked

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(17,42,123)
PAIRS=(("P1_MIX","P1_NORM"),("P1_MIX","P1_UNREG"),("P1_NORM","P1_UNREG"))


def analyze():
    previous,locked=verify_locked()
    rows=list(csv.DictReader((ROOT/'data/focused_norm/runs.csv').open()))
    out=dict(scope='Post hoc descriptive analysis; no dose, checkpoint, seed or held-aside prediction changed.',frontier={},loss={},loss_paired={},power={})
    critical=float(t.ppf(.975,2))
    power=lambda d:float(nct.sf(critical,2,d*np.sqrt(3))+nct.cdf(-critical,2,d*np.sqrt(3)))
    dz=float(brentq(lambda d:power(d)-.8,0,4))
    # Independent normal/chi-square mixture check; V/2 ~ Exponential(1) for df=2.
    integrated=quad(lambda v:(norm.sf(critical*np.sqrt(v)-dz*np.sqrt(3))+norm.cdf(-critical*np.sqrt(v)-dz*np.sqrt(3)))*np.exp(-v),0,np.inf)[0]
    assert abs(integrated-.8)<1e-9 and abs(power(0)-.05)<1e-9
    out['power_assumptions']=dict(n=3,df=2,alpha_two_sided=.05,target_power=.8,standardized_effect=dz,interpretation='Hypothetical Gaussian paired t-test; observed paired SD used as uncertain variance scenario, not achieved power or an exclusion bound.')
    for task in ('rte','mrpc'):
        cal=sorted([r for r in rows if r['task']==task and r['stage']=='calibration' and r['condition']=='P1_NORM'],key=lambda r:-float(r['rho_pooled']))
        assert len(cal)==7 and {int(r['seed']) for r in cal}=={31415}
        x=np.array([float(r['rho_pooled']) for r in cal]);y=np.array([100*(float(r['pooled_LT'])+float(r['pooled_TL'])) for r in cal])
        slope,intercept=np.polyfit(np.log(x),y,1)
        errors=[float(r['rho_pooled'])/float(next(m['rho_pooled'] for m in rows if m['stage']=='confirmation' and m['task']==task and m['condition']=='P1_MIX' and m['seed']==r['seed']))-1 for r in rows if r['stage']=='confirmation' and r['task']==task and r['condition']=='P1_NORM']
        out['frontier'][task]=dict(n_calibration_seeds=1,n_doses=7,norm_range=[float(min(x)),float(max(x))],norm_ratio=float(max(x)/min(x)),cross_range_pp=[float(min(y)),float(max(y))],monotone_in_decreasing_norm=bool(np.all(np.diff(y)>0)),ols_slope_pp_per_log_norm=float(slope),ols_intercept=float(intercept),ols_max_abs_residual_pp=float(max(abs(y-(slope*np.log(x)+intercept)))),illustrative_8percent_shift_pp=float(abs(slope)*np.log(1.08)),signed_confirmation_norm_errors=errors,illustrative_shifts_at_achieved_errors_pp=[float(slope*np.log1p(e)) for e in errors],interpretation='Single-seed descriptive fit, not a Lipschitz/causal bound or extrapolation rule.')
        lookup={(r['condition'],int(r['seed'])):r for r in locked if r['task']==task}
        old={(r['condition'],int(r['seed'])):r for r in rows if r['task']==task and r['stage']=='confirmation'}
        for arm in ('P1_UNREG','P1_MIX','P1_NORM'):
            out['loss'][task+'/'+arm]=summarize(float(lookup[arm,s]['held_aside_loss']) for s in SEEDS)
        for a,b in PAIRS:
            key=task+'/'+a+'_minus_'+b
            loss=summarize(float(lookup[a,s]['held_aside_loss'])-float(lookup[b,s]['held_aside_loss']) for s in SEEDS)
            out['loss_paired'][key]=loss
            metrics=dict(previous['paired_held_aside'][key])
            metrics['task_NLL']=loss
            metrics['probe_CE']=summarize(float(old[a,s]['probe_masked_ce'])-float(old[b,s]['probe_masked_ce']) for s in SEEDS)
            out['power'][key]={m:dict(paired_sample_sd=v['sample_sd'],effect_80pct=dz*v['sample_sd']) for m,v in metrics.items()}
    files=[ROOT/'data/focused_norm/runs.csv',ROOT/'data/locked_evaluation_20260916/held_aside_results.csv']
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return out


def tables(out):
    loss=[];power=[];frontier=[]
    for task in ('rte','mrpc'):
        f=out['frontier'][task]
        frontier.append(f"{task.upper()} & {f['norm_ratio']:.2f} & {f['cross_range_pp'][0]:.2f}--{f['cross_range_pp'][1]:.2f} & {f['ols_slope_pp_per_log_norm']:.3f} & {f['illustrative_8percent_shift_pp']:.3f} & {f['ols_max_abs_residual_pp']:.3f}"+r' \\')
        for a,b in PAIRS:
            key=task+'/'+a+'_minus_'+b;s=out['loss_paired'][key];contrast=a[3:]+'--'+b[3:]
            loss.append(' & '.join([task.upper(),contrast,*[f'{v:+.3f}' for v in s['values']],cell(s,3),f"$[{s['nominal_t95'][0]:.3f},{s['nominal_t95'][1]:.3f}]$"])+r' \\')
            for metric,m in out['power'][key].items():
                name={'accuracy':'Accuracy (pp)','f1':'F1 (pp)','task_NLL':'Task NLL (nat)','probe_CE':'Probe CE (nat)'}[metric]
                power.append(f"{task.upper()} & {contrast} & {name} & {m['paired_sample_sd']:.3f} & {m['effect_80pct']:.3f}"+r' \\')
    return {'REVIEW V3 LOSS':loss,'REVIEW V3 POWER':power,'REVIEW V3 FRONTIER':frontier}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    out=analyze();data=json.dumps(out,indent=2,sort_keys=True)+'\n';path=ROOT/'data/review_v3_analysis.json'
    texpath=ROOT/'neurips_2026.tex';tex=texpath.read_text()
    for name,rows in tables(out).items():
        start='% BEGIN GENERATED '+name;end='% END GENERATED '+name
        assert tex.count(start)==tex.count(end)==1,name
        new=re.sub(re.escape(start)+'.*?'+re.escape(end),lambda m:start+'\n% Generated by scripts/analyze_review_v3.py.\n'+'\n'.join(rows)+'\n'+end,tex,flags=re.S)
        if args.check:assert new==tex,'Stale '+name
        tex=new
    if args.write:path.write_text(data);texpath.write_text(tex)
    if args.check:assert path.read_text()==data,'Stale round-3 analysis'
    print('PASS: original locked data joins; all task-loss contrasts; single-seed frontier fits; noncentral-t sensitivity independently checked by quadrature.')
    print('Standardized 80%-power scenario:',out['power_assumptions']['standardized_effect'])
    for task,values in out['frontier'].items():print(task,values)


if __name__=='__main__':main()
