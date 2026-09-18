"""Reproduce review-v2 diagnostics and the separate locked-split evaluation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean

from analyze_final_evidence import summarize, cell

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/final_evidence_20260916"
LOCKED = ROOT / "data/locked_evaluation_20260916"
SEEDS = (17,42,123)
ARMS = ("P1_UNREG","P1_MIX","P1_NORM")
TASKS = ("rte","mrpc")


def csv_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def analyze():
    records = [json.loads(line) for line in (DATA/"focused_confirmation_modules.jsonl").read_text().splitlines()]
    by = defaultdict(dict)
    for m in records:
        by[m["task"],m["condition"],m["seed"]][m["module"]] = m
    assert len(by)==18 and all(len(ms)==48 for ms in by.values())
    out = dict(interpretation="Post-review descriptive analyses; thresholds were suggested after outcomes. No new training or retuning.", identity={}, sensitivity={}, held_aside={}, paired_held_aside={})
    for task in TASKS:
        for arm in ARMS:
            for view in ("total","learned_since_insertion"):
                eta, eqm, residual = [], [], []
                for seed in SEEDS:
                    ms = by[task,arm,seed].values()
                    for m in ms:
                        v=m[view]
                        assert abs(v["identity_residual_energy"]-v["block_energy"]["LL"]*(1-v["identity_alignment"])) < 1e-10
                    ll=sum(m[view]["block_energy"]["LL"] for m in ms)
                    res=sum(m[view]["identity_residual_energy"] for m in ms)
                    eta.append(100*(1-res/ll))
                    eqm.append(100*mean(m[view]["identity_alignment"] for m in ms))
                    residual.append(100*res/sum(m[view]["energy"] for m in ms))
                out["identity"][f"{task}/{arm}/{view}"] = dict(leading_energy_weighted_eta_percent=summarize(eta),equal_module_eta_percent=summarize(eqm),nonidentity_LL_fraction_of_delta_percent=summarize(residual))
        for threshold in (1e-6,1e-3):
            for view in ("total","learned_since_insertion"):
                counts, mix, norm = [], [], []
                for seed in SEEDS:
                    a,b=by[task,"P1_MIX",seed],by[task,"P1_NORM",seed]
                    selected=[k for k in a if min(a[k]["total"]["relative_frobenius"],b[k]["total"]["relative_frobenius"])>=threshold]
                    assert selected
                    counts.append(len(selected))
                    mix.append(100*mean(a[k][view]["p_cross"] for k in selected))
                    norm.append(100*mean(b[k][view]["p_cross"] for k in selected))
                out["sensitivity"][f"{task}/{threshold:g}/{view}"]=dict(seed_order=list(SEEDS),retained_modules=counts,mix=summarize(mix),norm=summarize(norm),norm_minus_mix=summarize(b-a for a,b in zip(mix,norm)))
    frozen = {r["run_id"]:r for r in json.loads((ROOT/"data/locked_evaluation_request_20260916.json").read_text())["population"]}
    old = {r["run_id"]:r for r in csv_rows(ROOT/"data/focused_norm/runs.csv") if r["stage"]=="confirmation"}
    results=csv_rows(LOCKED/"held_aside_results.csv")
    assert len(results)==len({r["run_id"] for r in results})==18 and set(frozen)=={r["run_id"] for r in results}
    manifest=json.loads((LOCKED/"evaluation_manifest.json").read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert manifest["results_csv_sha256"]==sha(LOCKED/"held_aside_results.csv")
    assert manifest["request_json_sha256"]==sha(ROOT/"data/locked_evaluation_request_20260916.json")
    assert manifest["request_markdown_sha256"]==sha(ROOT/"LOCKED_EVALUATION_REQUEST_20260916.md")
    assert manifest["population_source_sha256"]==sha(DATA/"focused_confirmation_learned.csv")
    assert json.loads((ROOT/"data/locked_evaluation_request_20260916.json").read_text())["source_sha256"]==manifest["population_source_sha256"]
    assert len(manifest["loading_checks"])==18 and {r["run_id"] for r in manifest["loading_checks"]}==set(frozen)
    for check in manifest["loading_checks"]:
        assert check["accuracy_reproduced"] and check["loss_reproduced"] and check.get("f1_reproduced",True)
        assert check["recomputed_accuracy"]==float(old[check["run_id"]]["fixed_accuracy"])
    examples={}
    for p in (LOCKED/"per_example").glob('*.json'):
        d=json.loads(p.read_text());assert d["run_id"] not in examples;examples[d["run_id"]]=d
    assert set(examples)==set(frozen)
    same_split={}
    for r in results:
        run=r["run_id"];d=examples[run];prior=old[run]
        assert r["validation_report_sha256"]==frozen[run]["validation_sha256"]
        assert r["checkpoint_identifier"]==Path(frozen[run]["fixed_checkpoint"]).name
        assert int(r["fixed_optimizer_step"])==int(prior["fixed_step"])
        assert float(r["inner_selection_accuracy"])==float(prior["fixed_accuracy"])
        assert d["checkpoint_sha256"]==r["checkpoint_sha256"] and d["split"]=="locked_evaluation"
        assert d["split_fingerprint"]==r["locked_split_fingerprint"]
        for key in ("task","condition","seed"):
            assert str(d[key])==r[key]==frozen[run][key]
        n=int(r["example_count"]);assert n=={"rte":277,"mrpc":408}[r["task"]]
        assert len(d["labels"])==len(d["predictions"])==len(d["sample_ids"])==len(set(d["sample_ids"]))==len(d["per_example_loss"])==n
        assert set(d["labels"]+d["predictions"]) <= {0,1}
        signature=(d["sample_ids"],d["labels"],d["split_fingerprint"])
        if r["task"] in same_split: assert signature==same_split[r["task"]]
        same_split[r["task"]]=signature
        accuracy=sum(y==p for y,p in zip(d["labels"],d["predictions"]))/n
        assert abs(accuracy-float(r["held_aside_accuracy"]))<1e-12
        assert abs(mean(d["per_example_loss"])-float(r["held_aside_loss"]))<1e-5
        if r["task"]=="mrpc":
            tp=sum(y==p==1 for y,p in zip(d["labels"],d["predictions"]))
            f1=2*tp/(sum(d["labels"])+sum(d["predictions"]))
            assert abs(f1-float(r["held_aside_f1"]))<1e-12
    for task in TASKS:
        lookup={(r["condition"],int(r["seed"])):r for r in results if r["task"]==task}
        metrics=("accuracy","f1") if task=="mrpc" else ("accuracy",)
        for arm in ARMS:
            out["held_aside"][f"{task}/{arm}"]={metric:summarize(100*float(lookup[arm,s]["held_aside_"+metric]) for s in SEEDS) for metric in metrics}
        for a,b in (("P1_MIX","P1_NORM"),("P1_MIX","P1_UNREG"),("P1_NORM","P1_UNREG")):
            out["paired_held_aside"][f"{task}/{a}_minus_{b}"]={metric:summarize(100*(float(lookup[a,s]["held_aside_"+metric])-float(lookup[b,s]["held_aside_"+metric])) for s in SEEDS) for metric in metrics}
    files=[DATA/"focused_confirmation_modules.jsonl",ROOT/"data/focused_norm/runs.csv",ROOT/"data/locked_evaluation_request_20260916.json",ROOT/"LOCKED_EVALUATION_REQUEST_20260916.md"]+sorted(p for p in LOCKED.rglob('*') if p.is_file())
    out["source_sha256"]={str(p.relative_to(ROOT)):sha(p) for p in files}
    return out,results


def tables(out,results):
    identity=[]; sensitivity=[];held=[];paired=[]
    for task in TASKS:
        for arm in ARMS:
            total=out["identity"][f"{task}/{arm}/total"];learned=out["identity"][f"{task}/{arm}/learned_since_insertion"]
            identity.append(' & '.join([task.upper(),arm[3:]]+[cell(x[k],2) for x,k in ((total,"leading_energy_weighted_eta_percent"),(learned,"leading_energy_weighted_eta_percent"),(learned,"equal_module_eta_percent"),(learned,"nonidentity_LL_fraction_of_delta_percent"))])+r' \\')
            for r in sorted((r for r in results if r["task"]==task and r["condition"]==arm),key=lambda r:int(r["seed"])):
                held.append(' & '.join([task.upper(),arm[3:],r["seed"],f'{100*float(r["held_aside_accuracy"]):.2f}',f'{100*float(r["held_aside_f1"]):.2f}' if r["held_aside_f1"] else '---',f'{float(r["held_aside_loss"]):.3f}'])+r' \\')
        for threshold in (1e-6,1e-3):
            for view,label in (("total","Total"),("learned_since_insertion","Learned")):
                s=out["sensitivity"][f"{task}/{threshold:g}/{view}"]
                sensitivity.append(' & '.join([task.upper(),f'${threshold:g}$',label,','.join(map(str,s["retained_modules"]))]+[cell(s[k],2) for k in ("mix","norm","norm_minus_mix")])+r' \\')
        for key,metrics in out["paired_held_aside"].items():
            if not key.startswith(task+'/'):continue
            label=key.split('/')[1].replace('P1_','').replace('_minus_','--')
            for metric,s in metrics.items():
                lo,hi=s["nominal_t95"]
                paired.append(' & '.join([task.upper(),label,metric]+[f'${v:.2f}$' for v in s["values"]]+[cell(s,2),f'$[{lo:.2f},{hi:.2f}]$'])+r' \\')
    return {"REVIEW IDENTITY":identity,"REVIEW SENSITIVITY":sensitivity,"HELD ASIDE SEEDS":held,"HELD ASIDE PAIRED":paired}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');p.add_argument('--check',action='store_true');args=p.parse_args()
    out,results=analyze();serialized=json.dumps(out,indent=2,sort_keys=True)+'\n';path=ROOT/'data/review_v2_analysis.json'
    texpath=ROOT/'neurips_2026.tex';tex=texpath.read_text()
    for name,lines in tables(out,results).items():
        begin=f'% BEGIN GENERATED {name}';end=f'% END GENERATED {name}'
        assert tex.count(begin)==tex.count(end)==1,name
        body=begin+'\n% Generated by scripts/analyze_review_v2.py.\n'+'\n'.join(lines)+'\n'+end
        new=re.sub(re.escape(begin)+r'.*?'+re.escape(end),lambda m:body,tex,flags=re.S)
        if args.check:assert tex==new,'Stale table '+name
        tex=new
    if args.write:path.write_text(serialized);texpath.write_text(tex)
    if args.check:assert path.read_text()==serialized,'Stale analysis'
    print('PASS: identity residuals; paired module subsets; all 18 frozen held-aside predictions, labels, metrics, checkpoint/validation joins and loading checks.')
    for key,value in out['held_aside'].items():print(key,{k:(round(s['mean'],3),round(s['sample_sd'],3)) for k,s in value.items()})


if __name__=='__main__':main()
