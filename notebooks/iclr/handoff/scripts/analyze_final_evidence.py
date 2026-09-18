"""Audit the final export and regenerate inline learned/band tables and figures.

This checks exported evidence, not the server's checkpoint reload procedure.
Pilots never enter confirmation summaries; missing comparisons remain pending.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
from statistics import mean, stdev

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/final_evidence_20260916"
SEEDS = (17, 42, 123)
TASKS = ("rte", "mrpc")
ARMS = ("P1_UNREG", "P1_MIX", "P1_NORM")
BLOCKS = ("LL", "LT", "TL", "TT")
VIEWS = ("total", "learned_since_insertion", "initial")
BANDS = ("LEAD", "MID", "TAIL")
BAND_ARMS = ("P1_HEAD_BASE",) + tuple(f"BAND_{b}_{f}" for b in BANDS for f in ("DIAG", "ROT64"))
LABELS = {"P1_HEAD_BASE": "Head only", **{f"BAND_{b}_{f}": f"{label} / {flex}" for b, label in zip(BANDS, ("Leading", "Middle", "Tail")) for f, flex in (("DIAG", "diagonal"), ("ROT64", "rotation"))}}


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def summarize(values):
    values = list(map(float, values))
    assert all(np.isfinite(values))
    n = len(values)
    out = dict(n=n, values=values, mean=mean(values) if n else None,
               sample_sd=stdev(values) if n > 1 else None, nominal_t95=None)
    if n == 3:
        half = float(t.ppf(.975, 2)) * out["sample_sd"] / np.sqrt(3)
        out["nominal_t95"] = [out["mean"]-half, out["mean"]+half]
    return out


def analyze():
    paths = sorted(p for p in DATA.rglob("*") if p.is_file())
    focused_path = ROOT / "data/focused_norm/runs.csv"
    invalid = set()
    for name in ("calibration", "confirmation"):
        invalid.update(json.loads((DATA / f"INVALIDATED_RUNS_{name}.json").read_text())["invalidated_run_ids"])
    assert len(invalid) == 39
    original = read_csv(focused_path)
    assert not invalid.intersection(r["run_id"] for r in original)
    original = {r["run_id"]: r for r in original if r["stage"] == "confirmation" and r["condition"] in ARMS}
    rows = read_csv(DATA / "focused_confirmation_learned.csv")
    assert len(rows) == len(original) == 18
    assert {r["run_id"] for r in rows} == set(original)
    assert {(r["task"], r["condition"], int(r["seed"])) for r in rows} == set(itertools.product(TASKS, ARMS, SEEDS))
    modules = defaultdict(list)
    for line in (DATA / "focused_confirmation_modules.jsonl").read_text().splitlines():
        m = json.loads(line)
        modules[m["run_id"]].append(m)
    assert set(modules) == set(original)
    checks = []
    for r in rows:
        old = original[r["run_id"]]
        assert r["validation_sha256"] == old["validation_sha256"]
        assert r["run_id"] in Path(r["fixed_checkpoint"]).parts
        assert Path(r["fixed_checkpoint"]).name == f'step_{int(old["fixed_step"]):08d}'
        assert re.fullmatch(r"[a-f0-9]{64}", r["validation_sha256"])
        for key in ("task", "condition", "seed"):
            assert r[key] == old[key]
        ms = modules[r["run_id"]]
        old_norms = json.loads(old["per_module_relative_frobenius"])
        assert {m["module"] for m in ms} == set(old_norms)
        assert all(abs(m["total"]["relative_frobenius"]-old_norms[m["module"]]) < 1e-12 for m in ms)
        assert len(ms) == len({m["module"] for m in ms}) == 48
        assert all((m["task"], m["condition"], int(m["seed"])) == (r["task"], r["condition"], int(r["seed"])) for m in ms)
        pretrained = [m["initial"]["energy"] / m["initial"]["relative_frobenius"]**2 for m in ms]
        for view in VIEWS:
            energies = [m[view]["energy"] for m in ms]
            assert min(energies) > 0
            for m, pre in zip(ms, pretrained):
                v = m[view]
                assert np.isclose(v["energy"], v["frobenius"]**2, rtol=1e-12, atol=0)
                assert np.isclose(v["energy"] / v["relative_frobenius"]**2, pre, rtol=1e-12, atol=0)
                assert abs(sum(v["fractions"].values())-1) < 1e-6
                for b in BLOCKS:
                    assert abs(v["block_energy"][b]/v["energy"]-v["fractions"][b]) < 1e-12
            rho = np.sqrt(sum(energies)/sum(pretrained))
            checks.append(abs(rho-float(r[f"{view}_pooled_rho"])))
            for b in BLOCKS:
                pooled = sum(m[view]["block_energy"][b] for m in ms)/sum(energies)
                eqm = mean(m[view]["fractions"][b] for m in ms)
                checks.extend((abs(pooled-float(r[f"{view}_pooled_{b}"])), abs(eqm-float(r[f"{view}_eqm_{b}"]))))
                if view == "total":
                    assert abs(pooled-float(old[f"pooled_{b}"])) < 1e-10
                    assert abs(eqm-float(old[f"eqm_{b}"])) < 1e-10
    assert max(checks) < 1e-12
    result = dict(source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[focused_path]},
                  audit_scope="Export reaggregation; server checkpoint validations not rerun.",
                  invalidated_runs_excluded=len(invalid), focused_runs=len(rows), module_rows=sum(map(len, modules.values())),
                  maximum_reaggregation_error=max(checks), focused={})
    for task, arm in itertools.product(TASKS, ARMS):
        rs = sorted((r for r in rows if r["task"] == task and r["condition"] == arm), key=lambda r: int(r["seed"]))
        values = {}
        for view in VIEWS:
            values[view] = {f"{weight}_{block}": summarize(100*(float(r[f"{view}_{weight}_LT"])+float(r[f"{view}_{weight}_TL"])) if block == "cross" else 100*float(r[f"{view}_{weight}_{block}"]) for r in rs) for weight in ("pooled", "eqm") for block in ("LL", "cross", "TT")}
            values[view]["rho"] = summarize(float(r[f"{view}_pooled_rho"]) for r in rs)
            # Every rank retained; top-three summaries are descriptive, not a selected hypothesis.
            curves = []
            for r in rs:
                energy = sorted((m[view]["energy"] for m in modules[r["run_id"]]), reverse=True)
                curves.append((np.cumsum(energy)/sum(energy)).tolist())
            values[view]["sorted_cumulative_module_energy"] = curves
            values[view]["top_three_share_percent"] = summarize(100*c[2] for c in curves)
        values["max_eqm_initialization_change_pp"] = max(abs(float(r[f"total_eqm_{b}"])-float(r[f"learned_since_insertion_eqm_{b}"]))*100 for r in rs for b in BLOCKS)
        result["focused"][task+"/"+arm] = values
    probe = json.loads((DATA / "probe_reference_original_backbone.json").read_text())
    assert len(probe["sample_ids"]) == len(probe["masked_token_count"]) == len(probe["per_example_loss_sum"]) == 256
    ce = sum(probe["per_example_loss_sum"])/sum(probe["masked_token_count"])
    assert abs(ce-probe["masked_token_cross_entropy"]) < 1e-12
    result["reference_probe"] = dict(ce=ce, examples=256, masked_tokens=sum(probe["masked_token_count"]))
    parity = json.loads((DATA / "tokenization_parity.json").read_text())["result"]
    assert parity["checked"] == 181 and parity["mismatches"] == 0
    result["tokenizer_parity"] = parity
    band = read_csv(DATA / "band_results.csv")
    coverage = json.loads((DATA / "band_coverage.json").read_text())
    assert len({r["run_id"] for r in band}) == len(band)
    assert len({r["entry_id"] for r in band}) == len(band)
    assert all(r["purpose"] in ("confirmation", "timing_pilot") for r in band)
    assert not invalid.intersection(r["run_id"] for r in band)
    assert {r["entry_id"] for r in band} == {k for k,v in coverage["entries"].items() if v["status"] == "completed"}
    manifests = [json.loads(p.read_text()) for p in sorted((DATA / "manifests").glob("*.json"))]
    recipes = {(m["task"], m["condition"]):m for m in manifests}
    assert len(recipes) == len(manifests)
    assert set(itertools.product(TASKS, ARMS)).issubset(recipes)
    for m in manifests:
        settings = m["train_settings"]
        assert m["batch_size"] * settings["accumulation_steps"] == 32
        assert settings["max_steps"] == {"rte":5670,"mrpc":2760}[m["task"]]
        assert settings["non_head_lr"] == .01
        assert settings["head_lr"] == {"rte":.0005,"mrpc":.001}[m["task"]]
        assert settings["weight_decay"] == settings["warmup_steps"] == 0
        assert settings["precision"] == "float32" and settings["max_gradient_norm"] == 1
        assert m["primary_endpoint"] == "fixed_optimizer_step"
        assert m["secondary_endpoint"] == "best_inner_accuracy_earliest_tie_excluding_step0"
    result["manifest_recipes"] = len(manifests)
    from check_band_provenance import check as check_band_provenance
    source_audit = check_band_provenance()
    result['band_source_audit'] = dict(source_revisions=len(source_audit['campaign_trees']),
        verified_source_files=len(source_audit['source_files']), frozen_confirmations=21)
    for r in band:
        recipe = recipes.get((r["task"],r["condition"]))
        assert recipe, "Missing executed recipe manifest"
        assert source_audit['campaign_trees'][r['source_revision']] == source_audit['campaign_trees'][recipe['source_revision']]
        assert int(r["fixed_steps"]) == recipe["train_settings"]["max_steps"]
        if r["condition"].startswith("BAND_"):
            cfg = recipe["band_config"]
            expected_rotation = 0 if r["condition"].endswith("DIAG") else 64
            assert cfg["band_size"] == 256 and cfg["rotation_size"] == expected_rotation
            assert cfg["band_start"] == {"LEAD":0,"MID":256,"TAIL":512}[r["condition"].split('_')[1]]
            assert int(r["trainable_adapter_params"]) == 48*(256+2*expected_rotation**2)
            assert int(r["trainable_head_params"]) == 592130
        entry = coverage["entries"][r["entry_id"]]
        assert any(a["run_id"] == r["run_id"] and a["status"] == "completed" for a in entry["attempts"])
        assert re.fullmatch(r"[a-f0-9]{64}", r["validation_sha256"])
        assert re.fullmatch(r"[a-f0-9]{40}", r["source_revision"])
        assert int(r["seed"]) in (SEEDS if r["purpose"] == "confirmation" else (31415,))
        assert ("/seed_" in r["entry_id"]) == (r["purpose"] == "confirmation")
        assert r["condition"] in BAND_ARMS
        if r["condition"].startswith("BAND_"):
            assert r["off_band_energy"] == "structural_zero_by_construction"
            # Preserve signed finite-precision diagonal residuals; do not misapply the off-band test tolerance.
            if r["condition"].endswith("DIAG"):
                assert abs(float(r["within_band_offdiagonal_fraction"])) < 1e-6
    confirmation = [r for r in band if r["purpose"] == "confirmation"]
    result["band"] = dict(snapshot_utc=coverage["generated_utc"], protocol_sha256=coverage["protocol_sha256"],
        coverage_counts=dict(Counter(v["status"] for k,v in coverage["entries"].items() if "/seed_" in k)),
        completed_confirmations=len(confirmation), completed_pilots=len(band)-len(confirmation), arms={}, contrasts={})
    contrasts = [(a+" minus head", {a:1, "P1_HEAD_BASE":-1}) for a in BAND_ARMS[1:]]
    contrasts += [(b+" rotation minus diagonal", {f"BAND_{b}_ROT64":1, f"BAND_{b}_DIAG":-1}) for b in BANDS]
    for lower, upper in itertools.combinations(BANDS, 2):
        for flex in ("DIAG", "ROT64"):
            contrasts.append((f"{upper} minus {lower} / {flex}", {f"BAND_{upper}_{flex}":1, f"BAND_{lower}_{flex}":-1}))
        contrasts.append((f"rotation gain: {upper} minus {lower}", {f"BAND_{upper}_ROT64":1, f"BAND_{upper}_DIAG":-1, f"BAND_{lower}_ROT64":-1, f"BAND_{lower}_DIAG":1}))
    for task in sorted({k.split('/')[0] for k in coverage["entries"]}):
        lookup = {(r["condition"], int(r["seed"])):r for r in confirmation if r["task"] == task}
        assert len(lookup) == sum(r["task"] == task for r in confirmation)
        for arm in BAND_ARMS:
            rs = [lookup[arm,s] for s in SEEDS if (arm,s) in lookup]
            counts = {int(r["trainable_adapter_params"]) for r in rs}
            heads = {int(r["trainable_head_params"]) for r in rs}
            assert len(counts) <= 1 and len(heads) <= 1
            result["band"]["arms"][task+"/"+arm] = dict(n=len(rs), adapter_entries=next(iter(counts), None), head_entries=next(iter(heads), None),
                accuracy=summarize(100*float(r["fixed_accuracy"]) for r in rs),
                task_loss=summarize(float(r['fixed_task_loss']) for r in rs),
                within_band_offdiagonal_percent=summarize(100*float(r['within_band_offdiagonal_fraction']) for r in rs if r['within_band_offdiagonal_fraction']),
                rho=summarize(float(r["pooled_relative_frobenius"]) for r in rs if r["pooled_relative_frobenius"]))
        for label, weights in contrasts:
            seeds = [s for s in SEEDS if all((a,s) in lookup for a in weights)]
            result["band"]["contrasts"][task+"/"+label] = dict(seeds=seeds,
                accuracy_pp=summarize(sum(w*100*float(lookup[a,s]["fixed_accuracy"]) for a,w in weights.items()) for s in seeds),
                task_loss=summarize(sum(w*float(lookup[a,s]['fixed_task_loss']) for a,w in weights.items()) for s in seeds))
    return result, rows, band


def cell(s, decimals=1):
    if not s["n"]:
        return "pending"
    value = f'{s["mean"]:.{decimals}f}'
    if s["sample_sd"] is not None:
        value += rf' \pm {s["sample_sd"]:.{decimals}f}'
    return "$"+value+"$"


def tables(result, rows, band):
    learned, seeds, bands, endpoints, contrasts = [], [], [], [], []
    for task in TASKS:
        for arm in ARMS:
            for view, label in (("total", "Total"), ("learned_since_insertion", "Learned")):
                stats = result["focused"][task+"/"+arm][view]
                learned.append(" & ".join([task.upper(), arm[3:], label]+[cell(stats[f"{w}_{b}"]) for w in ("pooled", "eqm") for b in ("LL", "cross", "TT")])+r" \\")
            for r in sorted((r for r in rows if r["task"] == task and r["condition"] == arm), key=lambda r:int(r["seed"])):
                seeds.append(" & ".join([task.upper(), arm[3:], r["seed"]]+[f'${100*float(r[f"learned_since_insertion_{w}_{b}"]):.2f}$' for w in ("pooled", "eqm") for b in BLOCKS])+r" \\")
        learned.append(r"\midrule")
        seeds.append(r"\midrule")
    for key, stats in result["band"]["arms"].items():
        task, arm = key.split('/')
        contrast = result["band"]["contrasts"].get(task+"/"+arm+" minus head")
        bands.append(" & ".join([task.upper(), LABELS[arm], f'{stats["n"]}/3', str(stats["adapter_entries"]) if stats["adapter_entries"] is not None else "pending", cell(stats["accuracy"],2), cell(contrast["accuracy_pp"],2) if contrast else "---", cell(stats["rho"],4)])+r" \\")
    for purpose in ("confirmation", "timing_pilot"):
        for r in sorted((r for r in band if r["purpose"] == purpose), key=lambda r:(r["task"], BAND_ARMS.index(r["condition"]), int(r["seed"]))):
            raw = r["within_band_offdiagonal_fraction"]
            off = f'{float(raw):.3g}' if raw else "---"
            rho = f'{float(r["pooled_relative_frobenius"]):.4f}' if r['pooled_relative_frobenius'] else '---'
            endpoints.append(" & ".join([r["task"].upper(), LABELS[r["condition"]], "Confirm." if purpose == "confirmation" else "Pilot", r["seed"], f'{100*float(r["fixed_accuracy"]):.2f}', r["best_step"], f'{100*float(r["best_accuracy"]):.2f}', rho, off])+r" \\")
    for key, value in result["band"]["contrasts"].items():
        task, label = key.split('/',1)
        contrasts.append(" & ".join([task.upper(), label.replace('_', r'\_'), str(len(value["seeds"]))+"/3", cell(value["accuracy_pp"],2)])+r" \\")
    return {"LEARNED SUMMARY":"\n".join(learned[:-1]), "LEARNED SEEDS":"\n".join(seeds[:-1]), "BAND SUMMARY":"\n".join(bands), "BAND ENDPOINTS":"\n".join(endpoints), "BAND CONTRASTS":"\n".join(contrasts)}


def figures(result, band):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9, "text.color":"black", "axes.labelcolor":"black"})
    fig, axes = plt.subplots(2, 2, figsize=(6.7,5.3), sharex=True, sharey=True)
    for i, task in enumerate(TASKS):
        for j, view in enumerate(VIEWS[:2]):
            ax = axes[i,j]
            for arm, style, color in zip(ARMS, (":", "-", "--"), (".55", "black", ".3")):
                curves = result["focused"][task+"/"+arm][view]["sorted_cumulative_module_energy"]
                for s, curve in enumerate(curves):
                    ax.plot(range(1,49),100*np.array(curve),style,color=color,lw=1,label=arm[3:] if s==0 else None)
            ax.set_title(task.upper()+" / "+("total" if j==0 else "since insertion"))
            ax.set_ylim(0,101); ax.set_xlim(1,48)
            ax.set_xticks([1,12,24,36,48]); ax.grid(alpha=.15)
            if j==0: ax.set_ylabel("Cumulative energy (%)")
            if i==1: ax.set_xlabel("Modules, largest energy first")
    axes[0,0].legend(fontsize=8,loc="lower right")
    fig.tight_layout(); fig.savefig(ROOT/"figures/focused_module_energy.pdf"); plt.close(fig)
    confirmation = [r for r in band if r["purpose"] == "confirmation"]
    lookup = {(r['condition'], int(r['seed'])): r for r in confirmation if r['task'] == 'rte'}
    fig, axes = plt.subplots(2, 2, figsize=(6.7, 5.4))
    for row, (column, multiplier, ylabel) in enumerate((
            ('fixed_accuracy', 100, 'Accuracy (%)'), ('fixed_task_loss', 1, 'Task loss (nat)'))):
        ax, cx = axes[row]
        for b, loc in enumerate(BANDS):
            differences = []
            for si, seed in enumerate(SEEDS):
                points = []
                for flex, marker, offset in (('DIAG', 'o', -.04), ('ROT64', 's', .04)):
                    r = lookup[f'BAND_{loc}_{flex}', seed]
                    x = b + (si-1)*.17 + offset
                    y = multiplier*float(r[column])
                    ax.plot(x, y, marker, color='black', fillstyle='none' if flex=='ROT64' else 'full', ms=4,
                            label=('Diagonal' if flex=='DIAG' else 'Rotation') if b==0 and si==0 else None)
                    points.append((x,y))
                ax.plot(*zip(*points), color='.55', lw=.6)
                delta = points[1][1]-points[0][1]
                differences.append(delta)
                cx.plot(b+(si-1)*.1, delta, ('o','s','^')[si], color='black', ms=4, label=str(seed) if b==0 else None)
            cx.errorbar(b, mean(differences), yerr=stdev(differences), fmt='_', color='black', capsize=3)
        heads = [multiplier*float(lookup['P1_HEAD_BASE',seed][column]) for seed in SEEDS]
        for value in heads:
            ax.axhline(value,color='.65',lw=.6,ls=':')
        ax.axhline(mean(heads),color='.4',lw=1,label='Head mean')
        cx.axhline(0,color='.65',lw=.7,ls=':')
        for axis in (ax,cx):
            axis.set_xticks(range(3),('Leading','Middle','Tail'))
            axis.set_xlim(-.45,2.45)
        ax.set_ylabel(ylabel)
        cx.set_ylabel('Paired difference ('+('pp' if row==0 else 'nat')+')')
    axes[0,0].set_title('RTE inner-selection endpoints')
    axes[0,1].set_title('Rotation minus diagonal')
    axes[0,0].legend(fontsize=7,loc='lower left')
    axes[1,1].legend(title='Seed',fontsize=7,loc='upper right')
    fig.tight_layout()
    fig.savefig(ROOT/'figures/band_confirmation.pdf', metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true");parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    result,rows,band=analyze()
    output=ROOT/"data/final_evidence_analysis.json"
    serialized=json.dumps(result,indent=2,sort_keys=True)+"\n"
    manuscript=ROOT/"neurips_2026.tex"; tex=manuscript.read_text()
    for name,body in tables(result,rows,band).items():
        begin=f"% BEGIN GENERATED {name}"; end=f"% END GENERATED {name}"
        if name.startswith("BAND ") and begin not in tex and end not in tex:
            # The incomplete block remains in the audited JSON, not numerical paper tables.
            continue
        assert tex.count(begin)==tex.count(end)==1, name
        replacement=begin+"\n% Generated by scripts/analyze_final_evidence.py.\n"+body+"\n"+end
        pattern=re.escape(begin)+r".*?"+re.escape(end)
        updated=re.sub(pattern,lambda match:replacement,tex,flags=re.S)
        if args.check: assert tex==updated, "Stale table: "+name
        tex=updated
    if args.write:
        output.write_text(serialized);manuscript.write_text(tex);figures(result,band)
    if args.check: assert output.read_text()==serialized, "Stale final evidence analysis"
    print(f'PASS: {result["focused_runs"]} focused runs, {result["module_rows"]} module records, {result["invalidated_runs_excluded"]} invalidated runs excluded; maximum reaggregation error {result["maximum_reaggregation_error"]:.3g}.')
    print('Band:',result["band"]["completed_confirmations"],'confirmations;',result["band"]["completed_pilots"],'pilots;',result["band"]["coverage_counts"])


if __name__=="__main__":
    main()
