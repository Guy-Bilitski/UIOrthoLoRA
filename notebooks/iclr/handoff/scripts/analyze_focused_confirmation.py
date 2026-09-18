"""Audit the exported focused study; never reads or rewrites server run artifacts.

Produces paired seed summaries and module norm comparisons independently of the
table builder. These checks audit the snapshot, not checkpoint reload validity.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, stdev

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/focused_norm"
SEEDS = (17, 42, 123)
TASKS = ("rte", "mrpc")
ARMS = ("P1_UNREG", "P1_MIX", "P1_NORM")


def summary(values):
    values = list(map(float, values))
    assert len(values) == 3 and all(np.isfinite(values))
    avg, sd = mean(values), stdev(values)
    half = float(t.ppf(0.975, 2)) * sd / np.sqrt(3)
    return dict(values=values, mean=avg, sample_sd=sd,
                nominal_t95=[avg-half, avg+half])


def metrics(row):
    f = lambda key: float(row[key])
    rho = f("rho_pooled")
    cross = f("pooled_LT") + f("pooled_TL")
    out = dict(rho=rho, cross_pp=100*cross,
               eqm_cross_pp=100*(f("eqm_LT")+f("eqm_TL")),
               LL_pp=100*f("pooled_LL"), TT_pp=100*f("pooled_TT"),
               accuracy_pp=100*f("fixed_accuracy"),
               best_accuracy_pp=100*f("best_accuracy"),
               probe_ce=f("probe_masked_ce"),
               normalized_cross_energy=rho*rho*cross)
    if row["fixed_f1"]:
        out["f1_pp"] = 100*f("fixed_f1")
    return out


def analyze():
    paths = [DATA / "runs.csv", DATA / "selection_final.json"]
    rows = list(csv.DictReader(paths[0].open()))
    selected = json.loads(paths[1].read_text())["selection"]
    rows = [r for r in rows if r["condition"] in ARMS]
    assert len(rows) == 36, len(rows)
    assert len({r["run_id"] for r in rows}) == 36
    assert all(re.fullmatch(r"[0-9a-f]{64}", r["validation_sha256"]) for r in rows)
    modules = None
    for r in rows:
        for prefix in ("pooled", "eqm"):
            values = [float(r[f"{prefix}_{b}"]) for b in ("LL", "LT", "TL", "TT")]
            # Exported float32 bases produce ~1e-7 coordinate/ambient error.
            assert min(values) >= 0 and abs(sum(values)-1) < 1e-6
        values = json.loads(r["per_module_relative_frobenius"])
        assert len(values) == 48 and all(v > 0 for v in values.values())
        if modules is None:
            modules = sorted(values)
        assert sorted(values) == modules
        assert abs(mean(values.values())-float(r["rho_eqm"])) < 1e-10
    out = dict(source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               seed_order=list(SEEDS), snapshot_rows=len(rows),
               validation_scope="Export consistency only; server checkpoint reports not rerun.",
               interval_scope="Nominal unadjusted paired t intervals, df=2; normal seed-difference assumption; not equivalence tests.",
               learned_since_insertion_available=any("learned" in key for key in rows[0]),
               learning_health={r["run_id"]: r["learning_health"] for r in rows}, tasks={})
    for task in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        cal = [r for r in task_rows if r["stage"] == "calibration"]
        conf = [r for r in task_rows if r["stage"] == "confirmation"]
        assert len(cal) == len(conf) == 9
        by = {(r["condition"], int(r["seed"])): r for r in conf}
        assert len(by) == 9 and set(by) == {(a, s) for a in ARMS for s in SEEDS}
        target_rows = [r for r in cal if r["condition"] == "P1_MIX"]
        assert len(target_rows) == 1
        target = float(target_rows[0]["rho_pooled"])
        assert target == selected[task]["target_pooled_relative_frobenius"]
        frontier = sorted([r for r in cal if r["condition"] == "P1_NORM"], key=lambda r: float(r["coefficient"]))
        assert len(frontier) == len(selected[task]["frontier"]) == 7
        for r, entry in zip(frontier, selected[task]["frontier"]):
            assert r["run_id"] == entry["run_id"]
            assert float(r["coefficient"]) == entry["coefficient"]
            assert abs(abs(float(r["rho_pooled"])/target-1)-entry["relative_error"]) < 1e-12
        best = min(frontier, key=lambda r: (abs(float(r["rho_pooled"])/target-1), float(r["coefficient"])))
        assert float(best["coefficient"]) == selected[task]["selected_coefficient"]
        assert all(float(by["P1_NORM",s]["coefficient"]) == float(best["coefficient"]) for s in SEEDS)
        vals = {key: metrics(r) for key, r in by.items()}
        result = dict(arms={}, paired={}, matching=[], module_pairs=[],
                      calibration_norm_range=[min(float(r["rho_pooled"]) for r in frontier), max(float(r["rho_pooled"]) for r in frontier)],
                      calibration_cross_range_pp=[min(metrics(r)["cross_pp"] for r in frontier), max(metrics(r)["cross_pp"] for r in frontier)])
        for arm in ARMS:
            result["arms"][arm] = {key: summary(vals[arm,s][key] for s in SEEDS) for key in vals[arm,SEEDS[0]]}
        for left, right in (("P1_MIX", "P1_NORM"), ("P1_MIX", "P1_UNREG"), ("P1_NORM", "P1_UNREG")):
            result["paired"][left+"_minus_"+right] = {key: summary(vals[left,s][key]-vals[right,s][key] for s in SEEDS) for key in vals[left,SEEDS[0]]}
        for seed in SEEDS:
            mix, norm = by["P1_MIX",seed], by["P1_NORM",seed]
            ratio = float(norm["rho_pooled"])/float(mix["rho_pooled"])
            result["matching"].append(dict(seed=seed, signed_error_percent=100*(ratio-1), error_percent=100*abs(ratio-1), passes_5pct=abs(ratio-1)<=0.05))
            m, n = (json.loads(r["per_module_relative_frobenius"]) for r in (mix,norm))
            ratios = np.array([n[key]/m[key] for key in modules])
            adjusted = ratios/ratio
            result["module_pairs"].append(dict(seed=seed,
                mix_median_rho=float(np.median(list(m.values()))),
                norm_median_rho=float(np.median(list(n.values()))),
                mix_modules_below_1e6=sum(v < 1e-6 for v in m.values()),
                norm_modules_below_1e6=sum(v < 1e-6 for v in n.values()),
                raw_ratio_quantiles=np.quantile(ratios,[0,.25,.5,.75,1]).tolist(),
                globally_rescaled_ratio_quantiles=np.quantile(adjusted,[0,.25,.5,.75,1]).tolist(),
                modules_within_5pct=int(np.sum(np.abs(ratios-1)<=.05)),
                rescaled_modules_within_5pct=int(np.sum(np.abs(adjusted-1)<=.05)),
                modules={key: dict(mix_rho=m[key], norm_rho=n[key], raw_norm_over_mix=n[key]/m[key], globally_rescaled_norm_over_mix=n[key]/m[key]/ratio) for key in modules}))
        result["energy_ratios_of_seed_means"] = {a+"_over_MIX": result["arms"][a]["normalized_cross_energy"]["mean"]/result["arms"]["P1_MIX"]["normalized_cross_energy"]["mean"] for a in ("P1_NORM", "P1_UNREG")}
        out["tasks"][task] = result
    return out


def module_figure(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 2, figsize=(6.7, 6.0), sharex=True, sharey=True)
    suffixes = ("self.query", "self.key", "self.value", "output.dense")
    for i, task in enumerate(TASKS):
        pairs = result["tasks"][task]["module_pairs"]
        for j, arm in enumerate(("mix_rho", "norm_rho")):
            values = np.array([[np.median([p["modules"][f"roberta.encoder.layer.{layer}.attention.{suffix}"][arm] for p in pairs]) for suffix in suffixes] for layer in range(12)])
            ax = axes[i,j]
            plot = ax.imshow(np.maximum(values, 1e-6), norm=LogNorm(1e-6, 1), cmap="Greys", aspect="auto")
            for layer, col in zip(*np.where(values < 1e-6)):
                ax.text(col, layer, "<", ha="center", va="center", fontsize=8)
            ax.set_title(f"{task.upper()} / {'MIX' if j == 0 else 'NORM'}", fontsize=10)
            ax.set_xticks(range(4), ("Query", "Key", "Value", "Output"))
            ax.set_yticks(range(12))
            if j == 0:
                ax.set_ylabel("Encoder layer (zero-based)")
            ax.tick_params(labelsize=8)
    fig.subplots_adjust(right=.84, hspace=.24, wspace=.15)
    bar = fig.colorbar(plot, cax=fig.add_axes([.87,.15,.025,.7]))
    bar.set_label("Module relative norm (median across 3 seeds)", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    fig.savefig(ROOT / "figures/focused_module_norms.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = analyze()
    output = json.dumps(result, indent=2, sort_keys=True)+"\n"
    path = DATA / "confirmation_analysis.json"
    if args.write:
        path.write_text(output)
        module_figure(result)
    if args.check:
        assert path.read_text() == output, "Analysis differs from frozen source snapshot"
    for task, r in result["tasks"].items():
        print(task, "match", r["matching"])
        for arm, values in r["arms"].items():
            print(arm, {k: (round(values[k]["mean"],4), round(values[k]["sample_sd"],4)) for k in ("cross_pp","eqm_cross_pp","accuracy_pp","probe_ce")})
        for pair, values in r["paired"].items():
            print(pair, {k: values[k] for k in ("cross_pp", "accuracy_pp")})
        print("energy ratios", r["energy_ratios_of_seed_means"])
        for p in r["module_pairs"]:
            print("modules", {k:v for k,v in p.items() if k != "modules"})
    print("PASS: unique seeds, 48 modules/run, fractions, norm means, calibration selection/frontier, validation-SHA format.")
    print("Learned-since-insertion columns available:", result["learned_since_insertion_available"])


if __name__ == "__main__":
    main()
