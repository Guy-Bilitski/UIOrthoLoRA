"""Faithful CLoRA reproduction job generator (handoff/20).

Reproduces CLoRA Table 3 (math, r64/alpha128, MetaMathQA 395K) and Table 2 (commonsense,
r32/alpha64, Commonsense170K) at the paper's EXACT recipe, and sweeps LR x weight-decay for
our LoRA+wd arm to test whether plain LoRA+wd matches/beats the fancy geometric adapters.

- Reproduction baselines run at CLoRA's fixed LR=3e-4 (their reported operating point).
- LoRA+wd runs the full LR x wd grid; its wd=0 column IS plain LoRA across the LR grid, so
  the LR=3e-4/wd=0 cell = the faithful plain-LoRA baseline (deduped).
- Resumable: skips any cell that already has results/<run>/summary.json.

  python make_frepro_jobs.py --table math --prefix frm \
      --base_model meta-llama/Llama-2-7b-hf \
      --data_path repro/LLM-Adapters/ft-training_set/metamathqa_395k.json --out jobs/frepro_math.txt
  python make_frepro_jobs.py --table cs --prefix frc \
      --base_model meta-llama/Llama-2-7b-hf --out jobs/frepro_cs.txt
"""
import os, glob, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/guy/UIOrthoLoRA/.venv/bin/python"

# ---- Table 3 (math): r=64, alpha=128 ----
MATH_BASELINES = {
    "lora":       "--method lora --lora_r 64 --lora_alpha 128",
    "pissa":      "--method lora --pissa 1 --lora_r 64 --lora_alpha 128",
    "milora":     "--method lora --milora 1 --lora_r 64 --lora_alpha 128",
    "dora":       "--method lora --use_dora 1 --lora_r 64 --lora_alpha 128",   # extra (not in Table 3)
    "clora_k64":  "--method clora --clora_k 64 --lora_r 64 --lora_alpha 128",
    "clora_k128": "--method clora --clora_k 128 --lora_r 64 --lora_alpha 128",
    "clora_k256": "--method clora --clora_k 256 --lora_r 64 --lora_alpha 128",
}
MATH_LORAWD = "--method lora --lora_r 64 --lora_alpha 128"

# ---- Table 2 (commonsense): r=32, alpha=64 ----
CS_BASELINES = {
    "lora_r32":    "--method lora --lora_r 32 --lora_alpha 64",          # the paper's MAIN LoRA baseline
    "pissa_r32":   "--method lora --pissa 1 --lora_r 32 --lora_alpha 64",
    "milora_r32":  "--method lora --milora 1 --lora_r 32 --lora_alpha 64",
    "dora_r32":    "--method lora --use_dora 1 --lora_r 32 --lora_alpha 64",
    "lora_r8":     "--method lora --lora_r 8 --lora_alpha 16",           # reduced-rank baseline
    "lora_r16":    "--method lora --lora_r 16 --lora_alpha 32",          # reduced-rank baseline
    "lora_l2":     "--method lora --lora_r 32 --lora_alpha 64 --weight_decay 1e-5",  # LoRA-L2 (~AdamW wd)
    "clora_k128":  "--method clora --clora_k 128 --lora_r 32 --lora_alpha 64",
    "clora_k256":  "--method clora --clora_k 256 --lora_r 32 --lora_alpha 64",
    "clora_k512":  "--method clora --clora_k 512 --lora_r 32 --lora_alpha 64",
    "clora_k1024": "--method clora --clora_k 1024 --lora_r 32 --lora_alpha 64",
    "clora_k2048": "--method clora --clora_k 2048 --lora_r 32 --lora_alpha 64",
}
CS_LORAWD = "--method lora --lora_r 32 --lora_alpha 64"

# OTHER adapters (NOT in CLoRA's tables) — added to show none beats LoRA+wd even when LR-swept.
# Each at the table's faithful r/alpha with its own paper-default calibration (CorDA/LoRA-Null = nq_open
# KPA / null-space; SC-LoRA = D+ task / D- = nq_open). Residual methods -> residual_save (scaling-generalized).
MATH_DATA_AWARE = {
    "milora":    "--method lora --milora 1 --lora_r 64 --lora_alpha 128",
    "sclora":    "--method lora --sclora 1 --sclora_beta 0.5 --lora_r 64 --lora_alpha 128",
    "lora_null": "--method lora --lora_null 1 --lora_r 64 --lora_alpha 128",
}
CS_DATA_AWARE = {
    "milora":    "--method lora --milora 1 --lora_r 32 --lora_alpha 64",
    "sclora":    "--method lora --sclora 1 --sclora_beta 0.5 --lora_r 32 --lora_alpha 64",
    "lora_null": "--method lora --lora_null 1 --lora_r 32 --lora_alpha 64",
}

BASELINE_LR = ("3e4", "0.0003")   # CLoRA's fixed LR
LORAWD_LRS = [("1e4", "0.0001"), ("2e4", "0.0002"), ("3e4", "0.0003"),
              ("5e4", "0.0005"), ("7e4", "0.0007"), ("1e3", "0.001")]
LORAWD_WDS = [("wd0", "0.0"), ("wd0p1", "0.1"), ("wd0p2", "0.2"), ("wd0p3", "0.3"), ("wd0p5", "0.5")]
# depth-first core (run first): brackets the expected optimum
CORE_LRS = {"2e4", "3e4", "5e4"}
CORE_WDS = {"wd0p1", "wd0p2", "wd0p3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["math", "cs"], required=True)
    ap.add_argument("--prefix", required=True, help="run-name prefix, e.g. frm / frc")
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--data_path", default="", help="train data; empty = train_cs default (commonsense_170k)")
    ap.add_argument("--seeds", default="42", help="comma list, e.g. 42 or 42,43,44")
    ap.add_argument("--cutoffs", default="256", help="comma list; first = faithful primary (all arms), "
                    "extras = sensitivity subset (lora baseline + lorawd core). Math: '256,512'.")
    ap.add_argument("--baselines", type=int, default=1, help="0 = skip fixed-3e-4 baselines "
                    "(CLoRA uses published numbers; DoRA/PiSSA dropped; LoRA anchor already reproduced).")
    ap.add_argument("--core", type=int, default=0, help="1 = depth-first core lorawd grid only (3x3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    seeds = a.seeds.split(",")
    cutoffs = [c.strip() for c in a.cutoffs.split(",")]
    primary = cutoffs[0]   # faithful primary (all arms); extra cutoffs = sensitivity (lora + lorawd core)
    if a.table == "math":
        baselines, lorawd_base, data_aware = MATH_BASELINES, MATH_LORAWD, MATH_DATA_AWARE
        adapt, rmg = "math_faithful", 256
    else:
        baselines, lorawd_base, data_aware = CS_BASELINES, CS_LORAWD, CS_DATA_AWARE
        adapt, rmg = "cs", 512

    done = {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(HERE, "results/*/summary.json"))}
    data_arg = f" --data_path {a.data_path}" if a.data_path else ""

    lines = []
    stats = {"target": 0, "skipped": 0}

    def add(run, flags, val, seed, cut):
        stats["target"] += 1
        if run in done:
            stats["skipped"] += 1
            return
        train = (f"{PY} train_cs.py {flags} --learning_rate {val} --cutoff_len {cut} "
                 f"--seed {seed} --base_model {a.base_model}{data_arg} --run_name {run}")
        ev = (f"{PY} eval_one_gpu.py --adapter /scratch/cf_models/{run} --run_name {run} "
              f"--base_model {a.base_model} --adapt_task {adapt} --ret_suite broad --ret_limit 0 --ret_max_gen {rmg}")
        lines.append(f"{train} && {ev}")

    for cut in cutoffs:
        sens = (cut != primary)   # non-primary cutoff = sensitivity: only lora baseline + lorawd core
        # reproduction baselines @ CLoRA's fixed LR=3e-4 (skipped with --baselines 0)
        for arm, flags in (baselines.items() if a.baselines else []):
            if sens and arm not in ("lora", "lora_r32"):
                continue
            for s in seeds:
                add(f"{a.prefix}_{arm}_lr{BASELINE_LR[0]}_c{cut}_s{s}", flags, BASELINE_LR[1], s, cut)
        # LoRA+wd LR x wd sweep (wd0 column == plain LoRA across the LR grid)
        for tok, val in LORAWD_LRS:
            for wtok, wval in LORAWD_WDS:
                # full grid at the primary cutoff; core-only at a sensitivity cutoff (or with --core)
                if (a.core or sens) and (tok not in CORE_LRS or wtok not in CORE_WDS):
                    continue
                for s in seeds:
                    add(f"{a.prefix}_lorawd_{wtok}_lr{tok}_c{cut}_s{s}",
                        f"{lorawd_base} --weight_decay {wval}", val, s, cut)

    # OTHER adapters (not in CLoRA's tables) — LR-swept at faithful recipe, primary cutoff only,
    # to show none matches LoRA+wd even when tuned. Paper-default calibration applied in train_cs.py.
    for arm, flags in data_aware.items():
        for tok, val in LORAWD_LRS:
            for s in seeds:
                add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, val, s, primary)

    with open(a.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"[{a.prefix}/{a.table}] target={stats['target']} seeds={a.seeds} core={a.core}  "
          f"done={stats['skipped']}  remaining={len(lines)} -> {a.out}")


if __name__ == "__main__":
    main()
