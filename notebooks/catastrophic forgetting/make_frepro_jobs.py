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

# ================================================================================================
# RESTART (frepro4) ADDITIONS — consortium Tier-A/B cells beyond the two original tables.
# All resumable (add() skips any run with results/<run>/summary.json). See handoff/21 sec 2-3.
# ================================================================================================
# LR token -> value lookup (LORAWD_LRS + the R3 low-LR "2e-5" anchor + B4's 5e-5).
LR_TOKENS = dict(LORAWD_LRS + [("2e5", "0.00002"), ("5e5", "0.00005")])

# CorDA++ arms (PI 2026-07-06: CorDA++ REPLACES CorDA as the paper's data-aware-SVD representative;
# old-CorDA re-run cells stay excluded). LR-swept over LORAWD_LRS + the 2e-5 native-LR cell.
# REQUIRES train_cs_cordapp.patch applied (now MANDATORY) and gates PASSED before dispatch:
# validate_cordapp_cpu.py (14/14) then validate_frepro_residual.py --cordapp (1 GPU). Emitted LAST
# in each table so the earlier CLoRA/lorawd cells run while the gates complete.
MATH_CORDAPP = {"cordapp": "--method lora --cordapp 1 --cordapp_n 5 --lora_r 64 --lora_alpha 128"}
CS_CORDAPP   = {"cordapp": "--method lora --cordapp 1 --cordapp_n 5 --lora_r 32 --lora_alpha 64"}
CORDAPP_LR_TOKS = [t for t, _ in LORAWD_LRS] + ["2e5"]

# B4 eval-matched calibration arm (PI approved ~35 GPU-h): tests whether SC-LoRA's below-curve
# deviation in the n=49 CS registry is a calibration artifact. Cells MATCH THE REGISTRY (lrsw)
# CONFIGS (sclora r32/a32 s=1, lora_null r16/a16 s=1 -- NOT the frepro r32/a64 recipe) with
# --calib_source eval_matched (windowed MMLU auxiliary_train text instead of nq_open train).
# REQUIRES b4.patch (adds --calib_source to train_cs.py). Emitted only via --b4 1 (CS table).
B4_CELLS = [
    ("b4_sclora_r32",    "--method lora --sclora 1 --sclora_beta 0.5 --calib_source eval_matched "
                         "--lora_r 32 --lora_alpha 32", ["5e5", "1e4", "3e4", "5e4"]),
    ("b4_lora_null_r16", "--method lora --lora_null 1 --calib_source eval_matched "
                         "--lora_r 16 --lora_alpha 16", ["2e5", "1e4", "3e4"]),
    ("b4_cordapp_r32",   "--method lora --cordapp 1 --cordapp_n 5 --calib_source eval_matched "
                         "--lora_r 32 --lora_alpha 64", ["1e4", "3e4"]),
]

# C2/R2: within-harness math CLoRA LR-sweep. Published-best k = k128, swept over LORAWD_LRS
# (the lr3e4 cell dedupes against the done fixed-3e-4 baseline). CLoRA needs --method clora --clora_k.
MATH_CLORA_SWEEP = {"clora_k128": "--method clora --clora_k 128 --lora_r 64 --lora_alpha 128"}

# R1: MiLoRA native-alpha (s=1, alpha==r) APPENDIX arm — labeled distinct from the PRIMARY alpha=2r
# `milora` arm (which keeps its name/config); LR-swept over LORAWD_LRS at the primary cutoff.
MATH_MILORA_A1R = {"milora_a1r": "--method lora --milora 1 --lora_r 64 --lora_alpha 64"}
CS_MILORA_A1R   = {"milora_a1r": "--method lora --milora 1 --lora_r 32 --lora_alpha 32"}

# Targeted EXTRA cells: (arm_name, flags, [lr_tokens]) at the primary cutoff, over --seeds.
#   - C5 SC-LoRA beta arms (math only): beta ENCODED in the run name so they never collide with the
#     primary beta=0.5 `sclora` sweep. sclora_b0p8 at {2e-5,1e-4,3e-4}; b0p9 + b0p5 at 2e-5.
#   - R3 low-LR (2e-5) disclaimer cells for lora_null + milora, one each per table.
MATH_EXTRA = [
    ("sclora_b0p8", "--method lora --sclora 1 --sclora_beta 0.8 --lora_r 64 --lora_alpha 128", ["2e5", "1e4", "3e4"]),
    ("sclora_b0p9", "--method lora --sclora 1 --sclora_beta 0.9 --lora_r 64 --lora_alpha 128", ["2e5"]),
    ("sclora_b0p5", "--method lora --sclora 1 --sclora_beta 0.5 --lora_r 64 --lora_alpha 128", ["2e5"]),
    ("milora",      "--method lora --milora 1 --lora_r 64 --lora_alpha 128", ["2e5"]),
    ("lora_null",   "--method lora --lora_null 1 --lora_r 64 --lora_alpha 128", ["2e5"]),
]
CS_EXTRA = [
    ("milora",    "--method lora --milora 1 --lora_r 32 --lora_alpha 64", ["2e5"]),
    ("lora_null", "--method lora --lora_null 1 --lora_r 32 --lora_alpha 64", ["2e5"]),
]

# Cutoff-2048 sensitivity anchors (math only): lora / clora_k128 / milora at lr3e4, cutoff 2048.
# REPLACES the planned c512 anchors — new finding: MiLoRA's released config = max_len 2048, which CLoRA
# follows (handoff/23 sec 2). The c512 lorawd cells already in the live pool stay as bonus data.
MATH_CUTOFF_ANCHORS = [
    ("lora",       "--method lora --lora_r 64 --lora_alpha 128"),
    ("clora_k128", "--method clora --clora_k 128 --lora_r 64 --lora_alpha 128"),
    ("milora",     "--method lora --milora 1 --lora_r 64 --lora_alpha 128"),
]

# R7: 3-seed headline cells (seeds 43,44; s42 already in the tables). (arm, flags, lr_token, table).
# The two LoRA+wd WINNERS are filled AFTER the LR sweep completes (see the commented placeholders).
HEADLINE_SEEDS = ["43", "44"]
HEADLINE_CELLS = [
    ("lorawd_wd0",  "--method lora --lora_r 64 --lora_alpha 128 --weight_decay 0.0", "1e4", "math"),
    ("clora_k2048", "--method clora --clora_k 2048 --lora_r 32 --lora_alpha 64",     "3e4", "cs"),
    # FILL AFTER SWEEP (uncomment + set winner wd/lr token):
    # ("lorawd_<wd>", "--method lora --lora_r 64 --lora_alpha 128 --weight_decay <wval>", "<lr>", "math"),
    # ("lorawd_<wd>", "--method lora --lora_r 32 --lora_alpha 64 --weight_decay <wval>",  "<lr>", "cs"),
]

# P2/#8: PiSSA math re-run — distinct 'frm2' prefix so it does NOT collide with the existing
# frm_pissa results dir (diagnose the collapse offline first, then this one clean re-run).
PISSA_RERUN = ("frm2_pissa_lr3e4_c256_s42",
               "--method lora --pissa 1 --lora_r 64 --lora_alpha 128", "0.0003")


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
    ap.add_argument("--b4", type=int, default=0, help="1 = emit ONLY the B4 eval-matched calibration "
                    "cells (CS table; needs b4.patch applied to train_cs.py). Ignores --prefix.")
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
    emitted = set()   # dedup: the same run may be requested by >1 block (e.g. clora_k128 lr3e4
                      # appears in both the baseline block and the CLoRA LR-sweep) -> emit once.

    def add(run, flags, val, seed, cut):
        if run in emitted:
            return
        emitted.add(run)
        stats["target"] += 1
        if run in done:
            stats["skipped"] += 1
            return
        train = (f"{PY} train_cs.py {flags} --learning_rate {val} --cutoff_len {cut} "
                 f"--seed {seed} --base_model {a.base_model}{data_arg} --run_name {run}")
        ev = (f"{PY} eval_one_gpu.py --adapter /scratch/cf_models/{run} --run_name {run} "
              f"--base_model {a.base_model} --adapt_task {adapt} --ret_suite broad --ret_limit 0 --ret_max_gen {rmg}")
        lines.append(f"{train} && {ev}")

    if a.b4:
        # B4 eval-matched calibration arm ONLY (9 cells @ 1 seed): separate file, appended to the
        # lean queue AFTER the CS grid. Resumable like everything else (skips done).
        assert a.table == "cs", "--b4 cells are CS-table (registry-matched) only"
        for name, flags, toks in B4_CELLS:
            for tok in toks:
                for s in seeds:
                    add(f"{name}_lr{tok}_s{s}", flags, LR_TOKENS[tok], s, primary)
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        print(f"[b4/{a.table}] target={stats['target']} seeds={a.seeds}  done={stats['skipped']}  "
              f"remaining={len(lines)} -> {a.out}")
        return

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

    # ===================== RESTART (frepro4) additions — consortium Tier-A/B =====================
    if a.table == "math":
        # C2/R2: within-harness math CLoRA LR-sweep (published-best k=k128); lr3e4 dedupes vs baseline.
        for arm, flags in MATH_CLORA_SWEEP.items():
            for tok, val in LORAWD_LRS:
                for s in seeds:
                    add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, val, s, primary)
        # R1: MiLoRA native-alpha (s=1) appendix arm, LR-swept.
        for arm, flags in MATH_MILORA_A1R.items():
            for tok, val in LORAWD_LRS:
                for s in seeds:
                    add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, val, s, primary)
        # C5 SC-LoRA beta arms + R3 low-LR (2e-5) disclaimer cells (targeted LR lists).
        for arm, flags, toks in MATH_EXTRA:
            for tok in toks:
                for s in seeds:
                    add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, LR_TOKENS[tok], s, primary)
        # C6 (superseded to c2048): cutoff-2048 sensitivity anchors at lr3e4.
        for arm, flags in MATH_CUTOFF_ANCHORS:
            for s in seeds:
                add(f"{a.prefix}_{arm}_lr3e4_c2048_s{s}", flags, "0.0003", s, "2048")
        # P2/#8: PiSSA math re-run (distinct frm2 prefix -> no results-dir collision).
        prun, pflags, pval = PISSA_RERUN
        add(prun, pflags, pval, "42", primary)
    else:  # cs
        # R1: MiLoRA native-alpha (s=1) appendix arm, LR-swept.
        for arm, flags in CS_MILORA_A1R.items():
            for tok, val in LORAWD_LRS:
                for s in seeds:
                    add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, val, s, primary)
        # R3 low-LR (2e-5) disclaimer cells (lora_null + milora).
        for arm, flags, toks in CS_EXTRA:
            for tok in toks:
                for s in seeds:
                    add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, LR_TOKENS[tok], s, primary)

    # R7: 3-seed headline cells (seeds 43,44) for THIS table's headline cells.
    for arm, flags, tok, tbl in HEADLINE_CELLS:
        if tbl != a.table:
            continue
        for s in HEADLINE_SEEDS:
            add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, LR_TOKENS[tok], s, primary)

    # CorDA++ arm — EMITTED LAST on purpose: the earlier CLoRA/lorawd cells give the cordapp gates
    # (validate_cordapp_cpu.py 14/14 -> validate_frepro_residual.py --cordapp) time to complete
    # before any cordapp cell dispatches. If a gate FAILS, comment these lines out and relaunch.
    for arm, flags in (MATH_CORDAPP if a.table == "math" else CS_CORDAPP).items():
        for tok in CORDAPP_LR_TOKS:
            for s in seeds:
                add(f"{a.prefix}_{arm}_lr{tok}_c{primary}_s{s}", flags, LR_TOKENS[tok], s, primary)

    with open(a.out, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"[{a.prefix}/{a.table}] target={stats['target']} seeds={a.seeds} core={a.core}  "
          f"done={stats['skipped']}  remaining={len(lines)} -> {a.out}")


if __name__ == "__main__":
    main()
