"""Tier A job-file generator (handoff/TIER_A_SPEC_2026-08-23.md, Phase 0).

Emits per-cell CHAINED job lines (train && eval && CE && evacuate) for:
  jobs/tierA_exp1_slice.txt    Exp 1 intruder slice, 18 cells, coverage-first order
  jobs/tierA_exp2_anchors.txt  Exp 2 Qwen rescale-ladder anchors, 4 cells

Method configurations are the FROZEN POOL arms verbatim (verified 2026-08-26
against surviving job lines + paper Appendix A; the spec's prose paragraph had
stale numbers — pool truth wins):
  LoRA+wd   r=32 alpha=64 wd=0.3            (both families)
  MiLoRA    r=32 alpha=32, --milora 1       (both families)
  SC-LoRA   r=32, --sclora 1 beta=0.5, pooled nq_open calib (NOT eval-matched);
            alpha=64 on Llama-CS, alpha=32 on Qwen-CS (family difference is
            real pool history — keep it)
Seed 43 everywhere (42 is the SC-LoRA magnitude outlier). Standard recipe:
3 epochs, cutoff 256, batch 16 (train_cs.py defaults).

Regenerate on the target machine with the real paths:
  python gen_tierA_jobs.py --python-bin /path/to/venv/bin/python \
      --out-root /scratch/cf_models --evac-dest user@host:/backups/tierA
(--evac-dest may also be a local dir; empty = evacuation step OMITTED, which
violates spec rule 2 — only do that for smoke tests.)

Cells whose results/<run>/summary.json already exists are emitted commented
out (dedupe against results/, spec Phase 0).
"""
import os
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA = "meta-llama/Llama-2-7b-hf"
QWEN = "Qwen/Qwen2.5-7B"
MATH_DATA = "repro/LLM-Adapters/ft-training_set/metamathqa_100k.json"

METHOD_FLAGS = {
    ("llama", "lorawd"): "--method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3",
    ("llama", "milora"): "--method lora --milora 1 --lora_r 32 --lora_alpha 32",
    ("llama", "sclora"): ("--method lora --sclora 1 --sclora_beta 0.5 --sclora_calib_size 256 "
                          "--calib_source nq_open --lora_r 32 --lora_alpha 64"),
    ("qwen", "lorawd"): "--method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3",
    ("qwen", "milora"): "--method lora --milora 1 --lora_r 32 --lora_alpha 32",
    ("qwen", "sclora"): ("--method lora --sclora 1 --sclora_beta 0.5 --sclora_calib_size 256 "
                         "--calib_source nq_open --lora_r 32 --lora_alpha 32"),
    # CLoRA pool arm verbatim (frc_clora_k1024_lr3e4 / qwsw_clora_k1024_lr2e4
    # registry args): r32 a64, k=1024, lambda=1.0, wd=0 (train_cs default)
    ("llama", "clora"): "--method clora --lora_r 32 --lora_alpha 64 --clora_k 1024 --clora_lambda 1.0",
    ("qwen", "clora"): "--method clora --lora_r 32 --lora_alpha 64 --clora_k 1024 --clora_lambda 1.0",
}

# Exp 1 v2 (PI-approved redesign 2026-08-27, in-session with Guy):
#  - cells 4/7/13/16 (MiLoRA+SC-LoRA below-knee) DROPPED, replaced by 2 CLoRA
#    cells at the pool operating points (retention-aware design with an explicit
#    directional constraint — the sharpest intruder-formation contrast).
#  - CLoRA cells run SEED 44: pool frc_clora_k1024_lr3e4_s43 is a documented
#    adaptation-collapse outlier (cs_avg 59.7 vs ~80 at s42/s44).
#  - non-coverage cells run the retention battery at ret_limit 1500 (recorded
#    in registries); coverage cells keep the FULL battery (pool-comparable).
# (spec cell #, family, method, lr, expected pool-median F_delta, zone, seed)
EXP1 = [
    # RE-SPECIFIED 2026-08-27 (Guy: conservative "weak / best / strong, not too much").
    # The original far-above rates were 4-7x too hot: pool frc_sclora lr1e-3 gives
    # F_delta 7.38 / retention 0.64 (annihilated); qwsw milora lr1e-3 -> ret 6.41;
    # qwsw sclora lr1e-3 -> ret 2.23. Useless for R2 (retention floor) and NaN-prone.
    # Every rate below comes from MEASURED pool (lr -> F_delta, retention) curves:
    #   weak = below knee, retention ~ base | best = operating point |
    #   strong = above knee, retention still >= ~60-77% of base.
    # LoRA+wd(0.3) is magnitude-capped by construction (Llama F<=0.47, Qwen F<=0.28
    # even at lr 1e-3) -- that IS its natural span; we do not lower wd to force a
    # bigger update because wd is what defines the arm.
    # --- Llama ---
    (1, "llama", "lorawd", "1e-4", 0.234, "weak (pool ret 104%)", 43),
    (2, "llama", "lorawd", "5e-4", 0.410, "BEST (pool ret 96%, cs 82.1)", 43),
    (3, "llama", "lorawd", "1e-3", 0.465, "strong (pool ret 98%; wd caps F)", 43),
    (4, "llama", "milora", "1e-4", 0.20, "weak (est; ours lr1e3 = 1.50)", 43),
    (5, "llama", "milora", "3e-4", 0.50, "BEST (est)", 43),
    (6, "llama", "milora", "1e-3", 1.501, "strong (OURS: ret 16.5 = 64% of base)", 43),
    (7, "llama", "sclora", "2e-5", 0.15, "weak (est; sclora is cliff-prone)", 43),
    (8, "llama", "sclora", "1e-4", 0.35, "BEST (est; pool lr1e-3 = F7.4 ret0.6)", 43),
    (9, "llama", "sclora", "3e-4", 1.20, "strong (est; NOT 1e-3 -> annihilation)", 43),
    # --- Qwen ---
    (10, "qwen", "lorawd", "2e-5", 0.100, "weak (pool ret 87%)", 43),
    (11, "qwen", "lorawd", "3e-4", 0.213, "BEST (pool ret 92.6%, cs 86.3)", 44),
    (12, "qwen", "lorawd", "1e-3", 0.282, "strong (pool ret 84%; wd caps F)", 43),
    (13, "qwen", "milora", "5e-5", 0.132, "weak (pool ret 89%)", 43),
    (14, "qwen", "milora", "1e-4", 0.184, "BEST (pool ret 89.5%, cs 87.5)", 43),
    (15, "qwen", "milora", "3e-4", 0.386, "strong (pool ret 77%; NOT 1e-3 = ret 6.4)", 43),
    (16, "qwen", "sclora", "2e-5", 0.170, "weak (pool ret 89.6%)", 43),
    (17, "qwen", "sclora", "5e-5", 0.275, "BEST (pool ret 88%; cliff right after)", 43),
    (18, "qwen", "sclora", "7e-5", 0.35, "strong (est; pool lr1e-4 already ret 21%)", 43),
    # --- CLoRA (retention-aware, explicit directional constraint) ---
    (19, "llama", "clora", "3e-4", 0.441, "operating point (pool ret 93.5%)", 44),
    (20, "qwen", "clora", "2e-4", 0.211, "operating point (pool ret 87.7%)", 44),
]
COVERAGE_FIRST = [2, 6, 11, 14, 17, 9]  # six model x method arms at best/strong
TAIL_ORDER = [3, 15, 12, 5, 18, 19, 20, 8, 1, 10, 13, 16, 4, 7]
# Speed pass 2 (Guy 2026-08-27, cost-focused): ONLY cells 2+6 keep the full
# battery (both already run/running — they calibrate subsample-vs-full);
# everything else, coverage included, runs ret_limit 1500.
FULL_EVAL_CELLS = set()   # trimmed proxy protocol everywhere (2026-08-27)
NONCOV_RET_LIMIT = 50     # PER-SUBTASK in lm-eval; 50 -> ~2050 gen reqs (~28min) vs 18543 (~4h16)

# Exp 2 anchors: (setting, method) -> lr per spec; qwsw = Qwen-CS, qwswm = Qwen-math.
# Plain-LoRA anchor is r32/a64 (capacity-matched to LoRA+wd; the pool's CS plain-LoRA
# arm was r16, this anchor is deliberately r32 per the PI-approved spec).
EXP2 = [
    ("qwsw", "lora", "3e-4"),
    ("qwsw", "lorawd", "5e-4"),
    ("qwswm", "lora", "3e-4"),
    ("qwswm", "lorawd", "5e-4"),
]


def lr_token(lr):
    return "lr" + lr.replace("-", "").replace("e0", "e")  # 5e-4 -> lr5e4


def chain(py, root, evac, rn, base, train_flags, adapt_task, ret_max_gen, data=None,
          seed=43, ret_limit=0):
    seg = [f"{py} train_cs.py {train_flags} --learning_rate {{lr}} --cutoff_len 256 "
           f"--seed {seed} --base_model {base} --out_root {root} --run_name {rn}"]
    if data:
        seg[0] += f" --data_path {data}"
    # health gate (2026-08-27): abort the chain before the expensive eval if the
    # adapter came out non-finite. A Qwen NaN cost ~10 GPU-h before this existed.
    seg.append(f"{py} adapter_health.py --adapter {root}/{rn} "
               f"--quarantine /home/kfir/cf_models_failed")
    seg.append(f"{py} eval_one_gpu.py --adapter {root}/{rn} --run_name {rn} "
               f"--base_model {base} --adapt_task {adapt_task} --ret_suite broad "
               f"--ret_limit {ret_limit} --eval_limit 200 --ret_max_gen {ret_max_gen}")
    seg.append(f"{py} forgetting_ce.py --runs {rn} --adapters_root {root} "
               f"--base_model {base} --max_length 1024 --max_blocks 0 --batch_size 2")
    if evac:
        seg.append(f"bash evacuate_cell.sh {root}/{rn} {evac}")
    return " && ".join(seg)


def maybe_done(line, rn):
    if os.path.exists(os.path.join(HERE, "results", rn, "summary.json")):
        return f"# DONE (results/{rn}/summary.json exists): {line}"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-bin", default="python")
    ap.add_argument("--out-root", default="/scratch/cf_models")
    ap.add_argument("--evac-dest", default="",
                    help="rsync destination for per-cell checkpoint evacuation (spec rule 2)")
    args = ap.parse_args()
    py, root, evac = args.python_bin, args.out_root, args.evac_dest
    if not evac:
        print("WARNING: --evac-dest empty -> job lines have NO evacuation step (spec rule 2).")

    lines1 = ["# tierA_exp1_slice.txt — intruder slice, 18 cells, coverage-first order.",
              "# Generated by gen_tierA_jobs.py; spec: handoff/TIER_A_SPEC_2026-08-23.md.",
              "# Run STRICTLY SERIAL (one GPU): python gpu_pool.py --gpus 1 --tag tierA1 --jobs jobs/tierA_exp1_slice.txt",
              "# Qwen lorawd 1e-3 (cell 12): if F_delta explodes (pooled precedent F=14.3),",
              "# retrain ONCE at 5e-4 (rename lr5e4f), flag in the run log, never silently substitute."]
    order = COVERAGE_FIRST + TAIL_ORDER
    by_id = {c[0]: c for c in EXP1}
    cov_pairs, tail_pairs = [], []
    for cid in order:
        _, fam, meth, lr, fexp, zone, seed = by_id[cid]
        base = LLAMA if fam == "llama" else QWEN
        prefix = "frc" if fam == "llama" else "qwsw"
        wdtok = {"lorawd": "_wd0p3", "clora": "_k1024"}.get(meth, "")
        rn = f"tia1_{prefix}_{meth}{wdtok}_{lr_token(lr)}_s{seed}"
        flags = METHOD_FLAGS[(fam, meth)]
        rl = 0 if cid in FULL_EVAL_CELLS else NONCOV_RET_LIMIT
        line = chain(py, root, evac, rn, base, flags, "cs", 512,
                     seed=seed, ret_limit=rl).format(lr=lr)
        cmt = (f"# cell {cid}: {fam} {meth} lr={lr} expected F_delta~{fexp} ({zone})"
               f"{' [ret_limit ' + str(rl) + ']' if rl else ' [FULL eval]'}"
               f"{' [seed ' + str(seed) + ']' if seed != 43 else ''}")
        pair = (cmt, maybe_done(line, rn))
        (cov_pairs if cid in COVERAGE_FIRST else tail_pairs).append(pair)
        lines1.append(cmt)
        lines1.append(pair[1])

    lines2 = ["# tierA_exp2_anchors.txt — Qwen rescale-ladder anchors, 4 cells, seed 43.",
              "# After all 4 land: rescale ladder (~16) + random-direction controls (8) are",
              "# built on CPU from the frozen qwsw/qwswm family curves, then eval-only jobs.",
              "# See handoff/TIER_A_SPEC_2026-08-23.md Exp 2 + handoff/H200_BOOTSTRAP.md."]
    for setting, meth, lr in EXP2:
        wdtok = "_wd0p3" if meth == "lorawd" else "_r32"
        rn = f"tia2_{setting}_{meth}{wdtok}_{lr_token(lr)}_s43"
        flags = ("--method lora --lora_r 32 --lora_alpha 64" if meth == "lora"
                 else METHOD_FLAGS[("qwen", "lorawd")])
        data = MATH_DATA if setting == "qwswm" else None
        task, mg = ("gsm8k", 256) if setting == "qwswm" else ("cs", 512)
        line = chain(py, root, evac, rn, QWEN, flags, task, mg, data=data).format(lr=lr)
        lines2.append(f"# anchor: {setting} {meth} lr={lr}")
        lines2.append(maybe_done(line, rn))

    # Staged queues (speed pass 2): stage1 = intruder coverage + Exp 2 anchors
    # (the paper-gating work, runs first); tail = R2-enrichment cells (run last,
    # after the Exp 2 ladder/control evals which are generated when anchors land).
    anchor_pairs = [(lines2[i], lines2[i + 1]) for i in range(4, len(lines2), 2)]
    lines_s1 = ["# tierA_stage1.txt — coverage cells (reduced eval) then Exp 2 anchors.",
                "# Generated by gen_tierA_jobs.py (speed pass 2, 2026-08-27). Serial."]
    for c, l in cov_pairs + anchor_pairs:
        lines_s1 += [c, l]
    lines_tail = ["# tierA_tail.txt — R2-enrichment cells, run AFTER Exp 2 ladder evals.",
                  "# Order: above/near-knee, CLoRA, below-knee last."]
    for c, l in tail_pairs:
        lines_tail += [c, l]

    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    for fn, lines in (("tierA_exp1_slice.txt", lines1), ("tierA_exp2_anchors.txt", lines2),
                      ("tierA_stage1.txt", lines_s1), ("tierA_tail.txt", lines_tail)):
        path = os.path.join(HERE, "jobs", fn)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        n = sum(1 for l in lines if l and not l.startswith("#"))
        print(f"wrote {path}: {n} live cells")


if __name__ == "__main__":
    main()
