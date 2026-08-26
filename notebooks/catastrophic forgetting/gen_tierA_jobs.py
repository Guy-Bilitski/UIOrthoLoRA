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
}

# Exp 1: (spec cell #, family, method, lr, expected pool-median F_delta, zone)
EXP1 = [
    (1, "llama", "lorawd", "1e-4", 0.23, "below"),
    (2, "llama", "lorawd", "5e-4", 0.40, "knee (pooled)"),
    (3, "llama", "lorawd", "1e-3", 0.46, "above (wd caps it)"),
    (4, "llama", "milora", "1e-4", 0.25, "below"),
    (5, "llama", "milora", "3e-4", 0.56, "near"),
    (6, "llama", "milora", "1e-3", 1.51, "far above"),
    (7, "llama", "sclora", "2e-5", 0.17, "below"),
    (8, "llama", "sclora", "2e-4", 0.60, "near"),
    (9, "llama", "sclora", "1e-3", 1.74, "far above"),
    (10, "qwen", "lorawd", "2e-5", 0.10, "below"),
    (11, "qwen", "lorawd", "3e-4", 0.21, "knee"),
    (12, "qwen", "lorawd", "1e-3", 0.28, "above (DIVERGENCE RISK, fallback 5e-4)"),
    (13, "qwen", "milora", "5e-5", 0.13, "below"),
    (14, "qwen", "milora", "2e-4", 0.29, "above"),
    (15, "qwen", "milora", "1e-3", 0.96, "far above"),
    (16, "qwen", "sclora", "2e-5", 0.12, "below"),
    (17, "qwen", "sclora", "2e-4", 0.43, "above"),
    (18, "qwen", "sclora", "1e-3", 1.09, "far above"),
]
COVERAGE_FIRST = [2, 11, 6, 15, 9, 18]  # all six model x method arms, knee-or-above

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


def chain(py, root, evac, rn, base, train_flags, adapt_task, ret_max_gen, data=None):
    seg = [f"{py} train_cs.py {train_flags} --learning_rate {{lr}} --cutoff_len 256 "
           f"--seed 43 --base_model {base} --out_root {root} --run_name {rn}"]
    if data:
        seg[0] += f" --data_path {data}"
    seg.append(f"{py} eval_one_gpu.py --adapter {root}/{rn} --run_name {rn} "
               f"--base_model {base} --adapt_task {adapt_task} --ret_suite broad "
               f"--ret_limit 0 --ret_max_gen {ret_max_gen}")
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
    order = COVERAGE_FIRST + [c[0] for c in EXP1 if c[0] not in COVERAGE_FIRST]
    by_id = {c[0]: c for c in EXP1}
    for cid in order:
        _, fam, meth, lr, fexp, zone = by_id[cid]
        base = LLAMA if fam == "llama" else QWEN
        prefix = "frc" if fam == "llama" else "qwsw"
        wdtok = "_wd0p3" if meth == "lorawd" else ""
        rn = f"tia1_{prefix}_{meth}{wdtok}_{lr_token(lr)}_s43"
        flags = METHOD_FLAGS[(fam, meth)]
        line = chain(py, root, evac, rn, base, flags, "cs", 512).format(lr=lr)
        lines1.append(f"# cell {cid}: {fam} {meth} lr={lr} expected F_delta~{fexp} ({zone})")
        lines1.append(maybe_done(line, rn))

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

    os.makedirs(os.path.join(HERE, "jobs"), exist_ok=True)
    for fn, lines in (("tierA_exp1_slice.txt", lines1), ("tierA_exp2_anchors.txt", lines2)):
        path = os.path.join(HERE, "jobs", fn)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        n = sum(1 for l in lines if l and not l.startswith("#"))
        print(f"wrote {path}: {n} live cells")


if __name__ == "__main__":
    main()
