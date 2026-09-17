# Qwen intruder campaign — COMPLETE (closing state, 2026-09-17 06:30 UTC)

Supersedes `QWEN_HANDOFF_2026-09-16.md`, which described the campaign mid-flight. Nothing is
running; the H200 is idle and everything is pushed. Narrative of every result and decision is in
`TIERA_RUN_LOG.md` (entries 2026-09-16 08:04 through 2026-09-17 01:42).

## 1. What was completed

**Main campaign** (pool `tQd`, drained 2026-09-16 20:55): 50 evaluation jobs, rc=0 on every one,
0 failures, 0 OOM, 0 retries. Earlier: 7 trainings, each skipping the identical 129 of 31,956
steps. Contents:

* Six informative Qwen2.5-7B rows at seven arms each (A, B, C, D, E, Ep, F1):
  MiLoRA 1e-4 · LoRA{+}wd 1e-4 · CLoRA k1024 2e-4 · CLoRA k1024 3e-4 · LoRA-Null 2e-4 ·
  LoRA r16 5e-5.
* SC-LoRA 2e-5: A and Ep are real; B/C/D/E/F1 are a six-fold null (the adapter has 3 intruders
  holding 1.5 % of the energy, so the builder's edits are arithmetic no-ops).
* Qwen2.5-7B base zero point: retention 48.00.

**Scaling ray** (pool `tQr`, drained 2026-09-17 01:42): 7 jobs, rc=0 on all. Four pure rescalings
of CLoRA 3e-4 plus three random-direction controls. Built by `rescale_qwen_ray.py` (new file,
modelled on `rescale_adapters.py`); queue `jobs/qwen_ray_queue.txt`.

## 2. Where everything is

| artefact | location |
| --- | --- |
| GitHub `ortho_new` | `67142e28` (local == origin, 0 unpushed) |
| Overleaf project 6a46eb1b48498302a1ab34db, branch `main` | `69c9a71`, table byte-identical to local |
| Per-arm results | `results/tia1_qwsw_*/summary.json` (headline.cs_avg, retention_mean, fdelta) |
| Watchdog final tables | `results/FINAL_TABLE_qwen.{md,csv}` |
| Generated table | `paper/table_intruder.tex` → Overleaf `tables/table_intruder.tex`; no `\dots` cells remain |
| Geometry per run | `results/intruder/<run>.json` |
| Arm norm ratios / feasibility | `logs/verify_<run>.log` |
| Pool logs | `logs/tQd_*.log` (campaign), `logs/tQr_*.log` (ray) |

Regenerate with `python paper_table.py` then `python make_table_intruder_tex.py` — never hand-edit
the table. **Rebase before regenerating:** Guy's commits eecf4e37 and 7f47a3c3 (2026-09-16 08:44)
rewrote the generator's caption and column labels, and a stale local copy will silently revert
them.

## 3. Results that matter

* **Intruder load is a function of update size.** Qwen top-ten slot share runs 0.2 % (SC-LoRA) →
  3 → 6 → 12 → 38 → 48 → 62 % (LoRA-Null); Llama 57-94 %.
* **C ≥ B on retention in 11 of 11 configurations** with a real deletion (5 Llama, 6 Qwen). The
  margin grows with intruder energy share; below ~1.3 points it is inside measurement noise, so
  the claim's weight comes from the three high-load rows. Task accuracy is the sharper axis
  (noise 0.25 pt): +2.4, +4.8 and +54 pp there.
* **D is worse than A in 12 of 12 configurations.** The F_delta penalty orders with intruder
  energy share; the task penalty does **not** (LoRA r16 at energy 0.018 loses 3.9 pp, more than
  CLoRA 2e-4 at 0.28). Mechanism is arithmetic: restoring the norm amplifies every surviving
  direction by 1/‖B‖.
* **Deletion collapses the task only once intruders dominate the top spectrum** — inert at ≤12 %,
  −1.3 to −2.4 pp at 38-48 %, destroyed at 62 % and on every Llama row (57-94 %).
* **Scaling ray (CLoRA 3e-4).** Retention vs F_delta fits at R² 0.9870 (quadratic; the relation is
  convex, so the linear 0.9774 understates it). Residuals: arm B +0.86, arm D −0.96, arm E −0.02,
  arm Ep −0.30 — **all inside the 1.28 pt noise**; on task accuracy B and D fall 4.08 and 4.82
  points below the curve while E and Ep stay on it.
* **The headline number.** At one matched magnitude (F_delta ≈ 0.24) task accuracy spans **80.94
  points** (86.44 shrunk-trained → 5.50 random direction) while retention spans **2.35**. Magnitude
  sets forgetting; direction sets capability.
* **Measurement floor.** SC-LoRA A/B/C/D/E/F1 are six evaluations of one arithmetically unchanged
  adapter: task spread 0.31 pp, retention **1.28 pt**, F_delta 0.001. Quote this as the protocol's
  resolution. The same row is a negative control: where the deletion is real D < A, where it is a
  no-op D sits on A, so the effects are not artifacts of the arm builder.

## 4. Open items (none block shutdown)

1. **Table vs caption, SC-LoRA.** The caption says "only A is reportable; other placeholders are
   not valid comparisons", but B/C/D/E now print numbers (they were evaluated). The substantive
   warning still stands, so nothing is misleading — but either add those `(run, arm)` tuples to
   the generator's `NOT_BUILT` or reword the clause. Printed as a number, SC-LoRA D (86.81 vs A's
   86.50) reads as a counterexample to "D is never better than A"; it is a no-op.
2. **Qwen LoRA-Null arm E** prints 48.57/85.31 at norm 0.788 against arm B's 0.706 (`[INFEASIBLE]`,
   rel diff 0.2464) while the caption asserts B/C/E share a norm. Needs a dagger or `--`.
3. **Manuscript numbers to refresh** (writing agent): "nine configurations" → eleven; the arm-D
   "more than a tenth of the energy" qualifier is falsified; state the 1.28 pt noise floor; lead
   C-vs-B with task accuracy; Ep is not independent of E on CLoRA 2e-4 or LoRA-Null; the
   count-matched (F1) severity ordering is not established and should stay qualitative.
4. **Llama vs Qwen divergence that must not be smoothed over.** Llama's arm B sits 2.16 retention
   points *below* its ray; Qwen's sits +0.86 *above*. The magnitude claim replicates; that sign
   does not.
5. **Not started:** second seed on Qwen LoRA-Null (arms A/B/C) to answer "all runs single-seed",
   ~9.5 h including training. Awaiting Guy.
6. **Off-node evacuation** of adapters under `/home/kfir/tierA_evac` remains open from the
   original Tier A handoff.

## 5. Restarting anything

The daemons are gone by design (`qwen_campaign3.sh`, `qwen_watchdog.sh` and both pools exited
normally). To run a new eval queue:

```
setsid /home/kfir/guyb/UIOrthoLoRA/.venv/bin/python gpu_pool_dyn.py --queue <queue.txt> \
  --tag <tag> --slots 2 --slots_training 0 --min_free_mb 40000 --stagger 150 \
  --lock logs/gpu_slot.lock >> logs/<tag>_pool.log 2>&1 < /dev/null &
```

Queue format is tab-separated `<run>\t<cmd>` with `STOP` last. Use `^`-anchored pgrep patterns for
health checks, and never `pkill -f <pattern>` from a shell whose own command line contains it.
