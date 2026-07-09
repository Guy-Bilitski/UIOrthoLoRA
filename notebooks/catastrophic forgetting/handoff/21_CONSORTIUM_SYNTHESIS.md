# 21 — Consortium Synthesis & Action Plan (Supervisor ruling)

**Date:** 2026-07-06 · **Author:** Supervisor (final technical authority below the human PI)
**Inputs:** 7 Wave-1 expert audits (CLoRA, MiLoRA, SC-LoRA, LoRA-Null, CorDA++, Paper-writer, Docs) + 1 adversarial critique.
**Scope of this doc:** reconcile all findings into ONE plan. This file is the only new artifact; nothing here has been applied. All `.py`/jobs/results are READ-ONLY while pool PID 2932862 (`frepro3`) drains `jobs/frepro_lean.txt` (103 cells).

---

## 0. Independent verification (what I recomputed before ruling)

I re-derived the load-bearing numbers directly from `results/frm_*/summary.json` and the job files rather than trusting the memos.

**Magnitude law on the 7 faithful math points** (recomputed, pure-python Pearson):
| set | r(‖ΔW‖, ret) | r(log‖ΔW‖, ret) | r(‖ΔW‖, BBH) |
|---|---|---|---|
| all 7 | **−0.933** | −0.792 | −0.873 |
| 5-point blob (drop LoRA+wd0 & PiSSA) | **+0.058** | +0.054 | **+0.720** |

The blob (LoRA, MiLoRA, CLoRA-k64/128/256) sits in ‖ΔW‖∈[1.019,1.283], ret∈[18.04,19.86]. The full-range r=−0.93 is carried entirely by two leverage points: **PiSSA (2.206, ret 3.62)** and **LoRA+wd0 (0.434, ret 22.58)**. Inside the blob the sign of r(‖ΔW‖,BBH) actually **flips positive (+0.72)**, and **MiLoRA retains 1.82 pp better than LoRA at essentially identical ‖ΔW‖ (1.257 vs 1.283)**. → The critic's kill-shot #1 is **CONFIRMED** and if anything understated.

**PiSSA is a collapsed run, not a smooth law point** (critic #8 CONFIRMED): BBH 7.23, MMLU-Pro 0.0, GSM8K 49.66 vs published 58.23. One of the two law-carrying leverage points is a broken run; the other (LoRA+wd0) is at a *different LR* (1e-4 vs the 3e-4 of every other point), so it also confounds LR with magnitude.

**MMLU-Pro is near-floor post-finetune** (critic #2 CONFIRMED): math MMLU-Pro = {LoRA 6.93, PiSSA 0.0, MiLoRA 9.54, CLoRA 8.51/9.49/9.42, LoRA+wd0 14.19} → 6/7 ≤ random (10%); base = 18.96 (works). `retention_mean = mean(BBH, MMLU-Pro)` (verified: LoRA (29.14+6.93)/2 = 18.04). Values **below** random are the signature of answer-extraction failure on the instruction-tuned generation format, not graceful forgetting. >50% of the headline retention delta rides on this.

**Queue structure** (verified against `jobs/frepro_lean.txt`, `frepro_cs.txt`, `frepro_math.txt`):
- 103 cells: math frm_ (lines 1–55), CS frc_ (lines 56–103). `frm_lora_null` = **lines 50–55**; all `frc_*` from **line 56** → critic #13 ordering CONFIRMED.
- **NO CLoRA anywhere in the queue.** Every "clora" grep hit is `sclora`. `frepro_cs.txt`/`frepro_math.txt` were generated with `--baselines 0`, so the entire CS reproduction baseline block (`clora_k128/256/512/1024/2048`, `pissa_r32`, `dora_r32`, `lora_r8/16`, `lora_l2`) was dropped, and the math CLoRA cells exist only at the 3 done fixed-3e-4 points. → **CS CLoRA k-grid NOT QUEUED** (paper-writer #1) and **CLoRA math LR-sweep does not exist** (critic #3) — both CONFIRMED.
- 7 faithful cells done, all math (the 7 `frm_*` summaries). Zero faithful CS done.

**Operational facts:** `gpu_pool.py` reads the whole job file into an in-memory FIFO queue **once at launch** (no live re-read, no per-line resume) → editing `frepro_lean.txt` now does NOT affect the running pool; only a kill+relaunch changes what runs. Resume/skip is external: `make_frepro_jobs.py` skips cells with `summary.json`; `build_lean.py` converts trained-but-uneval'd adapters to eval-only and skips done. No `nq_open load failed`/`falling back to wikitext` events in any log → SC-LoRA/LoRA-Null/CorDA calibration is genuinely using nq_open (SC-LoRA expert's silent-fallback worry: not occurring). `cordapp` is **not wired into `train_cs.py`** (only legacy `--corda`); `cordapp_init.py` and its CPU gate `validate_cordapp_cpu.py` are complete but uncalled.

---

## 1. RULINGS (supervisor confirms/overrides the critic)

| # | Conflict | Ruling | One-paragraph justification |
|---|----------|--------|------|
| **R1** | **MiLoRA α doctrine** (expert: s=1/α=r primary; critic: α=2r primary, s=1 appendix) | **CONFIRM critic** | Comparability is the whole point of a head-to-head table. If MiLoRA runs at its native s=1 while SC-LoRA/LoRA-Null/CorDA run at α=2r, the ‖ΔW‖ axis and the "identical recipe" claim both break. `residual_save.py` already preserves any scaling, so α=2r is exact for all data-aware arms. The queued `frm_milora`/`frc_milora` cells are **already α=2r (correct)** — no change. Add MiLoRA-s1 (`--lora_alpha 64` math / `32` CS) as a **labeled appendix reproduction row** only. |
| **R2** | **CLoRA published-vs-reproduced headline** ("LoRA 64.97 > CLoRA pub 64.59") | **CONFIRM critic** | Verified there is **no within-harness CLoRA best-LR anywhere** (all CLoRA cells fixed 3e-4). A 0.38 pp margin, cross-harness, while our own harness deflates CLoRA ~5 pp, is not a defensible headline. Soften to "matches/edges," add the **CLoRA math LR-sweep (Tier A)** so the comparison is within-ruler, and put seeds on the headline pair. Keep the published number only in a separately labeled row (CLoRA expert's rule). |
| **R3** | **2e-5 "concession" cells vs native anchors** | **CONFIRM critic; take the disclaimer route for the body** | Running competitors at 2e-5 inside CLoRA's bs16/3-epoch recipe is ~40× their native optimizer steps at 8× smaller batch — not their operating point. True native anchors need a different training config (bs128/1ep/~100K) and are Tier C. For submission, keep the 2e-5-in-our-recipe cells **with an explicit disclaimer** and **drop the "we covered their operating point" defense** entirely. Do not headline them. |
| **R4** | **LoRA-Null calibration** (expert: max_len 1024 + longer text; critic: max_len irrelevant, clone repo, match actual calib, run-as-is-if-degenerate) | **CONFIRM critic on mechanism and procedure; expert's fix is wrong** | Verified in `lora_null_init.py`: C is the in×in input Gram over 256 nq_open **questions** (~10 tokens each ≈ 2.5k tokens ≪ d_in 4096/11008), so C is rank-deficient and the "null space" (smallest-eigval eigenvectors) is dominated by *unsampled* directions. Raising `max_len` does nothing — the questions are already short. The driver is total calib tokens ≪ d_in. **Clone `github.com/HungerPWAY/LoRA-Null` (zero-GPU) and match its actual calibration.** If the repo is itself degenerate, run as-is and disclose (do not improve a competitor beyond its published behavior); if it uses richer/longer calib or a threshold null_dim, patch our calib to match and re-run. **Add a rank/effective-dim diagnostic** either way. This decision must be made inside the 30 h window (it changes queued cells 50–55 / 98–103). |
| **R5** | **Law-carrier dataset** (which evidence headlines the ‖ΔW‖ law) | **CONFIRM critic, and STRENGTHEN** | The 7-point law is leverage-driven (see §0: +0.058 on the blob; +0.72 wrong-sign on BBH; carried by a collapsed PiSSA and an off-LR LoRA+wd0). **Do not headline the law on the 7 faithful points.** Carry it on **(a)** the mature n=49 CS sweep (r=−0.86, disclosed non-faithful) **AND (b)** the completing faithful within-method LR sweeps (`lorawd` gives ‖ΔW‖ 0.4–2.0 for LoRA; `milora`/`sclora`/`lora_null` LR sweeps give within-method magnitude ranges) — these are faithful-recipe and fill the mid-range the 7 points leave empty. Lead with the **ANCOVA/partial framing**: magnitude explains the gross trend; at matched magnitude a **residual ~1.8 pp method effect remains** (MiLoRA > LoRA). Report that residual honestly — it is the honest boundary, already half-written in INTERESTING_INSIGHTS §9. |
| **R6** | **MMLU-Pro handling** | **CONFIRM critic** | With 6/7 math models ≤ random and base=18.96 working, post-tune MMLU-Pro measures extraction failure, not retention. **Report BBH-only as the primary math retention axis** (re-report is 0-GPU from existing summaries), disclose the MMLU-Pro floor, and keep MMLU-Pro only as a flagged secondary. Optionally re-run MMLU-Pro with a fixed extractor (Tier B) — but the paper cannot let >50% of its retention signal ride on it. Note: BBH-only preserves every qualitative story (LoRA+wd0 30.96 > MiLoRA 30.18 > LoRA 29.14 > CLoRA 27.55–28.61 ≫ PiSSA 7.23). |
| **R7** | **Selection protocol** (single-seed LR pick vs multi-seed) | **CONFIRM critic** | Single-seed LR selection is gameable by collapse basins (documented: seed 44 collapsed clora_k2048→23, dora_r8→22, lorawd_wd0p5→51). For the ~9 headline cells, **select the operating LR by the mean over ≥3 seeds, or require the win to hold at ≥2 adjacent LRs** — not std-added-after-the-fact. Table body stays seed-42 (matches CLoRA); the guard applies only to headline cells that carry a claim. |

**Kill-shots I uphold outright (not conflicts — the critic is simply right):** #1 (law leverage-driven), #2 (MMLU-Pro broken), #8 (PiSSA collapsed — diagnose/re-run before citing), #9 (sclora+lora_null missing from the s=2 gate), #10 (non-CLoRA FAITHFUL verdicts are paper-memory-based → Limitations), #11 (fix "r=−0.93"/"monotone" wording), #13 (don't restart blindly; ~30 h runway), #14 (BBH-only, cutoff-512 anchors, key_numbers hygiene).

**Where I temper the critic:** the campaign is **not dead**. Once the queued within-method LR sweeps land, R5(b) gives a genuine faithful mid-range for the law, and the honest thesis (R5) — "magnitude governs the trend; a small residual method effect exists; LoRA+wd matches the fancy adapters on math and mid-regularization CS, and loses only at high-k CS" — is publishable and better-defended than the original overclaim. The fix is reframing + ~1 day of new GPU, not abandonment.

---

## 2. CONSOLIDATED DIFF SET (apply at next pool restart; ordered; conflict-free)

Experts' overlapping diffs (make_frepro_jobs `data_aware` loop, `train_cs.py`, validators) are merged below. **[CHANGES QUEUED]** = alters cells already in the queue (requires kill+relaunch to take effect); **[ADDS]** = new cells/behavior only; **[GUARD/DOC]** = no cell change.

### Group A — `train_cs.py` (init correctness + CorDA++ wiring)
- **A1 [GUARD] Init-error gate.** After every data-aware init (`corda/milora/pissa/sclora/lora_null/cordapp`), the code already computes `err`. Add: `assert err < INIT_ERR_TOL` (recommend `1e-2`, bf16 round-trip) and abort loudly on failure; log `err` to `run_config.json`. Closes the MiLoRA-expert gap "validators don't gate on init err." No cell change; catches degenerate/exploded inits at train time.
- **A2 [CHANGES QUEUED, conditional on R4] LoRA-Null calibration.** After the repo clone (§5), if the repo's calibration is richer than 256 short questions, change the `lora_null` calib block (`train_cs.py` ~L262–284) to match (more samples and/or full-text passages so total tokens ≳ d_in) and add a **rank diagnostic** (`torch.linalg.matrix_rank`/eigenvalue-tail print of C per layer). If the repo is degenerate, leave the block as-is and only add the diagnostic + a disclosure comment. Affects queued cells 50–55, 98–103.
- **A3 [ADDS] CorDA++ 7-diff wiring** (wire the finished `cordapp_init.py`; ordering is load-bearing — `finalize_dynamic_rank_config` MUST run AFTER `residual_save.convert_saved_to_w0_relative`):
  1. **CLI args:** `--cordapp {0,1}`, `--cordapp_n` (default **5** per critic), `--cordapp_calib_size` (= `cordapp_n * 256`), reuse `--corda_calib_size` semantics.
  2. **Precompute BEFORE peft-wrap:** load `cordapp_n*256` nq_open prompts; `res = cordapp_init.precompute_cordapp(raw_model, prompts, tok, targets, fixed_rank=lora_r, N=cordapp_n, scaling=lora_alpha/lora_r)`.
  3. **Build with patterns:** `LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=targets, rank_pattern=res["rank_pattern"], alpha_pattern=res["alpha_pattern"])` → `get_peft_model`. (rank_pattern keys are fully-qualified names — already handled in cordapp_init.)
  4. **Inject KPM:** `err = cordapp_init.apply_cordapp(model, res["chosen_covs"], res["ranks"])`; gate via A1.
  5. **Mark residual:** add `cordapp` to the `residual_method` OR-clause (L288–290) so `capture_init_adapter` runs pre-train.
  6. **Finalize after conversion:** in the post-train residual block (L321–325), after `convert_saved_to_w0_relative`, call `cordapp_init.finalize_dynamic_rank_config(out_dir)` (**MANDATORY** — doubles per-layer rank_pattern/alpha_pattern to match the rank-2r stacking; without it a reload rebuilds each layer at the wrong per-layer rank and eval explodes).
  7. **Gate:** add a dynamic-rank cordapp case to `validate_frepro_residual.py` (see B1) — 0-step reload delta per layer ≈ 0.

### Group B — Validators
- **B1 [CHANGES/ADDS] s=2 residual gate hardening** (`validate_frepro_residual.py`): currently only PiSSA+MiLoRA at r64/α128. **Add `sclora` and `lora_null`** (critic #9) and **`cordapp`** (dynamic rank_pattern) to the s=2 (scaling=2) 0-step reload gate. This is cheap and must pass *before* those cells' outputs are trusted. (`validate_residual_zero_step.py` already covers corda/milora/sclora/lora_null at s=1; add cordapp there too.)
- **B2 [GUARD] Wire the gate into the pipeline** so a residual arm cannot silently produce a mis-scaled adapter (run B1 once per residual method family before/with its first cell).

### Group C — `make_frepro_jobs.py` (job generation) — the fix for the two missing blocks
- **C1 [ADDS] CS CLoRA k-grid (Tier A):** regenerate CS with `--baselines 1` (or add a targeted CLoRA-only pass) so `frc_clora_k128/256/512/1024/2048_lr3e4_c256_s42` are emitted. This is paper-writer's #1 gap.
- **C2 [ADDS] CLoRA math LR-sweep (Tier A):** add CLoRA (best-adapt k = k256) to the math `data_aware`-style loop so it sweeps `LORAWD_LRS` minus the done 3e-4 → `frm_clora_k256_lr{1e4,2e4,5e4,7e4,1e3}_c256_s42`. Requires teaching the loop that `clora` takes `--method clora --clora_k …` (it currently only emits `--method lora …` variants in the data_aware loop).
- **C3 [ADDS] CS baselines for a complete table:** at minimum `frc_pissa_r32_lr3e4` (high-magnitude CS anchor). `dora_r32`, `lora_r8/16`, `lora_l2` → Tier B.
- **C4 [ADDS] 3-seed headline cells:** emit seeds 43,44 for the ~4 headline cells (R7); dedupe against done.
- **C5 [ADDS] MiLoRA-s1 appendix + SC-LoRA β=0.8 arm (Tier B):** `--lora_alpha == --lora_r` MiLoRA reproduction rows (labeled), and `--sclora_beta 0.8` cells (SC-LoRA paper-best) — new cells, do not touch the queued β=0.5 sweep.
- **C6 [ADDS] cutoff-512 math anchors:** `lora`, `milora`, `clora_k256` at `--cutoff_len 512` to anchor the ~3 pp MATH offset (lorawd c512 cells already queued).

### Group D — `residual_save.py`
- **D1 [NO CHANGE]** Verified already scaling-generalized (rank-2r stack, alpha'=2α, r'=2r; validated by `validate_frepro_residual.py` at s=2). No edit needed.

### Group E — Wording / hygiene fixes (0-GPU, docs only)
- **E1** `paper/writing/INTERESTING_INSIGHTS.md` **L37** and **L75**: replace "r = −0.93 (n=7)" with the honest split — full-range r=−0.93 is leverage-driven; on the 5 same-LR competitors r≈+0.06 (blob); law carried by n=49 CS + within-method LR sweeps; matched-magnitude residual ≈1.8 pp (MiLoRA>LoRA). Reframe §3/§7 accordingly.
- **E2** `paper/writing/FINAL_TABLE_PLAN.md` **L44**: "monotone in ‖ΔW‖_F across all 7 faithful points" → qualify as leverage-driven; point to the n=49 + within-method carrier. (L277 already carries the "near-circularity → lead with ANCOVA residual" caveat — keep and elevate it.)
- **E3** key_numbers hygiene (critic #14): fix any `lora_null`-mislabeled-as-`lora` legacy entry; annotate MATH ~3 pp-low as cutoff-256 artifact with c512 anchors; note math `ret_max_gen 256` vs CS `512` is within-table-consistent.
- **E4** Add a Limitations bullet: non-CLoRA "FAITHFUL" verdicts are paper/memory-based pending repo clones (critic #10).

**Which diffs gate the pool restart:** A2 (if R4 → patch), C1/C2/C3/C4/C5/C6 (new cells) — all require regenerating the job file and relaunching. A1/A3/B1/B2 change code but only affect *future* processes (safe to land before relaunch). E-group is pure docs, do now.

---

## 3. RUN QUEUE (prioritized; 8×B200; math ≈5.5 GPU-h/cell, CS ≈3.5 GPU-h/cell)

### Tier A — must-run for submission
| Item | Cells | GPU-h | Notes |
|---|---|---:|---:|
| Finish current queue **minus** 12 degenerate lora_null | ~91 | ~385 | already in flight (lorawd/milora/sclora math+CS); ~48 wall-h |
| **CS CLoRA k-grid** (C1) k128/256/512/1024/2048 @3e4 | 5 | ~18 | #1 gap; the table is about CLoRA |
| **CLoRA math LR-sweep** (C2) k256 @ {1e4,2e4,5e4,7e4,1e3} | 5 | ~28 | within-harness CLoRA best-LR (R2) |
| **LoRA-Null re-run** (R4, if repo≠degenerate) math+CS | ~12 | ~54 | replaces killed cells 50–55/98–103; trim to best 3 LRs if budget-tight |
| **3-seed headlines** (R7) seeds 43,44 × 4 cells | 8 | ~36 | math LoRA+wd winner, closest structured competitor, CS LoRA+wd winner, CS CLoRA winner |
| **PiSSA math re-run** (#8) after generation inspection | 1 | ~6 | diagnose collapse first (0-GPU), then one clean re-run |
| **cutoff-512 math anchors** (C6) lora/milora/clora_k256 | 3 | ~17 | anchor the ~3 pp MATH offset |
| **CS PiSSA baseline** (C3) | 1 | ~4 | high-magnitude CS anchor |
| **BBH-only re-report** (R6) | 0 | 0 | recompute from existing summaries |
| **Tier-A NEW work (beyond the running pool)** | **~35** | **~163** | **≈20 wall-h ≈ 1 day on 8 GPUs** |
| **Tier-A total incl. draining the pool** | **~126** | **~548** | **≈68 wall-h ≈ 2.8 days** |

### Tier B — strengthen (after Tier A / gates)
| Item | Cells | GPU-h |
|---|---|---:|
| CorDA++ controlled arm (A3) — **only after** `validate_cordapp_cpu.py` PASS **and** a 1-GPU 0-step integration gate (B1) PASS; N via `--cordapp_n 5` | ~6–8 | ~40 |
| SC-LoRA β=0.8 arm (C5) math+CS at a few LRs | ~6 | ~30 |
| 2e-5-in-recipe cells with disclaimer (R3) for the top competitors | ~4 | ~22 |
| MiLoRA-s1 appendix reproduction rows (C5) | ~2–4 | ~15 |
| CS DoRA/lora_r8/16/lora_l2 baselines (C3) | ~5 | ~18 |
| MMLU-Pro extraction fix + re-eval (R6 optional) | eval-only | ~10 |

### Tier C — deferred (labeled future / strong-tier)
Native-recipe anchors (bs128/1ep/~100K) for competitors (R3); DoRA math LR-sweep; CorDA++ **native** track (needs paper's N + a TriviaQA/NQ/WebQS EM harness we do not have); full β×LR SC-LoRA grid.

---

## 4. POOL ACTION (precise, timed)

**Do NOT restart now.** The running pool is doing valid work (lines 1–49: lorawd/milora/sclora math, all correct at α=2r). The only queued liabilities are the 12 degenerate `lora_null` cells (math 50–55, CS 98–103); the first is ~30 h out because `gpu_pool` pulls FIFO and lines 1–49 (~30 wall-h at 8×5.5 GPU-h) must clear first.

**Within the 30 h window (all zero-GPU, does not touch the pool):**
1. Clone `github.com/HungerPWAY/LoRA-Null` → decide R4 (run-as-is vs patch calib). This is the gating decision.
2. Land code diffs A1, A3, B1/B2, E-group (they only affect *future* processes; in-flight processes already loaded their source, so they are unaffected).
3. If R4 → patch, prepare A2. Prepare the regenerated job file via `make_frepro_jobs.py` (C1/C2/C3/C4/C5/C6) — it auto-skips the ~40 cells that will have completed by then; run `build_lean.py` to convert any trained-but-uneval'd adapters to eval-only.

**Kill + relaunch (execute at ~T+26 h, before line 50 dispatches):**
```
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
pkill -f '[g]pu_pool'                     # bracket trick: the pkill's own cmdline won't self-match
# regenerate (skips done, drops degenerate lora_null in favor of patched cells, appends Tier-A):
/home/guy/UIOrthoLoRA/.venv/bin/python make_frepro_jobs.py --table math --prefix frm \
    --base_model meta-llama/Llama-2-7b-hf \
    --data_path repro/LLM-Adapters/ft-training_set/metamathqa_395k.json --out jobs/frepro_math.txt
/home/guy/UIOrthoLoRA/.venv/bin/python make_frepro_jobs.py --table cs --prefix frc \
    --base_model meta-llama/Llama-2-7b-hf --baselines 1 --out jobs/frepro_cs.txt
/home/guy/UIOrthoLoRA/.venv/bin/python build_lean.py       # dedup + eval-only conversion -> frepro_lean.txt
grep -c . jobs/frepro_lean.txt                              # sanity: expected remaining count
setsid /home/guy/UIOrthoLoRA/.venv/bin/python gpu_pool.py --gpus 8 --tag frepro4 \
    --jobs jobs/frepro_lean.txt > logs/frepro4_pool.log 2>&1 </dev/null &
grep rc=127 logs/frepro4_pool.log                          # MUST be empty (venv-python gotcha, handoff/00 #10)
```
**Must be patched before this relaunch:** A2 (lora_null calib per R4), and the new-cell generation (C1/C2). A1/A3/B1 should also be in place so the re-run cells are gated. **Fallback if the window is missed:** letting the 12 degenerate lora_null cells run wastes ~72 GPU-h and they must be re-run anyway — so relaunching before line 50 is strictly better. If a clean kill point can't be hit, kill anyway and rely on `build_lean.py` (it re-runs only cells lacking `summary.json`; in-flight cells are cheap to redo).

---

## 5. ZERO-GPU ACTIONS (do now, in parallel, no pool impact)

1. **BBH-only re-report (R6):** recompute math retention as BBH-only from the 7 existing `summary.json` (ranking: LoRA+wd0 30.96 > MiLoRA 30.18 > LoRA 29.14 > CLoRA-k256 28.61 > k64 28.24 > k128 27.55 ≫ PiSSA 7.23). Regenerate any figure/table that used `retention_mean`.
2. **Wording fixes (E1–E4):** INTERESTING_INSIGHTS L37/L75, FINAL_TABLE_PLAN L44; add the Limitations bullet on paper-memory-based faithfulness.
3. **Clone the 4 reference repos** (CLoRA already present in `repro/CLoRA`): `LoRA-Null` (gates R4), `MiLoRA`, `SC-LoRA`, `CorDA` → verify the FAITHFUL verdicts against actual source (critic #6/#10). Priority: LoRA-Null first (it gates the pool action).
4. **PiSSA generation inspection (#8):** read the saved eval artifacts for `frm_pissa_lr3e4_c256_s42` (BBH 7.23 / MMLU-Pro 0.0 → check for empty/degenerate generations, tokenizer/pad issue, or a genuine collapse) before re-running.
5. **Confirm SC-LoRA calibration source** (already verified here: no `nq_open`→wikitext fallback in any log; re-check after the sclora cells at lines 44–49 run).
6. **Run the CPU CorDA++ gate** `validate_cordapp_cpu.py` (no GPU) to confirm the math is sound before wiring A3.
7. **Prepare the regenerated job files** (dry-run `make_frepro_jobs.py` prints target/skipped/remaining) so the restart is instant.

---

## 6. PI DECISION LIST (human-only; each with recommendation + default)

| # | Decision | Recommendation | Default if no answer |
|---|----------|----------------|----------------------|
| **P1** | **Thesis framing.** Strong ("adapter wins are pure LR/magnitude artifacts") vs narrow-honest ("magnitude governs the retention trend; a ~1.8 pp residual method effect exists at matched magnitude; LoRA+wd matches the fancy adapters on math & mid-reg CS, loses at high-k CS"). | **Narrow-honest** — it survives the kill-shots and the within-blob +0.72/+1.8 pp evidence; the strong claim does not. | Narrow-honest |
| **P2** | **Budget / timeline.** Tier A only (~1 day new GPU + ~2 days drain) vs Tier A + selective Tier B (+~2 days) vs full wishlist (~6–8 days). | **Tier A now; add CorDA++ controlled + SC-LoRA β0.8 from Tier B only if a reviewer needs them.** | Tier A only |
| **P3** | **LoRA-Null (R4).** Run-as-is-and-disclose vs patch-calibration-to-repo. | **Match the repo exactly**; if the repo is genuinely degenerate, run as-is and disclose (don't improve a competitor). Clone gates this. | Match repo exactly |
| **P4** | **Pool restart.** Kill+relaunch before line 50 (recommended) vs let 12 degenerate cells run then re-run. | **Kill+relaunch before line 50** after patching (saves ~72 GPU-h). | Kill+relaunch |
| **P5** | **2e-5 concession cells (R3).** Include in body / appendix-with-disclaimer / omit / do full native anchors. | **Appendix-with-disclaimer (Tier B); drop the operating-point defense.** Full native anchors only if a reviewer demands (Tier C). | Omit from body; disclaimer cells in appendix |
| **P6** | **CorDA++ scope.** Controlled-arm only (Tier B, after gates) vs full native (Tier C, needs QA-EM harness). | **Controlled arm only**, contingent on `validate_cordapp_cpu.py` + 1-GPU 0-step gate passing; native = future work. | Controlled arm only |
| **P7** | **MMLU-Pro (R6).** BBH-only primary vs also fix+re-eval the extractor. | **BBH-only primary now**; attempt an extraction fix + re-eval (Tier B) only if time permits. | BBH-only primary |

---

### Bottom line
The critic's audit holds on every checked number: the 7-point magnitude law is leverage-driven (a collapsed PiSSA + an off-LR LoRA+wd0), MMLU-Pro is a broken retention half, no within-harness CLoRA best-LR exists, and neither the CS CLoRA k-grid nor a CLoRA math LR-sweep is queued. None of this is fatal. The path to a defensible submission is: reframe to the honest narrow thesis, re-report retention as BBH-only, add ~35 cells (CS CLoRA k-grid + CLoRA math sweep + fixed LoRA-Null + 3-seed headlines + PiSSA re-run + c512 anchors) at ~1 day of new GPU, wire+gate CorDA++ for a Tier-B arm, and restart the pool before the 12 degenerate lora_null cells run.
