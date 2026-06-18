# 12 — STATE DUMP (resume here) — 2026-06-17 ~10:30

## ONE-LINE STATUS
The bold "LoRA+weight-decay DOMINATES CLoRA" claim **deflated at full scale to a TIE**; the "geometry
doesn't matter, only magnitude" claim is **NOT supported** (universal curve doesn't collapse). Live
question: does simple LoRA+wd **MATCH** the *data-aware* methods (SC-LoRA, CorDA++) too? → needs faithful
ports. Clean-wd full-scale reruns running (fixing a data-contamination bug). Two gates open.

## WHERE WE LANDED (honest findings ledger)
| finding | status | evidence |
|---|---|---|
| retention ∝ ‖ΔW‖_F (magnitude) WITHIN a structure | SOLID | corr −0.96..−0.98 (CLoRA, UIO, LoRA); rank sweep ret 25.4→8.5 as r 4→256 |
| weight-SVD DIRECTION (μ_E) predicts retention | REFUTED | r=−0.09 (D1) — but this is the WEIGHT basis (wrong basis per CorDA/SC-LoRA) |
| "rank mitigates CF" surprise | REFUTED (LoRA) | higher rank → MORE forgetting via larger ‖ΔW‖ (conventional); no diffusion |
| LoRA+wd ≥ CLoRA | **GATE-1 PASS (clean full-scale)** | CLEAN reruns: wd0.1 80.4/24.86 TIES CLoRA-k1024 79.8/24.85; wd1.0 76.7/**26.87** DOMINATES CLoRA-k2048 65.4/25.7; DoRA-r8 79.8/**25.38** beats both at CS80. (earlier "24.42 tie" was the CONTAMINATED checkpoint.) CAVEAT: wd-frontier NOISY/non-monotone (single seed; wd0.3 anomalous) → SEEDS needed before "beats". ret>base partly real transfer (BBH 35.3>33.1). |
| CorDA protocol omits weight-decay baseline | **VERIFIED (GATE-2 win)** | CorDA official scripts: wd=0 across CorDA/LoRA/full-FT, NO ‖BA‖_F-matched baseline → their retention gain CONFOUNDED with magnitude. The decisive test (CorDA-KPA vs LoRA+wd at matched ‖ΔW‖_F) was NEVER run by them. |
| "geometry irrelevant, only F-norm" (universal collapse) | NOT SUPPORTED | universal-curve: methods spread 3–5 at matched F (LoRA+wd/UIO > CLoRA), ≈ fast-noise; full-scale needed |
| directional norm ‖ΔW·C_retain^½‖ beats raw ‖ΔW‖_F | MARGINAL | −0.79 vs −0.77 (n=8) — within noise |
| DoRA beats the frontier | PENDING | dora_r8/16/32/64 landing on GPU0 |
| UIO corrected best Pareto | DONE | k410_v205_dE1 71/25.7 ; k410_v410_dE0 69/26.3 ; ties CLoRA, CS-ceiling ~71-74 < CLoRA 80 |

**Live defensible claim = "simple LoRA+wd MATCHES CLoRA" (strong-baseline / 'do we need this machinery?').**
NOT "dominates", NOT "geometry irrelevant". Extending to data-aware methods (SC-LoRA/CorDA) is the
decisive next test (they could TIE → magnitude suffices, or BEAT → data-basis geometry genuinely helps).

## RUNNING NOW (detached pools; check with `grep -c DONE logs/<tag>_pool.log`)
- GPU 1,2,3 — `wdclean`: clean retrains lora_wd{0p3,0p5,1p0}_clean (UNIQUE names) + FULL-scale eval + forensics. Resolves GATE 1 high-wd region + the contamination.
- GPU 4,6,7 — `wdcleanlo`: clean lora_wd{0p01,0p05,0p1}_clean + FULL eval. Completes the clean full-scale wd frontier.
- GPU 0 — `dora`: 3 full-scale wd confirms (done) → DoRA r8/16/32/64 (train+eval+forensics+databasis).
- GPU 5 — `frontier`: half-rotation UIO gap-fill (fr_k*_*), confirms UIO ceiling.
- Analysis scripts: `analyze_headline.py`, `universal_curve.py` (the collapse test), `analyze_magnitude_law.py` (join FIXED via _norm), `analyze_d1_d2.py`.

## ⚠️ DATA-INTEGRITY BUG (caused the false-positive)
Duplicate run_names wrote to the SAME checkpoint path: GPU0 `magctl` AND the (killed) `phase2` pool both
trained `lora_wd0p3` → `/scratch/cf_models/lora_wd0p3` → overwrite/corruption. Symptom: `lora_wd0p3_full`
CS=52 vs fast CS=80.7 (same "checkpoint"). ⇒ wd0.3/1.0 FULL numbers UNTRUSTWORTHY. FIX = the `*_clean`
reruns (unique names). LESSON: never reuse run_name/checkpoint paths across pools.

## THE TWO GATES (must close before the big campaign)
- **GATE 1 (premise):** does LoRA+wd ≥ well-reproduced CLoRA at full scale? Clean points so far say TIE
  (not dominate). The `*_clean` reruns (running) give the trustworthy verdict, esp. high-wd.
- **GATE 2 (novelty + fidelity) — needs web/source:** 
  - SC-LoRA: arXiv 2505.23724 ; openreview KAE9YDK0t8. Recipe (partial, VERIFY input-vs-output cov):
    B_init=Q_r, A_init=Q_rᵀW0, residual=W0−Q_rQ_rᵀW0; Q_r=top-r eigvecs of (1−β)Cov₊ − β·Cov₋ over
    fine-tuning (D+) vs knowledge (D−) activations; β swept. (Repo not yet located.)
  - CorDA: **github.com/iboing/CorDA** (NeurIPS'24), arXiv 2406.05223. KPA mode = SVD of (W·Cov_input)
    over a QA/knowledge calib set; init adapter from the **SMALLEST-r** components, freeze the rest.
  - **CRITICAL LIT-CHECK:** do SC-LoRA / CorDA(++) / CLoRA / OPLoRA papers include a TUNED weight-decay
    (or dropout) LoRA baseline? If they OMIT it → "field missed the obvious baseline" = the story. If
    included & wd lost → our "matches" claim is in trouble. NOT yet resolved.
  - (WebFetch of these was interrupted; re-fetch to extract exact init code before porting.)

## KEY NUMBERS (full-scale unless noted)
- BASE (no FT) retention = **26.0** (BBH-AO 33.1 + MMLU-Pro 19.0) = the ceiling.
- CLoRA full: k1024 **79.8/24.85**, k2048 65.4/25.7. LoRA(wd0) ~CS79/ret21.7.
- LoRA+wd full CLEAN: wd0.05 79.5/24.1, wd0.1 80.2/24.4. (wd0.3/1.0 contaminated → reruns.)
- Rank sweep (fast): r4 77/25.4 (F19) … r256 58/8.5 (F129).
- Oracle (perfect input-gate) = (CS79, ret26) — full adapt + zero forgetting (motivated doc 10 gate).

## NEW CODE (this session)
- `forensics.py` (weight-basis spectral UᵀΔWV), `forensics_databasis.py` (DATA-basis; --cov_source
  retain[MMLU-Pro]/task; data_resp=‖ΔW·C^½‖²), `universal_curve.py` (the collapse test),
  `norm_trace.py` (per-step ‖ΔW‖_F+loss callback, wired into train_cs+uio_inprocess), `leakage.py`
  (full-ΔW preserved metrics + grad-enabled λ penalty), `data_aware_init.py` (SC-LoRA/CorDA scaffolding —
  ⚠️ INJECTION BUG: toy reconstruction err 0.2 not ~0, FIX before use), `analyze_*.py`.
- train_cs.py: added --weight_decay, --use_dora, norm_trace. uio_inprocess.py: drop_major, λ penalty,
  in-process forensics, norm_trace.

## HOW TO RESUME
1. `cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"` ; venv=/home/guy/UIOrthoLoRA/.venv/bin/python
2. Check pools: `for t in wdclean wdcleanlo dora frontier; do echo $t $(grep -c DONE logs/${t}_pool.log); done`
3. When `*_clean` land: `python analyze_headline.py` + `python universal_curve.py` → GATE 1 verdict (tie vs dominate, full-scale, contamination-free).
4. GATE 2: fetch SC-LoRA (2505.23724) + CorDA (github.com/iboing/CorDA) source for exact init + the wd-baseline lit-check.
5. If GATE 1 ties + GATE 2 novel: fix data_aware_init.py injection, port SC-LoRA+CorDA faithfully, run full-scale, overlay on universal curve.

## OPEN DECISIONS (for the user)
- Plant which flag: "LoRA+wd MATCHES the forgetting adapters (incl. data-aware)" [strong-baseline] vs
  the failed "dominates"/"geometry irrelevant". Likely the strong-baseline, pending SC-LoRA/CorDA.
- Worth the multi-method port (SC-LoRA/CorDA/MiLoRA/OPLoRA) or wrap as the magnitude-measurement +
  strong-baseline note? Decision after GATE 1 (clean) + GATE 2 (lit-check).
- Honest risk: data-aware methods may genuinely BEAT wd (data-basis matters) → then the claim is the
  TAXONOMY ("random/weight-basis methods = magnitude-in-disguise; data-aware = real"), not "all magnitude".
