# Experimental Fairness & External-Comparability Audit

*Produced 2026-07-03. Two questions: (1) is the systematic set fair across adapters? (2) can we
compare the original papers' reported numbers against our trained LoRA+wd? Evidence-based; external
numbers web-sourced with URLs; code claims carry file:line.*

---

## Q1 — Is our protocol fair across adapters?

**Verdict: fair for the LAW, not yet for the RANKING — and the authors already know it.**

### Held identical across all 8 arms (the load-bearing controls) — FAIR ✔
- Base model / tokenizer / dtype / target modules (`q,k,v,up,down`) — `train_cs.py:124,134,177`.
- Training data (commonsense_170k, cutoff 256, no packing) — `train_cs.py:85,197`.
- **Same number of optimization updates on the same data** (3 epochs, eff. batch 16, warmup 100); no
  arm overrides — `train_cs.py:127,131-133`.
- **The same 7-LR grid {2e-5…1e-3} applied to every arm** — `make_campaign_jobs.py:30-31`.
- One shared eval harness (few-shot counts, gen_cap=1024, max_len=4096, bbh metric fix) — `eval_one_gpu.py`.
- One shared fdelta probe (7 CS sets, n=100, token-weighted formula) — `fdelta.py:62-103`.
- Residual-save converts PiSSA-family adapters to W0-relative rank-2r so ΔW is measured consistently;
  0-step→ΔW=0 self-check — `residual_save.py:37-64`. (Subtlest correctness point; handled.)

Because the law is analyzed on the ‖ΔW‖ axis, the rank/scaling/wd differences below move a method
*along* the curve, not off it — so **"retention lies on one ‖ΔW‖ curve" rests on a fair comparison.**

### Real asymmetries (ranked by damage; direction = who it favors)
1. **Calibration↔eval mismatch (HIGH).** CorDA/SC-LoRA/LoRA-Null calibrate on nq_open (256 samples,
   `train_cs.py:218,239,260`); retention is BBH/MMLU-Pro (reasoning). Disjoint (no leakage) but
   distribution-mismatched → data-aware inits protect the wrong subspace → **biased AGAINST them, FOR
   LoRA+wd.** This is the asymmetry that plausibly *creates* the "SC-LoRA off-curve" effect.
   → **Acknowledged**; closed by **B4** (eval-matched calibration). Off-curve language embargoed until B4.
2. **Rank/capacity + wd-knob confound (HIGH for ranking).** LoRA+wd is r32 (56.1M params) vs the r16
   arms (28.05M), and is the only arm with the wd knob — `make_campaign_jobs.py:18-26`. Confounds the
   *ranking* claim. → **Acknowledged**; closed by **B5a** (param-matched {r16,r32}×{wd0,wd0.3}).
3. **Scaling asymmetry 2.0 vs 1.0 (MED) — NOT YET ACKNOWLEDGED.** Plain arms use α=2r (scaling 2.0);
   residual arms are forced α=r (scaling 1.0) by the residual conversion — `residual_save.py:59-62`,
   asserts `train_cs.py:213,230,237,257`. So a fixed nominal LR maps to different ΔW growth per family;
   the 7-LR grid is not equally "centered." Largely absorbed by the ‖ΔW‖ reparametrization, but it is
   an unstated axis. → **Recommend one sentence in Setup/Limitations.**
4. **Single seed (MED).** s42 for all — `make_campaign_jobs.py:34`. Fragile for rankings.
   → **Acknowledged**; closed by **B5c** (seeds 43/44).
5. **DoRA fdelta under-measurement (MED) — NOT YET ACKNOWLEDGED; potential correctness issue.**
   `fdelta` calls `get_delta_weight("default")` (`fdelta.py:79`), which for the LoRA `Linear` layer
   returns `B@A * scaling` **only** (`src/peft/tuners/lora/layer.py:930`) and omits DoRA's
   magnitude-vector rescaling (applied in `forward`, not in `get_delta_weight`). DoRA's *retention*
   (y) is correct (eval uses forward), but its *magnitude coordinate* (x) is under-measured → DoRA
   sits too far LEFT on the hero curve and in the ANCOVA. Scope: one of the six on-curve adapters.
   → **Options:** (a) recompute DoRA's true ΔW = m·(W0+BA)/‖W0+BA‖_c − W0 from the saved DoRA
   checkpoints (needs checkpoints + GPU; changes one series' x-values); or (b) add an explicit
   measurement caveat for the DoRA arm. **This one needs a decision.**

**Single most important fairness fix:** B4 (eval-matched calibration) — the only asymmetry that
plausibly manufactures the headline effect; currently biases FOR LoRA+wd.

---

## Q2 — Can we compare the original papers' numbers to our LoRA+wd?

**Verdict: on RETENTION it is a category error for all adapters except CLoRA; on ADAPTATION it is
narrowly valid (GSM8K broadly; CS-8 only via MiLoRA/DoRA), always with a single-LR caveat. CLoRA is
the one clean external anchor on both axes.**

### The decisive structural fact
Our retention axis is **BBH + MMLU-Pro**. Of the competitor papers, **only CLoRA reports these.** The
init/data-aware family reports closed-book QA EM (TriviaQA/NQ/WebQS) or a WikiText loss proxy, or
nothing. So their "retention" numbers **cannot be placed on our axis** — which is precisely why a fair
forgetting comparison *requires* running every adapter ourselves under one harness. (This is a
positive justification for the systematic set, not merely a caveat.)

### Per-paper (all web-sourced)
| Adapter | Llama-2-7B? | CS-8? | GSM8K (MetaMath)? | Forgetting metric | Comparable to our LoRA+wd? |
|---|---|---|---|---|---|
| **CLoRA** (2410.16801) | ✔ | ✔ (LoRA 79.9…k2048 83.7) | ✔ (LoRA 60.6, k128 64.6) | **BBH+MMLU-Pro via lm-eval** (LoRA 26.69/14.46; base 34.91/18.56; k2048 38.67/20.59) + an **L2/wd baseline** + a **𝔽=‖ΔWx‖/‖x‖ table** | **YES, both axes** — the one true anchor |
| **DoRA** (2402.09353) | ✔ | ✔ (LoRA 77.6 / DoRA 79.7@r32, 80.5@r16) | LLaMA-1 arith | **None** | Adaptation: YES. Retention: N/A |
| **MiLoRA** (2406.09044) | ✔ | ✔ (79.2 @r32, lr3e-4 untuned) | ✔ (63.5@r64) | WikiText loss proxy only | Adapt: YES (best init-family match). Retention: NO |
| **CorDA** (2406.05223) | ✔ | ✗ | ✔ (IPM 53.9/KPM 44.6 @r128) | TriviaQA/NQ/WebQS | GSM8K: yes w/ caveat. Retention: NO |
| **SC-LoRA** (2505.23724) | ✔ | ✗ | ✔ (53.5 @r128,β0) | TriviaQA/NQ/WebQS + safety | GSM8K: yes w/ caveat. Retention: NO |
| **PiSSA** (2404.02948) | ✔ | ✗ | ✔ (53.1) | **None** | GSM8K: yes w/ caveat. Retention: impossible |
| **LoRA-Null** (2503.02659) | ✔ | ✗ | ✔ (44.4 @r128) | TriviaQA/NQ/WebQS | GSM8K: yes w/ caveat. Retention: NO |

Cross-cutting caveat on every adaptation "yes": all competitors report a **single fixed LR** (2e-5;
MiLoRA 3e-4) with **no sweep**, vs our best-of-7 — comparing their one point to our swept best is
asymmetric and structurally favors us. (This single-LR practice is itself Exhibit A for our
LR-artifact diagnosis.)

### Scale validation — our LoRA reproduces the canonical anchor ✔
- BoolQ: ours **69.97** vs canonical Llama-2-7B LoRA **69.8** (DoRA repro under Hu-2023 recipe).
- CS-8: ours **79.1** sits between the two published Llama-2-7B LoRA baselines DoRA **77.6** and
  CLoRA **79.9**; LoRA+wd 81.6 above all LoRA baselines, below CLoRA's best k2048 83.7 (expected order).
- Our rank 32 / LR 3e-4 = the exact canonical LLM-Adapters commonsense recipe.
Sources: [LLM-Adapters #64](https://github.com/AGI-Edgerunners/LLM-Adapters/issues/64),
[NVlabs/DoRA](https://github.com/NVlabs/DoRA), [CLoRA arXiv:2410.16801](https://arxiv.org/abs/2410.16801).

### The CLoRA opportunity
CLoRA is not just comparable — it independently corroborates our thesis from a competitor's own data:
(i) it reports forgetting on our exact BBH+MMLU-Pro axis; (ii) it includes an **L2-regularization
(≈ weight-decay) baseline** — the closest published analog to LoRA+wd; (iii) its forgetting proxy
𝔽 = ‖ΔWx‖/‖x‖ **is our fdelta magnitude axis**, and its Table 4 shows 𝔽 tracking forgetting across
methods. Caveat for any number-level line-up: CLoRA's base BBH = 34.91 vs our BBH-AO base = 33.10
(a ~1.8pt harness-config gap), so compare **relative degradation from each paper's own base**, or use
CLoRA as triangulating evidence — do not merge raw numbers 1:1.

---

## Bottom line — are we on the right way?
**Yes.** The systematic set is fair where the *law* needs it, the accuracy scale is validated against
the canonical anchor, and the one place cross-paper comparison is valid (CLoRA) both works and
corroborates us. The known ranking-level asymmetries are already scoped to B4/B5a/B5c. Two new items
to handle: disclose the scaling-asymmetry (one sentence) and decide on the DoRA-fdelta measurement
(recompute vs caveat). And the paper should (a) state explicitly that cross-paper *forgetting*
comparison is impossible for all adapters but CLoRA — the justification for running them all
ourselves — and (b) add the CLoRA published-number cross-check + the scale-validation anchor.
