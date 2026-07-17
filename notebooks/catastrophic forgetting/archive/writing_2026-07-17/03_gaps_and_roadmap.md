# 03 — Gaps & Roadmap: what's still needed to make this referee-proof

Audit date: 2026-07-02. Scope: honest assessment of the head-to-head CF study against the
central thesis (magnitude governs forgetting, geometry does not; LoRA+wd matches/beats fancy
adapters; their "wins" are an LR/magnitude artifact; wake-up call to control magnitude not geometry).

Provenance discipline: [DONE]=measured & mature, [PARTIAL]=data exists but incomplete,
[TODO]=not run, [RISK]=referee attack surface. ‖ΔW‖_F everywhere means the token-weighted
Frobenius norm, field `fdelta_token_weighted`.

---

## (a) DONE and mature — the publishable core (Llama-2, seed 42)

**The magnitude law, 2 domains.** [DONE]
- Commonsense (n=49 = 7 methods × 7 LRs): r(retention, log‖ΔW‖_F) ≈ −0.87 pooled / −0.93 on the
  5 well-behaved methods; slope ≈ −10pp/decade. Math/gsm8k (n=8): r ≈ −0.93. Dual-domain
  generality is the headline that lifts this from "a CS observation" to "a law".
- Axis choice is settled and defensible: ‖ΔW‖_F beats dw_sv_max (σmax is confounded — CorDA's
  spiky spectrum inflates it) and beats LR as a predictor (retention~‖ΔW‖ R²=0.75 vs
  retention~LR R²=0.35). This R² gap IS the "LR is only a proxy" evidence.
- Per-benchmark decomposition: MMLU forgets fastest (−23pp/decade), TruthfulQA ~immune. Adds
  texture and shows the law isn't an artifact of one benchmark.

**LoRA+wd wins the Pareto plane.** [DONE, with caveats below]
- LoRA+wd0.3 bounds ‖ΔW‖_F ≈ 72 at lr1e-3 (vs 200–1395 for the structured arms) → adapts to
  CS ≈ 81 AND retains ≈ 34. Main table (`paper/table_main_cs.tex`): LoRA+wd 81.6/25.6 leads;
  well-behaved cluster (MiLoRA, LoRA, CLoRA, DoRA) within ~1pp retention.
- Ports are audited faithful (Phase-0 audit): CorDA-KPA, MiLoRA, CLoRA, SC-LoRA, LoRA-Null all
  verified vs reference repos; residual-init save-conversion gated by a 0-step ΔW→0 check. This
  "we proved our port is faithful before claiming a method fails" discipline is a real referee
  shield — keep it prominent in the methods section.

Bottom line: the LAW is mature and defensible on Llama-2. The "LoRA+wd wins" and "fancy wins are
LR" claims are drafted but NOT yet referee-proof — see (c).

---

## (b) IN PROGRESS — Qwen2.5-7B replication + Llama-2 math fill-in

**Qwen 2×2.** [PARTIAL] Target = 2 domains × 8 arms × 7 LRs = 112 Qwen cells. Done as of audit:
- Qwen-CS (qwsw): 8/56 cells (7 lora LRs + 1 corda). Qwen-math (qwswm): 5/56 cells (lora only).
- Total Qwen ≈ **13/112** cells. So far the law replicates on the thin slice we have:
  Qwen-CS r(retention, log‖ΔW‖) = −0.92 (matches Llama). Math sweep is barely started.
- One combined 8-GPU pool (`camp4`, jobs/combined.txt, 155 jobs) is live and resumable; 14 DONE
  in ~24h; Qwen cells run 6–10h each (slower than Llama). Four Qwen-specific eval bugs are now
  fixed and verified no-op on Llama (pad token, BBH fewshot normalization, gen_cap=1024,
  max_len=4096) — so the 56 L2-CS + 14 L2-math cells stay valid.
- **ETA: ~6–8 days** to drain the current combined pool (Qwen CS+math + L2-math gaps), single
  scheduler. This is the critical-path clock for a complete 2-model result.

**Llama-2 math fill-in.** [PARTIAL] lrswm has only lora (7) + lorawd (6) + dora (1) = 14/56; the
5 structured arms (corda/clora/milora/sclora/lora_null) × 7 LRs are still missing and are in the
combined pool. The math-domain law currently rests on n=8 — thin; the fill-in is what makes math
a genuine second domain rather than an anecdote.

Honesty framing for the paper: present Qwen as **supporting/in-progress replication**, not a
completed second model. The law-replicates-on-Qwen claim is currently ~13 cells and mostly the
easy (LoRA) arm — do NOT present a Qwen Pareto/method-ranking table until the structured arms land.

---

## (c) REQUIRED for the strong claim to survive review

The LAW survives on Llama-2 today. The three *aggressive* claims — (i) LoRA+wd surpasses fancy
adapters, (ii) their wins are an LR artifact, (iii) it's a field-wide wake-up call — each has a
live referee attack surface. In rough load-bearing order:

### C1. The adapt–retention Pareto figure + the fixed-LR-vs-swept-LR "gotcha" exhibit [PARTIAL→TODO] — LOAD-BEARING
This is the single exhibit that carries claims (i) and (ii). Two panels:
- **Pareto panel:** adaptation (CS-8 or gsm8k) vs retention (core), all 8 arms, Pareto frontier
  drawn; show LoRA+wd on or above the frontier. (fig3/fig1 drafts exist in figs_v2.)
- **The gotcha panel (BUILD THIS EXPLICITLY):** for each fancy adapter, plot its *single-best-LR*
  point (what its paper would report) vs the *full LR sweep* collapsing onto the shared magnitude
  curve. The visual "at one LR it looks better; sweep the LR and it falls onto the same line" IS
  the wake-up-call argument. Without it, claim (iii) is an assertion.
- [RISK] A referee will ask "did you tune LoRA+wd's LR as hard as you tuned theirs?" — the sweep
  is symmetric (7 LRs per method), so answer yes, but the figure must make the symmetry obvious.

### C2. Calibration↔eval-distribution fairness pass for data-aware adapters [TODO] — LOAD-BEARING, NOT OPTIONAL
This is the biggest threat to intellectual honesty and the most likely referee kill-shot.
- The current "SC-LoRA (−3.3pp) and CorDA (−3.0pp) forget MORE than their ‖ΔW‖ budget predicts"
  (ANCOVA p<0.001) is CONFOUNDED. CorDA/SC-LoRA/LoRA-Null are calibrated on nq_open (factoid QA),
  but retention eval is BBH/MMLU/MMLU-Pro/ARC/TruthfulQA (academic/reasoning). KPM only protects
  directions the calibration covariance exercises → nq_open protects the WRONG subspace for our
  eval → "data-aware inits forget more" may just be "we handicapped them with a mismatched
  calibration set." CorDA was ALSO earlier mis-calibrated on wikitext (now fixed to nq_open); the
  nq_open runs are re-running.
- REQUIRED: re-run all calibration-using arms with **eval-distribution-matched** calibration
  (MMLU/ARC auxiliary-train, disjoint from test), 256 samples, shared across arms; PLUS a
  sensitivity arm (nq_open vs eval-matched) to quantify the effect. See handoff/14 §8 Fix 1.
- [RISK] If eval-matched calibration moves CorDA/SC-LoRA back ONTO the curve, the honest paper
  says "the law is method-free once calibration is fair" — a CLEANER result. If they stay off,
  we have a real second-order effect. Either way we must run it; we CANNOT publish "LoRA beats
  CorDA" while CorDA is calibrated on the wrong distribution. Until this lands, all off-curve /
  "fancy adapters forget more" language stays out.
- Also required here: PEFT-CorDA residual round-trip via `path_initial_model_for_weight_conversion`
  and the init-output-invariance check run AFTER reload, not in-memory (handoff/14 Fix 2/3) — the
  in-memory check passed once on a reloaded-corrupt model already (the residual_save bug class).

### C3. Param-matched LoRA+wd control [TODO] — LOAD-BEARING for claim (i)
- [RISK] The arms MIX ranks: LoRA/DoRA/CorDA r16; MiLoRA/SC-LoRA r32; CLoRA k1024. And ONLY LoRA
  has the wd knob. A referee will say "LoRA+wd wins because it has 2× the rank of some arms and an
  extra regularizer no one else got." This is the fairness gap the LAW framing was chosen to
  sidestep — but the moment we headline "LoRA+wd surpasses fancy adapters," the gap is back.
- REQUIRED minimum: a LoRA+wd r32 control (param-matched to the r32 arms) AND a LoRA+wd r16
  control, so the win is shown at matched capacity. Stronger: give every method the wd knob at
  ≥2 wd values (the deferred "fairness experiment") — at least for the 2–3 arms nearest the
  frontier, so the claim is "wd helps everyone, and geometry adds nothing on top."
- If we do NOT run this, retreat claim (i) to "LoRA+wd lands on the same frontier as fancy
  adapters at far lower engineering cost" — still a real result, less attackable.

### C4. Seeds 43/44 where a result is interesting [TODO]
- Single seed (s42) is fine for the LAW (49 points trace the curve). It is NOT fine for any
  head-to-head *ranking* claim: the 3-seed mtx_ matrix already exposed single-seed collapse basins
  (seed 44 fragile; wd0.5, odd ranks collapse on one seed). REQUIRED for: the Pareto winner claim,
  the off-curve (C2) verdict, and any per-method delta we headline. Add 43/44 only to the ~6–8
  cells that appear in a headline table/figure, not the full grid (cost control).
- [RISK] "n=1 seed for a method comparison" is an easy desk-reject line. Error bars on the
  headline points are cheap insurance.

### C5. Base-ceiling calibration for MMLU / ARC / TruthfulQA [TODO]
- Retention numbers are uninterpretable without the base-model (0-adapter) score per benchmark to
  define "no forgetting = 100% retained." BBH/MMLU-Pro have it; MMLU/ARC/TruthfulQA do not.
- Cheap (5 eval-only runs, no training). Do it before any camera-ready retention percentage.

### C6. CorDA++ as the advanced arm [TODO — post-2×2]
- handoff/14: CorDA++ (arXiv:2506.13187) = dynamic covariance selection + dynamic rank allocation
  under a fixed param budget, built as Path C on PEFT static CorDA. Algorithms are now transcribed
  (compactness π(C)=√(d_out·σmax)/σmin; Eq 7–10). This is the "we didn't strawman the SOTA" arm:
  it makes the paper's negative result about geometry land against the *strongest* geometric method,
  not a weak one.
- [RISK] Without CorDA++, a referee says "you beat 2023-era adapters; the 2025 dynamic ones escape
  your law." CorDA++ closes that. Note dynamic rank breaks nominal param parity → must report
  REALIZED trainable-param count and match to our r16-equivalent budget (28,049,408 params).
- STILL OPEN before building: candidate-pool size N (paper appendix) + PEFT `covariance_file`
  load-if-exists and `rank_pattern` bottom-r slicing behaviors (handoff/14 Fix 3).

---

## (d) Prioritized critical path: now → submittable

Costs are rough GPU-days on the single 8-GPU B200 box (one scheduler rule). "days" = wall-clock.

| # | Item | Depends on | Cost | Why it's here |
|---|------|-----------|------|---------------|
| 1 | Drain combined pool: finish Qwen CS+math + L2-math structured arms | live | ~6–8 d (running) | completes the 2×2; unblocks every 2-model figure |
| 2 | Build the Pareto + fixed-vs-swept-LR gotcha exhibit (C1) | Llama data (have it) | ~0.5 d compute + fig work | carries claims (i)+(ii); can start NOW on Llama |
| 3 | Base-ceiling calibration MMLU/ARC/TQA (C5) | none | ~0.5 d (eval-only) | makes retention numbers interpretable; do in parallel |
| 4 | Eval-matched calibration re-run, all data-aware arms + sensitivity arm (C2) | #1 scheduler free | ~2–3 d (5 arms × 7 LR, CS first) | LOAD-BEARING fairness; gates all off-curve language |
| 5 | Param-matched LoRA+wd controls (r16 & r32) + wd knob on 2–3 frontier arms (C3) | #1 scheduler free | ~2 d | makes "LoRA+wd wins" fair |
| 6 | Seeds 43/44 on the ~6–8 headline cells (C4) | #2/#4 identify which | ~1.5 d | error bars; kills n=1 desk-reject |
| 7 | CorDA++ arm + ablations (C6) | #1, algorithms ready | ~3–4 d | beats the strongest geometry, not a strawman |

**Two submittable tiers:**
- **Minimum defensible ("the law" paper):** items 1–4 + 6-lite. ≈ 2 weeks wall-clock from now.
  Headline = the dual-domain, two-model magnitude law + LoRA+wd on the frontier, WITH the fairness
  pass so no off-curve overclaim. Aggressive method-ranking softened.
- **Strong ("wake-up call") paper:** all of 1–7. ≈ 3.5–4 weeks. Supports the full "their wins are
  LR, control magnitude not geometry, and it holds against CorDA++" thesis.

**Sequencing note:** items 2 and 3 need no GPU scheduler slot and should start immediately on the
mature Llama data while the combined pool (item 1) runs. Items 4/5/7 contend for the single
scheduler — order them 4 → 5 → 7 (fairness before controls before the advanced arm), because 4
can retroactively invalidate off-curve claims that 5 and 7 would otherwise be built to explain.

---

## Referee attack surfaces — the honest list (memorize these)
1. **Near-circularity:** "retention falls with ‖ΔW‖ is true by construction." Rebuttal: the
   non-trivial claim is that method identity adds ~nothing beyond ‖ΔW‖ (residual/ANCOVA test) and
   that direction/geometry doesn't modulate the slope. Lead with the residual test, not the raw r.
2. **Unfair calibration (C2):** biggest kill-shot; nq_open vs academic eval. MUST fix.
3. **Rank + wd asymmetry (C3):** "LoRA+wd got extra capacity + an extra knob." MUST control.
4. **Single seed (C4):** collapse basins already observed; n=1 ranking is fragile.
5. **Strawman SOTA (C6):** only beat old adapters. CorDA++ closes it.
6. **"Known result":** CLoRA already links magnitude to CF. Our delta = it's *method-free* across
   8 adapters × 2 domains × 2 models, that geometry is causally inert, and that the field's
   per-method wins are an LR artifact — framed as a controlled/negative wake-up call, not a
   discovery that magnitude matters.

---

## 10-line summary
1. The magnitude law is DONE and mature on Llama-2: retention ~ log‖ΔW‖_F, r≈−0.87 (−0.93 clean),
   dual-domain (CS + math), ‖ΔW‖ beats LR as predictor (R² 0.75 vs 0.35).
2. LoRA+wd0.3 sits on/above the Pareto frontier (adapts ~81 CS, retains ~34) — drafted, not yet fair.
3. Qwen replication is IN PROGRESS: ~13/112 cells; law replicates on the thin slice (Qwen-CS r=−0.92);
   ETA ~6–8 days to drain the live combined pool. Present as supporting, not complete.
4. LOAD-BEARING #1: build the Pareto + fixed-LR-vs-swept-LR "gotcha" exhibit — it carries the
   "their wins are an LR artifact" claim; can start now on Llama data.
5. LOAD-BEARING #2: eval-matched calibration re-run for CorDA/SC-LoRA/LoRA-Null — the current
   off-curve finding is confounded by nq_open↔academic-eval mismatch; NOT optional; gates all
   "fancy adapters forget more / LoRA beats CorDA" language.
6. LOAD-BEARING #3: param-matched LoRA+wd control (arms mix r16/r32; only LoRA has wd) — needed
   before headlining "LoRA+wd surpasses fancy adapters."
7. Seeds 43/44 on the ~6–8 headline cells (single-seed collapse basins already observed) + base-
   ceiling calibration for MMLU/ARC/TruthfulQA (cheap, eval-only).
8. CorDA++ (handoff/14, algorithms transcribed) is the advanced arm that stops "you strawmanned SOTA."
9. Critical path: start figs (#2) + base ceilings (#3) NOW on Llama; then single-scheduler order
   fairness(4)→controls(5)→CorDA++(7); seeds(6) on headline cells only.
10. Two tiers: minimum-defensible "law" paper ≈2 weeks (items 1–4,6-lite, softened ranking);
    strong "wake-up call" paper ≈3.5–4 weeks (all items). Top attack surfaces: unfair calibration,
    rank/wd asymmetry, single seed, strawman SOTA, near-circularity.
