# 11 — THE WEIGHT-DECAY FINDING: "the field missed the obvious magnitude baseline" — VALIDATION PLAN

Status: PRELIMINARY + EXCITING (2026-06-16). The #3 falsifier returned a THIRD, un-pre-registered
outcome: plain **LoRA + weight-decay appears to DOMINATE CLoRA** on the adaptation–retention frontier.

## The claim (calibrated — do NOT overstate)
DEFENSIBLE: weight-decay on the LoRA adapter **massively improves retention while MAINTAINING adaptation
(~CS 80)**, dominating CLoRA's subspace regularization — because forgetting is governed by ‖ΔW‖, which
weight decay controls **directly and efficiently**, while CLoRA controls it indirectly (and worse).
NOT YET DEFENSIBLE: "improves adaptation" (CS 79→80.7 is within fast-eval ±1 noise) and the absolute
numbers (ret 28.4 > base 26.0 is fast-eval noise, stderr ≈ ±4 at 64 samples).

Preliminary fast-scale (±4 noise): wd 0/0.05/0.1/0.3 → ‖ΔW‖_F 40.8/33.5/28.0/15.3, ret 23.4/24.4/26.6/28.4,
CS ~79–80.7. At matched ‖ΔW‖≈28: LoRA+wd ret 26.6 vs CLoRA-k256 22.7. vs CLoRA-k1024 (79.9/25.6): LoRA+wd
0.1 (80.2/26.6) dominates.

## WHY this matters / the headline framing
If a properly-tuned weight-decay LoRA beats CLoRA (ACL'25) — and OPLoRA, CorDA, SC-LoRA all build elaborate
subspace machinery — then **the field has been missing the simplest baseline.** That's an
"emperor-has-no-clothes" result, directly explained by our magnitude law.

## ⚠️ THE NOVELTY CRUX (do this FIRST — it's make-or-break)
**Did CLoRA / OPLoRA / CorDA / SC-LoRA include a TUNED weight-decay (L2-on-adapter) baseline?**
- If NO (they omitted the obvious baseline) → genuine "the field missed it" finding. STRONG.
- If YES and wd lost in their setup → we must explain the discrepancy (their wd untuned? different
  setting? our CLoRA weak?) before claiming anything.
Weight decay is SO obvious that its omission would itself be the story. LIT-CHECK their baseline tables.

## ⚠️ FAIRNESS CAVEAT (our CLoRA repro)
Our CLoRA-k2048 over-constrains (CS 65 vs paper's ~83.7) — a weak point of our repro. So "beat CLoRA"
must be argued against (a) our WELL-reproduced CLoRA points (k1024 = 79.9/25.6, a faithful repro) AND
(b) CLoRA's PAPER-reported numbers, not just our weak k2048. Don't cherry-pick our worst CLoRA point.

## VALIDATION EXPERIMENTS (full-scale, rigorous)
1. **FULL-scale retention (ret_limit 0) on the wd sweep** — eval the saved wd checkpoints (0.01/0.05/0.1/
   0.3/1.0) on FULL BBH+MMLU-Pro. CONFIRM/KILL the dominance vs CLoRA's existing FULL numbers. [ASAP, eval-only]
2. **Full wd sweep + finer grid** — add wd ∈ {0.5, 0.7, 2.0} if 0.3 is the sweet spot; trace the full
   LoRA+wd frontier.
3. **SEEDS (≥3)** on the 2-3 headline wd points — error bars so "dominates" isn't one lucky seed.
4. **Mechanism plot** — (‖ΔW‖_F → retention) and (‖ΔW‖_F → CS): show wd reduces ‖ΔW‖ WITHOUT cutting CS,
   while CLoRA's constraint cuts CS (k2048→65). The "why wd wins" figure.
5. **Magnitude-knob ablation** — is it specifically weight decay, or any magnitude control? Compare
   wd vs (smaller LR) vs (smaller rank, from the rank sweep) vs (explicit ‖ΔW‖ penalty). If they ALL
   trace the same frontier → "magnitude control (any form) beats subspace methods" (stronger, cleaner).
6. **Other-method overlay** — place DoRA, MiLoRA on the same frontier; does wd dominate the whole family?
7. **GENERALIZATION (for the strong claim)** — ≥1 more setting: 2nd model (Llama-3-8B) or 2nd task. Needed
   before "the field is doing it wrong" — a single-setting result is a curiosity.

## Pre-registration (predicted + both branches)
- PREDICT: full-scale confirms LoRA+wd dominates CLoRA's frontier (higher ret at matched CS / matched ‖ΔW‖).
- CONFIRMED → the constructive headline: "regularize magnitude directly; subspace machinery is unnecessary
  and inferior." Pair with the magnitude law (mechanism) + the lit-gap (novelty).
- REFUTED (full-scale wd ≈ or < CLoRA) → fast-eval inflated it; fall back to "magnitude governs" measurement
  paper. Kill the strong claim cleanly.

## Kill criteria
- Full-scale wd does NOT beat well-reproduced CLoRA (k1024) → no dominance claim.
- Lit-check shows the field already reported tuned-wd baselines beating/comparable → not novel.
- Doesn't generalize to a 2nd setting → scope down to "in this setting" / curiosity.

## ⚠️ UNIVERSAL-CURVE TEST (2026-06-16, fast-scale preview) — TEMPERS the bold claim
The bold claim "geometry doesn't matter, only ‖ΔW‖_F" predicts ALL adapters COLLAPSE onto one
retention-vs-F curve. Preview over 36 points (CLoRA/LoRA-rank/LoRA+wd/UIO-corr): at matched F, retention
SPREADS 3–5.6 pts, systematically (LoRA+wd & UIO-corr ABOVE CLoRA). That is NOT a collapse → HINTS
geometry DOES matter → bold claim likely FALSE. BUT the spread ≈ fast-eval noise (±4), so UNRESOLVED until
full-scale (noise ~±1.5). 
**CLAIM DISCIPLINE:** lead with the DEFENSIBLE claim — "simple LoRA+wd ≥ elaborate forgetting adapters"
(strong-baselines). Do NOT lead with "geometry irrelevant" — the data leans AGAINST it. The universal
collapse is an ADDITIONAL claim, only if full-scale shows the spread vanish. Two SEPARATE axes:
  - retention vs F  : universal? (tests "geometry irrelevant for forgetting")
  - CS vs F         : differs by method (adaptation EFFICIENCY) — LoRA+wd best (CS80.7@F15), UIO worst.

## ═══ FULL CAMPAIGN (2026-06-16): "Are forgetting-mitigation adapters necessary vs regularized LoRA?" ═══
Goal: a fair, full-scale, multi-axis comparison showing simple **LoRA+weight-decay** matches/dominates the
elaborate forgetting adapters on the adaptation–retention frontier. If it holds + is novel → a strong
"rethinking strong baselines" paper.

### TWO GATES (enforce BEFORE committing the full compute)
- **GATE 1 — premise (running now):** full-scale LoRA+wd vs well-reproduced CLoRA. If full-scale does NOT
  show LoRA+wd ≥ CLoRA → fast-eval inflated it; STOP the campaign, fall back to measurement paper.
- **GATE 2 — novelty (lit-check, do ASAP):** has anyone shown TUNED regularized-LoRA (wd / dropout) is a
  strong baseline that matches CLoRA/CorDA/DoRA/OPLoRA for FORGETTING? The "PEFT baselines are underrated"
  genre EXISTS — must confirm this specific forgetting comparison is unclaimed. If claimed → not novel.

### Comparison matrix (full-scale retention = FULL BBH+MMLU-Pro; CS = 8-task; report ‖ΔW‖_F + F∆)
Protagonist — **LoRA+wd**: rank ∈ {8,16,32,64} × wd ∈ {0,0.05,0.1,0.3,1.0} (+ LR sensitivity {3e-4,1e-3}
at rank32). Traces LoRA+wd's full frontier + rank×wd interaction.
Competitors (each TUNED, not strawman; main HP swept):
  - LoRA (wd=0) — forgetful baseline ✓        - CLoRA (k128..2048) ✓ + PAPER numbers (our k2048 weak)
  - DoRA (matched ranks) — running             - UIOrthoLoRA-corrected ✓ (best ~71/26)
  - MiLoRA (minor-SV init) — PORT+run          - [stretch] PiSSA / LoRA-Null / SC-LoRA / OPLoRA
Seeds ≥3 on headline points (error bars on "dominates"). GENERALIZATION (for the strong claim): ≥1 more
setting — Llama-3-8B and/or a math task (harder task = bigger ‖ΔW‖ needed → does wd still win, or hurt
adaptation? this is where the claim could break — important regime test).

### Mechanism (the "why" figure)
Overlay ALL methods on (‖ΔW‖_F → retention): do they collapse onto ONE magnitude curve? LoRA+wd reduces
‖ΔW‖ WITHOUT cutting CS; CLoRA's constraint cuts CS (k2048→65). Ties the result to the magnitude law (08).

### Sequencing (gated)
- A [running]: full-scale LoRA+wd vs CLoRA (GATE 1).
- B [queued, training proceeds in parallel]: LoRA+wd rank×wd grid (full-scale).
- C [ONLY if GATE 1 ✓ and GATE 2 ✓]: port MiLoRA(+others), seeds, generalization (2nd model/task).
HONEST: a math/hard-task generalization is the make-or-break for the strong claim — commonsense is easy
(small ‖ΔW‖ suffices), so wd looks free here; a task needing large ‖ΔW‖ may show wd trading CS for ret.

## Relation to other docs
- DEPRIORITIZES 10 (gated adapter): if wd already breaks the frontier, the gate must beat wd-LoRA (higher
  bar) — keep as "can we do even better" / future work, not the lead.
- This is the constructive payoff of the magnitude finding (08/09). 06/07/08/09 unchanged.
