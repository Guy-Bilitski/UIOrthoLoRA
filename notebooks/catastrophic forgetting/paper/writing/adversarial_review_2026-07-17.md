# ADVERSARIAL EVIDENCE-BASE REVIEW — 2026-07-17 (post-addendum panel)

`[Independent hostile-reviewer pass over PAPER_BLUEPRINT.md, analysis_final/01–07,
key_numbers §18–§19, EXPERIMENTAL_FAIRNESS_AUDIT.md, baseline_fidelity_audit.md.
Run blind to the same-day supervisor memo. Reconciled verdict: analysis_final/
08_thesis_memo_2026-07-17.md. Bar: NeurIPS/ICLR AC. CLoRA-faithful constraint
honored — no attack rests on CLoRA's numbers being wrong. A4 was RESOLVED
same-day (06 §5 commonality split); other dispositions in 08 §4.]`

## 1. ATTACKS (ranked by severity)

### SEVERITY-HIGH

**A1 — Causal-ordering: own mediation says drift, not magnitude, is proximal.**
C1 is labeled "mechanism" but 05 §2 + dyn2_mediation.txt: controlling KL, the direct
magnitude→retention path collapses in 4/6 families (lrswm −0.14, qwsw −0.08, qwswm
+0.05; survives lrsw −0.63, frc −0.68). Textbook "magnitude → behavioral drift →
forgetting, drift proximal." E1 does not rescue "mechanism" — scalar rescaling moves
F_Δ, spec_max, AND KL together. Partially answered (05 flags it; Ladder B: KL adds
+0.005 after F_Δ at run level). NOT answered for "mechanism". Fix: relabel C1
"first-order predictor / control variable"; state drift as proximal channel.
Writing-fixable, mandatory.

**A2 — Interventional keystone is n=15(+9), one model, one task.**
E1 is Llama-CS only. No interventional evidence on Qwen, math, or full-FT.
UNANSWERED for generalization. Needed: ≥1 rescale ladder + random-direction control
on Qwen-CS and a math arm. Absent that, scope the causal claim to Llama-CS
explicitly in the abstract.

**A3 — "6 model×task families" double-counts Llama.**
2 architectures × 2 task types = 4 settings; frc/frm are recipe variants of
lrsw/lrswm. As-worded false. Fix: "4 model×task settings under 6 training recipes
across 2 architectures."

**A4 — Ladder ΔR² order-dependence; F_Δ↔geometry collinearity (r(F_Δ,spec_max)=0.92).**
"+0.017" is a last-entered floor. Sub-attacks: restriction-of-range (not fatal —
geometry spreads widely, dyn4 G1); metrics-scale-with-F_Δ (true only for spec_max;
e_top/stable_rank are scale-invariant); collinearity (real for spec_max only).
Answered by standardized betas + seed-stability; was missing a commonality/Shapley
split. → **RESOLVED same-day, 06 §5:** shape-only geometry: unique(magnitude)
+0.296 vs unique(shape) +0.016 (18×), shared +0.099, order-independent; spec_max
reclassified as a magnitude measure (r=+0.931) and dropped from the geometry ledger.

**A5 — Geometry-relevant methods are the ones not cleanly assessed.**
CorDA excluded (own port bug — wikitext vs nq_open calibration, fidelity audit
Finding 1); SC-LoRA is the one deviator, rescued by E4 (n=20). Clean on-curve set is
essentially the LoRA-variants. NOT answered for CorDA. Fix: scope sentence —
geometry verdict established on LoRA-variant family, provisional for data-aware
inits (SC-LoRA via E4, CorDA withheld, no CorDA number cited).

### SEVERITY-MEDIUM

**A6 — CE "corroboration" is re-description.** r(F_Δ,CE)=+0.79…+0.92 is "partly
mechanical" (§18.6) — CE is a third measurement of the same drift. Fix: frame CE as
a behavior-space view of the same drift, not independent confirmation.

**A7 — Quarantine convention self-contradiction.** §18.1 says "quarantine-filtered"
but the pool is quarantine-INCLUDED (32 divergent runs inside n=1035); ΔR²(F_Δ)
0.395→0.328 when excluded (V1); r robust (−0.847→−0.864). Fix: "finite-value-
filtered, quarantine-included"; report both ΔR²s.

**A8 — Qwen CE coverage ~60%, missingness mechanism unshown; "123" (§17.8) vs
"136" (§18.6) count discrepancy.** UNANSWERED: reconcile count + show CE-present vs
CE-absent Qwen cells have matched F_Δ/retention distributions, or scope the claim.

**A9 — Heterogeneity.** Knees span ~a decade; normalized slopes −0.33…−0.70 (2×).
Answered by the pre-registered "relation" rule — if enforced everywhere.

**A10 — E4 may be teaching-to-the-test.** Eval-matched calibration gives SC-LoRA
eval-distribution information the LoRA arms never get. Fix: fair target = held-out
task-representative calibration; E4 is the upper-bound-favorable control; keep
"provisional" at 20/24.

**A11 — 284B recurrence overstated.** Per-family Spearman on n=7 methods: 5/6 below
the p=0.05 critical value (0.786). Defensible statistic: sign test (6/6 positive,
p≈0.03), not "+0.86 pooled". Clusters partly design-determined (SVD/subspace
methods spread by construction); spec/fro orderings anti-correlate and are excluded
— selective-reporting-adjacent if not disclosed. Says nothing about the magnitude
relation at scale. Fix: one honest sentence, sign-test framing, appendix table.

**A12 — Headline n changes across docs** (1299/1001/1018/1004/1035/1034/911).
Fix: one appendix reconciliation table (date, filter, join, n); 1,035 is the spine.

**A13 — Method-count chaos** (8 designed / 7 assessed / 10 ladder tokens).
Fix: canonical statement — "8 designed adapters; 7 assessed (CorDA withheld);
+3 control/ablation arms (lorawdr16, milorawd, pissa) in specific analyses only."

**A14 — C4 rescaling marginal and single-arm.** +1.09±1.80 pp (n=15, t≈2.3);
robust part is the asymmetry (downscaling safe, upscaling −3.86). Two floating
numbers (+1.09 vs-twins, +1.29 on-curve residual) — label them distinctly.
Demote to discussion.

### LOW / NITS
- Stale "single seed s42" line (§13) contradicts §16 3-seed state — purge.
- Qwen within-cell SD 2.1–2.7 pp: attach error bars to any Qwen ranking.
- DoRA F_Δ lower-bound caveat (magnitude vector uncounted) should appear at
  submission, not camera-ready.

## 2. GO/NO-GO VOTES (independent)

1. Title: GO "relation/first-order predictor"; NO-GO "Not Geometry" → prefer
   "Magnitude First, Geometry Second".
2. C4: NO-GO as headline; discussion paragraph.
3. Qwen-math SC-LoRA standout: GO.
4. 284B: GO as designed-but-lost in Limitations; NO-GO on foregrounding recurrence;
   one sentence, sign-test framing.
5. E5 replay: GO, explicitly CE-only/partial.
6. DoRA/MiLoRA: GO retention-relevant OPs + collapse disclosure; NO-GO mean-rule rows.

## 3. FLAGS

**Lost-DeepSeek-retention dependence:** 07 correctly withholds retention/CE/
magnitude claims; keep recurrence out of Results; verify no abstract line implies
284B generalization of the relation.

**CLoRA-faithful guardrail (risk is overclaiming a win, not doubting CLoRA):**
"beats best published (CLoRA 64.6) by +2.2 pp" compares our best-of-7-swept
LoRA+wd to CLoRA's single reported LR, cross-harness (base BBH 34.91 vs 33.10) —
the exact asymmetry EXPERIMENTAL_FAIRNESS_AUDIT Q2 calls "structurally favors us".
Fix: anchor on in-harness CLoRA (both swept, frm_clora_k256 = 60.8); say
"matches/edges"; disclose swept-vs-single.

## BOTTOM LINE

Statistical spine genuinely strong (multi-seed, micro-test r=−0.713/n=954, LOMO,
standardized betas, E1 2×2). Three things that draw blood if unfixed: (1) the
"mechanism" overclaim vs own KL-mediation; (2) "6 families" + n/method bookkeeping;
(3) honest scoping of the geometry verdict to the LoRA-variant family. 284B and C4
should be dialed down, not up. Only A2 (one Qwen/math rescale ladder) requires new
compute and is the single highest-value missing run.
