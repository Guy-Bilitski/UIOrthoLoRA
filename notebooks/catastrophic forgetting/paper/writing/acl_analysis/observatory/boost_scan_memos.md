# Boost scan — unused material in the ANALYSIS MEMOS (2026-08-06)

Scope: analysis_final/01-09 + PAPER_BLUEPRINT + dyn1-5 + e5_e6_salvage + seed_stability +
op_points outputs, INTERESTING_INSIGHTS.md, ACL_CAMPAIGN_INSIGHTS_2026-07-19.md,
fleet_findings.md, REBUTTAL_PREP.md, MISSING_EXPERIMENTS.md, FINAL_TABLE_PLAN.md, and the two
adversarial reviews — checked against `paper/.overleaf-git/main.tex` (full read, both halves)
and against `observatory/artifact_gap_map.md` (its items 1-7 are EXCLUDED here as duplicates:
signature erosion, mediation table, intervention figure + shrinking pairs, principal-touch
tax / slope split, fit-constants table, exchange-rate residuals, dose-benefit panels).

Verification key: FROZEN = appears in key_numbers.md §18-§19. VERIFIED = confirmed in
acl_analysis/verification/verification_report.md (safe-to-quote ledger). Anything else =
UNVERIFIED (including EXTERNAL published numbers, marked as such). Nothing here edits
main.tex or any exhibit; these are candidates only.

---

## TOP 10 (best first, one line each)

1. **E6 wd-transfer split verdict** — weight decay caps MiLoRA's magnitude exactly as LoRA's (+1.75/+2.36 pp above the family curve at adaptation 80.2) but destabilizes DoRA as implemented; FROZEN; two sentences in 5.2 + Limitations.
2. **CLoRA never diverges in-band (0/121)** — the one measured benefit geometry does buy is optimization stability; VERIFIED; one sentence in 5.1 or the cost paragraph, and it is CLoRA-positive.
3. **E4 quantified: eval-matched SC-LoRA ladder sits +0.92 above the curve vs -3.39 below with nq_open** — upgrades the paper's vague "moved it by about 4 points"; FROZEN; one appendix sentence.
4. **Free-lunch quantification: 99-100% of peak adaptation is reachable below the retention knee in all six families** — VERIFIED; one sentence sharpening the RQ2 "training past the knee is a pure loss" claim.
5. **E5 replay CE salvage: 5% replay lowered CE-forgetting in 4 of 4 matched cells (about -0.05 to -0.09)** — FROZEN; one Limitations/Related-Work sentence that preempts the "where is the replay baseline" review.
6. **Biderman et al. weight-decay reconciliation** — their wd (5e-5 to 1e-4, full FT, one rate) barely moves the update, so their null is what the magnitude relation predicts; UNVERIFIED-EXTERNAL; one Related-Work sentence defusing the best-known apparent contradiction.
7. **Pre-registered quote-both rule violated: qwswm r = -0.830 must carry its collapse-clean twin -0.695** — FROZEN rule, currently absent from body and fig:family-scatter caption; one clause to add.
8. **SC-LoRA Qwen-CS seed vignette: one recipe, three seeds, 28 pp retention swing tracking per-seed magnitude 0.44/0.30/0.30** — cell VERIFIED, per-seed triplet UNVERIFIED; one concrete sentence for the within-cell test in 5.2.
9. **Qwen-CS-specific below-knee adaptation tax of about 5.3 pp to ceiling (all other arms within 0.8 pp)** — UNVERIFIED (computed in 01); quantifies the paper's existing qualitative Qwen-CS note; needs a verification pass.
10. **Cross-literature corroboration beyond CLoRA** (MiLoRA Tables 7-8 magnitude/CE ordering; LoRA-Null Table 4b CorDA rank collapse) — EXTERNAL/UNVERIFIED; one sentence or tiny appendix table extending app:clora's external-evidence argument.

---

## DETAILED ENTRIES

### 1. E6 wd-generalization split verdict (MiLoRA+wd transfers; DoRA+wd breaks)
- **What:** wd0.3 applied to MiLoRA lands +1.75/+2.36 pp ABOVE the lrsw family curve (2/2 cells, adaptation up to 80.2 at lr5e-4, F_delta capped 0.84 -> 0.30); wd0.3 applied naively to DoRA (which decays its magnitude vector) is degenerate by CE salvage (forgetting-CE 20.8/10.4 vs 2.1/2.6 twins, spec_max to 1183; benchmark evals lost).
- **Lives in:** key_numbers §18.3 E6; analysis_final/01 §4.4, 02 §4d, 03 §5.7 and fig-ready stmt 7; e5_e6_salvage.txt (raw rows); PAPER_BLUEPRINT §7.
- **Could become:** two sentences in §5.2 (the knob paragraph, after "weight decay, CLoRA's k, and LoRA rank each move F_delta monotonically"): observed transfer of the wd knob to a second method, and the observed boundary; DoRA+wd caveat also fits Limitations item 5. The paper currently mentions the "two-run MiLoRA+wd transfer arm" only as a roster item and never uses its result.
- **Verification:** FROZEN (§18.3). MiLoRA+wd raw values also in e5_e6_salvage.txt.
- **Effort:** sentence(s).
- **Risk:** DoRA half rests on CE-only salvage (0/2 benchmark evals) — must be stated as such; keep observational ("applied naively to DoRA it destabilized training"), not prescriptive ("do not use wd with DoRA"). n=2 cells for the MiLoRA half; disclose. No adjudication conflict.

### 2. CLoRA zero in-band divergences (0/121)
- **What:** across all in-band rates CLoRA has 0 diverged runs of 121, vs lora 7/179, milora 9/162, sclora 4/113, lora_null 3/97, dora 3/73 (LoRA+wd numerator 6, denominator unverified — quote as about 4% or omit).
- **Lives in:** verification_report.md B6 (all counts exact) + safe-to-quote ledger; ACL_CAMPAIGN_INSIGHTS item 4 ("geometry buys optimization stability, not retention").
- **Could become:** one sentence in §5.1 (after the head-to-head result) or next to the cost sentence in §5.2: at matched protocol the orthogonality constraint is the only arm that never diverged in-band. Balances the paper's null result with an observed benefit and is guardrail-positive toward CLoRA.
- **Verification:** VERIFIED (exact, safe-to-quote).
- **Effort:** sentence.
- **Risk:** do NOT quote LoRA+wd's 6/146 (denominator does not reproduce; verification DO-NOT-QUOTE item 10). Phrase as an observation about training stability, not a claim about why.

### 3. E4 calibration control, quantified (+0.92 above vs -3.39 below the curve)
- **What:** the full eval-matched SC-LoRA ladder (20/24 cells, 6 rates, up to 5 seeds) sits at mean residual +0.92 pp above the pooled lrsw magnitude curve; the same recipe with the default nq_open calibration sits -3.39 pp below (n=24). The whole ladder moved, not one point.
- **Lives in:** key_numbers §18.3 E4; analysis_final/03 §1 (fig-ready stmt 1); dyn5_pareto.txt (eval-matched cells reach the observed Llama-CS frontier).
- **Could become:** one sentence in app:sweep's "eval-matched calibration control" paragraph or in §5.1's reversal discussion, replacing/precising "the choice of SC-LoRA's calibration corpus alone moved it by about 4 points" with the two residuals. Optionally note the eval-matched cells reach the observed frontier (dyn5) — but that membership is UNVERIFIED.
- **Verification:** FROZEN (§18.3) for +0.92/-3.39; dyn5 frontier membership UNVERIFIED.
- **Effort:** sentence.
- **Risk:** carry the A10 qualifier (eval-matched calibration is the favorable control; a held-out task-representative calibration is the fair target) and "20 of 24 cells" provisionality. Framing guardrail: a fairness property of data-aware inits, never an SC-LoRA deficiency.

### 4. Free-lunch quantification (99-100% of peak adaptation below the knee)
- **What:** in every one of the six run families, the best below-knee adaptation is 99.0-100% of the family's global best (81.8/81.8, 81.4/81.9, 87.8/87.8, 58.5/59.1, 68.5/68.5, 77.2/77.2).
- **Lives in:** acl_analysis insights (03_freelunch_exchange.py), confirmed exactly in verification_report.md B7 + safe-to-quote ledger; ACL_CAMPAIGN_INSIGHTS item 3.
- **Could become:** one sentence in §5.2's "Adaptation ... peaks at the retention knee" clause or in app:exhibits "The adaptation side of the same relation": the observed peak below the knee is 99-100% of the global peak in all six families. Pairs with the already-present reachability counts (plain LoRA 0/4 vs LoRA+wd 12/26 below-knee cells).
- **Verification:** VERIFIED (exact).
- **Effort:** sentence.
- **Risk:** avoid the "free lunch" slogan (verdict-flavored); state as an observation about where the adaptation optimum sits. Use 12/26, never 12/31 (DO-NOT-QUOTE item 7).

### 5. E5 replay CE salvage
- **What:** LoRA + 5% replay lowered CE-forgetting relative to matched plain-LoRA twins in 4 of 4 cells (lr3e-4: 2.248/2.204 vs 2.307/2.254; lr5e-4: 2.465/2.462 vs 2.551/2.526; KL likewise lower); benchmark-retention comparison lost with the fleet.
- **Lives in:** key_numbers §18.3 E5; e5_e6_salvage.txt (raw); analysis_final/03 fig-ready stmt 8; panel GO vote in 08 §3(v).
- **Could become:** one sentence where the paper says "no data is replayed" (§2 Related Work) or in Limitations: a small consistent drift-side reduction from replay was observed in four matched cells; its benchmark-retention comparison did not complete. Preempts the standard "compare against replay" reviewer ask.
- **Verification:** FROZEN (§18.3).
- **Effort:** sentence.
- **Risk:** CE-only, 4 cells, one setting; must be stated as partial. Observational framing straightforward.

### 6. Biderman et al. weight-decay reconciliation
- **What:** their finding that weight decay does not mitigate forgetting used coefficients 5e-5 to 1e-4 on full fine-tuning at a single tuned rate; at those coefficients the penalty barely moves the update magnitude, so under the magnitude relation their null is the predicted outcome; ours is wd 0.3 on adapter matrices, verified to bound F_delta.
- **Lives in:** REBUTTAL_PREP O11 (checked against arXiv:2405.09673v2 §4.5/Fig 4); PAPER_BLUEPRINT §4 outline note.
- **Could become:** one sentence in §2 Related Work attached to the existing biderman2024lora citation. The paper currently cites Biderman only for "LoRA forgets less than full fine-tuning" and leaves the wd contradiction unaddressed — a likely reviewer objection with a ready answer.
- **Verification:** UNVERIFIED-EXTERNAL (characterization of their setup; verify the coefficients/setup against the paper once more before quoting).
- **Effort:** sentence.
- **Risk:** must characterize their experiment exactly and neutrally (Guy's rule: extremely accurate, never offend); no claim that they were wrong — the two designs sit at different points of the same relation.

### 7. qwswm clean-subset twin (quote-both rule)
- **What:** the pre-registered rule (§17.7/§18.1/§18.6) is that Qwen-math's pooled r = -0.830 is always quoted together with its format-collapse-clean value -0.695 (9 degenerate runs dropped). main.tex quotes -0.830 in §5.2 prose and in the fig:family-scatter caption with no clean twin anywhere (Limitations gestures at the convention but gives no number).
- **Lives in:** key_numbers §18.1/§18.6; analysis_final/01 §1.3; 04 consistency note 3.
- **Could become:** one clause in the fig:family-scatter caption or app:exhibits fit-constants paragraph: "(Qwen math -0.830 pooled, -0.695 after excluding nine format-collapsed runs; we quote both)".
- **Verification:** FROZEN (both numbers and the rule).
- **Effort:** clause.
- **Risk:** none stylistically; omitting it is the risk (a reader of the released stats scripts will find the rule).

### 8. SC-LoRA Qwen-CS seed vignette (the within-cell test made concrete)
- **What:** one recipe, three seeds: SC-LoRA Qwen-CS at lr1e-4 retains 9.4/36.2/37.9 as its per-seed F_delta lands at 0.441/0.299/0.302 — seed-level magnitude fluctuations alone traverse the relation (cell mean 27.85 +/- 15.96).
- **Lives in:** analysis_final/02 §3a + fig-ready stmt 4; op_points_output_2026-07-17.txt; cell mean confirmed in verification_report B6.
- **Could become:** one sentence in §5.2's "not an artifact of the training recipe" paragraph, as the concrete instance of the r = -0.713 within-cell result the paper already states abstractly; also motivates the seed-instability monitor of §5.3.
- **Verification:** cell mean VERIFIED; the per-seed retention/F_delta triplet is UNVERIFIED against the authorities (memo-verified only) — needs a one-cell verification pass before quoting exact per-seed values.
- **Effort:** sentence (plus a small verification pass for the triplet).
- **Risk:** stats-in-words rule (state the swing in words before numbers); keep it observational about the cell, not a verdict on SC-LoRA; the paper already calls SC-LoRA the most seed-fragile method, so this adds mechanism-adjacent texture without new claims.

### 9. Qwen-CS below-knee adaptation tax (about 5.3 pp), all other arms within 0.8 pp
- **What:** below each family knee, mean retention sits within +/-0.8 pp of the base ceiling in five of six families (frc slightly above); Qwen-CS alone plateaus about 5.3 pp under its 44.35 ceiling (best single below-knee run 41.03) — a model-and-task-specific intercept paid before any magnitude-driven forgetting.
- **Lives in:** analysis_final/01 §5 (table + fig-ready stmt 9, marked "computed"); 03 §5.8; PAPER_BLUEPRINT §8.
- **Could become:** one or two sentences quantifying what the paper already says qualitatively in the safe-band paragraph ("no method stays within 2 points of the Qwen base at any rate ... a fact about the family"); natural home is that app:exhibits paragraph or the §5.2 knee discussion.
- **Verification:** UNVERIFIED (computed in 01 from the frozen pool, not in §18/§19 or the verification report) — needs a verification pass; also note §18.7's Llama ceiling is 26.0 vs the exact 25.89 (01 §8 flags the rounding).
- **Effort:** sentence + needs-verification-pass.
- **Risk:** never write "Qwen pays a tax" unqualified — qwswm contradicts it (01's explicit warning); phrase per-family. Knee positions are fit-dependent; keep the numbers coarse ("about 5 points").

### 10. Cross-literature corroboration beyond CLoRA
- **What:** the magnitude account is visible in the other adapter papers' own published tables: MiLoRA Table 7/8 (ΔW-amplification LoRA 68.2 / PiSSA 55.8 / MiLoRA 44.9 tracking CE-to-base PiSSA 6.07 > LoRA 3.24 > MiLoRA 2.54); LoRA-Null Table 4b (CorDA rank collapse, r256 89% -> 73% retention: more capacity, more forgetting); CorDA++ Eqs 5-6 bound loss by the norm of moved directions.
- **Lives in:** INTERESTING_INSIGHTS §6; fleet_findings.md (MiLoRA and LoRA-Null expert blocks); an unused generator exists (paper/writing/fig_cross_literature.py).
- **Could become:** one sentence at the end of §2's "What governs forgetting" paragraph, or a three-row companion note in app:clora ("published third-party numbers consistent with the same ordering"). Extends the paper's external-corroboration argument past the single CLoRA anchor.
- **Verification:** EXTERNAL published numbers (key_numbers §12 class: cite, do not recompute) = UNVERIFIED under the stated authorities; transcriptions were expert-checked in fleet_findings but should be re-checked against the PDFs before use.
- **Effort:** sentence or tiny table.
- **Risk:** highest style/accuracy risk on the list: every number must be transcribed exactly and framed as those papers' own faithful measurements (same guardrail as CLoRA); avoid any hint that their designs were flawed. Do NOT include the MiLoRA alpha=2r-vs-alpha=r confound observation (fleet_findings) unless separately cleared — it edges toward criticizing their design.

---

## SECOND TIER (worth a look if space remains)

11. **Harness sensitivity: +19.5 pp GSM8K from evaluation protocol alone** (INTERESTING_INSIGHTS §9; FINAL_TABLE_PLAN B-eval). The same MetaMath adapter scores about 46.6 under lm-eval default vs 60-66 under the train-matched protocol; motivates the unified harness and the RQ4 theme that measurement choices set reported numbers. Could be one sentence in app:traincfg "Decoding and evaluation". UNVERIFIED (July-9 era, pre-freeze) — needs a verification pass before quoting. Risk: none stylistically if framed as measurement hygiene.

12. **The training suite itself degrades in a fixed order** (ACL_CAMPAIGN_INSIGHTS item 6: hellaswag/ARC gains collapse first, social_i_qa last — part of cs_avg behaves like a hidden retention benchmark). One sentence extending §5.4. UNVERIFIED — verification report explicitly did not re-derive the adaptation-side ordering (insights 8); needs a pass. Risk: low.

13. **PiSSA's BBH 7.2 is partly generation-format collapse, not pure knowledge loss** (INTERESTING_INSIGHTS §8: 37/270 correct targets, about 22% empty generations, likelihood-scored MMLU parity 24.5 shows recognition survives). One footnote to tab:math softening a harsh number and echoing the format-damage channel. UNVERIFIED (July-8 gate, pre-freeze); needs re-derivation. Risk: low; it is PiSSA-favorable, which helps the paper's fairness posture.

14. **Operating-point KL spread: LoRA+wd sits at the lowest drift in 4 of 6 families; the Llama math grid spans about 21x (0.216 vs 4.455 nats)** (observatory m4_op_points; VERIFIED in report B8). The body already says "lowest drift in each Llama block"; the 21x spread is a vivid one-clause quantification for §5.3. Risk: footnote the lorawd-r16 split convention if the "15x" frc variant is used.

15. **MiLoRA Table 6 init ablation (Principal 60.7 < Random 63.2 < Minor 64.0 at fixed budget)** (fleet_findings). An honest published counterpoint that direction matters at fixed budget, fitting the paper's "we do not claim geometry can never help" close. EXTERNAL/UNVERIFIED; transcription check needed. Risk: keep as their finding, stated positively.

## Compliance note found during the sweep (not a boost, flagging for the PI loop)
- The §17.7 quote-both rule (item 7 above) is currently unmet in main.tex; fixing it is
  defensive rather than additive, which is why it ranks despite adding no new result.
