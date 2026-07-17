# 05 — Review Notes on `paper_draft.tex`

*Critic pass over the drafted paper (2026-07-02), checked against
`CONCLUSIONS_AND_IDEAS.md`, `04_critique.md`, `data/key_numbers.md`, and the actual
contents of `figures/` and `tables/`. Numbered, most-important first. Each item says
WHAT is wrong, WHERE, and the concrete FIX.*

**Overall verdict.** The draft is a large, honest improvement over the pre-draft state:
it already softens "surpasses" → "matches/edges," carries an unflinching Limitations
section, marks every data gap with a visible `%TODO(data)`, and demotes SC-LoRA/CorDA to
provisional. The remaining problems are (a) a handful of *hard* internal inconsistencies
that a reviewer will catch by cross-checking the draft against its own tables/figures, and
(b) several places where the prose is still a half-step ahead of what the shipped
figures/tables actually show. Fix the numbered items below in order.

---

## TIER 1 — Must fix before the draft is internally consistent (a reviewer catches these by reading only the PDF)

### 1. The math table on disk contradicts the draft prose and key_numbers — three-way mismatch.
- `tables/table_main_math.tex` contains **only 2 rows** (LoRA+wd 49.1/24.4/fdelta 0.359 @lr3e-4; LoRA 46.5/22.9/0.520 @lr3e-4). **No DoRA row.**
- The draft §Pareto prose (line ~409) and abstract-of-math claim quote LoRA+wd math = **50.6/24.6/fdelta 0.399 @lr5e-4** (from key_numbers §4) — a *different cell* than the table.
- The draft §law prose (line ~245) and Table caption (line ~424) say math "covers only LoRA, LoRA+wd, and DoRA" (n=14, 3 adapters) — but the table shows only 2 adapters, and n=14 = 2 adapters × 7 LRs, NOT 3.
- **This is the single most damaging internal contradiction: the paper's own table, its prose, and its stat (n=14) disagree on how many math adapters exist and on the headline LoRA+wd math point.**
- FIX: pick ONE source of truth (key_numbers.md). Decide whether math is 2 or 3 adapters and whether n=14 is 2×7. Regenerate `table_main_math.tex` to match (add DoRA if it belongs; reconcile the LoRA+wd cell to either 49.1/24.4/0.359 OR 50.6/24.6/0.399 — not both). Then make §law's "n=14 (3 adapters)" and §Pareto's quoted point and the table agree. The existing `%TODO(data)` at line ~412 flags this but the body still asserts the unreconciled numbers as fact.

### 2. Draft says "seven learning rates" but lists/uses only the sweep after dropping 2e-3, 5e-3 — verify n arithmetic is consistent.
- §Setup (line ~178) sweeps 7 LRs {2e-5 … 1e-3}. Good.
- But the pooled CS law is n=49 = 7 series × 7 LRs, and the draft (correctly, per §Limits item 5) admits LoRA-Null is pooled into LoRA — i.e. the "6 adapters" the caption of fig0 names actually contribute **7 series** to reach n=49. The draft never states this arithmetic, so "six visible adapters" (§Limits item 4) and "n=49" look contradictory (6×7=42, not 49).
- FIX: add one clause where n=49 first appears (fig0 caption / §law line ~228): "n=49 = 7 LR × 7 series (LoRA-Null is presently pooled into the LoRA series; see §Limitations)." Otherwise a reviewer does 6×7=42 and flags the mismatch — exactly critique item B6.6.

### 3. Abstract/§intro still say "eight adapters" while the headline figures show six.
- Abstract (line ~46), §intro (line ~95), §related, and §conclusion (line ~582) all say "eight adapters." But CorDA is excluded from every figure/table (§Limits item 4 admits this) and LoRA-Null is pooled into LoRA (§Limits item 5). The hero fig0 and ANCOVA fig2 show **six visible series, of which one (SC-LoRA) is off-curve**.
- The draft *does* footnote this honestly in §Limits, which is a big improvement over the pre-draft. But per critique B1, "eight" must not sit next to "one curve" in the load-bearing sentences.
- FIX (choose one, consistently): (a) change the count in the abstract/intro/conclusion to "eight adapters (six on the headline curve; CorDA re-running, LoRA-Null pooled — see §Limitations)", OR (b) say "eight adapters, of which six appear on the current curve." Do NOT leave the bare "eight … one curve" phrasing in the abstract while the figure shows six. This is the highest-visibility honesty gap.

### 4. The pooled CS correlation is quoted with two different framings that a reader will read as inconsistent.
- Abstract (line ~55): "r=-0.86 pooled, -0.92 on the well-behaved adapters; slope ≈ -10 to -15 pp/decade."
- §law (line ~235): pooled "slope -14.8 pp/decade (n=49)"; on-curve "slope -10.0 (n=42)".
- These ARE consistent (the -10 to -15 range brackets both), but the fig0 caption (line ~228) says "slope -10 pp/decade" for the *pooled on-curve* fit while labeling it "pooled fit on the on-curve adapters … r=-0.92, R²=0.84, n=42" and then "all-adapters pooled r=-0.86 (n=49)". Verify the fig0 image's drawn line matches the -10.0 (on-curve, n=42) slope, not -14.8. If the black line in the PNG is the n=49 pooled fit (-14.8) but the caption says -10, that is a figure/caption mismatch.
- FIX: confirm which fit fig0's black line actually is, and make the caption's r/R²/slope/n quartet match that exact fit. Round the headline r consistently to -0.86 everywhere (critique B6.4: recomputed value is -0.858).

---

## TIER 2 — Claims still slightly ahead of the shipped evidence (soften or add the missing piece)

### 5. Claim 3 ("wins are an LR artifact") still rests on an exhibit the paper does not have.
- The draft is commendably honest here — the `%TODO(data)` at line ~362 explicitly states the "reproduce-a-published-win-then-dissolve-it" figure is not built, and that Claim 3 currently rests on fig4 (best-LR-not-shared) + fig7 (R² contrast). This directly addresses critique B3.
- BUT the abstract (line ~61-66), §intro item 3 (line ~101), and §conclusion (line ~589) still state Claim 3 as an established diagnosis ("the fancy adapters' reported wins are largely a learning-rate artifact"), not as a hypothesis supported by indirect evidence.
- FIX: align the strength of the abstract/intro/conclusion phrasing with the honest `%TODO` in §lr. Recommended: "we show the *ingredients* of the artifact — best-LR is not shared, and dW predicts retention far better than LR does; reproducing a specific published win and dissolving it is deferred (§lr, §Limitations)." As written, the abstract over-claims relative to the body's own admission.

### 6. The Pareto "win" margin is sub-noise on a single seed — the draft says so in Limits but the abstract/§Pareto still lead with "wins the upper-right corner."
- §Pareto (line ~389) header "LoRA+wd wins the upper-right corner"; the win is +1.5pp adapt / +0.8pp ret (§Limits item 1), smaller than the documented seed-collapse variance (tens of pp).
- The draft's "honest verb" paragraph (line ~428) and §Limits item 1 handle this well ("matches/edges"). But the *section header* and the abstract's "matches or edges the … Pareto frontier of every elaborate geometric adapter" still read as a ranking claim.
- FIX: change the §Pareto header from "wins the upper-right corner" to "lands on the upper-right frontier" or "occupies the sweet-spot corner." Keep the body's honest verb. Consistency between header and body.

### 7. SC-LoRA is "ringed as the only significant deviator" in figure captions — but §Limits calls the whole calibration↔eval comparison confounded.
- fig0 caption (line ~230) "SC-LoRA is ringed as the single below-curve deviator"; fig2 caption (line ~276) reports "-4.15 pp (p=0.006, the only significant deviator, provisional)." Draft text (line ~259-263) does add "provisional" and cites the calibration confound. Good.
- Per critique B4, the concern is that presenting the ONE significant result (SC-LoRA) as a ringed finding, even labeled provisional, still visually headlines a result the paper admits it cannot interpret (nq_open vs academic-eval mismatch).
- FIX: this is acceptable *if* every SC-LoRA mention carries "provisional / confounded by calibration mismatch." Audit that fig0's caption (currently just "ringed as the single below-curve deviator") also gets the "(provisional)" qualifier — right now fig2 has it but fig0 does not. Do NOT let any SC-LoRA sentence read as "SC-LoRA's geometry forgets more" without the calibration caveat inline.

### 8. Qwen: abstract says "two base models" / "in-progress replication"; math anti-replication is buried.
- Abstract (line ~50) "two base models (Llama-2-7B, complete; Qwen2.5-7B, an in-progress replication)." §qwen (line ~504-508) is admirably honest about the +0.67 wrong-sign Qwen-math result.
- Per critique I2: the abstract's "two base models" phrasing lets a skimmer count Qwen as a completed second model. The steeper Qwen-CS slope (-34.8 vs -10/-15) means only *sign+monotonicity* replicate, not the quantitative law — the draft §qwen says exactly this (good), but the abstract does not qualify.
- FIX: abstract → "Qwen2.5-7B (a partial replication: ~13/112 cells; CS-LoRA replicates the sign, math-LoRA does not yet)." Keep the §qwen body as is — it is the model for how to report an anti-replication.

---

## TIER 3 — Rigor / polish (strengthen; not fatal)

### 9. The sweet-spot band [0.31, 0.62] is post-hoc and (by the draft's own numbers) excludes LoRA's best point.
- §mech (line ~448) and fig8 caption define the band from the same 49 points, then celebrate LoRA+wd (0.394) landing in it. Critique I3. The draft's `%TODO` at line ~454 already flags this and notes LoRA's own best point (0.623) falls outside.
- FIX: either derive the band from an a-priori utility criterion, or present it descriptively ("adaptation and retention curves cross in [0.31,0.62]") and drop the "wd lands in it for free" causal spin. At minimum keep the `%TODO` visible until resolved.

### 10. Near-circularity is defended only by an in-sample R² bump (0.74→0.87).
- §law (line ~253-266) reports the ANCOVA intercept test; the `%TODO` at line ~264 already promises the slope-interaction ANCOVA and leave-one-method-out predictive check. Critique I1.
- FIX: add the two deferred tests (slope-interaction term; LOO fit-on-5-predict-6th RMSE) before camera-ready, or explicitly frame the current ANCOVA as "intercept-only, in-sample" so the claim's strength is honest. An in-sample R² jump from adding 5 free intercepts is expected and a reviewer will say so.

### 11. Base-ceiling calibration missing for MMLU/ARC/TruthfulQA → "broad retention" and per-benchmark slopes not normalizable; "TruthfulQA immune" may be a floor artifact.
- §setup `%TODO` (line ~190) and §law per-benchmark `%TODO` (line ~288) and §Limits item 7 all flag this. Critique I5.
- The draft still prints Ret-broad numbers in both tables (33.2, 33.6, etc.) and headlines "TruthfulQA is essentially flat … close to immune" (line ~284).
- FIX: either add the 5 eval-only base runs, or add an explicit inline caveat wherever Ret-broad appears ("uncalibrated; shown for texture only") and soften "immune" → "flat, pending a floor check." Do not print broad-retention percentages as if normalized.

### 12. Data hygiene: shipped `data/campaign_summary.jsonl` still contains the CorDA explosion + duplicate rows (critique B2).
- Not visible in the PDF, but if the data file is released with the paper, `fdelta=515.77, ret=0` CorDA rows and duplicate LR cells destroy pipeline credibility.
- FIX: purge exploded/duplicate rows, document the dedup rule, and tag each surviving CorDA row with its calibration set (wikitext vs nq_open) before any data release. (Independent of whether CorDA appears in figures.)

### 13. LoRA-Null labeling bug contaminates the reported LoRA residual and within-r (critique I6).
- §Limits item 5 (line ~559) correctly flags that LoRA-Null is pooled into "lora." But the draft still quotes the LoRA ANCOVA residual (+0.79) and LoRA within-r (-0.97) in §law (line ~240, ~257) as if they are plain LoRA — they are actually LoRA∪LoRA-Null (14 points). Pooling a *geometric* null-space method into the vanilla baseline flatters "geometry inert."
- FIX: after the labeling fix, recompute and re-quote the LoRA-only residual and within-r, and report LoRA-Null's own residual. Until then, add "(currently pooled with LoRA-Null; see §Limitations)" to the first LoRA residual mention.

### 14. arXiv IDs unverified; the `%TODO` at line ~127 notes `2603.02224` is "almost certainly wrong" (impossible March-2026 date). Bibliography is placeholder-only (line ~627). Critique N1.
- FIX: verify all IDs (LoRA 2106.09685, MiLoRA, CorDA 2406.05223, DoRA, CLoRA, SC-LoRA, OPLoRA 2510.13003, CorDA++ 2506.13187) and build `references.bib` before submission. Mechanical but mandatory.

### 15. Missing: compute/repro statement for a paper whose method IS a sweep (critique N2).
- FIX: add a short appendix with total run count (~288 Llama cells + Qwen), wall-clock/GPU, and a repro pointer. Cheap credibility for a sweep-based paper.

### 16. Tone vs evidence mismatch (critique N3).
- The "wake-up call / ships an adapter every week" framing (abstract, intro, conclusion) is combative. With Claim 3 deferred (item 5) and Claim 2 sub-noise (item 6), the polemic currently outruns the evidence.
- FIX: keep the aggressive framing only if you ship the strong tier (seeds + CorDA + gotcha figure + Qwen 2×2). For the minimum-defensible tier, lead with the measurement-methodology contribution (LR-sweep-as-instrument + fair dW axis) and reserve the polemic.

---

## What the draft already gets right (keep)
- Honest scope stated up front (§intro line ~109; Llama mature, Qwen in progress).
- Visible `%TODO(data)` markers on every unfinished number — an excellent discipline; keep them until each is resolved.
- "Matches/edges," not "dominates," as the Pareto verb (§Pareto line ~428).
- SC-LoRA and CorDA demoted to provisional/excluded with reasons.
- The fallback measurement-contribution framing in §conclusion (line ~601) — this is the guaranteed-defensible core; foreground it more if seeds/CorDA don't land.
- All 10 cited figures (fig0–fig8, op_points) and both tables exist in `figures/`+`tables/`. No missing-asset problem. The problem is content mismatch (item 1), not missing files.
