# Area-chair review — two figure decisions + title/section framing (2026-07-09)

Reviewer role: skeptical NeurIPS AC, adversarial. Thesis under test: *retention is governed by
update magnitude F_Δ, not adapter geometry; geometry = fingerprint; rank = modest 2nd lever;
published "wins" sit on the same magnitude law once update size is controlled.*

Sources read: `artifact_status_report.html`, `artifact_review_round_final.md`,
`handoff/27_GEOMETRY_DRIFT_2026-07-09.md`, `key_numbers.md`. Numbers re-derived this pass from
`results/geo_drift/{adapter_metrics,master_labeled,permatrix/*}.jsonl` and the CLoRA Table-4 rows
hardcoded in `paper/writing/fig_cross_literature.py`.

## What I independently checked before ruling (load-bearing)
- **CLoRA Table-4 fit (all 10 rows):** r = −0.980, slope −14.65 pp/dec — reproduces the artifact's
  −0.98 / −14.7. BUT the slope is subset-sensitive: their **5 independent baselines only** (LoRA r8/r16/r64,
  LoRA-L2, MiLoRA) → r −0.954, slope **−12.7**; their **CLoRA-k series only** → r −0.984, slope **−18.9**.
  The "−14.7 ≈ our −14.8" headline is a *blend* of a shallow baseline slope and a steep own-method knob.
  Sign/rank are rock-solid (r ≤ −0.95 in every subset); the *precise slope match is partly fortuitous.*
- **y-axis mismatch:** their axis is BBH-only, base **34.91**; our retention is BBH+MMLU-Pro, base **26.0**
  (~9 pp offset). The clean apples-to-apples is BBH↔BBH (our BBH slope −14.3, key_numbers §7; theirs −14.7).
- **x-support mismatch:** CLoRA's tightest points (k1024 F_Δ 0.21→36.5, k2048 0.14→38.7) sit at F_Δ our
  sweep barely reaches, and **both EXCEED their own base (34.91)** — i.e. the strongest "agreement" region
  is CLoRA's own constrained method beating base.
- **Qwen geometry: does not exist.** All 320 reconstructed adapters are `Llama2-7b`
  (`master_labeled.jsonl`); the Qwen sweep is **LoRA-only** (key_numbers §11, ~13 cells). There is no
  Qwen per-layer/per-method geometry to plot.
- **Per-layer data exists** (`permatrix/`, 320 files). SC-LoRA erosion is real: `lrsw_sclora_r32_lr5e5`
  has ein_top mean 0.63, max 0.985, 86/160 matrices > 0.6 — matches handoff/27's "early layers 0.61–0.75,
  erodes 0.70→0.21 with LR."
- **Geometry-as-independent-axis is rejected in your own data:** partial ΔR² ≈ 0.0002 (p=0.53) beyond
  magnitude+method; the preliminary amp_top/e_top signal flips sign when PiSSA/SC-LoRA are dropped
  (handoff/27). This is the fact that governs both the title and the section decision.

---

## (#2a) Cross-literature overlay (our F_Δ↔retention + CLoRA Table-4 F_Δ↔OOD) — **ADD, but reframe**

**Verdict: ADD (conditionally).** A competitor's *own published table* independently reproducing your
law is the single highest-value external-validity card you have; skipping it forfeits your best answer to
"is this just your pipeline?" But the *current framing* ("same slope, one line, two datasets") is the
attackable part, not the figure itself.

**The one objection that matters most:** *"This is not one line — it is two lines with different
y-metrics (their BBH base 34.91 vs your BBH+MMLU-Pro base 26.0) whose tightest agreement lives in a
low-F_Δ regime your own sweep never populates and where CLoRA's constrained method beats its own base.
The '−14.7 ≈ −14.8' match is a coincidence of pooling a −12.7 baseline slope with a −18.9 own-method
slope."* A reviewer who recomputes the subset slopes (as I did) can make the "identical slope" claim look
oversold in one paragraph, and the y-offset makes "one literal line" false on its face.

**How to preempt (do all four):**
1. Plot **BBH↔BBH** (ours −14.3, theirs −14.7), not retention↔"OOD-acc". Same benchmark on both axes
   kills the y-metric objection and the slopes genuinely match.
2. Lead the caption with **r and direction** ("both datasets: r ≤ −0.95, negative, F_Δ→retention"),
   which is robust in every subset; report slope as a **range/CI**, not a point-matched "−14.7≈−14.8".
3. Keep the artifact's honest "**parallel law, not one literal line**" wording — but fix the imprecise
   "~2× lower" note: their LoRA F_Δ (0.79) ≈ ours; only their *constrained* k-series is lower.
4. **Annotate that CLoRA's k1024/k2048 exceed their base BBH** and say plainly why (they climb the curve
   by shrinking F_Δ) — otherwise a reviewer weaponizes those two points as "geometry buys retention above
   the law." Owning it converts your weakest points into on-curve confirmation.

---

## (#2b) Per-layer / per-model (Qwen) geometry-drift inset — **SKIP as framed**

**Verdict: SKIP the "per-model (Qwen)" inset outright; FOLD the per-layer part into the geometry
section (panel D), do not ship it as a standalone cross-model figure.**

**The one objection that matters most (fatal for the Qwen half):** *"There is no Qwen geometry data.
Every reconstructed adapter is Llama-2-7B, and your Qwen sweep is LoRA-only — LoRA is precisely the
neutral-baseline method whose fingerprint is flat by construction. A 'per-model geometry-drift' inset
would therefore be either fabricated or a trivially-flat LoRA panel that demonstrates nothing."* You
cannot answer "which Qwen adapters?" — that question sinks the panel. Do not imply cross-model geometry.

**Secondary objection (the per-layer half):** a detailed, richly-structured per-layer geometry figure
*works against your own thesis* — it makes geometry look intricate and important right where you are
arguing it is "just a fingerprint." As a standalone inset it over-weights geometry; as **panel D inside
the geometry section**, subordinate to the null-result panel, it is an asset (mechanistic story for the
SC-LoRA outlier) rather than a liability.

**How to preempt:** (a) Drop "per-model / Qwen" from any geometry figure; if you must gesture at Qwen,
keep it to the *magnitude law* replication (r=−0.88, CS, LoRA), which you do have, and say geometry was
not run on Qwen. (b) Ship the per-layer SC-LoRA erosion only as panel D, captioned as "the *mechanism*
of the one below-law outlier," explicitly not as evidence geometry is a general lever.

---

## (#3a) Title: bold "It's the Magnitude, Not the Geometry" — **do not use the bold form**

**Verdict: the bold title will draw an overclaim attack and violates your own guardrails.**
"Not the Geometry" is the banned "geometry doesn't matter" claim in title form (framing-guardrails memory),
and it is refuted by *your own paper* three ways:
- **SC-LoRA:** §2 says its input-principal concentration "is the mechanism behind its poor retention"
  (−5.7 pp below the law). For SC-LoRA, geometry is *causally implicated* in retention. A title saying
  "not the geometry" is contradicted on your own page.
- **Rank** is a real 2nd lever (partial r ≈ −0.56). It is not *only* magnitude.
- Your headline geometry contribution ("fingerprint recovers each method's design") is *itself a geometry
  result* — the bold title undersells your own novelty while overclaiming the negative.

**The one objection that matters most:** *"You disproved a strawman. Your data show geometry correlates
with the deviation of SC-LoRA (and PiSSA), and rank is a documented second lever — so 'not the geometry'
overclaims exactly where your own analysis is most careful."*

**Least-likely-to-draw-rejection framing** — make the *positive* claim bold and demote geometry to a
subtitle, never negate geometry in the main title:
> **"The Magnitude Law: Update Size Governs Retention in Parameter-Efficient Fine-Tuning"**
> subtitle: *"adapter geometry is a fingerprint of each method, not an independent lever; rank is a modest second."*

If you want the contrast in the title, the defensible ceiling is the **first-order** qualifier, which your
data support (rank = 2nd-order, outliers geometry-explained):
> "Update Magnitude, Not Geometry, Is the **First-Order** Lever on Forgetting in PEFT."
Avoid unqualified "not the geometry"; the single word "first-order" is what stops the overclaim reject.

---

## (#3b) Promote geometry to a full body section + 4-panel figure — **HELPS, if it leads with the null**

**Verdict: PROMOTE it.** The obvious reviewer question ("but doesn't geometry matter?") is asked
regardless of section size. A full section that *proactively* runs the stress-test is the strongest
possible preemption; a two-paragraph treatment reads as dodging the question and invites a desk-level
"geometry is under-analyzed" complaint. The fingerprint result (design recovered from *trained* weights,
persisting 3 epochs) is genuine added novelty. Net: a bigger geometry section **reduces** exposure *iff*
framed correctly.

**The one objection that matters most (the internal tension a sharp reviewer will exploit):** *"You call
geometry 'just a fingerprint,' yet you also say SC-LoRA's geometry is the mechanism of its below-law
forgetting. So geometry IS causal — for at least two methods. Which is it?"*

**How to preempt (this is the load-bearing move):** draw the precise line — **geometry sets *where* a
method starts on the magnitude axis; it is not a second axis orthogonal to magnitude.** Init design
(principal-space) makes SC-LoRA/PiSSA place a *large, mis-allocated* update → they act *through* magnitude
and misallocation. *Among methods matched on magnitude, geometry adds nothing* (ΔR² ≈ 0.0002, p=0.53;
permutation; the amp_top signal flips sign when the two principal-init outliers are dropped). Make **panel
B (the drop-outlier collapse/flip) the centerpiece**, not an appendix — it is your proof, and shown first
it converts "geometry matters" from an attack into a resolved question. Frame the flip as *rigor*
("a naive geometry–retention correlation exists but is fully attributable to two principal-init
outliers"), never as a retraction.

**One consistency landmine in the 4-panel plan:** the fingerprint heatmap (panel C) and the per-layer
panel (D) prominently feature **CorDA**, which key_numbers §8 says is **excluded / not publishable**
(wikitext-calib bug, contaminated dedup, F_Δ 515 explosion). Showing CorDA as a headline geometry
fingerprint while excluding it from every law/table is an inconsistency a cross-referencing reviewer will
flag. Fix: either drop CorDA from the figure, or label it "fingerprint only — excluded from all
quantitative law claims (calibration mismatch)," matching the §2 footnote already in the artifact.

---

## The 3 changes that most increase credibility
1. **Reframe #2a from "same slope / one line" to "same sign, comparable magnitude, two datasets," plotted
   BBH↔BBH with slope reported as a range.** This is the difference between an unassailable external
   replication and a paragraph a reviewer picks apart with a 3-line recompute (baselines-only −12.7,
   CLoRA-k-only −18.9). Highest ratio of credibility gained to effort.
2. **Retire the bold "not the geometry" title; adopt the magnitude-law-positive title with the
   geometry-as-fingerprint / first-order qualifier.** Removes the one framing that is contradicted by your
   own SC-LoRA mechanism and banned by your own guardrails.
3. **Make the geometry section's drop-outlier null (ΔR²=0.0002 / sign-flip) the first thing the reader
   sees, and resolve the "fingerprint vs mechanism" tension explicitly** (geometry sets the starting point
   on the magnitude axis; it is not an orthogonal axis). This is what lets a *larger* geometry section
   protect rather than expose the thesis — and it also neutralizes the SC-LoRA counterexample to the title.
