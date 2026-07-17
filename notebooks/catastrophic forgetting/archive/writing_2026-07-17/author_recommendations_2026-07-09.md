# Author recommendations — three presentation decisions (2026-07-09)

Lead author -> PI. Grounded in: `paper.tex`, `artifact_status_report.html`,
`handoff/27_GEOMETRY_DRIFT_2026-07-09.md`, `artifact_review_round_final.md`,
`data/key_numbers.md`. PI guardrail respected throughout: **no bold unprovable
claims; constructive framing; nothing beyond what the data supports.**

Thesis being served: retention in PEFT is governed by weight-update magnitude
F_Δ (first-order lever); rank is a modest second lever; adapter geometry is a
fingerprint that explains outliers, not an independent knob; published adapter
"wins" carry the ingredients of a learning-rate/recipe artifact and land on the
same magnitude law once F_Δ is controlled.

---

## Summary of recommendations

| # | Decision | Recommendation |
|---|----------|----------------|
| 2a | Cross-literature overlay figure | **BUILD** — highest-value figure for the thesis; build as a *parallel-slope* overlay, not one line |
| 2b | Per-layer / per-model geometry inset | **SPLIT** — build the SC-LoRA per-layer + LR-erosion inset; **do not** build a Qwen geometry row |
| 3a | Title | Drop the absolute "Not the Geometry"; recommend a "first-order predictor / magnitude law" title (ranked below) |
| 3b | Geometry: full section vs subsection | **Compact fingerprint subsection + one consolidated figure**; do not promote to a full co-equal body section |

---

## (#2a) Cross-literature overlay figure — BUILD IT (highest priority)

**Recommendation: build it, as the single most reviewer-persuasive exhibit in the
paper.** Plot our (log F_Δ, retention) points + fitted line together with CLoRA
Table-4's (log F_Δ, out-of-domain BBH) points + fitted line, and annotate the two
slopes: **ours −14.8 pp/decade, theirs −14.7 pp/decade**.

Why it materially helps (not redundant):
- It is the paper's **only fully independent** evidence — a different codebase,
  a different pipeline, a different research group's published numbers — landing
  on the same slope. That is the strongest available answer to the review's
  sharpest latent objection ("you built a harness to get the answer you wanted")
  and to the near-tautology worry the paper itself flags (limitation 7,
  `sec:limits`). Two datasets converging on −14.7 / −14.8 is provable and
  constructive.
- The review flagged this exact gap twice (comment E = WEAK; FIX 7): the r=−0.98
  external replication currently lives only in prose and an appendix table
  (`tab:clora`), not as a figure. It is the campaign's best fact and it is buried.
- It costs almost nothing: every number already exists in `tab:clora` and
  `key_numbers.md`.

Non-negotiable honesty constraint (do it this way or don't do it):
- **It is a parallel-slope overlay, not one shared line.** CLoRA's absolute F_Δ
  levels run ~2× lower than ours (probe/harness offset) and their base BBH is
  34.91 vs our 33.10. Plot two point clouds with two own-fit lines; caption:
  "same slope from two independent datasets; horizontal F_Δ offset and different
  base reflect probe/harness differences — a *parallel* law, not one curve."
  Naively overlaying raw points as if they coincide would be the kind of overclaim
  a CLoRA-reading reviewer catches instantly, and would invert the figure's value.
- Optional cleaner variant: put retention on each dataset's own "Δ-from-base"
  scale so the two lines share a y-origin; keep both slopes annotated.

This figure earns a place **in the body** (near the magnitude-law section), and
the "−14.7 vs −14.8, two datasets" fact deserves promotion to the abstract-level
framing / a header claim, not an appendix.

---

## (#2b) Per-layer / per-model geometry drift inset — SPLIT

### (i) SC-LoRA input-principal concentration by layer + LR-erosion — BUILD IT

**Recommendation: build it as an inset/panel inside the geometry subsection.**
It converts the paper's single biggest exception into a constructive win.

Why it materially helps:
- SC-LoRA is the **one provisional off-curve deviator** (−4.15 pp ANCOVA
  residual, p=0.006) and the only real threat to "geometry adds nothing beyond
  magnitude." Showing the *mechanism* — ein_top 0.41 mean, early q/k layers
  0.61–0.75, eroding **0.70→0.21 as LR rises (r=−0.96)** — turns the exception
  from an unexplained hole into evidence that the fingerprint tool *explains its
  own outlier*. That strengthens the constructive "geometry = fingerprint" story
  rather than undermining the magnitude law.
- It independently reproduces SC-LoRA's own paper's stated limitation ("the
  constraint erodes during training"), which validates the measurement tool — a
  guardrail-positive, provable point.
- The review flagged per-layer drift as asserted-not-shown (comment J = WEAK;
  FIX 9). The data already exists in `handoff/27`.

Framing guardrail: present it strictly as *fingerprint + outlier explanation*,
never as "principal-direction concentration is a second forgetting axis" — that
axis was **rejected** (handoff/27: amp_top flips to +0.25 when the two
principal-init outliers are dropped; ΔR²=0.0002 beyond magnitude+method). The
inset explains *why SC-LoRA specifically* deviates, not a universal geometric law.

### (ii) Qwen geometry row — DO NOT BUILD

**Recommendation: skip it.**
- Per-model geometry is off-thesis: the central claim is magnitude-governs-
  forgetting; geometry is a supporting fingerprint. A Qwen geometry row advances
  neither the law nor the fingerprint story materially.
- Qwen is already the weakest, most-caveated arm (~13/112 cells, LoRA-mostly,
  **math anti-replicates** r=+0.67 ns). Adding a thin second-model geometry claim
  expands attack surface for near-zero thesis gain, and risks reading as an
  overclaim about a model that doesn't yet replicate. Keep Qwen scoped to the
  law's *sign/monotonicity* replication and leave geometry single-model.

---

## (#3a) Title

**The current title "It's the Magnitude, Not the Geometry" overclaims and should
change.** "Not the Geometry" is an absolute negative that the paper's own results
contradict in three places: (1) rank is a real, modest 2nd lever (partial
r=−0.56); (2) SC-LoRA's geometry genuinely correlates with its poor retention
(−4.15 pp off-curve, mechanistically explained by ein_top); (3) high-k CLoRA
genuinely wins high-rank commonsense by buying adaptation-efficiency per unit
update. It is the "geometry doesn't matter" framing the PI explicitly banned, and
a hostile reviewer will use the paper's own geometry section against its title —
a self-inflicted rejection risk. (Minor: "Weight-Update Norm" is also now
imprecise; the axis is F_Δ / effective update *magnitude*, not the Frobenius norm.)

**Ranked options:**

1. **"The Magnitude Law: Weight-Update Size Is the First-Order Predictor of
   Forgetting in Parameter-Efficient Fine-Tuning"**
   *Risk: LOW.* "First-order predictor" is exactly what the data supports
   (magnitude 1st, rank modest 2nd); it does not deny geometry; it names the
   paper's central artifact. Provable and constructive, minimal skim-misread risk.

2. **"It's Mostly the Magnitude: Update Size, Not Adapter Geometry, Governs
   Forgetting in PEFT"**
   *Risk: LOW–MEDIUM.* Keeps the memorable "It's … the Magnitude" cadence;
   "Mostly" + "Governs" are defensible and inoculate against the geometry-
   concession nitpick. Residual risk: a skim reader could still latch onto "Not
   Adapter Geometry" and ignore "Mostly."

3. **"It's the Magnitude, Not the Geometry: …" (current)**
   *Risk: HIGH.* Asserts the banned absolute; contradicted by the paper's own
   rank lever, SC-LoRA outlier, and high-k CLoRA concession. Not recommended.

Recommended: **#1** (safest against the guardrail while still headline-strong).
If the PI wants to preserve the memorable cadence, **#2** is the acceptable
punchier fallback; #3 should not ship.

---

## (#3b) Geometry: full section vs compact subsection

**Recommendation: keep geometry a COMPACT fingerprint subsection, and give it ONE
consolidated figure — do not promote it to a full co-equal body section.**

Rationale:
- The geometry verdict is deflationary by design: geometry is a *fingerprint +
  outlier explanation*, not a competing axis (principal-direction axis rejected;
  amp_top ΔR²=0.0002). Elevating it to a full section with a sprawling 4-panel
  figure over-weights the paper's **most fragile, most caveated** analysis and
  creates a tonal contradiction — "geometry is a fingerprint, not a knob," told
  across a full headline section — that invites exactly the scrutiny it can least
  afford. That cuts against the guardrail.
- The load-bearing geometry result — the ANCOVA "geometry adds nothing beyond
  magnitude" test (the thing that separates this work from prior single-method
  papers) — **already has its body figure** (fig2 residuals) and stays in the
  body. That is the right amount of geometry prominence.
- The constructive fingerprint payoff is real and worth *one* figure: build a
  single consolidated exhibit combining (C) the per-method fingerprint heatmap
  (recovers each method's design from trained weights) and (D) the SC-LoRA
  per-layer + LR-erosion inset from #2b. Optionally add the (B) stress-test panel
  (amp_top/ein_top collapse+flip when PiSSA/SC-LoRA are dropped) — that panel is
  a *guardrail-positive* honest negative and pre-empts "your geometry axis is
  carried by two outliers." Panel (A) is the hero and already exists; do not
  duplicate it.

Net: geometry's footprint = fig2 (ANCOVA, body, unchanged) + one fingerprint/
erosion figure in a compact subsection. This keeps the magnitude law the
headline, keeps geometry constructive and honest, and stays inside the guardrail.
