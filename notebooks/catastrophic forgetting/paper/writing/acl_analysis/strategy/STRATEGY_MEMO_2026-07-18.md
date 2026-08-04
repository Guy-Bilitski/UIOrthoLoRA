# STRATEGY MEMO — Repositioning the PEFT-forgetting paper for ACL/ARR

**To:** Guy (PI) · **Re:** Reconciling the "field-consolidation / first systematic multi-metric review" vision with the panel-approved "Magnitude First, Geometry Second" thesis · **Date:** 2026-07-18
**Author:** paper-supervisor agent (ACL repositioning campaign)

**One-line verdict:** Your vision and the panel's thesis are not in tension — the panel gave you the *finding*; you are now asking for the right *packaging*. For ACL specifically, that packaging is a net win. But four of your six contributions are load-bearing and two are thinner than the framing implies. The magnitude relation must stay the spine; the four metrics are *instruments of one protocol that converge on one answer*, never four co-equal headlines. Everything below enforces that.

---

## 1. FRAMING RECONCILIATION

**The two framings decompose cleanly onto the two-layer contribution the panel already blessed (08 §1, BLUEPRINT §2):**

- **Your "consolidation + first systematic multi-metric, multi-config review"** = the **measurement/protocol layer** (Layer 1). This is the guaranteed-novel core: a fair magnitude axis (F_Δ, CLoRA's own Eq-3 diagnostic), the LR-sweep-per-method instrument, eval-matched calibration, and — the new packaging you want — the *four instruments run identically on every adapter* (retention/adaptation trade-off, magnitude, major/minor geometry, CE drift).
- **The panel's "Magnitude First, Geometry Second"** = the **finding layer** (Layer 2): what the protocol reveals when you point it at the zoo.

These are the same paper described from two ends. The protocol is *why anyone should believe* the finding; the finding is *what makes the protocol worth building*. A pure consolidation/benchmark paper with no thesis dies at ACL as "a table, not a result." A pure thesis paper invites "incremental over CLoRA." Bolting them together is what makes this an A*-shaped submission.

**How to weight them for ACL.** ACL/ARR rewards resource, measurement, and meta-evaluation papers far more than NeurIPS does — "we re-ran the zoo fairly and here is what survives" is a genuinely ACL-native genre (cf. the reproducibility/meta-eval track record). So **give the protocol co-equal top billing in the title and abstract**, which the current NeurIPS-flavored draft does not. But *inside the paper, the magnitude relation remains the single organizing spine* — the four instruments are introduced as one apparatus, and each section closes by returning to "and this is why magnitude is first-order." Concretely: **35% of the intro/abstract real estate to the protocol-as-contribution, 65% to the finding.** The metrics get equal *methodological* treatment (each defined and applied to all adapters) but unequal *rhetorical* treatment (magnitude is the answer; geometry and CE are shown to be, respectively, a second-order fingerprint and a re-description of the same drift).

**Recommended title (primary):**
> **Magnitude First, Geometry Second: A Fair, Multi-Metric Audit of Catastrophic Forgetting Across PEFT Adapters**

This keeps the panel's exact thesis phrase (survives A4/A5, "first-order predictor" not "mechanism") and prepends the ACL-native "fair, multi-metric audit" that encodes your consolidation vision.

**Alternates:**
- (A) *What Actually Controls Forgetting in Parameter-Efficient Fine-Tuning? Auditing the Adapter Zoo on a Common Magnitude Axis* — question-form, audit-forward; strongest "is this a measurement paper" signal.
- (B) *Measuring Forgetting Fairly: Update Magnitude, Not Geometry, Governs Retention in PEFT* — leads on the measurement contribution, states the finding as the payoff.

Avoid the bare "The Magnitude Law" (current title) — the panel already ruled "law" is only licensed with the knee caveat (§18.2; normalized slopes −0.33…−0.70 do not converge), and it buries the protocol.

**150-word abstract skeleton:**
1. *(Hook, the zoo problem)* A new PEFT adapter ships almost weekly, each claiming to curb catastrophic forgetting through a geometric prior (data-aware init, null-space projection, orthogonality penalty). Their wins are reported on incompatible axes, at single learning rates. (2 sentences)
2. *(Protocol contribution — the consolidation)* We build the first fair, common-harness audit: every adapter run identically under four instruments — the adaptation/retention trade-off, a method-neutral update-magnitude axis (F_Δ), update geometry, and behavioral (CE) drift — swept over learning rates and seeds across two architectures and two task types (1,035 adapters). (2 sentences)
3. *(Finding — the spine)* Retention tracks update magnitude to first order (pooled r=−0.847, flat-then-falling with a per-family knee); once magnitude is fixed, geometry adds ~18× less unique variance, and reported per-method wins carry a learning-rate-tuning ingredient a sweep dissolves. (1–2 sentences)
4. *(Consequence + honest scope)* Plain LoRA+weight-decay — the cheapest magnitude control — matches or edges the frontier of every elaborate adapter. Control the size of the update, not its geometry. (1 sentence)

---

## 2. CONTRIBUTION LIST — PI's six mapped to evidence, with severity flags

| # | PI's contribution | Supporting evidence | Verdict / flag |
|---|---|---|---|
| **1** | "Gathering them all together" (consolidation) | 7 of 8 adapters assessed + full-FT anchor (E2) + off-recipe bridging (E7) on 4 instruments; common harness is a *precondition*, only CLoRA reports our axis (03 §4/O12, paper §Setup) | **SOLID as a protocol claim.** But **never say "all"** — CorDA is withheld (port mis-calibration, 03 §2), PiSSA/OPLoRA positioned-not-swept. Canonical wording (A13): "8 designed; 7 assessed; +3 control arms." |
| **2** | "First systematic review against LRs and seeds" | LR-sweep-per-method instrument (§18.5: partials, fixed-LR strata r≤−0.7); 287 cells with n≥3 seeds (§18.1) | **STRONGEST genuine novelty.** This + the fair axis is the guaranteed-defensible core. Foreground it. Caveat: Qwen within-cell SD 2.1–2.7 pp; some arms single-seed — state per-arm. |
| **3** | "Magnitude + geometry + CE, compared against each other" | Nested ΔR² ladder (§19.1): magnitude +0.395, geometry +0.017, method +0.006; mediation (05): KL is proximal in 4/6 families | **SOLID but requires the A1 fix.** Magnitude is the *first-order predictor/control variable*, **not the mechanism** — your own mediation says drift (KL) is proximal. CE is a *behavior-space view of the same drift* (A6), not independent confirmation. |
| **4** | "How activation methods behave against pre-forwarding of the data" | E4: SC-LoRA eval-matched → +0.92 pp above curve (n=20, provisional 20/24) vs nq_open −3.39; CorDA withheld; LoRA-Null on-curve | **THINNER THAN THE FRAMING.** Real and interesting — *data-aware/activation inits inherit their calibration distribution* — but the evidence is **one clean positive (SC-LoRA), one withheld (CorDA), one already-on-curve (LoRA-Null)**. Cannot carry "we systematically characterized activation methods." Frame as a **fairness *requirement* the protocol exposes**, demonstrated on SC-LoRA, with CorDA honestly declared unassessable. Not a headline contribution. |
| **5** | "Training, init, and memory overheads compared" | Appendix efficiency paragraph (paper §app:repro): DoRA ~2.1× step time; CLoRA memory tax up to 6.7 GB at k2048 (analytical, peak not instrumented); data-aware inits one-time SVD/calibration tax; CorDA++ ~5× | **REAL but under-exhibited and partly analytical.** Good ACL material (practitioners care). Promote to a compact main-body "cost of geometry" table. **Flag:** CLoRA memory numbers are *analytical resident sizes, not metered peak* — must be labeled as such. |
| **6** | "Compare all with vanilla LoRA+wd" | C3: CS 81.8±0.2 / ret 25.9±0.4 @ frontier; math 66.8±0.8; param-matched r16 control (§16); E6 wd transfers to MiLoRA, breaks DoRA | **SOLID as "matches/edges the frontier," NOT as "LoRA+wd is best."** Single seed on the elaborate arms, rank/wd asymmetry, CorDA excluded. Honest verb "matches or edges" (C3 fix); "beats CLoRA +2.2" must anchor on **in-harness** CLoRA (60.8), both swept. |

**Net:** Contributions 2 and 3 (with the A1/A6 fixes) plus the protocol are the unassailable core. Contribution 6 is safe only under the "matches/edges" verb. **Contribution 4 is the one to dial back**, and **5 is the one to dial up** (cheap, concrete, ACL-friendly).

---

## 3. SECTION OUTLINE — 8-page ACL long paper + appendix

The current draft is a 9-page NeurIPS-shaped manuscript on stale numbers (n=49, single-seed, title "The Magnitude Law"). **The restructure is also the vehicle for landing the frozen n=1035 / multi-seed numbers — do both in one pass.** Target section budget (ACL 8-page body):

1. **Introduction (~1 p).** The zoo problem → two-layer contribution (protocol + finding) stated as such → the four instruments named → the three-claim spine + message. *Lands: A1 ("first-order predictor," not mechanism), A3 ("4 model×task settings under 6 recipes, 2 architectures"), A13 (method count), coverage sentence (7 of 8).* **Survives from current intro, retuned to the consolidation framing.**

2. **Related Work & Positioning (~0.75 p).** Keep the "adapter zoo organized by basis" taxonomy — it directly serves the consolidation frame. Keep the Biderman reconciliation and the CLoRA/Lee positioning. *Lands: O6 (delta vs CLoRA/Lee), O11 (Biderman).* **Survives, lightly compressed.**

3. **The Four Instruments (NEW, ~1 p) — the protocol section.** *The section the vision demands and the current paper lacks as a unit.* Define, once, the four measurements applied identically to every adapter: (i) adaptation/retention trade-off; (ii) F_Δ (fair magnitude axis, Eq-3, why over spectral norm — R² 0.72 vs 0.56/0.58); (iii) update geometry (SVD-alignment); (iv) CE-to-base drift. Then the two comparison instruments: per-method LR sweep, eval-matched calibration. Fold in the common-harness-is-a-precondition argument (O12) and scale validation. *Lands: A12 (n reconciliation pointer), the metrology of 03 §6.* **New assembly of material currently scattered across Setup.**

4. **The Magnitude Relation (~1.25 p, the hero).** §18.1 pooled r=−0.847, per-family; flat-then-falling with a knee (A9/§18.2); within-cell micro-test r=−0.713; **interventional E1** (15/15 on-curve, random-direction −3.05 pp — this is what lifts you above CLoRA); Qwen + math replication with honest scope. *Lands: A2 (scope causal claim to Llama-CS explicitly), the "relation not law" wording.*

5. **Geometry Is Second-Order (~1 p).** The nested ΔR² ladder (§19.1, magnitude 18× geometry) as the one-table headline; ANCOVA/LOMO; geometry-as-fingerprint. **Cut the fingerprint battery to its single best panel.** *Lands: A4 (commonality split, spec_max reclassified), A5 (scope geometry verdict to LoRA-variant family), 09-Q1 (drop the positive-spec_max sentence — it dies under clustering).*

6. **Learning Rate Is Only a Proxy (~0.75 p).** The rewritten battery (§18.5). **Retire the old "R² 0.74 vs 0.32" strawman** — lead with fixed-LR strata + partials + decoupling grids. *Lands: the §2 rewrite.*

7. **Behavioral Drift and the Two Channels (NEW/compressed, ~0.5 p).** Where the CE-metric contribution earns main-body space *without* over-claiming: KL is the proximal channel (05 §2), MMLU-Pro format damage is a second channel neutral-text drift can't see. Framed as *deepening* the magnitude finding, not an independent result. *Lands: A6 (CE as behavior-space view), A1 (drift is proximal).* Could merge into §5 if space is tight.

8. **The Practical Corollary + The Cost of Geometry (~1 p).** LoRA+wd matches/edges frontier (honest verb); param-matched control; wd-transfer boundary (E6). **Add the compact overhead table here** (contribution 5). *Lands: C3 (in-harness anchor), O4 (capacity), O7 (margin-not-the-point).*

9. **Discussion, Limitations, Repro (~0.75 p).** The honest ledger: single-arm causal test (Llama-CS), geometry verdict scoped to LoRA-variants, CorDA withheld, 284B designed-but-lost (sign-test framing, appendix only — 08 decision iv/A11), DoRA F_Δ lower bound *at submission not camera-ready*. Claims-evidence table.

**To fit 8 pages:** the current draft is over-long on the geometry fingerprint prose and the CLoRA cross-check — both compress heavily (cross-check → appendix, one sentence + fig in body).

---

## 4. EXHIBIT PLAN — MAIN (≤7) vs APPENDIX

ACL 8-page body realistically supports **6–7 exhibits**. Choose for the two-layer story.

**MAIN (7):**
1. **fig0 hero — retention vs F_Δ (the magnitude relation).** Non-negotiable. Update to multi-seed, all-arms, flat-then-falling with knee marked. Star the LoRA+wd point.
2. **fig1 axis — the fair-axis justification (F_Δ vs spectral).** Serves the protocol contribution; can be a single panel. Keep.
3. **NEW: ΔR² ladder / league table (magnitude 0.395 → geometry 0.017 → method 0.006).** The *one-table version of the title claim* (08 highlight #1), currently only in prose. The single most reviewer-proof exhibit — promote it. Include cluster-bootstrap CIs (09-Q1: 2000/2000 replicates order magnitude>geometry).
4. **fig9 LR-artifact** (the single-LR illusion trajectory) — the diagnosis, and the sharpest ACL "aha."
5. **fig3 Pareto / operating points** — the corollary; commonsense + math panels.
6. **NEW: "Cost of geometry" overhead table** (contribution 5) — train-time / init-time / memory per method vs LoRA+wd. Label CLoRA memory as analytical.
7. **table_main_cs** (per-method operating points, multi-seed, robustness column). The consolidation made tangible.

**Demote/merge:** fig2 (ANCOVA residuals) is now *redundant with the ΔR² ladder* — appendix. fig7 (LR proxy) overlaps fig9 — fig9 body, fig7 appendix. fig8 (budget) — one sentence + appendix.

**APPENDIX:** fig2, fig7, fig8, fig_geometry_4panel (one panel may surface in §5 if space allows), fig_cross_literature + full CLoRA Table-4 (one sentence + pointer in body), fig4/5/6, fig_ce_vs_magnitude, table_op_points, table_lr_artifact, table_claims_evidence, the n-reconciliation table (A12), the 284B sign-test table (A11), the DeepSeek recurrence.

**Rulings on anticipated NEW exhibits:**
- **Correlation heatmap/league table** → YES, as the ΔR² ladder (main #3). A raw metric-vs-metric heatmap is weaker; the *nested* ladder is the defensible form.
- **Metric-comparison table** → appendix (supports the four-instruments section).
- **Overhead "cost of geometry" table** → MAIN (#6).
- **CLoRA k & wd dose-response** → appendix (corroboration, not headline; guard the CLoRA-faithful line).
- **Per-benchmark fragility** (fig5) → appendix, but surface one sentence in §4.

---

## 5. ACL-SPECIFIC RISKS (ARR reviewer panel) + one-line mitigations

- **"Is this NLP / does it fit ACL?"** → Frame as forgetting *of language-model capabilities* (BBH/MMLU-Pro/TruthfulQA) under instruction/task fine-tuning; lead the intro with the deployment problem. The measurement-methodology genre is ACL-native — lean in.
- **"Limited task diversity — CS + math only."** → State it plainly; point to E7 bridging (MedMCQA, r=−0.878) as the third-domain signal; promise broader tasks. Don't over-sell "universal."
- **"Single epoch / one recipe scope."** → Disclose; the faithful-recipe (frm) block *is* each method's published recipe.
- **"7B scale only."** → The 284B wound. **Do not imply 284B generalizes the relation.** Sign-test recurrence framing (A11), appendix only. Promise the scale arm.
- **"Single seed on the elaborate arms / sub-noise Pareto margin."** → Pre-empt in Limitations: 287 cells n≥3; the *relation* is not a ranking; verb is "matches/edges"; vs plain LoRA (3-seed both sides) the edge is outside noise.
- **"CorDA excluded — you hid the method most likely to break your thesis."** → Pre-empt with the honest-coverage sentence *in the intro*, the port-bug disclosure, "7 of 8 assessed; CorDA withheld, not off-curve." Own it before they find it.
- **"Incremental over CLoRA."** → The protocol + the *cross-method* geometry-null + interventional E1 are the delta. Make the ladder and E1 visible early.

**Pending experiment → promise in paper:** B4/E4 completion retires calibration-fairness/CorDA gaps; **Qwen-CS + math rescale ladder (A2)** = single highest-value missing run (causal claim cross-architecture); Qwen base ceilings retire the slope-scale caveat.

---

## 6. NUMBERED DECISIONS — arbitration where the vision meets panel rulings

**D1 — "Four metrics checked equally" vs the single magnitude headline.** *Instruments, not co-equal findings.* Present the four metrics as one apparatus (main §3) applied identically to all adapters. The *narrative* stays single-spine. **Equal method, unequal message.**

**D2 — "First systematic review" risks a thesis-free benchmark.** *Keep the review framing; the magnitude relation is the load-bearing result.* The consolidation is the *contribution*; the law is the *point*. Both in the title; the finding drives every section.

**D3 — Contribution #4 (activation methods vs pre-forwarding).** *Demote from headline to a subsection of the protocol/fairness section.* Honest frame: "data-aware/activation initializations inherit their calibration distribution — a fairness requirement for any cross-method forgetting benchmark, demonstrated for SC-LoRA; CorDA honestly withheld."

**D4 — Contribution #5 (overheads).** *Promote to a compact main-body table.* Condition: label CLoRA memory as analytical resident size; DoRA 2.1× / CorDA++ 5× with source.

**D5 — Contribution #6 (LoRA+wd).** *Keep, under "matches or edges the frontier" only.* Anchor the CLoRA comparison in-harness (60.8), both swept.

**D6 — Title.** *Adopt the primary recommendation* ("Magnitude First, Geometry Second: A Fair, Multi-Metric Audit…").

**D7 — "Significant change in the current state of the paper."** *Restructure framing/packaging aggressively (new §3 four-instruments, new overhead table, consolidation-forward intro/title, land the frozen n=1035 numbers) — but do NOT relitigate the settled thesis or numbers.* The panel's rulings are guardrails, not suggestions. The big change is in *how the paper is sold*, not in *what it claims*.

---

## Assessment of the ceiling

The evidence base is genuinely A*-viable and honest enough to survive the adversarial pass with no fatal hit (08 §7): 1,035 adapters, multi-seed, a within-cell micro-test, an interventional arm, and a fairness battery that turned its one significant deviator into a fairness *win*. The ceiling at ACL is a strong accept / award-adjacent **if and only if** the paper is sold as what it provably is — a fair-measurement protocol that consolidates a fragmented field and discovers that magnitude, not geometry, is first-order — rather than as "LoRA+wd beats the zoo." To reach the ceiling: (1) execute the consolidation-forward reframe and land the frozen numbers; (2) make the ΔR² ladder and interventional E1 visible early (the delta over CLoRA/Lee); (3) wear the two real scope boundaries on the sleeve (single-arm causal test; geometry verdict scoped to LoRA-variants); (4) the one experiment that most raises the ceiling is the **Qwen-CS/math rescale ladder (A2)**.
