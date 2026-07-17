# CONCLUSIONS AND IDEAS — the master reference for the paper

*The single, self-contained document that details everything we concluded and everything we
considered, so the paper can be drafted directly from it. Written 2026-07-02 by the lead scientist.
Synthesizes `01_project_narrative.md`, `02_figures_tables_explained.md`, `03_gaps_and_roadmap.md`,
and `data/key_numbers.md` (the verified numbers file — its values override any older scale).*

---

## 0. Notation and a load-bearing units note (read first)

- The magnitude axis is the **token-weighted Frobenius norm of the weight update, `‖ΔW‖_F`**.
  Conceptually this is the field the brief calls `fdelta_token_weighted`.
- **In the actual campaign registry the field is named `fdelta`**, and it ranges **~0.05–3.7**.
  There is **no** `fdelta_token_weighted` column; `fdelta` *is* the token-weighted Frobenius norm.
  The "72–1395" magnitudes that appear in some handoffs / the brief are an **older, unnormalized
  `‖ΔW‖` scale**; every current figure, table, and number below uses the normalized `fdelta` scale.
  When the paper quotes a magnitude, it must quote the `fdelta` value (e.g. LoRA+wd at `‖ΔW‖_F ≈ 0.39`),
  not 72. This is the one place the older narrative and the verified numbers disagree — trust the
  numbers file.
- Retention (**core**) = mean(BBH answer-only 3-shot, MMLU-Pro 5-shot CoT). Base ceiling core = **26.0**
  (BBH-AO 33.10 + MMLU-Pro 18.96).
- Retention (**broad**) = mean(BBH, MMLU-Pro, MMLU, ARC-c, TruthfulQA).
- Adaptation = 8-task commonsense accuracy (CS-8) for the commonsense domain; GSM8K exact-match for math.
- All mature numbers are **Llama-2-7B, seed 42**, unless explicitly marked Qwen. Qwen is a
  replication **in progress**.

---

## 1. Thesis and the four claims

**Thesis (one paragraph).** Catastrophic forgetting under parameter-efficient fine-tuning is governed
by the **magnitude of the weight update**, `‖ΔW‖_F`, and *not* by the geometric structure of the
adapter that produced it. Across eight adapters — plain LoRA, LoRA+weight-decay, DoRA, CorDA, MiLoRA,
SC-LoRA, LoRA-Null, and CLoRA — spanning two task domains (commonsense, math) and (in replication) two
base models (Llama-2-7B, Qwen2.5-7B), retention collapses onto **one** curve set by `‖ΔW‖_F`. Because
the update magnitude is the true lever, the *simplest* possible magnitude control — plain LoRA plus a
weight-decay term — matches or surpasses the adaptation-vs-retention Pareto frontier of every elaborate
geometric adapter. The elaborate adapters' published "wins" are largely a **learning-rate artifact**:
evaluated at a single LR they look better, but sweeping the LR per method collapses them back onto the
shared magnitude curve. The message to a field that ships a new adapter every week is a wake-up call:
**control the magnitude, not the geometry.**

**The four claims (the paper's spine — keep this exact framing throughout):**

1. **MECHANISM.** Forgetting is governed by `‖ΔW‖_F`, not adapter geometry. All adapters fall on one
   retention-vs-`‖ΔW‖` curve.
2. **CONSEQUENCE.** Plain LoRA + weight decay matches or surpasses the Pareto frontier of the
   elaborate geometric adapters, because weight decay is the simplest way to bound `‖ΔW‖`.
3. **DIAGNOSIS.** The fancy adapters' reported wins are largely an LR/magnitude artifact; the
   LR-sweep-per-method is the instrument that exposes it.
4. **MESSAGE.** Control the magnitude, not the geometry.

**Study design.** Llama-2-7B (primary, complete) + Qwen2.5-7B (replication, in progress) ×
{commonsense, math} × 8 adapters (lora r16; lora+wd r32/wd0.3; dora r16; corda r16 KPA; milora r32;
sclora r32; lora_null r16; clora k1024) × 7 LRs (2e-5 … 1e-3) × seed 42.

---

## 2. Evidence for the magnitude law

### 2.1 The core correlations (mature, Llama-2, seed 42)

`r(retention, log10 ‖ΔW‖_F)`:

| Dataset | r | R² | slope (pp/decade) | n | p | status |
|---|---|---|---|---|---|---|
| **Llama-2 CS, pooled (6 methods)** | **-0.86** | 0.74 | -14.8 | 49 | 3e-15 | **mature** |
| Llama-2 CS, on-curve (excl. SC-LoRA) | **-0.92** | 0.84 | -10.0 | 42 | — | **mature** |
| **Llama-2 math, pooled** | **-0.97** | 0.93 | -10.1 | 14 | 2e-8 | mature (LoRA/LoRA+wd/DoRA only; sparse) |
| Qwen-2.5 CS (LoRA only) | **-0.88** | 0.78 | -34.8 | 7 | 9e-3 | replication, in progress |
| Qwen-2.5 math (LoRA only) | +0.67 | 0.45 | +2.0 | 5 | 0.21 | **does NOT yet replicate** (ns, flat, sparse) |

Headline number to quote: **r ≈ -0.86 pooled, -0.92 on the five well-behaved methods; slope ≈ -10 to
-15 pp per decade of `‖ΔW‖_F`.** Within-method correlations (Llama CS) are -0.86 to -0.98 (LoRA -0.97,
LoRA+wd -0.95, CLoRA -0.98, DoRA -0.86, CorDA -0.91, MiLoRA -0.96, SC-LoRA -0.88), i.e. every adapter
*individually* traces the same downward line, and the points **interleave** across methods.

### 2.2 Dual-domain generality

The law holds on **math (GSM8K)** as well as commonsense: Llama-2 math pooled r ≈ -0.97 (n=14, LoRA /
LoRA+wd / DoRA only). This dual-domain result is what lifts the finding from "a commonsense observation"
to "a law." **Caveat:** the math arm is sparse (only 3 adapters, n=14); the five structured arms
(CorDA/CLoRA/MiLoRA/SC-LoRA/LoRA-Null × 7 LRs) are still in the pool. Present math as a genuine second
domain but flag its thinness.

### 2.3 Two-model replication (in progress — present as supporting, not complete)

- Qwen-2.5 **CS** LoRA sweep (7 LRs): **r = -0.88** — the law replicates on the thin slice we have.
- Qwen-2.5 **math** LoRA sweep (5 LRs): r = +0.67 (ns, p=0.21) — **does not yet replicate**; needs the
  higher-LR cells to reach the forgetting regime.
- **~13 of ~112 planned Qwen cells complete**, and mostly the easy (LoRA) arm. No Qwen figures
  generated yet. ETA ~6–8 days to drain the live combined pool.
- Honesty rule: never present a Qwen Pareto or method-ranking table until the structured arms land.

### 2.4 Per-benchmark decomposition (mature, Llama-2 CS) — the law acts per-capability

Slope of each benchmark vs `log ‖ΔW‖_F`:

| Benchmark | slope (pp/decade) | r | base ceiling |
|---|---|---|---|
| **MMLU** | **-23.4** | -0.93 | uncalibrated |
| MMLU-Pro | -15.2 | -0.89 | 18.96 |
| ARC-Challenge | -14.9 | -0.93 | uncalibrated |
| BBH | -14.3 | -0.79 | 33.10 |
| **TruthfulQA** | **-0.5** | -0.10 | uncalibrated (flat / immune) |

Broad factual knowledge (MMLU) dies fastest with magnitude; truthfulness is essentially untouched. The
law is monotone in `‖ΔW‖` for every capability except TruthfulQA, which is flat — texture that shows the
law is not an artifact of a single benchmark.

### 2.5 Magnitude beats its confounds

- **Magnitude beats LR as a predictor** (the "LR is only a proxy" result): retention ~ log(`‖ΔW‖_F`)
  **R² = 0.74** vs retention ~ log(LR) **R² = 0.32** — R² more than doubles when you switch the x-axis
  from the learning rate to the update magnitude it produces. (Brief cites 0.75 vs 0.35; the verified
  numbers are 0.74 vs 0.32.)
- **`‖ΔW‖_F` beats `σ_max` as the axis.** σ_max is method-confounded — spiky inits (CorDA, DoRA,
  SC-LoRA) inflate it, which would mis-rank CorDA — so token-weighted Frobenius is the fair, tight axis.
- **No independent rank effect.** A LoRA rank sweep (r4 → r256) drives retention 25.4 → 8.5 purely by
  raising `‖ΔW‖`, with σ₁ growing with rank and no "diffusion." The earlier "rank surprisingly
  mitigates CF" surprise is dead — it folds entirely into the magnitude law.
- **Weight-SVD direction is causally irrelevant.** The directional leakage thermometer `μ_E`
  (`‖U_Rᵀ·E·Ū_r‖₂`, output leak into the preserved subspace) predicts retention at **r ≈ -0.09** — not
  at all. Direction, as measured in the static weight-SVD basis, does not modulate forgetting.

### 2.6 The non-circularity argument (must lead with this, not the raw r)

The raw correlation "bigger update → more forgetting" is near-tautological (a bigger perturbation
perturbs more). The **non-trivial, publishable** claim is that *method identity adds essentially
nothing beyond `‖ΔW‖`*, shown by the fairness/ANCOVA residual test (§5). Lead the results with the
residual test, present the raw r only as setup.

**Figure/table pointers:** `fig0_hero.png` (the law), `fig1_magnitude_law.png` (axis choice — three
candidate magnitude measures), `fig5_per_benchmark.png` (per-capability slopes), `fig7_lr_is_the_proxy.png`
(R² contrast), `fig8_magnitude_budget.png` (dual budget), `key_numbers.md` §1, §2, §7.

---

## 3. The Pareto result and the "wins-are-LR" diagnosis

### 3.1 LoRA+wd on/above the frontier (Llama-2 commonsense, best-adapt LR per method)

Each row is the method at its own adaptation-optimal LR (the fair, per-method-tuned comparison):

| Method | best LR | CS-8 | Ret-core | Ret-broad | `‖ΔW‖_F` | σ_max | robust (ret≥24, /7) |
|---|---|---|---|---|---|---|---|
| **LoRA+wd(0.3)** | 5e-4 | **81.6** | **25.6** | 33.2 | **0.394** | 34.3 | **6/7** |
| SC-LoRA | 5e-5 | 80.1 | 22.5 | 32.5 | 0.559 | 11.3 | 1/7 |
| MiLoRA | 3e-4 | 79.9 | 24.7 | 33.6 | 0.543 | 48.5 | 5/7 |
| LoRA | 3e-4 | 79.1 | 24.4 | 32.7 | 0.623 | 40.7 | (see note) |
| CLoRA | 5e-4 | 78.4 | 21.9 | 30.1 | 0.643 | 29.8 | 5/7 |
| DoRA | 2e-4 | 78.3 | 24.8 | 32.9 | 0.445 | 38.8 | 4/7 |

(CorDA excluded from all figures/tables pending re-validation — see §6.)

LoRA+wd(0.3) is the ONLY method that is simultaneously: highest adaptation (81.6), highest core
retention among the high-adapters (25.6, essentially at the base ceiling 26.0), **lowest** update
magnitude (0.394), and the widest safe operating band (6/7 LRs keep ret ≥ 24). It wins the upper-right
of the Pareto plane.

### 3.2 Math confirms it (sparse)

| Method | best LR | GSM8K | Ret-core | Ret-broad | `‖ΔW‖_F` | σ_max |
|---|---|---|---|---|---|---|
| **LoRA+wd(0.3)** | 5e-4 | **50.6** | **24.6** | 34.0 | 0.399 | 27.5 |
| LoRA | 3e-4 | 46.5 | 22.9 | 31.5 | 0.520 | 35.3 |
| DoRA | 2e-5 | 33.3 | 25.2 | 32.9 | 0.327 | 7.1 |

LoRA+wd again wins both axes; math sweep sparse (LoRA / LoRA+wd / DoRA only).

### 3.3 The "wins are LR" diagnosis

- **Best LR is not shared across methods** — it ranges over {5e-5, 1e-4, 2e-4, 3e-4, 5e-4}
  (SC-LoRA peaks at 5e-5, DoRA at 2e-4, LoRA/MiLoRA at 3e-4, LoRA+wd/CLoRA at 5e-4). A single fixed LR
  therefore *flatters whichever method happens to be well-tuned there* — this is the mechanism of the
  reported "wins."
- **Sweeping LR per method collapses the fancy adapters onto the shared magnitude curve.** The
  LR-sweep-per-method is the instrument that exposes the artifact. The "gotcha" exhibit to build:
  for each fancy adapter, plot its single-best-LR point (what its paper reports) beside its full 7-LR
  sweep collapsing onto the shared `‖ΔW‖` line. This visual carries claim 3.
- The sweep is **symmetric** — 7 LRs per method, including LoRA+wd — so we can answer the obvious
  referee question "did you tune LoRA+wd as hard as theirs?" with yes; the figure must make the
  symmetry obvious.

### 3.4 The honest strength of the claim

The defensible verb is **"matches / lands on the frontier at far lower engineering cost"**, and at best
**"edges out"** — NOT "dominates." Earlier full-scale reads that looked like domination
(e.g. "LoRA+wd dominates CLoRA") deflated to a tie/edge once a checkpoint-collision data bug was fixed
(§6). Full-scale clean reference points: wd0.1 = 80.4/24.86 ties CLoRA-k1024 79.8/24.85; wd1.0 =
76.7/**26.87** beats CLoRA-k2048 65.4/25.7. Frontier is noisy/non-monotone on a single seed, so
"matches" is the safe claim until seeds land.

**Figure/table pointers:** `fig3_pareto.png` (Pareto, both domains), `fig4_lr_sensitivity.png` (LR
optima not shared), `fig7_lr_is_the_proxy.png` (the mechanism), `op_points.png` (operating-point table,
robustness), `table_main_cs.tex`, `table_main_math.tex`, `key_numbers.md` §3, §4.

---

## 4. Per-method characterization

Where each adapter sits on the curve, whether it is "on" (indistinguishable from the law) or "off," and
why. Residual = pp above/below the pooled magnitude curve (ANCOVA, §5); positive = retains better than
budget predicts.

- **LoRA (r16), the reference.** Vanilla Gaussian-A / zero-B. Within-method r = -0.97; ANCOVA residual
  **+0.79 pp (ns) → ON the law.** Best point 79.1 / 24.4 at lr3e-4, `‖ΔW‖_F` = 0.623 (near the right
  edge of the sweet-spot band, so it pays retention for its adaptation). The baseline the whole story is
  built around.

- **LoRA+wd (r32, wd0.3), the protagonist.** AdamW weight decay applied to the adapter matrices only
  (base + LayerNorm frozen) — a subspace-free direct magnitude knob. Within-method r = -0.95; residual
  **+0.06 pp (ns) → ON the law**, i.e. it wins by *moving along* the curve to a smaller `‖ΔW‖`
  (0.394, inside the sweet-spot band), not by beating the curve. Wins CS (81.6/25.6) and math
  (50.6/24.6), lowest magnitude, widest robustness (6/7). The existence proof for claim 2.

- **DoRA (r16).** Magnitude-direction-decomposed LoRA. Within-method r = -0.86 (loosest single-method
  fit); residual **+1.37 pp (ns) → ON the law.** Best 78.3/24.8 at lr2e-4, `‖ΔW‖_F` = 0.445 (low).
  Spiky σ_max (38.8) — one reason σ_max is an unfair axis. Robustness 4/7. Solid but does not beat
  simple wd.

- **CorDA (r16, KPA/nq_open calib).** Data-driven: SVD of `W·Cov_input` over a calibration set, freeze
  the principal (knowledge) components, adapt the smallest-r (KPA retention mode). Port verified
  faithful to PEFT; its large A-norm (~49) is *genuine* KPA (C_inv un-whitening), not a bug; 0-step → ΔW=0.
  **Currently EXCLUDED from every figure/table** because it was mis-calibrated on wikitext-2 and is
  re-running on the paper's nq_open. Older wikitext point 77.9/19.9. Earlier "off-curve by ~3pp" reading
  is **pending and confounded** — do not report it. On the diagnosis: as a data-aware init it transmits
  the same LR into a larger `‖ΔW‖`, which is why it drifts toward the wrong end of the frontier.

- **MiLoRA (r32).** Init from the **bottom-r (minor)** singular triples of W₀; residual = principal.
  Port faithful (selects minor, not PiSSA principal). Within-method r = -0.96; residual **+1.04 pp (ns)
  → ON the law.** Best 79.9/24.7 at lr3e-4, `‖ΔW‖_F` = 0.543, robustness 5/7. A clean example of an
  elaborate init that adds nothing beyond its magnitude.

- **SC-LoRA (r32, β0.5), the one off-curve deviator.** Subspace-constrained init:
  `M=(1-β)Cov₊ - β·Cov₋`, top-r eigenvectors as B, `A = Qᵀ W₀`, residual `W₀ - QQᵀW₀`. Math faithful.
  Within-method r = -0.88; ANCOVA residual **-4.15 pp (p=0.006) → the ONLY statistically significant
  below-curve deviator.** Best 80.1/22.5 at lr5e-5, `‖ΔW‖_F` = 0.559, and **brittle: robustness 1/7**
  (its only safe LR forces adaptation to collapse). It transmits more `‖ΔW‖` per LR (fig7) — the
  candidate mechanism for the extra forgetting. **PROVISIONAL:** confounded by the calibration↔eval
  mismatch (§6) and single seed; ringed as an outlier in figures, not headlined as a real effect yet.
  Open fidelity nuance: per-sample norm `Y.max().abs()` (|max|) vs the CorDA-family `Y.abs().max()`
  (max|·|), and calib max_len 2048 vs repo 512/1024.

- **LoRA-Null (r16).** Input-covariance **null-space** init: `BA = W₀·U_null·U_nullᵀ` from the
  smallest-SV left vectors; residual `W₀ - BA`. Ported this session (unit-tested: recon 2.4e-7,
  silent-on-used-dirs 3.6e-7); calibrated nq_open. **Not shown as its own series** because of a data
  labeling bug: `lrsw_lora_null_*` is classified as method `"lora"` in the figure generator, so its 7
  points silently pool into plain LoRA (see §6). The pooled law is identical either way; the LoRA
  legend/robustness/n must be corrected before publication. Fidelity flag: `null_dim` default = r is an
  assumption.

- **CLoRA (k1024).** The one loss-term method (vanilla init): penalty
  `λ·Σ(½‖A·Pv‖² + ½‖Bᵀ·Pu‖²)` against a frozen random-orthonormal P of width k, λ=1. Port faithful.
  Within-method r = -0.98; residual **+0.09 pp (ns) → ON the law.** Best 78.4/21.9 at lr5e-4,
  `‖ΔW‖_F` = 0.643 (right edge), robustness 5/7. Over-constrains at extreme k (k2048 CS → 65 vs paper
  83.7). The prior-art anchor: CLoRA already links magnitude to CF, so our delta is *method-freeness
  across 8 adapters × 2 domains × 2 models*, not the discovery that magnitude matters.

- **UIOrthoLoRA / UILinLoRA (dead as a method, kept for provenance).** Our home-grown
  orthogonal-rotation / linear-scaling adapter in a truncated-SVD spectral-tail basis. Dominated
  (adaptation ceiling ~74 CS vs ~79–80), removed from the active campaign. It served its purpose as the
  *instrument* whose independent knobs (directional leakage vs magnitude) let us prove direction is inert
  and magnitude is the budget — the empirical origin of the whole thesis.

**Summary of ANCOVA residuals (Llama-2 CS):** LoRA +0.79, LoRA+wd +0.06, MiLoRA +1.04, CLoRA +0.09,
DoRA +1.37 (all ns → on the law); **SC-LoRA -4.15 (p=0.006, the sole deviator, provisional).** Five of
six adapters are statistically indistinguishable from the magnitude law.

---

## 5. The mechanistic story

Why does the same recipe transmit into different `‖ΔW‖`, and why does weight decay fix it?

1. **The same LR becomes a different-sized update depending on the init.** Data-aware inits (CorDA,
   SC-LoRA) start the adapter in a subspace where the loss gradient is large, so at a *fixed* learning
   rate they accumulate a **larger** `‖ΔW‖` than a vanilla init would (fig7 panel A: LR → resulting
   `‖ΔW‖`, method-dependent). This is the transmission channel: geometry does not change the *shape* of
   the retention curve, it changes *where on the curve* a given LR lands you.

2. **Retention is a monotone function of the resulting `‖ΔW‖` — full stop.** Once you know `‖ΔW‖`, the
   init that produced it carries essentially no extra information about how much the model forgot
   (ANCOVA residuals ~0 for 5/6 methods). So the elaborate geometry buys you a different *operating
   point*, not a different *law*.

3. **Adaptation also rises with `‖ΔW‖`, so magnitude is a two-edged budget.** Adaptation slope = **+20.3
   pp/decade**, retention slope = **-14.8 pp/decade** of `‖ΔW‖`. There is a narrow **sweet-spot band
   `‖ΔW‖_F ∈ [0.31, 0.62]`**: near-max adaptation while retention stays near the base ceiling. Adapters
   that fall off the frontier do so because their (init × LR) combination pushes `‖ΔW‖` past the right
   edge of this band.

4. **Weight decay lands you in the sweet spot for free.** wd directly penalizes `‖A‖² + ‖B‖²`, i.e. it
   *bounds* `‖ΔW‖` regardless of subspace. LoRA+wd(0.3) sits at `‖ΔW‖_F` = 0.394 — inside the band —
   while the un-regularized high-adapters (LoRA 0.623, CLoRA 0.643, SC-LoRA 0.559) sit at or past the
   right edge. That is the entire mechanism of claim 2: the geometric adapters are elaborate ways to
   choose *where* the update goes; weight decay is the trivial way to choose *how big* it is, and how
   big is what matters.

5. **Why the weight-SVD basis is the wrong place to look for "direction."** Our directional thermometers
   live in the *static weight-SVD* basis and see nothing (`μ_E` r=-0.09). The field's counter-claim
   (CorDA/SC-LoRA) is that knowledge lives in the **data/activation-covariance** basis. So "direction
   doesn't matter" is more precisely "direction *in the weight-SVD basis* doesn't matter"; the honest
   framing is that even the data-basis methods, once you control `‖ΔW‖`, add ~nothing — but the
   calibration↔eval fairness question (§6) must be settled before that is a hard claim.

**Figure pointers:** `fig7_lr_is_the_proxy.png` (transmission channel + R² contrast),
`fig8_magnitude_budget.png` (dual budget + sweet-spot band), `fig6_supporting_structure.png` (adaptation
requires magnitude; σ_max spikiness; efficiency at fixed retention).

---

## 6. Honest limitations and fairness caveats

State every one of these in the paper. They are the referee attack surface; owning them is the defense.

1. **Single seed (s42).** Fine for the LAW (49 points trace the curve robustly). **NOT** fine for any
   head-to-head *ranking* claim: the earlier 3-seed matrix exposed single-seed "collapse basins"
   (seed 44 collapsed clora_k2048 → 23, dora_r8 → 22, lorawd_wd0p5 → 51). Error bars (seeds 43/44) are
   required on the ~6–8 headline cells before headlining any per-method delta or the off-curve verdict.

2. **Rank / knob asymmetry.** Ranks are NOT matched: LoRA/DoRA/CorDA/LoRA-Null r16; MiLoRA/SC-LoRA r32;
   CLoRA k1024. And **only LoRA has the wd knob.** A referee will say "LoRA+wd wins because it has more
   rank and an extra regularizer no one else got." The LAW framing sidesteps this (it is a statement
   about `‖ΔW‖`, not a ranking); but the moment we headline "LoRA+wd surpasses fancy adapters," the gap
   is live. A **param-matched LoRA+wd control (r16 and r32)** and, ideally, the wd knob given to every
   frontier arm, are the honest prerequisites.

3. **Calibration↔eval distribution mismatch (the biggest open fairness question).** CorDA, SC-LoRA, and
   LoRA-Null calibrate on **nq_open** (factoid QA), but retention is evaluated on BBH/MMLU/MMLU-Pro/
   ARC/TruthfulQA (academic/reasoning). KPM/KPA only protects directions the calibration covariance
   exercises, so nq_open may protect the *wrong* subspace for our eval — meaning "data-aware inits
   forget more than their budget" could be an artifact of handicapping them with a mismatched
   calibration set, not a real geometric penalty. Until an **eval-matched calibration** re-run (plus an
   nq_open-vs-matched sensitivity arm) lands, **all off-curve / "fancy adapters forget more" / "LoRA
   beats CorDA" language stays out.**

4. **CorDA is doubly caveated and currently excluded.** It was first mis-calibrated on wikitext-2
   (understating its retention), fixed to nq_open, and is re-running; it is excluded from every current
   figure/table. Do not report any CorDA number or the "CorDA off-curve" finding until the nq_open runs
   complete AND the calibration↔eval question (caveat 3) is addressed.

5. **The SC-LoRA off-curve deviation is PROVISIONAL.** The single significant ANCOVA deviator (-4.15 pp)
   is confounded by both caveats 3 (calibration mismatch) and 1 (single seed). Present it as a ringed
   outlier and an open question, not a result.

6. **Qwen is in progress.** ~13/112 cells, mostly LoRA; CS-LoRA law replicates (r=-0.88), math-LoRA does
   not yet (r=+0.67, ns). No Qwen figures. Present as supporting replication, never as a completed
   second model or a second ranking.

7. **Math domain is sparse** (n=14, LoRA/LoRA+wd/DoRA only). A real second domain once the structured
   arms land, an anecdote until then.

8. **Base-ceiling calibration missing** for MMLU / ARC / TruthfulQA (BBH and MMLU-Pro have it).
   Retention percentages for those three are uninterpretable without their 0-adapter base scores. Cheap
   (5 eval-only runs) — do before any camera-ready percentage.

9. **Near-circularity of the raw correlation.** "Bigger update → more forgetting" is partly true by
   construction. Rebut by leading with the *residual/ANCOVA* test (method identity adds ~nothing beyond
   `‖ΔW‖`), not the raw r.

10. **Data-labeling bug in the figure generator.** `lrsw_lora_null_*` is classified as method `"lora"`,
    so LoRA-Null's 7 points pool into plain LoRA in the legend/robustness/n. The pooled law and the
    best-adapt LoRA point are unaffected, but this must be corrected before camera-ready and all figures
    regenerated.

11. **Minor fidelity flags to confirm if challenged:** SC-LoRA per-sample norm |max Y| vs max|Y|, and
    SC-LoRA calib max_len 2048 vs repo 512/1024; LoRA-Null `null_dim` default = r is an assumption.

---

## 7. Ideas and future directions (everything we considered)

1. **Param-matched LoRA+wd control (C3).** A LoRA+wd r16 and r32 control so the win is shown at matched
   capacity; stronger, give the wd knob to the 2–3 frontier arms at ≥2 wd values, turning the claim into
   "wd helps everyone, and geometry adds nothing on top." Fallback if not run: retreat claim 2 to
   "LoRA+wd lands on the same frontier at far lower engineering cost."

2. **Eval-matched / calibration-matched arms (C2).** Re-run all calibration-using arms (CorDA, SC-LoRA,
   LoRA-Null) with MMLU/ARC auxiliary-train calibration (disjoint from test), 256 samples, shared across
   arms, plus an nq_open-vs-matched sensitivity arm. If they move back onto the curve → the *cleaner*
   result ("the law is method-free once calibration is fair"). If they stay off → a real second-order
   effect. Either way it gates all off-curve language. Also required: PEFT-CorDA residual round-trip via
   `path_initial_model_for_weight_conversion` with the init-output-invariance check run *after* reload.

3. **CorDA++ as the advanced arm (C6).** Dynamic covariance selection + dynamic rank allocation under a
   fixed param budget (arXiv:2506.13187, algorithms transcribed; compactness
   `π(C)=√(d_out·σ_max)/σ_min`). This is the "we didn't strawman the SOTA" arm — it lands the negative
   result about geometry against the *strongest* 2025-era geometric method, not a 2023 one. Dynamic rank
   breaks nominal param parity, so report the **realized** trainable-param count matched to our
   r16-equivalent budget (28,049,408 params). Execute after the 2×2.

4. **Seeds 43/44 on headline cells only (C4).** Error bars on the ~6–8 cells that appear in a headline
   table/figure — cheap insurance against the "n=1 method comparison" desk-reject, not the full grid.

5. **Base-ceiling calibration (C5).** 5 eval-only runs for MMLU/ARC/TruthfulQA base scores.

6. **The retention-corner / gated-magnitude adapter.** The constructive follow-up: an input-conditional
   gate g(x) scaling only the adapter delta, with a dual loss against a preservation corpus *disjoint
   from the eval set*; oracle ≈ (CS 79, ret 26) dominates the whole frontier. Deprioritized after LoRA+wd
   (the gate now has to beat wd-LoRA, a higher bar) — future work, not the lead.

7. **Measurement-tool framing (the fallback paper).** If the ranking claims cannot be made
   referee-proof, the paper still stands as a *measurement / diagnosis* contribution: the
   LR-sweep-per-method is an instrument that decomposes any adapter's reported win into its true
   magnitude effect plus an LR-tuning artifact, and the `‖ΔW‖_F` axis (over σ_max / LR) is the fair way
   to compare adapters. This is the guaranteed-defensible core even if fairness experiments change the
   rankings.

8. **The directional norm `‖ΔW·C_retain^½‖` (data-basis magnitude).** Hoped to be the non-circular
   headline axis; currently only marginally better than raw `‖ΔW‖_F` (-0.79 vs -0.77, n=8, within
   noise). Worth revisiting with more points as a *less circular* magnitude measure.

**Prioritized critical path (single 8-GPU B200, one scheduler):** (1) drain the combined pool
(Qwen CS+math + L2-math structured arms, ~6–8 d, running) → (2) build the Pareto + fixed-vs-swept-LR
gotcha exhibit (start NOW on mature Llama data, no GPU needed) + (3) base-ceiling calibration (parallel,
eval-only) → then single-scheduler order (4) eval-matched calibration → (5) param-matched controls →
(7) CorDA++, with (6) seeds on the headline cells identified along the way. **Two tiers:** minimum
defensible "law" paper (items 1–4 + seeds-lite, ~2 weeks, softened ranking); strong "wake-up call"
paper (all items, ~3.5–4 weeks).

---

## 8. Paper narrative outline (section-by-section for the writer)

**Title direction:** "It's the Magnitude, Not the Geometry: A Wake-Up Call for PEFT Forgetting."

- **Abstract.** State the four claims in order. Lead with the law (r≈-0.86 pooled / -0.92 clean,
  dual-domain, replicating on a second model), then the consequence (LoRA+wd matches/beats the
  frontier), then the diagnosis (wins are an LR artifact), then the message. Teaser graphic = `fig0_hero`.

- **§1 Introduction.** The field ships a new forgetting-mitigation adapter every week, each claiming a
  win over the last. We show those wins are largely an LR/magnitude artifact and that a single scalar —
  `‖ΔW‖_F` — governs forgetting method-independently. Frame as a controlled, negative, wake-up-call
  contribution. State the honest scope up front (Llama-2 mature, Qwen in progress).

- **§2 Related work & positioning.** Basis axis: static weight-SVD (OPLoRA) vs data/activation
  covariance (CorDA/SC-LoRA/CorDA++) vs gradient/Fisher. CLoRA already links magnitude to CF — our delta
  is *method-freeness across 8 adapters × 2 domains × 2 models*, that geometry is causally inert, and
  that per-method wins are an LR artifact. **Verify every arXiv ID before submission** (all currently
  `[VERIFY]`; `2603.02224` is almost certainly wrong).

- **§3 Method & measurement.** The shared-trainer design (every structured adapter is "LoRA with a
  different init"; only CLoRA adds a loss term). The magnitude axis `‖ΔW‖_F` and why it beats σ_max and
  LR (`fig1`, `fig6b`). The retention suite (core = BBH-AO + MMLU-Pro; broad adds MMLU/ARC/TruthfulQA;
  base ceiling 26.0). The **port-fidelity audit** — "we prove our port is faithful before claiming a
  method fails" — as a referee shield; mention the residual-save 0-step check.

- **§4 Result 1 — the magnitude law.** `fig0_hero` + `fig8_magnitude_budget` + `fig5_per_benchmark`.
  Correlations, dual-domain, per-capability (MMLU fastest, TruthfulQA immune). Lead the non-circularity
  argument here.

- **§5 Result 2 — geometry doesn't matter (fairness/ANCOVA).** `fig2_fairness_residuals`. 5/6 adapters
  indistinguishable from the law; SC-LoRA the sole (provisional) deviator. This is the load-bearing
  non-circular result.

- **§6 Result 3 — LoRA+wd wins the Pareto.** `fig3_pareto` + `table_main_cs` + `table_main_math` +
  `op_points`. State the honest verb ("matches / edges"). Fold in the fairness caveats (rank/wd
  asymmetry) explicitly.

- **§7 Result 4 — the LR artifact.** `fig4_lr_sensitivity` + `fig7_lr_is_the_proxy` + the
  fixed-LR-vs-swept-LR "gotcha" exhibit. Best LR is not shared; the R² contrast (0.74 vs 0.32); the
  symmetric-sweep answer to "did you tune LoRA+wd as hard."

- **§8 Discussion / the wake-up call.** Control the magnitude, not the geometry; weight decay lands you
  in the sweet-spot band for free; practical operating-point guidance (`op_points`, robustness).

- **§9 Limitations.** The full §6 list, unflinching: single seed, rank/wd asymmetry, calibration↔eval
  mismatch (biggest), CorDA excluded, SC-LoRA provisional, Qwen in progress, sparse math, base-ceiling
  gaps, near-circularity, the labeling bug.

- **§10 Future work.** CorDA++, calibration-matched arms, param-matched controls, the gated-magnitude
  retention-corner adapter, the measurement-tool framing.

- **Appendix.** `fig6_supporting_structure`; full per-method within-correlations and residuals; the
  port-fidelity audit details; the four Qwen eval-bug fixes (pad token, BBH normalization, gen_cap=1024,
  max_len=4096, each proven a no-op on Llama-2); the units note (§0).
