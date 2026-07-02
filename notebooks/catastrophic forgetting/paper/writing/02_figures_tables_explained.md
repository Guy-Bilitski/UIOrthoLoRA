# Figures & Tables, Explained

Every figure and table curated for the paper, made self-explanatory. For each: **filename**,
**what it plots** (axes/series), the **exact key numbers**, the **one-sentence takeaway**, and
**which thesis point (1–4) it supports**.

**The thesis spine** (referenced by number throughout):
1. **MECHANISM** — forgetting is governed by weight-update magnitude ||dW||_F, not adapter geometry; all adapters fall on ONE retention-vs-||dW|| curve.
2. **CONSEQUENCE** — plain LoRA + weight decay matches or beats the adapt-vs-retention Pareto frontier of elaborate geometric adapters, because wd is the simplest way to control ||dW||.
3. **DIAGNOSIS** — fancy adapters' reported "wins" are largely an LR/magnitude artifact; sweeping LR collapses them onto the same magnitude curve.
4. **MESSAGE** — control the magnitude, not the geometry.

All figures: **Llama-2-7B, seed 42**, commonsense LR-sweep (`lrsw_*`, 6 methods × 7 LRs, n=49
points incl. LoRA-Null mislabeled as LoRA — see mismatch #1) unless stated. Math uses `lrswm_*`.
Magnitude axis = `fdelta` = token-weighted ||dW||_F. Base retention ceiling (no-FT) core = **26.0**.
CorDA is **excluded** from every figure (data under re-validation).

Files live in `writing/figures/*.png` and `writing/tables/*`. Raw registry: `writing/data/campaign_summary.jsonl`.
All load-bearing numbers are in `writing/data/key_numbers.md`.

---

## FIGURES

### fig0_hero.png — THE hero figure
- **Plots:** retention (core, mean of BBH & MMLU-Pro) [y] vs ||dW||_F [x, log]. Black line = magnitude law fit on the 5 on-curve adapters + 95% bootstrap CI; all 6 methods by color/marker; SC-LoRA ringed as the single below-curve outlier; green dotted base ceiling.
- **Key numbers:** on-curve fit **r=-0.92, R²=0.84, slope=-10 pp/decade** (n=42); all-6 pooled r=-0.86; SC-LoRA sits ~4pp below; base ceiling 26.0.
- **Takeaway:** Retention collapses along a single tight curve set by the SIZE of the weight update, not by which adapter produced it.
- **Supports:** Thesis **1** (the mechanism, front and center).

### fig1_magnitude_law.png — choosing a fair magnitude axis
- **Plots:** 3 panels, retention [y] vs three magnitude measures [x, log]: (chosen) ||dW||_F token-weighted Frobenius, mean spectral norm σ̄, max spectral norm σ_max. Pooled log-fit + bootstrap CI + all 6 methods each panel.
- **Key numbers:** ||dW||_F is the tightest fit (r=-0.86, R²=0.74); σ_max is loosest and method-confounded (spiky inits inflate it). Blue-highlighted panel = the axis we adopt.
- **Takeaway:** Of the candidate magnitude measures, token-weighted Frobenius ||dW||_F gives the tightest, most method-neutral fit — so it is the fair x-axis for the whole paper.
- **Supports:** Thesis **1** (justifies the measurement instrument).

### fig2_fairness_residuals.png — ANCOVA fairness test
- **Plots:** Left = data + pooled spline "law" (5 adapters trace it; SC-LoRA ringed below). Right = per-method residual box/strip from the pooled curve; green band = "indistinguishable from the law" zone; μ labels per method.
- **Key numbers:** pooled R²=0.74 → +per-method intercepts R²=0.87, **ANCOVA F(5,42)=8.3, p<0.001**. Residual means: LoRA +0.8, LoRA+wd +0.1, MiLoRA +1.0, CLoRA +0.1, DoRA +1.4 (all ns); **SC-LoRA -4.15 pp, p=0.006** (only significant deviator).
- **Takeaway:** 5 of 6 adapters are statistically indistinguishable from the magnitude law; only SC-LoRA forgets more than its budget predicts (and that is provisional).
- **Supports:** Thesis **1** (geometry adds ~nothing once magnitude is controlled) + honesty caveat.

### fig3_pareto.png — adaptation–retention Pareto
- **Plots:** 2 panels (CS, math). Retention [y] vs adaptation [x] per method; each point = one LR; open ring = each method's best-LR (max adapt); red rings = training-collapse outliers; green base ceiling.
- **Key numbers:** CS best-adapt points — LoRA+wd (81.6, 25.6) dominates the upper-right; LoRA (79.1, 24.4), MiLoRA (79.9, 24.7), DoRA (78.3, 24.8), SC-LoRA (80.1, 22.5), CLoRA (78.4, 21.9). Math — LoRA+wd (50.6, 24.6) beats LoRA (46.5, 22.9). Collapse rings: clora k1024@1e-4, dora r16@2e-5.
- **Takeaway:** Plain LoRA+wd sits on or above the Pareto frontier of every geometric adapter, on both domains.
- **Supports:** Thesis **2** (the consequence — simplest method wins the tradeoff).

### fig4_lr_sensitivity.png — LR optima are not shared
- **Plots:** 2 panels vs LR [x, log]: adaptation [left] and retention [right] per method; rings = each method's adaptation-optimal LR; retention monotone-decreasing in LR.
- **Key numbers:** best-adapt LRs differ across methods (SC-LoRA 5e-5, DoRA 2e-4, LoRA/MiLoRA 3e-4, LoRA+wd/CLoRA 5e-4) — the optimum is NOT shared.
- **Takeaway:** Because each adapter peaks at a different LR, evaluating everyone at one fixed LR biases the comparison — the root of the artifact.
- **Supports:** Thesis **3** (mechanism of the LR artifact).

### fig5_per_benchmark.png — which knowledge dies first
- **Plots:** 5 panels (BBH, MMLU, MMLU-Pro, ARC-c, TruthfulQA), each accuracy [y] vs ||dW||_F [x, log] + pooled slope; 6th panel = degradation-slope bar chart.
- **Key numbers:** slope pp/decade — MMLU -23.4 (r=-0.93, fastest), MMLU-Pro -15.2, ARC-c -14.9, BBH -14.3, **TruthfulQA -0.5 (flat, immune)**.
- **Takeaway:** Broad factual knowledge (MMLU) is the first casualty of magnitude; truthfulness is essentially untouched — the law acts per-capability but is monotone everywhere except TruthfulQA.
- **Supports:** Thesis **1** (mechanism generalizes across capabilities).

### fig6_supporting_structure.png — the tension, spectral fairness, efficiency
- **Plots:** 3 panels — (a) adaptation vs ||dW|| (rises: magnitude is REQUIRED for adaptation); (b) σ_max/σ̄ spikiness per method (why σ_max is an unfair axis: DoRA/SC-LoRA spike, LoRA flat); (c) adaptation achievable at fixed retention budget (ret=24) per method.
- **Key numbers:** adaptation rises with ||dW|| (slope +20 pp/decade, see fig8); spectral spikiness varies widely by method → σ_max confounds; efficiency bars rank methods at equal retention.
- **Takeaway:** Magnitude is a two-edged budget (needed for adaptation, fatal to retention), and σ_max is unfair because inits differ in spikiness — supporting the choice of ||dW||_F.
- **Supports:** Thesis **1** (axis fairness) + **2** (efficiency framing).

### fig7_lr_is_the_proxy.png — LR is only a proxy; ||dW|| is the cause
- **Plots:** 3 panels — A: LR → resulting ||dW|| per method (data-aware inits, e.g. SC-LoRA, turn the same LR into a LARGER update); B: retention vs LR (loose); C: retention vs ||dW|| (tight). B/C share y so the R² contrast is the message.
- **Key numbers:** **retention~log(LR) R²=0.32** vs **retention~log(||dW||) R²=0.74** — R² more than doubles. SC-LoRA transmits more ||dW|| per LR (candidate mechanism for its extra forgetting).
- **Takeaway:** Learning rate predicts forgetting only loosely; the ||dW|| it produces predicts it tightly — LR is a confounded proxy for the true cause.
- **Supports:** Thesis **3** (the diagnosis — this is the instrument that exposes the artifact).

### fig8_magnitude_budget.png — one axis sets both adaptation and forgetting
- **Plots:** 2 stacked panels sharing ||dW||_F [x, log]: adaptation (rises) [top], retention (falls) [bottom]; green sweet-spot band; base ceiling + safe threshold marked.
- **Key numbers:** adaptation slope **+20.3**, retention slope **-14.8** pp/decade; **sweet-spot band ||dW||_F ∈ [0.31, 0.62]**. LoRA+wd (0.394) sits inside it; un-regularized high-adapters sit at/past the right edge.
- **Takeaway:** A single knob — ||dW||_F — trades adaptation against retention, and there is a narrow sweet-spot band that weight decay lands you in for free.
- **Supports:** Thesis **1** + **2** + **4** (control the magnitude).

### fig9_lr_artifact.png — THE LR-artifact exhibit (Claim 3's headline figure)
- **Plots:** the adapt–retention plane directly (x = commonsense accuracy, y = mean(BBH, MMLU-Pro)). Each method's **full 7-LR trajectory** is drawn as a faint connected line + markers; LoRA (plain, black-edged) and LoRA+wd (bold diamonds) are emphasized. **LoRA+wd's Pareto frontier is highlighted as a thick blue band**, and the region it dominates is lightly shaded. Each fancy method's *best-looking single-LR point* is **ringed** (the "if you only ran one LR" illusion point). Green dotted base-retention line at 26.0. CorDA deduped to its latest nq_open-calibrated eval; its diverged lr1e-3 point (fdelta≈516) dropped.
- **What it proves:** at a **fixed** LR a structured adapter can look better than LoRA (its ringed point), but once you **sweep** the LR, LoRA+wd traces a frontier that sits at or above every ringed point — the "win" evaporates. The long tails plunging to ret≈3–5 (CorDA, SC-LoRA at high LR) show the same methods self-destruct once their weight-update magnitude blows up: it was never geometry, it was the LR-controlled ‖dW‖.
- **Key numbers (best single-LR point → dominating LoRA+wd point):** DoRA 78.3/24.8 (lr2e-4) → LoRA+wd 81.6/25.6; MiLoRA 79.9/24.7 (lr3e-4) → 81.6/25.6; SC-LoRA 79.5/25.3 (lr2e-5) → 81.6/25.6; LoRA-Null 73.0/26.2 (lr2e-5) → LoRA+wd 80.6/26.2 (equal retention, +7.5 adapt); CLoRA 64.9/24.3 (lr3e-4) → 81.6/25.6; CorDA 76.2/19.4 (lr5e-5, never clears ret≥24) → 81.6/25.6. **LoRA+wd's swept frontier Pareto-dominates all six** methods' best single-LR points. LoRA+wd frontier vertices: (adapt/ret) 45.4/27.8 (lr5e-5), 77.0/26.9 (lr1e-4), 80.6/26.2 (lr3e-4), 81.6/25.6 (lr5e-4).
- **Companion:** `tables/table_lr_artifact.tex` gives the single-LR-vs-swept comparison numerically (see TABLES).
- **Honest caveats (do not overstate):** the *single-LR illusion* is strong for **SC-LoRA (+26.0 adapt vs LoRA at the same LR), LoRA-Null (+19.5), DoRA (+5.9)**; **weak for MiLoRA (+0.8 adapt, +0.3 ret)**; and **absent for CLoRA** — at its best LR CLoRA is already −14.2 adapt *behind* plain LoRA at the same LR, so no "win" ever existed to be an artifact (plain LoRA dominates it outright, and LoRA+wd more so). **CorDA** never Pareto-beats LoRA at a matched LR (it trades +38 adapt for −7.9 ret), and its interpretation is still confounded by the calibration↔eval fairness question (B4) — so CorDA reads as *magnitude-driven collapse*, not a clean LR-artifact win. Single seed (s42), n=7 LRs/method, Llama-2-7B commonsense only (math sweep too sparse for the fancy arms).
- **Supports:** Thesis **3** (the diagnosis) — this is the figure Claim 3 was missing.

### op_points.png — operating-point table (figure form)
- **Plots:** table — per method: best-LR (adapt/ret), safe-LR (max adapt with ret≥24), robustness (#LRs of 7 keeping ret≥24), color-coded green→red.
- **Key numbers:** robustness — LoRA+wd 6/7 (widest); MiLoRA/CLoRA 5/7; DoRA 4/7; **SC-LoRA 1/7 (brittle)**. SC-LoRA's only "safe" LR forces adaptation to collapse (⚠).
- **Takeaway:** LoRA+wd has by far the widest safe operating window; data-aware inits transmit too much ||dW|| at every usable LR, leaving them brittle.
- **Supports:** Thesis **2** + **4** (practical payoff of controlling magnitude).

---

## TABLES

### table_main_cs.tex / .txt — main commonsense results
- **Contents:** one row per method at its best-adapt LR (Llama-2 CS, s42): Method, Config, CS-8, Ret-core, Ret-broad, ||dW||_F, σ_max. Base (no-FT) row at top.
- **Key numbers:** LoRA+wd(0.3) lr5e-4 → **CS 81.6, Ret 25.6, ||dW|| 0.394** (top of table); then SC-LoRA 80.1/22.5, MiLoRA 79.9/24.7, LoRA 79.1/24.4, CLoRA 78.4/21.9, DoRA 78.3/24.8.
- **Takeaway:** LoRA+wd leads on adaptation AND retention while carrying the smallest weight update.
- **Supports:** Thesis **2**.
- **Note:** `.txt` is a STALE older draft (different selection rule "max core-retention at CS≥70", includes CorDA, shows ±std). The `.tex` is current. Prefer the `.tex`; treat `.txt` as legacy.

### table_main_math.tex / .txt — main math (GSM8K) results
- **Contents:** same schema, GSM8K domain, best-adapt LR per method.
- **Key numbers:** LoRA+wd(0.3) lr5e-4 → **GSM8K 50.6, Ret 24.6, ||dW|| 0.399**; LoRA 46.5/22.9. (Registry now also has DoRA math @ 33.3/25.2 — see mismatch #2.)
- **Takeaway:** LoRA+wd wins both axes on math too, confirming the CS result in a second domain.
- **Supports:** Thesis **2**.
- **Note:** `.tex` currently shows only LoRA & LoRA+wd; `.txt` is a stale draft. Math sweep is sparse (preliminary).

### table_lr_artifact.tex — the LR-artifact comparison (companion to fig9)
- **Contents:** one row per fancy method (DoRA, MiLoRA, SC-LoRA, LoRA-Null, CLoRA, CorDA). Columns: **Single-LR view** — the method at its best-looking single LR (adapt/ret) vs plain LoRA *run at the same LR*, and the Δadapt gap; **Swept view** — the LoRA+wd Pareto-frontier point that dominates the method's best single-LR point, and a yes/no domination verdict. Llama-2 CS, s42, 7 LRs. Rows where no LR clears ret≥24 are daggered (CorDA).
- **Key numbers:** Δadapt (method − LoRA at the *same* LR): SC-LoRA **+26.0**, LoRA-Null **+19.5**, DoRA **+5.9**, MiLoRA **+0.8**, CLoRA **−14.2**, CorDA **+38.1** (but −7.9 ret, no clean Pareto win). Swept-view domination = **yes for all six** — LoRA+wd@lr5e-4 (81.6/25.6) or @lr3e-4 (80.6/26.2) dominates each best single-LR point.
- **Takeaway:** numerically pins Claim 3 — the fixed-LR advantage (columns 2–4) is real for most methods, and it is fully erased by sweeping the LR for LoRA+wd (columns 5–6). CLoRA has no fixed-LR advantage to begin with; CorDA's is a magnitude/retention loss, not a win.
- **Supports:** Thesis **3**.
- **Note:** auto-generated (dedup by latest `evaluated_at`, CorDA nq_open eval, diverged lr1e-3 dropped). Single seed — treat as illustrative, not seed-averaged.

---

## Figure → paper-section mapping

| Paper section | Primary figures/tables |
|---|---|
| Teaser / abstract graphic | fig0_hero |
| §Method — the magnitude axis | fig1_magnitude_law, fig6 (panel b) |
| §Result 1 — the magnitude law | fig0_hero, fig8_magnitude_budget, fig5_per_benchmark |
| §Result 2 — geometry doesn't matter (fairness) | fig2_fairness_residuals |
| §Result 3 — LoRA+wd wins the Pareto | fig3_pareto, table_main_cs, table_main_math, op_points |
| §Result 4 — the LR artifact | **fig9_lr_artifact, table_lr_artifact** (headline), fig4_lr_sensitivity, fig7_lr_is_the_proxy |
| §Discussion / wake-up call | fig8_magnitude_budget, op_points |
| §Appendix — supporting structure | fig6_supporting_structure |

## Model coverage & readiness

| Item | Model | Status |
|---|---|---|
| fig0–fig8, op_points, table_main_cs | Llama-2-7B | **publication-ready** (single-seed caveat) |
| table_main_math | Llama-2-7B | ready but SPARSE (LoRA/LoRA+wd; DoRA now available, needs regen) |
| Qwen replication | Qwen-2.5-7B | **IN PROGRESS** — no Qwen figures generated yet; CS-LoRA law replicates (r=-0.88), math-LoRA does NOT yet (r=+0.67 ns). Regen a Qwen panel once ≥5 adapters swept. |

**Regen triggers (do NOT regenerate now — figures current as of 2026-06-29):**
- Add a Qwen replication panel/appendix once Qwen has ≥5 adapters on CS and higher-LR math cells.
- Regenerate table_main_math to include DoRA (already in registry).
- Fix the LoRA-Null labeling (mismatch #1) before final camera-ready, then regen all figures.
- If CorDA re-validation completes, add it back and regen.
