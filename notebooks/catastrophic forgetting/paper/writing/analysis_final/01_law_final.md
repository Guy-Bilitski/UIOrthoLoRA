# 01 — THE MAGNITUDE RELATION: functional form + universality (FINAL)

`[WRITTEN 2026-07-17 — law-final analyst. Sources: results/*/summary.json (final synced dataset,
1,500 evaluated runs; quarantine 71), key_numbers.md §18 (frozen), analysis_final/analyze_full_output.txt,
analyze_adversarial_output.txt, analyze_ebatch_output.txt. Every number below was either
independently recomputed from results/*/summary.json (marked ✓verified) or taken from the frozen
analyzer outputs (marked [§18.x]/[A#]/[E#]). NO numbers from campaign_summary.jsonl, results_book/,
or the 07-14 artifact.]`

**Wording rule (pre-registered, §17.1/§18.2):** headline = **"magnitude relation
(flat-then-falling with a knee)"**. "Law" may be used only with the knee caveat, because
normalized slopes do not converge across families. CLoRA's published numbers are faithful and
are never framed as suspect; the research question is only whether magnitude at matched capacity
explains the retention pattern.

---

## HEADLINE NUMBERS BOX

| Claim | Stat | Source |
|---|---|---|
| The relation, pooled all-seeds | r(ret, log10 F_Δ) = **−0.847** (rank −0.923), n=1035, 6 families | §18.1 ✓verified |
| Strongest family | frm r=−0.929 (rank −0.969), n=144 | §18.1 ✓verified |
| Weakest family (headline / clean) | qwswm **−0.830 / −0.695**, n=164/155 — quote both | §18.1, A7 ✓verified |
| Functional form | 2-segment beats linear in all 6 families (F=1.6–40.0); below-knee slopes −13.8…+2.0, above-knee −7.5…−40.8 pp/dec | §18.2, A1 ✓verified |
| Interventional (E1) | 15/15 trained rescales land on curve (mean resid +1.29±2.07 pp, within-set r=−0.732); 9/9 random directions −1.76±1.32 ⇒ **direction penalty −3.05 pp** | §18.3, E1 ✓verified (+1.27±2.14 on my refit) |
| Rescale > retrain | fd-matched twins: rescale ret − trained ret = **+1.09±1.80 pp (n=15)**; flagship pair +2.51 pp at −3.7 adapt | ✓computed |
| Universality | full-FT monotone but −4.1…−8.6 pp below LoRA curve; brl r=−0.878, brq r=−0.995 (off-recipe); Qwen anti-replication dead (bottom-half r ≈ 0, flat not positive) | §18.3, E2/E3/E7 |
| Ceilings | Llama core 25.89 (broad 35.26); Qwen core 44.35 (broad 53.61). Qwen-CS below-knee plateau sits **5.3 pp below** its ceiling; all other arms ≤0.8 pp | ✓computed |
| X-axis | R²: log F_Δ **0.72** > dw_sv_max 0.58 > log‖ΔW‖_F 0.56 (n≈1018–1034) | §18.6, A9 ✓verified (0.717/0.587/0.567, n=1034) |

---

## 1. THE RELATION

### 1.1 Pooled, all seeds (verifies §18.1 exactly)

Recomputed from `results/*/summary.json` with the freeze convention (non-corda/smoke, finite
F_Δ>0 and retention; retention = `headline.retention_mean` = mean(BBH, MMLU-Pro);
x = log10 `headline.fdelta`):

| family | r | rank-r | n | seeds |
|---|---|---|---|---|
| Llama-2 CS (lrsw) | −0.886 | −0.908 | 180 | 42–45 |
| Llama-2 math (lrswm) | −0.865 | −0.833 | 120 | 42–44 |
| Qwen-2.5 CS (qwsw) | −0.840 | −0.778 | 151 | 42–45 |
| Qwen-2.5 math (qwswm) | −0.830 | −0.582 | 164 | 42–44 |
| Llama CS grid (frc) | −0.928 | −0.952 | 276 | 42–46 |
| Llama math-395k (frm) | −0.929 | −0.969 | 144 | 42–45 |
| **ALL pooled** | **−0.847** | **−0.923** | **1035** | |

✓verified: my independent recomputation reproduces every cell of this table to 3 decimals.

**Quarantine-convention note (disclose in reproducibility appendix).** The freeze filter is
"finite F_Δ & retention"; `quarantine_diverged.txt` (71 runs) is mostly caught by that filter,
but 32 quarantined family runs have finite (exploded-magnitude / collapsed-retention) values and
ARE inside n=1035 — they are legitimate far-end points under the finite-value convention.
Strict additional exclusion of all 71 (✓computed): lrsw −0.864 (170), lrswm −0.865 (120),
qwsw −0.837 (149), qwswm −0.743 (160), frc −0.921 (268), frm −0.877 (136), ALL −0.864 (1003).
The relation is unchanged; qwswm moves toward its clean-subset value (see 1.3).

### 1.2 Seed-averaged (cell-level) — the noise-free version

r(mean ret, log10 mean F_Δ) per recipe-cell [§18.1, ✓verified]: lrsw −0.916 (56 cells),
lrswm −0.871 (42), qwsw −0.799 (58), qwswm −0.832 (62), frc −0.928 (74), frm −0.896 (51).
Within-cell seed SD(ret): lrsw 0.94 pp, lrswm 0.33, frc 0.75, frm 1.00; Qwen 2.73/2.07 pp,
inflated by seed-unstable F_Δ cells (SD(F_Δ) up to 0.98 in qwswm) [full-output §3].

### 1.3 Qwen math — the anti-replication is dead, with an honest asterisk

qwswm is now 3 seeds, n=164, r=−0.830 — no longer qualitative-only [§18.1]. Its
format-collapse-clean subset is −0.695 (9 degenerate runs dropped; A7) and its rank-r is −0.582:
the pooled value is partly tail-anchored (drop-top-F_Δ-quartile r=−0.277, A1). Per §17.7/§18.1:
**always quote −0.830 and −0.695 together.**

---

## 2. FUNCTIONAL FORM — flat-then-falling with a knee (A1, §18.2)

Two-segment continuous hinge fit vs single line, knee grid = 20–80th pctile of log10 F_Δ
(✓verified — I reran the fit and reproduce every knee/slope/F below):

| family | knee (log10 F_Δ) | knee (F_Δ) | s_below (pp/dec) | s_above | F(2seg vs lin) | lin slope | lin R² |
|---|---|---|---|---|---|---|---|
| lrsw | −0.02 | 0.95 | −13.8 | −7.5 | 15.6 | −9.34 | 0.79 |
| lrswm | −0.48 | 0.33 | −2.4 | −12.0 | 8.6 | −10.30 | 0.75 |
| qwsw | −0.69 | 0.20 | **+2.0** | **−40.8** | 31.6 | −29.18 | 0.71 |
| qwswm | −0.91 | 0.12 | **+0.9** | **−24.4** | 40.0 | −17.61 | 0.69 |
| frc | −0.45 | 0.35 | −3.7 | −16.5 | 20.5 | −14.98 | 0.86 |
| frm | −0.50 | 0.32 | −4.5 | −12.5 | 1.6 | −12.15 | 0.86 |

The story: **below the knee retention is flat-to-mildly-declining (Qwen: literally flat, slope
+0.9…+2.0); above it, it falls steeply (−7.5…−40.8 pp per decade of F_Δ).** Two-segment beats
linear in every family (frm weakest, F=1.6 — the one family where a single line is adequate).
lrsw is the outlier in knee location (−0.02, i.e. F_Δ≈1): most of its sweep sits below the knee
and its "below" slope (−13.8) is already the fall — its knee marks saturation of collapse,
not onset.

**Why the wording is "magnitude relation", not "log-linear law" (pre-registered §17.1):**
normalized linear slopes (pp/dec ÷ family retention range) do **not** converge:
lrsw −0.33, lrswm −0.52, qwsw −0.70, qwswm −0.38, frc −0.54, frm −0.46 [A1, ✓verified].
The *shape* (flat, knee, fall) and the *ordering variable* (F_Δ) are universal; the slope
coefficient is family-specific. A single log-linear law would require both.

Robustness of monotonicity above the knee [A1]: healthy-only (ret≥15) r = −0.678…−0.942;
drop-top-F_Δ-quartile r = −0.277…−0.942 (weakest qwswm, disclosed); bottom-half-of-log-range
r = −0.722…−0.916 in every family.

---

## 3. INTERVENTIONAL UPGRADE (E1) — the relation survives do(F_Δ)

Design: take a trained adapter, rescale ΔW to target F_Δ ∈ {0.15, 0.40, 0.80} without any
retraining (15 = 5 methods × 3 targets, COMPLETE); controls = 9 random-direction ΔW at the same
three F_Δ targets. Residuals vs the observational lrsw fit ret = 19.91 − 9.47·log10(F_Δ)
(n=201) [E1 output; my refit on n=203: 19.92 − 9.49, residual stats within 0.07 pp].

- **Trained rescales: n=15, mean on-curve residual +1.29 ± 2.07 pp, within-set r = −0.732**
  [§18.3, ✓verified +1.27±2.14 / −0.732]. Setting the magnitude *places the run on the curve*.
- **Random directions at matched F_Δ: mean residual −1.76 ± 1.32 pp ⇒ direction penalty
  −3.05 pp vs trained** [§18.3, ✓verified −1.78±1.39, penalty −3.05]. Direction is real but
  bounded — consistent with the observational A3 method offsets (±1.2–4.6 pp) and
  partial r(spec_max, ret | F_Δ)=+0.117 [§18.4]. Random directions also buy essentially zero
  adaptation (adapt 0.5–7.0 vs 13–80 trained) — magnitude forgets, direction adapts.
- **Upscaling asymmetry:** the one clearly off-curve trained rescale is the *upscale*
  e1_clora_f080_s45 (0.65→0.78): −3.86 pp below curve; random directions at f080 are likewise
  the worst controls (−1.8…−3.8 below) [E1 table, ✓verified]. Downscaling a trained direction is
  safe; magnifying one is not free.

### 3.1 Rescale > retrain — matched pairs (✓computed from the final dataset)

Across all 15 rescales, each paired with the same-method lrsw trained run nearest in log F_Δ:
**mean Δretention = +1.09 ± 1.80 pp in favor of the rescale** (13/15 pairs ≥ −0.5; the two
negatives are the clora upscale above and dora_f015). Flagship pairs (fd / ret / adapt):

| rescaled | trained twin | Δret | note |
|---|---|---|---|
| e1_lora_f040_s43 — 0.436 / 26.93 / 75.4 | lrsw_lora_r16_lr3e4_s42 — 0.623 / 24.42 / 79.1 | **+2.51** | pre-registered exemplar: −3.7 adapt buys +2.5 ret |
| e1_clora_f040_s45 — 0.416 / 25.88 / 79.6 | lrsw_clora_k1024_lr3e4_s44 — 0.447 / 24.09 / 79.7 | **+1.79** | matched fd AND adapt (Δadapt −0.1) |
| e1_milora_f080_s43 — 0.806 / 22.12 / 76.2 | lrsw_milora_r32_lr5e4_s42 — 0.840 / 20.76 / 76.7 | **+1.36** | matched fd AND adapt (Δadapt −0.5) |
| e1_dora_f080_s43 — 0.821 / 24.05 / 79.3 | lrsw_dora_r16_lr3e4_s44 — 0.684 / 23.27 / 79.1 | **+0.78** | matched adapt at *higher* fd |
| e1_sclora_f080_s43 — 0.859 / 20.52 / 77.8 | lrsw_sclora_r32_lr1e4_s42 — 0.813 / 16.38 / 79.9 | +4.14 | inflated by the E4 nq_open calibration artifact in the trained twin — cite with that caveat |

Reading: at matched effective magnitude (and, in the clora/milora/dora pairs, matched
adaptation), a post-hoc rescaled adapter retains **as much or more** than an adapter trained to
that magnitude. Forgetting follows where the weights *end up*, not the path training took.
Counter-case to keep honest: e1_clora_f080 (upscale) loses 3.97 pp to its fd-matched twin.

---

## 4. UNIVERSALITY — where the relation does and doesn't travel

### 4.1 E2 full-FT anchor (COMPLETE 3/3) — universal in form, family-specific in level

fft_full lr1e-5/3e-5/1e-4: F_Δ 0.023→0.080→0.395, ret 26.90→26.16→17.12 — **monotone in F_Δ**,
but −8.57/−4.12/−6.61 pp BELOW the LoRA-family curve at matched F_Δ [E2, §18.3].
Caveat (disclose): full-FT ΔW is dense — dw_sv_max ≈4 vs 30–40 for adapters — so F_Δ, built on
top singular directions, **under-counts dense update mass**; the fair statement is that the fft
points obey the shape on their own curve, shifted down, not that they land on the LoRA line.

### 4.2 E7 bridging arms (7/8) — off-recipe replication

New task (MedMCQA) + new placement (attention-only), both models [E7, §18.3]:
- **brl** (Llama): r = −0.878 (n=4); fd 0.161→0.777, ret 26.06→19.40.
- **brq** (Qwen): r = −0.995 (n=3, lr1e3 lost); fd 0.083→0.231, ret 42.06→39.70.
The relation is not an artifact of the commonsense recipe or of full-module adapters.

### 4.3 Qwen arms — final status

qwswm r = −0.830 (n=164, 3 seeds), clean −0.695 [§18.1/A7]. **The earlier "Qwen fails to
replicate" concern is dead** — what looked like anti-replication was the below-knee plateau:
E3 densification (13/26 landed; 2nd wave lost to the fleet kill) shows bottom-half r ≈ 0
(**qwsw −0.04, qwswm −0.03**) — *flat below the knee, not positive* [E3, §18.3]. The 13 new
mid-LR cells fall exactly in sequence (e.g. qwsw lora_null fd 0.116 ret 40.03 → sclora fd 0.374
ret 35.28). Remaining hole is figure density only, ledgered in §18.7.

### 4.4 Boundary notes (for the limitations section)

E6: MiLoRA+wd0.3 lands +1.75/+2.36 pp above the lrsw curve at adapt up to 80.2 (wd transfers);
DoRA+wd0.3 is degenerate (CE 20.8/10.4, spec_max→1183) — wd is not a universally free knob
[§18.3]. E5 replay: CE-only salvage (−0.05…−0.09 CE vs twins in 4/4 cells); benchmark retention
lost. DSV4 284B: designed-but-lost (0/21 synced).

---

## 5. CEILINGS, INTERCEPTS, AND THE QWEN ADAPTATION TAX

Base ceilings (results/base_*_noft/summary.json, ✓verified):

| model | core ret (BBH+MMLU-Pro)/2 | broad ret | components |
|---|---|---|---|
| Llama-2-7B | **25.89** | 35.26 | bbh 32.96, mmlu_pro 18.82 |
| Qwen-2.5-7B | **44.35** | 53.61 | bbh 47.93, mmlu_pro 40.77 |

(§18.7 quotes "Llama 26.0" — the exact summary.json value is 25.89; fix to 25.9 in the paper.
Base-ceiling coverage is 4/22 dirs evaluated; the retshard/bbhAO ladders never synced.)

**Adaptation tax = base ceiling − mean retention of below-knee runs** (knees from §2;
✓computed, n = runs with log10 F_Δ < knee):

| family | n below knee | mean ret | median | p90 | max | gap to ceiling (mean) |
|---|---|---|---|---|---|---|
| lrsw | 144 | 25.29 | 26.11 | 27.31 | 27.99 | +0.60 |
| lrswm | 51 | 25.26 | 25.40 | 26.03 | 26.40 | +0.63 |
| **qwsw** | **64** | **39.08** | **39.02** | 40.56 | 41.03 | **+5.27** |
| qwswm | 93 | 43.81 | 43.72 | 44.82 | 45.86 | +0.54 |
| frc | 88 | 26.66 | 26.89 | 27.53 | 27.99 | −0.77 |
| frm | 28 | 25.55 | 25.48 | 26.16 | 26.43 | +0.34 |

Refined claim: **the "Qwen adaptation tax" is specific to the Qwen-CS arm** — even below the
knee, qwsw plateaus ~5.3 pp under the 44.35 ceiling (best single below-knee run 41.03, still
−3.3), while Llama arms and Qwen-math plateau within ±0.8 pp of their ceilings (frc slightly
*above* base). I.e. instruction-format CS adaptation costs an instruction-tuned model a fixed
~5 pp of BBH/MMLU-Pro before any magnitude-driven forgetting begins; Llama (low ceiling) and
Qwen-math (gsm8k adapt) pay ≈0. Do not write "Qwen pays a 6 pp tax" unqualified — qwswm
contradicts it.

---

## 6. WHAT THE X-AXIS IS

F_Δ (CLoRA's adapt-distribution-weighted update magnitude) is the right axis, not raw update
size [§18.6/A9, ✓verified on n=1034 family rows with geo.json]:

| predictor | R²(ret ~ log10 x) |
|---|---|
| **F_Δ** | **0.72** (0.717) |
| dw_sv_max | 0.58 (0.587) |
| ‖ΔW‖_F (fro_total) | 0.56 (0.567) |

The 0.56→0.72 gap IS the direction weighting: label the axis **"effective update magnitude on
the adaptation distribution"** [§17.9]. Alignment (F_Δ/‖ΔW‖_F, ×1e-3) is NOT method-invariant —
dora 2.71±2.53, milora 2.22±2.26, lorawd 2.15±2.47, lora 2.06±2.03, sclora 2.03±0.89, pissa
1.86±0.15, lora_null 1.61±0.93, clora 1.54±1.07 [A9] — but within-method spread ≳ between-method
gap, so F_Δ is not a method label in disguise. Consistent with this, LR is only a proxy
(R² 0.22–0.52 vs 0.69–0.86 for F_Δ; partial r(F_Δ|LR) −0.58…−0.91 vs r(LR|F_Δ) −0.17…+0.29;
fixed-LR strata r ≤ −0.67 at every LR ≥ 1e-4 in every family) [§18.5/A2], and within-cell
seed-level F_Δ fluctuations predict retention at fixed recipe (demeaned r = −0.713, n=954 obs /
290 cells, t=−31.3) [§18.6/A5].

---

## 7. FIGURE-READY STATEMENTS (one sentence each, with exact support)

1. **Main relation:** "Across 1,035 adapters spanning 6 model×task families, 8 methods, and 3–5
   seeds, retention of held-out capability tracks effective update magnitude with pooled
   r = −0.847 (rank −0.923; per-family −0.830…−0.929)." [§18.1 ✓]
2. **Shape:** "The relation is flat-then-falling: below a per-family knee (F_Δ ≈ 0.12–0.95)
   retention slopes are −4…+2 pp/decade, above it −7.5…−40.8, and a two-segment fit beats a
   line in all six families (F = 1.6–40.0)." [A1 ✓]
3. **Not a single law:** "Normalized slopes range −0.33…−0.70 across families, so we claim a
   universal magnitude *relation*, not a universal log-linear *law*." [A1 ✓, §17.1 rule]
4. **Intervention:** "Directly setting F_Δ by rescaling trained adapters reproduces the curve
   (15/15 runs, mean residual +1.29±2.07 pp, within-set r = −0.732), while random directions at
   matched magnitude fall 3.05 pp lower and buy no adaptation." [E1 ✓]
5. **Rescale beats retrain:** "At matched effective magnitude, post-hoc rescaled adapters retain
   +1.09±1.80 pp more than adapters trained to that magnitude (n=15 pairs; e.g. LoRA 0.44:
   26.9 ret / 75.4 adapt rescaled vs 24.4 / 79.1 trained)." [✓computed]
6. **Full FT:** "Full fine-tuning follows the same monotone shape but sits 4.1–8.6 pp below the
   adapter curve at matched F_Δ — F_Δ under-counts dense update mass (dw_sv_max ≈4 vs 30–40)."
   [E2]
7. **Off-recipe:** "On MedMCQA with attention-only adapters the relation reappears untuned:
   r = −0.878 (Llama, n=4) and −0.995 (Qwen, n=3)." [E7]
8. **Qwen replication:** "Qwen-2.5 math now replicates at 3 seeds (r = −0.830, n=164; −0.695
   excluding 9 format-collapsed runs), and densification shows its below-knee segment is flat
   (bottom-half r = −0.03…−0.04), not anomalous." [§18.1, A7, E3]
9. **Adaptation tax:** "Below the knee, Llama arms and Qwen-math retain to within ±0.8 pp of
   their base ceilings (25.89 / 44.35), while Qwen-CS pays a constant ≈5.3 pp adaptation tax
   before magnitude-driven forgetting begins." [✓computed]
10. **Axis choice:** "Adaptation-weighted magnitude F_Δ explains retention (R² = 0.72) better
    than raw ‖ΔW‖_F (0.56) or the top singular value (0.58) — the update's *projection onto the
    adaptation distribution* is what forgets." [A9 ✓]
11. **Direction second-order:** "At matched F_Δ, method identity shifts retention by a bounded
    ±1.2–4.6 pp (partial r of top-SV +0.117) — real, but an order smaller than the 20+ pp
    magnitude effect." [§18.4]

---

## 8. PROVENANCE / DISCREPANCY LEDGER

- §18.1 table: reproduced exactly (all 6 families + pooled, 3 decimals). Convention: finite-value
  filter; 32 finite quarantined runs included in n=1035; strict-71 exclusion variant in §1.1.
- §18.2 knees/slopes/F: reproduced exactly with the seg2 hinge fit (20–80 pctile knee grid).
- E1 residuals: analyzer +1.29±2.07 on its n=201 lrsw fit; my n=203 refit gives +1.27±2.14,
  penalty identical (−3.05). Quote the analyzer values.
- Llama base core ceiling: exact 25.89 (base_llama2_7b_noft and base_llama2_noft agree);
  §18.7's "26.0" is a rounding slip — use 25.9.
- E3 ebatch print shows qwswm pooled −0.807 (n=165) vs freeze −0.830 (n=164): sample-set
  difference inside the ebatch script; the frozen §18.1 value (−0.830) is canonical.
- Stale sources NOT used: campaign_summary.jsonl (645 rows), results_book/, the 07-14 .bak.
