# 04 — Story integrity: artifact discrepancy list + final claim verdicts

Scope: the published results artifact (saved HTML, "Last updated 2026-07-14 10:45") audited
against the final post-fleet-kill dataset. Ground truth precedence: `data/key_numbers.md` §18
(FINAL FREEZE 2026-07-17) > §17 > §16; analyzer outputs in `analysis_final/`
(`analyze_full_output.txt`, `analyze_adversarial_output.txt`, `analyze_ebatch_output.txt`,
`final_census.txt`, `e5_e6_salvage.txt`); `assessment_2026-07-17.md`;
`adversarial_review_2026-07-16.md`.

Guardrails observed throughout: CLoRA's published numbers are faithful and are never framed as
suspect; the pre-registered wording rule (§17.1/§18.2) governs "law" vs "magnitude relation".

Convention: "OP-RECOMPUTE" = the operating-points agent recomputes the cell values; my job here
is flagging which artifact cells are at risk, not supplying replacements.

---

## A. ARTIFACT DISCREPANCY LIST (section by section)

### A.0 Header / title / byline

| # | Artifact location | Old value / claim | New value / claim | Source |
|---|---|---|---|---|
| H1 | Title + eyebrow ("The Magnitude Law…"), and every "law" usage below | Headline framing is a log-linear "Magnitude Law" | RETIRED → "magnitude relation (flat-then-falling with a knee)"; the word "law" only with the knee caveat, per the **pre-registered decision rule** (normalized slopes do NOT converge: −0.33…−0.70). 2-segment beats linear in every family (F 1.6–40.0); knees (log10 F_Δ): lrsw −0.02, lrswm −0.48, qwsw −0.69, qwswm −0.91, frc −0.45, frm −0.50 | §18.2, §17.1; analyze_adversarial A1 |
| H2 | Dek: "at matched update size the assessed methods land on the same curve" (unqualified) | Direction/method is a nothing-effect | Must be qualified: direction is a real, bounded **second-order** effect — random-direction controls at matched F_Δ sit **−3.05 pp below** trained adapters (E1, 9/9 controls); partial r(log spec_max, ret \| log F_Δ) = **+0.117** (t=3.7, n=1018); method offsets at matched F_Δ bounded ±1.2–4.6 pp | §18.3 (E1), §18.4; analyze_ebatch E1; analyze_adversarial A3 |
| H3 | Byline "Last updated · 2026-07-14 10:45" | 07-14 registry state (622 rows) | Final freeze 2026-07-17; dataset = **1,661 result dirs, 1,500 with full evals**, quarantine 71; `campaign_summary.jsonl` (645 rows) and `results_book/` are STALE — numbers must source from `results/*/summary.json` | §18 preamble; final_census.txt |
| H4 | Byline "Adapters · 9" / "Analyses · magnitude · geometry · efficiency · CE-forgetting" | Evidence base as of 07-14 | Add: interventional rescaling (E1), full-FT anchor (E2), Qwen densification (E3), eval-matched SC-LoRA ladder (E4), replay (E5, CE-only), wd-generalization (E6), bridging arms (E7); PiSSA now also trains in the Llama CS grid (see A.1 #0-1) | §18.3; final_census.txt |

### A.1 Tiles + gloss

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| T1 | Tile "Magnitude law r = −0.86 … holds within every one of the 7 assessed adapters (r −0.86 to −0.97) … Qwen CS n = 49 pooled r = −0.86" | Single-seed pooled r, n=49-per-arm framing | Final per-family, all-seeds, quarantine-filtered table (§18.1): lrsw **−0.886** (n=180, s42–45), lrswm **−0.865** (n=120), qwsw **−0.840** (n=151), qwswm **−0.830** (n=164), frc **−0.928** (n=276), frm **−0.929** (n=144); **ALL pooled −0.847 (rank −0.923), n=1035**. Seed-averaged cell-level r: −0.80…−0.93. The single-seed within-method list (−0.86…−0.97) is superseded; note LoRA+wd is flat-by-construction within-method on Qwen (+0.09) | §18.1; analyze_full_output (1),(2) |
| T2 | Tile "Best CS operating point 81.80 ± 0.16 / 25.93 ± 0.42 (3-seed)" | 3-seed (42/43/44) point | OP-RECOMPUTE: lrsw now spans seeds 42–45 (180 rows; 49 cells with n≥3) — mean/SD/n at risk for every headline operating point | §18.1; final_census |
| T3 | Tile "Best math 66.8 vs 64.6 (3-seed)" | frm 3-seed | OP-RECOMPUTE: frm now seeds 42–45 (144 usable rows; 163 evaluated dirs) — headline cell may have gained seeds | §18.1, §18.7 |
| T4 | Gloss "seeds — mature single-model operating points are seed-42 point estimates" | Single-seed disclaimer | Outdated: **287 cells have n≥3 seeds** (§13.2; per-family 49/36/43/47/72/43). The disclaimer must be replaced with the multi-seed statement | §13.2; analyze_full (2) |
| T5 | Gloss "retention seed-SD is ≤ 0.5 for on-law cells (median ≈ 0.4)" | Old 3-seed matrix SD | Final within-cell SD(ret): lrsw **0.94 pp**, lrswm 0.33, frc 0.75, frm 1.00; Qwen qwsw **2.73** / qwswm **2.07** (inflated by seed-unstable-F_Δ cells). The relation's 15–30 pp range is 10–50× seed noise | §18.1; analyze_full (3) |
| T6 | Gloss knee reference (implicit "saturation near base") | One global knee at F_Δ ≈ 0.37 (§3 callout) | Per-family knees (log10 F_Δ): lrsw −0.02, lrswm −0.48, qwsw −0.69, qwswm −0.91, frc −0.45, frm −0.50 — the single 0.37 knee is superseded | §18.2 |

### A.2 §0 The adapters

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 0-1 | Config table row "PiSSA (math arm)" + caption "PiSSA runs in the math arm only" | PiSSA math-only | PiSSA now has Llama **CS-grid (frc)** cells too: significant method offset −5.9±1.0 pp in frc (and −11.4±2.1 in frm, collapse-driven) | §18.4; analyze_adversarial A3 |
| 0-2 | (missing) | No mention of full-FT, replay, bridging conditions | §0 (or a new "conditions" note) should acknowledge the non-adapter arms now in evidence: full-FT anchor, replay-5%, MedMCQA attention-only bridging | §18.3 E2/E5/E7 |

### A.3 §1 Commonsense — adaptation & retention

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 1-1 | Lead: "+3.4 CS, +4.3 retention over CLoRA … 8–10× either method's seed noise" | 3-seed margins | OP-RECOMPUTE (margins + noise multiples change with 4–5-seed cells and lrsw SD(ret)=0.94) | §18.1 |
| 1-2 | Llama CS operating-point table (all 8 rows: LoRA+wd 81.80±0.16/25.93±0.42/0.38–0.41; SC-LoRA 80.61±0.41/24.60±1.85; LoRA 79.08/23.81; LoRA-Null 78.93/22.14; CLoRA 78.41/21.59; DoRA 74.29±8.65/25.20; MiLoRA 57.69±22.67/24.20) | n=3, seeds 42/43/44 | **Every row at risk** — OP-RECOMPUTE on the final lrsw set (seeds 42–45, quarantine-filtered). Also "Safe band x/7" columns are s42-sweep-based and should be restated on the multi-seed data; the sweep also now contains extra LRs (7e-5, 1.5e-4 densification cells on Qwen; lr2e-3 stratum appears in lrsw A2 output) so "of the 7 LRs swept" needs a coverage caveat | §18.1; analyze_adversarial A2 (lr2e-3 stratum, n=6); analyze_ebatch E3 |
| 1-3 | Footnote: "LoRA-Null's spread sits past the law's knee (≈0.37) where residual SD grows 0.6→3.5" | Global knee 0.37, single-fit residual SDs | Superseded by per-family knees (lrsw knee at log10 F_Δ = −0.02, i.e. F_Δ ≈ 0.95 on the pooled multi-seed fit) and the family-level heteroscedasticity statement; do not reuse 0.37/0.6/3.5 without recompute | §18.2 |
| 1-4 | wd × lr grid paragraph ("single-seed faithful-recipe grid … Single-seed cells, disclosed as such") | Single-seed frc grid | frc is now **n=276, seeds 42–46**, r=−0.928 (rank −0.952) — the single-seed disclaimer is obsolete; the r16-collapse basin now has seed data available for check (replicate was queued; verify before reprinting the 13.5 story) | §18.1; §16 (W2/B5a) |
| 1-5 | "MMLU-Pro loses 3.24 pp while BBH loses 0.95 pp" | 07-14 operating-point decomposition | OP-RECOMPUTE (depends on the refreshed operating points) | — |
| 1-6 | CLoRA boundary callout: k1024 73.3±11.7, k2048 70.0±5.1 (3-seed); k-grid F_Δ 0.61→0.34, ret 22.6→25.0 | 3-seed boundary cells | OP-RECOMPUTE (seed counts may have grown); framing itself survives — high-k CLoRA on-relation via update shrinkage, faithful to their Table 4. Keep the "their harness, not directly comparable" commensurability note (CLoRA numbers faithful — required framing) | §18.1; guardrail |
| 1-7 | Qwen CS table + caption ("single seed 42 otherwise", "2-seed", SC-LoRA 9.4/36.2/37.9) | Mixed 1–3-seed rows | qwsw is now n=151 rows, seeds 42–45, **43 cells with n≥3** — the whole table must be rebuilt as multi-seed (OP-RECOMPUTE); note qwsw has **27 trained-not-evaluated** cells (ledger) and 1 degenerate cell excluded in the clean fit | §18.1, §18.7; final_census; analyze_adversarial A7 |
| 1-8 | Qwen caption: "even the smallest-update adapters give up ~3 pp on this recipe" | ~3 pp | **~6 pp adaptation tax**: Qwen base 44.35 vs adapter plateau ~37–39 even below the knee (model-dependent intercept; ≈0 on Llama). Worth its own paragraph per the assessment | assessment_2026-07-17 ("New results" #3) |
| 1-9 | Qwen plain-LoRA "best GSM8K at lr 1e-3, F_Δ 0.59, BBH collapses to 16.0" (in §1 tail note) | Real forgetting reading | Survives, but the format-collapse control now applies: qwswm clean-subset r −0.695 vs pooled −0.830 — quote both when using collapsed Qwen cells | §17.7, §18.6 |

### A.4 §2 Math — adaptation & retention

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 2-1 | Llama math table (LoRA+wd 66.8±0.8/33.6±1.0; MiLoRA 63.7±0.8; LoRA 59.6±1.5; SC-LoRA 60.5±0.5; single-seed CLoRA/LoRA-Null/CorDA++/DoRA/PiSSA rows) | 3-seed/single-seed mix | OP-RECOMPUTE: frm now seeds 42–45 (144 usable). The 66.79±0.79 headline was re-verified at §16 but must be restated at final seed counts | §18.1; §16 |
| 2-2 | Qwen math table + SC-LoRA-edge paragraph (76.3±1.0 etc.) | 1–3-seed rows, "single seed 42 otherwise" | OP-RECOMPUTE: qwswm now n=164, seeds 42–44, 47 cells n≥3; **22 trained-not-evaluated** qwswm cells go to the ledger. The SC-LoRA adaptation-edge finding is orthogonal to E4 (E4 is about *retention* residual) — keep, but re-verify cell values | §18.1, §18.7; final_census |
| 2-3 | "Across the faithful-recipe math cells, BBH vs log update size gives r = −0.92 … −0.82 excluded … ≈ −0.76 trained regime; LoRA+wd-dominated (32 of 44 cells)" | Old sparse-composition math law | Superseded: **frm r = −0.929 (rank −0.969), n=144**, multi-seed; lrswm r = −0.865, n=120. Composition disclosure must be recomputed; the "honest range" triplet is stale | §18.1 |
| 2-4 | "same slope family as commonsense (−14.8)" and Spearman battery "−0.87/−0.87/−0.96" | Slope-convergence story | RETIRED: pooled slopes now lrsw −9.34, lrswm −10.30, qwsw −29.18, qwswm −17.61, frc −14.98, frm −12.15; normalized slopes −0.33…−0.70 **do not converge** — replace "same slope family" with the knee-shaped relation statement | analyze_adversarial A1; §18.2 |
| 2-5 | Qwen math framing throughout §2 ("qualitative", anti-replication cleanup implied) | "Qualitative/directional replication — single seed, BBH-only" (§16 wording) | **Qwen math is no longer qualitative-only: 3 seeds, n=164, r = −0.830** (clean-subset −0.695 — quote both). The old "+0.67 parser artifact" story is fully dead; E3 shows the below-knee half is **flat (r −0.03), not positive** | §18.1, §18.3 E3; analyze_ebatch E3 |

### A.5 §3 Magnitude analysis

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 3-1 | Per-arm law table: Llama CS n=49 r=−0.86 slope −14.8; Llama math n=39 r=−0.82; Qwen CS n=49 r=−0.86 slope −32.0; Qwen math n=48 r=−0.69 | Single-seed table | Replace with §18.1 final table (six families incl. frc/frm, n=1035 pooled, rank-r column, seed ranges) + seed-averaged cell-level r + within-cell SDs | §18.1 |
| 3-2 | Slope-family bar chart "four independent fits land in one −14…−15 family" (incl. CLoRA Table 4 −14.7) | Slope universality | RETIRED as a headline: single-line slopes are no longer the object (knee model); normalized slopes −0.33…−0.70. The CLoRA Table-4 external overlay itself stays (their data, faithful) but may no longer be sold as "same slope family" | §18.2; analyze_adversarial A1 |
| 3-3 | "R² = 0.74 vs 0.32" panels + "partial correlation of LR flips to +0.46" | Old n=49 LR comparison | Superseded by the **rewritten LR battery** (§18.5): R²(F_Δ) 0.689–0.863 vs R²(LR-continuous) 0.223–0.516; LR-as-dummies nearly ties in sweep families (lrsw 0.761 vs 0.785 — the old comparison is a strawman as written); partials r(F_Δ\|LR) −0.58…−0.91 (\|t\|≥7.6) vs r(LR\|F_Δ) −0.17…+0.29 (\|t\|≤4); fixed-LR strata r ≤ −0.7 at every LR ≥ 1e-4 in every family; decoupling grids frc/frm 0.86 vs 0.39/0.37 | §18.5, §2 (rewritten); analyze_adversarial A2 |
| 3-4 | "replicates five separate ways" bullets (within-method −0.86…−0.97; Qwen CS n=49; Qwen math n=48 r=−0.69 with saturation excuse) | Single-seed replication list | Rebuild: (a) six families multi-seed (§18.1); (b) within-cell micro-test **r = −0.713 (n=954 obs / 290 cells, t=−31.3)** — the closest thing to a causal signal, → main text; (c) E1 interventional; (d) E7 bridging (brl −0.878, brq −0.995); (e) CE within-family r(CE,ret) −0.63…−0.92 | §18.6, §18.3, §17.5 |
| 3-5 | Broad-battery per-benchmark table (n=49/43) + "composite r=−0.937, slope −17.0" | Old n | At-risk: recompute at full n; final per-family broad-retention r available (−0.81…−0.92; w/o ARC-c moves ≤0.09 — add the contamination-control disclosure sentence) | analyze_adversarial A6; §18.6 |
| 3-6 | Callout "curve saturates … knee ≈ 0.37 … past-knee slope ≈ −21 … residual SD 0.6 vs 3.5 … partial r −0.87 … permutation p<5e-5" | One global saturating fit | Superseded by the per-family 2-segment model (§18.2: below-knee −13.8…+2.0, above-knee −7.5…−40.8) and the A1 robustness subsets (healthy-only r −0.68…−0.94; drop-top-quartile −0.28…−0.94, weakest qwswm — tail-anchored, disclose) | §18.2; analyze_adversarial A1 |
| 3-7 | (missing) | — | **E1 interventional upgrade is absent** from the artifact: 15/15 trained rescales, mean on-curve residual **+1.29 ± 2.07 pp**, within-set r=−0.732; 9/9 random-direction controls −1.76 ± 1.32 pp (**direction penalty −3.05 pp**); upscaling asymmetry (clora 0.65→0.78: −3.86 pp); **rescale > retrain** (e1_lora_f040 ret 26.93/adapt 75.4 vs trained lr3e-4 twin 24.4/79.1) — observational→interventional is the single biggest missing upgrade | §18.3 E1; analyze_ebatch E1; assessment |
| 3-8 | (missing) | — | **E2 full-FT anchor absent**: 3/3 monotone (26.9→26.2→17.1 as F_Δ 0.023→0.395) but −4.1…−8.6 pp BELOW the LoRA-family curve at matched F_Δ (dense ΔW, dw_sv_max ~4 vs 30–40; fft F_Δ under-counts dense mass — disclosed) → "universal in form, family-specific in level" | §18.3 E2; analyze_ebatch E2 |
| 3-9 | (missing) | — | **E7 bridging arms absent**: MedMCQA, attention-only targets — brl (Llama) r=−0.878 (4/4), brq (Qwen) r=−0.995 (3/4, lr1e3 lost). Off-recipe reproduction on both models | §18.3 E7; analyze_ebatch E7 |

### A.6 §4 Geometry analysis

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 4-1 | Lead: "geometry acts through the size … rather than as an independent protection mechanism"; bullet "(2) Placement adds no measurable retention benefit" | Direction ≈ zero effect | REFINED to bounded second-order: partial r(log spec_max, ret \| log F_Δ) = **+0.117** (t=3.7; cell-level +0.115); spec_mean carries no residual signal (+0.03 ns); method offsets ±1.2–4.6 pp (sig. sclora/pissa/dora/lora_null in Llama families; Qwen ns); E1 random-direction −3.05 pp at matched F_Δ. Never state "adds nothing" unqualified | §18.4, §18.3 E1; analyze_adversarial A3 |
| 4-2 | Bullet "(3) The two methods that fall below the size law do so because of their placement" (SC-LoRA + CorDA) and the SC-LoRA −5.7/−4.15 residual story | SC-LoRA below-law due to placement (mis-allocated update), calibration control on 3 LRs | **RESOLVED-AS-ARTIFACT, now on the full E4 ladder**: eval-matched calibration mean residual **+0.92 pp ABOVE the curve (n=20 cells, 6 LRs × up to 5 seeds)** vs nq_open-calibrated **−3.39 pp (n=24)**. The deviation is a property of the calibration distribution, not method geometry — and not primarily of placement-driven magnitude either; the "because of their placement" wording must be softened for SC-LoRA. (b4_sclora_lr2e5_s42 adapt=13.2 is a low-LR format fluke; exclude from adapt comparisons. 4 b4 cells unevaluated: lr1e3×2, s46×2) | §18.3 E4; analyze_ebatch E4; assessment |
| 4-3 | Old E4 callout numbers "+0.05 / +2.7 / +1.8 at lr 5e-5/1e-4/3e-4 vs +0.9 / −2.8 / −6.3" | 3-LR partial ladder | Superseded by the full ladder (per-cell residuals in analyze_ebatch E4; headline +0.92 vs −3.39) | analyze_ebatch E4 |
| 4-4 | "Second — rank: at matched update size … partial r ≈ −0.56" | Rank as the second lever, n=49 battery | At-risk/superseded: at full n, spread metrics are largely collinear with magnitude (stable_rank_w r −0.36 raw; §13.3: partial r(spec_max, ret \| log F_Δ)=+0.195 on the wide 1222-row match, family-only **+0.117**). The −0.56 rank partial is from the old battery — recompute or retire; the residual direction signal lives in **spec_max**, and its sign is positive | §13.3, §18.4; analyze_full (4) |
| 4-5 | "no alignment metric adds more than ~0.03 R² … vs 0.63 for size alone" | Old stress-test numbers | At-risk: final R² baseline is 0.69–0.86 per family (ret ~ log F_Δ); method dummies add +0.01…+0.14. Also new: F_Δ beats ‖ΔW‖_F (R² 0.72 vs 0.56) and dw_sv_max (0.58) — the gap IS the adaptation-distribution weighting; axis label should read "effective update magnitude on the adaptation distribution" | §17.4, §17.9, §18.6; analyze_adversarial A4/A9 |
| 4-6 | Fingerprint tables (Llama + Qwen medians) | 07-14 battery | Fingerprints are design-signature results, largely stable; geometry rows now 1398 merged — re-verify medians/n-per-row if reprinted, but no known contradiction | analyze_full header |
| 4-7 | (missing) | — | **DoRA F_Δ lower-bound caveat absent**: DoRA carries significant *positive* offsets at matched F_Δ (frm +4.6±1.4, lrsw +2.9±0.7) — consistent with F_Δ reconstructed from (α/r)B·A under-counting its learned magnitude-vector component (same disclosure class as E2's fft undercount). State as a measurement caveat, not a DoRA-protection claim | §18.4; analyze_adversarial A3; assessment (fft undercount) |
| 4-8 | (missing) | — | New sharper slogan available: **adaptation-efficiency ANCOVA** — retention cost of a unit of magnitude is near-universal (R² 0.69–0.86, method offsets ≤ a few pp); methods differ mainly in adaptation bought per unit (method spread 4.9–16.0 pp) | §17.4; analyze_adversarial A4 |

### A.7 §5 Compute & memory

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 5-1 | "LoRA + wd … none (wd is a free flag)" | wd universally free | Add the **E6 boundary**: wd0.3 transfers to MiLoRA (+1.75/+2.36 pp above curve, adapt 80.2 at lr5e4) but **breaks DoRA as-implemented** (degenerate: CE 20.8/10.4 vs DoRA twins 2.1/2.6, spec_max up to 1183; benchmark evals lost) — naive AdamW wd on DoRA incl. its magnitude vector is not free | §18.3 E6; e5_e6_salvage.txt |
| 5-2 | Rest of §5 (measured peaks, wall-clock ratios) | — | No contradiction found in the final data; keep | — |

### A.8 §6 CE-forgetting

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 6-1 | "independently confirms the magnitude law on a metric we did not design"; ρ=0.974 (53 cells), ρ=0.97 (282 CS cells), ρ=0.94 (n=106) | CE vs F_Δ correlations as the corroboration | **Framing repair (§17.8)**: r(F_Δ, CE) = +0.81…+0.92 is *partly mechanical* (same ΔW); lead with the evidential link **r(CE, retention)**: lrsw −0.862, lrswm −0.923, qwsw −0.631, qwswm −0.792, frc −0.858, frm −0.896. Coverage: CE merged n≈1304; **136 Qwen runs lack CE** (jobs/ce_backfill_qwen.txt — unfillable without GPUs; disclose) | §17.8, §18.6; analyze_adversarial A8 |
| 6-2 | CE table rows (per-method math anchors) | 07-14 values | Low risk (external reproduction anchors unchanged) but cell values OP-RECOMPUTE where seed counts grew (PiSSA/CorDA++ 3-seed rows) | — |
| 6-3 | (missing) | — | **E5 replay belongs here**: trained 4/4, benchmark evals LOST (0/4); CE salvage — replay-5% CE lower than matched plain-LoRA twins in all 4 cells (lr3e4: 2.248/2.204 vs 2.307/2.254; lr5e4: 2.465/2.462 vs 2.551/2.526; Δ ≈ −0.05…−0.09; KL likewise lower). State as **partially answered**: small consistent CE-forgetting reduction; the benchmark-retention comparison died with the fleet | §18.3 E5; e5_e6_salvage.txt |

### A.9 §7 Correctness assurance / §8 Independent verification / footer

| # | Artifact location | Old | New | Source |
|---|---|---|---|---|
| 7-1 | §7 table | 07-14 checks | Add: quarantine regenerated (71 diverged runs, `results/quarantine_diverged.txt`); registry declared stale — final numbers recomputed directly from `results/*/summary.json` (analyze_full/adversarial/ebatch re-run at freeze) | §18 preamble |
| 8-1 | §8 "two independent automated adversarial review passes … 140/143 verified" | 07-14 review state | Superseded: the **07-16 three-reviewer adversarial pass** (statistics / Reviewer-2 / DeepSeek-design) drove the §17 recompute, the wording rule, and the E1–E7 batch; §8 must describe that review + the E-batch verdicts, not the old numeric audit | adversarial_review_2026-07-16.md; §17, §18.3 |
| F-1 | Footer "Evidence base" | 07-14 inventory | Rewrite: 1,661 result dirs / 1,500 evaluated; six families n=1035 usable; E1/E2/E4/E6-MiLoRA/E7-Llama complete; E3 13/26, E5 0/4 benchmark (CE salvaged), E6-DoRA degenerate-by-CE, brq 3/4; **the missing-data ledger** (see below) | §18.7; final_census |
| F-2 | (missing everywhere) | — | **Data-state ledger absent**: DSV4 **284B 0/21 synced** (trains + geometry done on DeepSeek nodes; capped evals never synced — designed-but-lost, spec handoff/DEEPSEEK_GEN_EXPERIMENT.md); 284B base ceiling absent; base-ceiling dirs 4/22 evaluated (Llama core 26.0 / broad 35.26; Qwen 44.35); trained-not-evaluated: qwsw 27, qwswm 22, lrsw_lorarep05 4, lrsw_dorawd 2, b4_sclora 4 | §18.3, §18.7; final_census |

---

## B. CLAIM-BY-CLAIM FINAL VERDICT MATRIX

| Claim | Final status | Final supporting numbers | Where it goes in the paper |
|---|---|---|---|
| **Claim 1 — magnitude→forgetting mechanism** | **VALIDATED + UPGRADED (observational→interventional)** | Six families r −0.830…−0.929, ALL pooled −0.847 (rank −0.923), n=1035, seeds 42–46 (§18.1); within-cell micro-test r=−0.713 (n=954/290 cells, t=−31.3) (§18.6); **E1: 15/15 rescales, on-curve residual +1.29±2.07 pp, within-set r=−0.732**; upscaling asymmetry (clora −3.86 pp); rescale>retrain (26.93/75.4 vs 24.4/79.1) | §4 (sec:law) — headline + new interventional subsection; micro-test → main text |
| **Functional form (log-linear "law")** | **RETIRED → REPLACED (flat-then-falling with knee)** | 2-segment beats linear all 6 families (F 1.6–40.0); knees log10 F_Δ −0.02…−0.91; below-knee −13.8…+2.0, above-knee −7.5…−40.8; normalized slopes −0.33…−0.70 (no convergence) → pre-registered wording: "magnitude relation", "law" only with knee caveat; E3: Qwen bottom-half r −0.04/−0.03 (flat, not positive — anti-replication dead) | §4 (sec:law) functional-form subsection; title/abstract wording; Limitations |
| **"Magnitude, not geometry" (direction second-order)** | **REFINED (bounded second-order effect)** | Partial r(log spec_max, ret \| log F_Δ) = +0.117 (t=3.7; cell +0.115, t=2.1); spec_mean +0.03 ns; method offsets ±1.2–4.6 pp (frc: sclora −3.7±0.5\*, pissa −5.9±1.0\*; frm: pissa −11.4±2.1\*, dora +4.6±1.4\*; Qwen all ns); **E1 random-direction penalty −3.05 pp** at matched F_Δ | §5 (geometry section) — restated as "magnitude first-order (R² 0.69–0.86); direction/method 1–4 pp second-order"; title framing softened |
| **SC-LoRA below-curve deviation** | **RESOLVED-AS-ARTIFACT (calibration set, not method)** | E4 full ladder 20/24: eval-matched mean residual **+0.92 pp above curve** vs nq_open −3.39 pp (n=24); lr2e5_s42 adapt fluke excluded; 4 cells unevaluated (lr1e3×2, s46×2 — disclose) | §5 control-experiment callout + fig2 outlier discussion; fairness win to foreground |
| **Claim 2 — LR is only a proxy (LR-artifact diagnosis)** | **VALIDATED — REWRITTEN battery (old "R² doubles" framing retired as strawman)** | §18.5: R²(F_Δ) 0.689–0.863 vs R²(LR-cont) 0.223–0.516 (dummies nearly tie in sweeps: lrsw 0.761 vs 0.785); partials r(F_Δ\|LR) −0.58…−0.91 (\|t\|≥7.6) vs r(LR\|F_Δ) −0.17…+0.29; fixed-LR strata r ≤ −0.7 at every LR ≥ 1e-4, every family; decoupling grids frc/frm 0.86 vs 0.39/0.37 | §6 (sec:lr) — replace the R²-doubling exhibit with strata + partials + decoupling grids |
| **Claim 3 — LoRA+wd corollary (practical winner)** | **VALIDATED; wd-generalization SPLIT VERDICT** | CS/math operating-point wins carry over (values OP-RECOMPUTE at final seed counts); frc: wd is a clean magnitude knob (n=276, r=−0.928); **E6: MiLoRA+wd +1.75/+2.36 pp above curve (2/2, adapt 80.2)** = transfers; **DoRA+wd DEGENERATE** (CE 20.8/10.4 vs twins 2.1/2.6, spec_max 1183; benchmark lost) = boundary: wd is not a universally free knob | §7 (sec:pareto) + a boundary paragraph; Limitations for DoRA+wd |
| **Second-model replication (Qwen2.5-7B)** | **UPGRADED (CS multi-seed; math quantitative)** | qwsw r=−0.840, n=151, seeds 42–45 (cell-level −0.799); **qwswm r=−0.830, n=164, 3 seeds — no longer qualitative-only** (clean-subset −0.695, quote both per §17.7); Qwen seed SDs 2.73/2.07 pp disclosed | §3/§4 (setup + law); claims table row "Second model" rewritten; Limitations gets the clean-subset disclosure |
| **Universality beyond LoRA-family / recipe** | **SHARPENED + PARTIALLY-ANSWERED** | **E2 full-FT 3/3**: monotone (26.9→26.2→17.1 over F_Δ 0.023→0.395) but −4.1…−8.6 pp below LoRA curve (dense ΔW; fft F_Δ under-counts — disclosed) → "universal in form, family-specific in level"; **E7 bridging 7/8**: brl −0.878 (4/4), brq −0.995 (3/4) off-recipe on both 7Bs | §8 (discussion) scope subsection; E2 disclosure in Limitations |
| **Base ceilings / adaptation tax** | **VALIDATED (7B) + new finding** | Llama core 26.0 (in-registry 25.89) / broad 35.26; Qwen 44.35; **Qwen ~6 pp adaptation tax below the knee** (plateau 37–39 vs 44.35; ≈0 on Llama) — model-dependent intercept; 284B ceiling absent; base-ceiling dirs 4/22 evaluated | §3 (setup, normalization) + one paragraph on the tax; ledger in Limitations |
| **CE corroboration** | **VALIDATED — REFRAMED** | Evidential link r(CE, ret) per family −0.631…−0.923 (lead with this); r(F_Δ, CE) +0.81…+0.92 flagged partly mechanical; coverage 136 Qwen runs without CE (unfillable, disclose) | §5/§6 CE subsection (paper's CE exhibit) + Limitations coverage note |
| **Replay baseline (practitioner falsification test)** | **PARTIALLY-ANSWERED** | Trained 4/4, benchmark evals LOST 0/4; CE salvage: replay-5% lower CE in all 4 matched pairs (Δ −0.05…−0.09 nats; KL likewise) — small consistent CE-forgetting reduction | Limitations / practical section — stated exactly as partial (CE-only) |
| **DSV4 284B generalization** | **LOST-WITH-FLEET** | 0/21 synced (trains + geometry completed on DeepSeek nodes; capped evals never synced). No data in this repo. Ledger as designed-but-lost (spec: handoff/DEEPSEEK_GEN_EXPERIMENT.md) | Limitations + reproducibility appendix ledger; remove any 284B forward-references |
| **CorDA (original) placement claim** | **NOT-ASSESSED (unchanged)** | Zero post-07-11 clean nq_open CS re-run cells landed (§16 "still pending"); CorDA stays excluded from every law/fit (§8/§9 rules); CorDA++ math rows remain as disclosed single-seed anchors | Claims table "withheld" row unchanged; geometry fingerprint kept as fingerprint-only |
| **DoRA F_Δ lower-bound** | **REFINED — disclosed measurement caveat** | DoRA positive offsets at matched F_Δ (frm +4.6±1.4\*, lrsw +2.9±0.7\*) consistent with (α/r)B·A-reconstructed F_Δ under-counting the learned magnitude-vector component — same disclosure class as E2's fft undercount; do not read as DoRA "beating the relation" | §5 measurement subsection footnote + Limitations |

### Consistency notes for whoever rewrites the artifact

1. Never mix the retired single-fit constants (r −0.86 / slope −14.8 / knee 0.37 / R² 0.74-vs-0.32)
   with §18 numbers in one exhibit — they come from incompatible fits.
2. CLoRA's published Table 3/4 numbers stay framed as faithful external data; the harness-gap
   commensurability note stays; no "artifact/strawman" language about their results.
3. Everywhere a Qwen pooled r is quoted, the §17.7 rule applies: quote the clean-subset value
   alongside (qwswm −0.830 / clean −0.695).
4. `campaign_summary.jsonl` and `results_book/` are stale — every recomputed number must trace to
   `results/*/summary.json` via the three freeze analyzers.
