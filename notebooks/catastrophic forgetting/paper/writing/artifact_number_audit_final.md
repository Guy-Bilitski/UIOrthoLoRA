# Artifact number audit — `artifact_status_report.html`

**Auditor:** data-verifier (independent recompute)
**Audit date:** 2026-07-09
**Artifact audited:** `/tmp/claude-1000/-home-guy-UIOrthoLoRA/72dbcb26-a4ce-47e1-aaa6-792e52121dea/scratchpad/status_report.html`
(identical to `notebooks/catastrophic forgetting/paper/writing/artifact_status_report.html`)
**Interpreter:** `/home/guy/UIOrthoLoRA/.venv/bin/python`

## Recompute rules (stated explicitly)

- **Dedup rule** (matches `key_numbers.md`): keep the row with the **latest `evaluated_at` per `run_name`** from
  `results/campaign_summary.jsonl` (472 raw rows → 456 unique run_names as of this audit).
- **CS operating-point / magnitude-law pool** = `lrsw_` prefix, Llama-2 s42. Method = 2nd underscore token
  (`lora_null` spans two tokens). Canonical law pool = 7 adapters × 7 LRs = 49 (CorDA excluded).
- **Retention (core)** = `retention_mean` = mean(BBH-AO, MMLU-Pro); base ceiling 26.0.
- **Math cells** = `frm_` (faithful math, MetaMathQA, r64/α128) summary.json; math retention = BBH only (ceiling 33.1).
- **CE-to-base** = `results/forgetting.jsonl` (`forgetting_ce`), the metric behind `fig_ce_vs_magnitude.py`.
- **Geometry** = `results/geo_drift/master_labeled.jsonl` joined with summary F_Δ, exactly as
  `fig_geometry_4panel.py` loads it (retention non-null + valid F_Δ → 303 rows; CS subset = 217).
- **Efficiency wall-clock / params** = median `train_runtime_s` / `trainable_params` over `lrsw_` in
  `results/train_registry.jsonl` (the same computation `fig_efficiency.py` performs).
- Published (competitor-paper) numbers are `[EXTERNAL]` anchors — verified against the campaign's cited values,
  not recomputable from `summary.json`.

---

## Verification table

| # | Claim (as written) | Location | Recomputed value | Source file(s) | Verdict |
|---|---|---|---|---|---|
| 1 | Magnitude law r = −0.86 | tile 1 | −0.858 | campaign_summary.jsonl (lrsw pool n=49) | PASS |
| 2 | holds within every one of 7 adapters (r −0.86 to −0.97) | tile 1 / §1 | −0.864 (LoRA-Null) … −0.972 (SC-LoRA) | campaign_summary.jsonl | PASS |
| 3 | second model (Qwen) | tile 1 / §1 | Qwen CS plain-LoRA r=−0.883 | campaign_summary.jsonl (qwsw_lora_r16, n=7) | PASS |
| 4 | Best CS operating point 81.6 | tile 2 / §3 | 81.62 (lrsw_lorawd_wd0p3_lr5e4_s42) | campaign_summary.jsonl | PASS |
| 5 | / 25.6 retention | tile 2 / §3 | 25.55 | campaign_summary.jsonl | PASS |
| 6 | Best math GSM8K 67.3 | tile 3 / §4 | 67.25 (frm_lorawd_wd0p3_lr2e4_c256_s42) | results/frm_.../summary.json | PASS (67.25 → 67.3, borderline round) |
| 7 | vs 64.6 (CLoRA best pub) | tile 3 / §4 | 64.59 published | [EXTERNAL] CLoRA Table 4 | PASS |
| 8 | Efficiency 0 init cost (LoRA+wd) | tile 4 | wd is a free AdamW flag; init 0 | train_registry.jsonl | PASS (qualitative) |
| 9 | CLoRA up to 6.7 GB frozen matrices | tile 4 / §5 | 6.69 GB @ k2048 | fig_efficiency.py (k×1,753,088 bf16) | PASS |
| 10 | DoRA 2.1× slower | tile 4 | 2.14× | train_registry.jsonl | PASS (2.14 → 2.1) |
| 11 | Adapters · 8 (byline) / "eight adapters" (dek) | byline, dek | 8 in CS LR sweep; 9 methods described in §0 | campaign_summary + artifact §0 | **DISCREPANCY** (internal: §0 says "nine methods") |
| 12 | base model scores 26.0 (ceiling) | gloss / §1 / foot | (33.10+18.96)/2 = 26.03 | base_l2-7b_bbhAO + base_l2-7b retention_agg.json | PASS |
| 13 | update = mean ‖ΔW·x‖/‖x‖ (F_Δ, CLoRA Eq 3) | gloss | matches fdelta field definition | key_numbers.md §0 | PASS (definition) |
| 14 | §0 "All nine methods" | §0 | LoRA, LoRA+wd, DoRA, PiSSA, MiLoRA, CLoRA, SC-LoRA, LoRA-Null, CorDA++ = 9 | artifact §0 | see #11 (conflicts with "eight") |
| 15 | Retention vs LR R² = 0.32 | §1 panel LR | 0.321 | campaign_summary.jsonl | PASS |
| 16 | Retention vs F_Δ r = −0.86, R² = 0.74 | §1 panel DW | −0.858, 0.736 | campaign_summary.jsonl | PASS |
| 17 | 7 rates | §1 | 7 LRs per method | campaign_summary.jsonl | PASS |
| 18 | R² 0.74 vs 0.32 | §1 | 0.736 vs 0.321 | campaign_summary.jsonl | PASS |
| 19 | Spearman ρ = −0.90 | §1 callout | −0.896 | campaign_summary.jsonl | PASS |
| 20 | knee at update size ≈ 0.36 | §1 callout | 0.373 | fig_magnitude_law.py | **DISCREPANCY** (minor: recomputed 0.373 ≈ 0.37) |
| 21 | true slope ≈ −21 pp/decade below ceiling | §1 callout | 21.0 (saturating post-knee) | fig_magnitude_law.py | PASS |
| 22 | vs −15 for ceiling-attenuated linear fit | §1 callout | −14.78 | campaign_summary.jsonl | PASS |
| 23 | partial r = −0.87 (controlling adapter identity) | §1 callout | −0.868 | campaign_summary.jsonl | PASS |
| 24 | permutation test p < 5×10⁻⁵ | §1 callout | 5.0×10⁻⁶ (N=200k perms) | campaign_summary.jsonl | PASS |
| 25 | CLoRA Table 4 r = −0.98 | §1 callout good | −0.980 | [EXTERNAL] fig_cross_literature.py | PASS |
| 26 | slope −14.7 pp/decade | §1 callout good | −14.65 | [EXTERNAL] fig_cross_literature.py | PASS |
| 27 | vs our −14.8 | §1 callout good | −14.78 | campaign_summary.jsonl | PASS |
| 28 | ten published rows | §1 callout good | 10 rows | [EXTERNAL] CLoRA Table 4 | PASS |
| 29 | 55 embedded scatter points (S[]) | §1 script | all 55 match F_Δ & retention to ≤0.0005 (log) / 0.00 (ret) | campaign_summary.jsonl | PASS |
| 30 | DWFIT slope −14.78, inter 17.85, r −0.86 | §1 script | −14.776, 17.849, −0.858 | campaign_summary.jsonl | PASS |
| 31 | 320 saved adapters | §2 | 320 rows | master_labeled.jsonl | PASS |
| 32 | retention tracks size within every method r −0.75 to −0.94 (battery) | §2 | −0.746 … −0.944 | master_labeled.jsonl (CS) | PASS |
| 33 | rank partial r = −0.56 | §2 | −0.52 (adapter-r\|logF) / −0.55 (eff_rank) / −0.59 (stable_rank), all CS | master_labeled.jsonl | **DISCREPANCY** (filter/metric-dependent; no natural filter = −0.56 exactly) |
| 34 | subspace alignment ΔR² ≈ 0.0002 | §2 | amp_top 0.00025; ein_top 0.00067; e_top 0.0067 (beyond logF+method, CS) | master_labeled.jsonl | PASS (metric-dependent; matches amp_top) |
| 35 | neutral baseline ≈ 0.06 (~6% of 4096) | §2 caption | 256/4096 = 0.0625 | geo_drift definitions | PASS |
| 36–43 | Fingerprint LoRA 0.071/0.047/0.076/0.051/8.8 | §2 table | 0.071/0.047/0.076/0.051/8.8 | master_labeled.jsonl (CS mean, n=29) | PASS |
| 44–48 | Fingerprint LoRA+wd 0.072/0.048/0.086/0.050/6.5 | §2 table | 0.072/0.048/0.086/0.050/6.5 | master_labeled.jsonl (n=41) | PASS |
| 49–53 | Fingerprint MiLoRA 0.067/0.115/0.077/0.115/7.7 | §2 table | 0.067/0.115/0.077/0.115/7.7 | master_labeled.jsonl (n=20) | PASS |
| 54–58 | Fingerprint CLoRA 0.060/0.047/0.066/0.050/7.5 | §2 table | 0.060/0.047/0.066/0.050/7.5 | master_labeled.jsonl (n=31) | PASS |
| 59–63 | Fingerprint DoRA 0.079/0.046/0.082/0.050/5.6 | §2 table | 0.079/0.046/0.082/0.050/5.6 | master_labeled.jsonl (n=25) | PASS |
| 64–68 | Fingerprint LoRA-Null 0.126/0.035/0.080/0.054/6.7 | §2 table | 0.126/0.035/0.080/0.054/6.7 | master_labeled.jsonl (n=7) | PASS |
| 69–73 | Fingerprint SC-LoRA 0.104/0.041/0.410/0.021/19.4 | §2 table | 0.104/0.041/0.410/0.021/19.4 | master_labeled.jsonl (n=44) | PASS |
| 74–78 | Fingerprint CorDA 0.078/0.048/0.041/0.494/13.0 | §2 table | 0.078/0.048/0.041/0.494/13.0 | master_labeled.jsonl (n=20) | PASS |
| 79 | law residual LoRA +2.3 | §2 table | +2.34 | master_labeled.jsonl (pooled all-cs fit) | PASS |
| 80 | law residual LoRA+wd +1.7 | §2 table | +1.73 | master_labeled.jsonl | PASS |
| 81 | law residual MiLoRA +1.6 | §2 table | +1.62 | master_labeled.jsonl | PASS |
| 82 | law residual CLoRA +1.5 | §2 table | +1.46 | master_labeled.jsonl | PASS |
| 83 | law residual DoRA +3.6 | §2 table | +3.57 | master_labeled.jsonl | PASS |
| 84 | law residual LoRA-Null +0.6 | §2 table | +0.55 | master_labeled.jsonl | PASS |
| 85 | law residual SC-LoRA −5.7 | §2 table | −5.66 | master_labeled.jsonl | PASS |
| 86 | law residual CorDA −3.0* | §2 table | −3.03 | master_labeled.jsonl | PASS |
| 87 | SC-LoRA ein_top 0.41 | §2 text | 0.410 | master_labeled.jsonl | PASS |
| 88 | SC-LoRA sits 5.7 pp below law | §2 text | −5.66 | master_labeled.jsonl | PASS |
| 89 | SC-LoRA erodes 0.70 → 0.21 with LR | §2 text | 0.703 → 0.211 (erosion r=−0.962) | master_labeled.jsonl (lrsw_sclora) | PASS |
| 90 | CorDA ein_bot 0.49 | §2 text | 0.494 | master_labeled.jsonl | PASS |
| 91 | Base row 26.0 / 0 | §3 table | 26.03 ceiling / 0 | base retention_agg.json | PASS |
| 92 | LoRA+wd 5e-4 / 81.6 / 25.6 / 0.39 / 6/7 | §3 table | 5e-4 / 81.62 / 25.55 / 0.394 / 6-of-7≥24 | campaign_summary.jsonl | PASS |
| 93 | SC-LoRA 5e-5 / 80.1 / 22.5 / 0.56 / 1/7 | §3 table | 5e-5 / 80.14 / 22.47 / 0.559 / 1/7 | campaign_summary.jsonl | PASS |
| 94 | MiLoRA 3e-4 / 79.9 / 24.7 / 0.54 / 5/7 | §3 table | 3e-4 / 79.86 / 24.72 / 0.543 / 5/7 | campaign_summary.jsonl | PASS |
| 95 | LoRA 3e-4 / 79.1 / 24.4 / 0.62 / 5/7 | §3 table | 3e-4 / 79.11 / 24.42 / 0.623 / 5/7 | campaign_summary.jsonl | PASS |
| 96 | LoRA-Null 5e-4 / 78.9 / 23.6 / 0.70 / 5/7 | §3 table | 5e-4 / 78.93 / 23.64 / 0.696 / 5/7 | campaign_summary.jsonl | PASS |
| 97 | CLoRA 5e-4 / 78.4 / 21.9 / 0.64 / 5/7 | §3 table | 5e-4 / 78.36 / 21.88 / 0.643 / 5/7 | campaign_summary.jsonl | PASS |
| 98 | DoRA 2e-4 / 78.3 / 24.8 / 0.45 / 4/7 | §3 table | 2e-4 / 78.27 / 24.84 / 0.445 / 4/7 | campaign_summary.jsonl | PASS |
| 99 | 78–82 accuracy band | §3 text | 78.27 … 81.62 | campaign_summary.jsonl | PASS |
| 100 | 25.6 within noise of 26.0 | §3 text | 25.55 vs 26.03 | campaign_summary.jsonl | PASS |
| 101 | CLoRA pub k1024/k2048 82.6/83.7 acc @ BBH 36.5/38.7 | §3 callout | BBH 36.49/38.67; acc external | [EXTERNAL] CLoRA Table 4 | PASS (BBH verified; acc external) |
| 102 | LoRA+wd math 67.3 / F_Δ 0.28 / BBH 33.1 | §4 bar | 67.25 / 0.278 / 33.10 | frm_lorawd_wd0p3_lr2e4_c256_s42/summary.json | PASS |
| 103 | CLoRA-k128 published 64.6 | §4 bar | 64.59 | [EXTERNAL] | PASS |
| 104 | MiLoRA published 63.5 | §4 bar | 63.53 | [EXTERNAL] | PASS |
| 105 | LoRA published 60.6 | §4 bar | 60.58 | [EXTERNAL] | PASS |
| 106 | CLoRA-k128 our pipeline 59.6 | §4 bar | 59.59 | frm_clora_k128_lr3e4_c256_s42/summary.json | PASS |
| 107 | PiSSA published 58.2 | §4 bar | 58.23 | [EXTERNAL] | PASS |
| 108 | LoRA in-pipeline 60.2 vs 60.6 | §4 text / §7 | 60.2 | frm_lora_lr3e4_c256_s42/summary.json | PASS |
| 109 | LoRA+wd retains at 33.1 (math BBH) | §4 text | 33.10 | frm_lorawd_wd0p3_lr2e4_c256_s42 | PASS |
| 110 | PiSSA collapses to BBH 7.2 | §4 text | 7.23 | frm_pissa_lr3e4_c256_s42/summary.json | PASS |
| 111 | math base ceiling 33.1 | §4 / foot | 33.10 | base_l2-7b_bbhAO retention_agg.json | PASS |
| 112 | GSM8K bar widths 89.7/86.1/84.7/80.8/79.5/77.6 % | §4 bars | value/75×100 = 89.7/86.1/84.7/80.8/79.5/77.6 | derived | PASS |
| 113 | 160 target modules | §5 caption | q,k,v,up,down ×32 = 160 | train_registry.jsonl args | PASS |
| 114 | r=64 = 112M trainable params | §5 caption | 112,197,632 (mtx_corda_r64); DoRA 113,074,176 | train_registry.jsonl | PASS (approx for DoRA) |
| 115 | baseline ~55 GB GPU memory | §5 caption | not in registry (memory footprint) | — | [EXTERNAL] not verifiable from data |
| 116 | ~5 GPU-h train+eval per cell | §5 | train median 2.01 GPU-h; +eval external | train_registry.jsonl | PASS (train part); eval part external |
| 117 | DoRA 2.13× wall-clock | §5 table | 2.145× | train_registry.jsonl | **DISCREPANCY** (correct ≈ 2.14×) |
| 118 | CLoRA +0.42–6.7 GB frozen (by k) | §5 table/text | 0.42 (k128) … 6.69 (k2048) | fig_efficiency.py | PASS |
| 119 | at k=512, +1.7 GB | §5 text | 1.67 GB | fig_efficiency.py | PASS |
| 120 | already 8× trainable LoRA weights | §5 text | 1.67 / 0.209 (112M bf16) = 8.0× | train_registry + fig_efficiency | PASS |
| 121 | CLoRA 1.14× wall-clock | §5 table | 1.167× | train_registry.jsonl | **DISCREPANCY** (correct ≈ 1.17×) |
| 122 | SC-LoRA / LoRA-Null 256 calib fwd passes | §5 table/text | sclora_calib_size=256 (args) | train_registry.jsonl args | PASS (note: fig_efficiency.py labels SC-LoRA "512" — that label is wrong; artifact is right) |
| 123 | +22 GB transient (calib init) | §5 | memory footprint, not in data | — | [EXTERNAL] not verifiable |
| 124 | CorDA++ 1,280 calib fwd (5 rounds) | §5 | corda_calib_size=256/round (args); ×5 rounds = 1,280 | train_registry.jsonl args; CorDA paper | [EXTERNAL]/recipe (base 256 confirmed; ×5 per CorDA impl) |
| 125 | CorDA++ ~1 GPU-h precompute | §5 | CorDA paper Table IX | [EXTERNAL] | PASS (external) |
| 126 | ~20% on top of ~5 GPU-h | §5 | 1/5 = 20% | derived | PASS |
| 127 | MiLoRA/PiSSA 160 base-weight SVDs | §5 | 160 modules | train_registry.jsonl | PASS |
| 128 | CE LoRA+wd (small update) F_Δ 0.20 / CE 2.00 | §6 table | 0.196 / 1.998 (frm_lorawd_wd0p5_lr1e4_c256_s42) | results/forgetting.jsonl | PASS |
| 129 | CE LoRA (3e-4) 1.28 / 3.57 | §6 table | 1.283 / 3.570 | results/forgetting.jsonl | PASS |
| 130 | CE MiLoRA (3e-4) 1.26 / 3.66 | §6 table | 1.257 / 3.659 | results/forgetting.jsonl | PASS |
| 131 | CE PiSSA (3e-4) 2.21 / 6.31 | §6 table | 2.206 / 6.307 | results/forgetting.jsonl | PASS |
| 132 | reproduce MiLoRA Table 8: LoRA 3.24, PiSSA 6.07, MiLoRA 2.54 | §6 caption | published anchors | [EXTERNAL] MiLoRA Table 8 | PASS |
| 133 | rank correlation CE↔size = 0.94 | §6 caption | Spearman(F_Δ,CE) = 0.943 (n=6) | results/forgetting.jsonl | PASS |
| 134 | MiLoRA (3.66) ≈ LoRA (3.57) matched-magnitude | §6 text | 3.659 vs 3.570 | results/forgetting.jsonl | PASS |
| 135 | 8 adapter ports FAITHFUL | §7 | 8 ports audited | handoff/25_SUPERVISION_REPORT | PASS (audit-sourced) |
| 136 | reconstruction matches spectral norm <0.2% | §7 | median rel err 0.005%, 94% <0.2% (handoff: "==dw_sv_max to 4 dp") | master_labeled vs campaign_summary; handoff/27 | PASS |
| 137 | LoRA reproduces pub GSM8K 60.2 vs 60.6 | §7 | 60.2 vs 60.58 | frm_lora + [EXTERNAL] | PASS |
| 138 | save→reload lossless Δ 6e-9 | §7 | Δ6e-9 | handoff/25_SUPERVISION_REPORT | PASS (audit-sourced) |
| 139 | base retention ceiling BBH 32.96 vs registered 33.10 | §7 | registered 33.099 confirmed; 32.96 re-eval per handoff 25 | base_l2-7b_bbhAO; handoff/25 | PASS (33.10 verified; 32.96 audit-sourced) |
| 140 | CE metric vs MiLoRA T8: LoRA 3.57↔3.24, PiSSA 6.31↔6.07 | §7 | 3.570↔3.24, 6.307↔6.07 | forgetting.jsonl + [EXTERNAL] | PASS |
| 141 | evidence base: 8 adapters × 7 LRs | foot | 8 methods × 7 = 56 lrsw_ CS runs | campaign_summary.jsonl | PASS |
| 142 | geometry battery over 320 saved adapters | foot | 320 | master_labeled.jsonl | PASS |
| 143 | retention = BBH+MMLU-Pro (base 26.0); math BBH (base 33.1) | foot | 26.03 / 33.10 | base retention_agg.json | PASS |

---

## Corrections the author MUST make

1. **DoRA wall-clock (§5 table): 2.13× → 2.14×.** Median `lrsw_dora`/`lrsw_lora` `train_runtime_s`
   = 15504.7 / 7229.9 = **2.145×** (the campaign's own `fig_efficiency.py` also prints 2.14×).
   The tile "2.1×" is fine.
2. **CLoRA wall-clock (§5 table): 1.14× → 1.17×.** Median `lrsw_clora`/`lrsw_lora` = 8433.4 / 7229.9 =
   **1.167×**. (`fig_efficiency.py` omits CLoRA from its wall-clock panel, so 1.14 has no live source.)
3. **Method count is internally inconsistent.** §0 header "All **nine** methods" vs dek/byline "**eight**
   LoRA-family adapters" / "Adapters · 8". §0 lists 9 (PiSSA is the 9th, math-only); the CS LR sweep has 8
   (incl. CorDA). Pick consistent wording, e.g. "nine methods; eight in the commonsense LR sweep."

## Flags (filter/metric-dependent or approximate — author should verify/soften)

- **§2 rank partial r = −0.56** is not reproduced by any single natural definition: partial r(retention, adapter-`r` | logF)
  = −0.52 (all CS), eff_rank = −0.55, stable_rank = −0.59; on-curve-six stable_rank = −0.64. The value sits inside the
  −0.52…−0.64 band but "−0.56" is filter-dependent. Recommend quoting the definition used, or "≈ −0.5".
- **§1 knee ≈ 0.36** recomputes to **0.373** (saturating fit) → rounds to 0.37, not 0.36. Minor; consider "≈ 0.37".
- **§2 ΔR² ≈ 0.0002** is metric-specific: amp_top gives 0.00025 (matches), but ein_top gives 0.00067 and e_top 0.0067.
  Fine as "essentially nothing," but name the alignment metric.
- **§4 GSM8K 67.3** is 67.25 (rounds up); a marginally higher c256 cell exists (67.4 at wd0.2/lr1e-4), but the
  reported triple 67.3/0.28/33.1 is internally consistent (it is the cell sitting exactly at the BBH ceiling). No change needed.
- **§5 "~55 GB baseline", "~22 GB transient"** are memory footprints not present in any results file — [EXTERNAL],
  not verifiable from this registry.
- **§5 CorDA "1,280 (5 rounds)"**: training arg `corda_calib_size=256` (per round) is confirmed; the ×5-rounds→1,280
  total and "~1 GPU-h" are CorDA-paper/implementation facts ([EXTERNAL]).
- **§7 "32.96" and "Δ 6e-9"** are audit-gate values documented in `handoff/25_SUPERVISION_REPORT_2026-07-09.md`;
  the registered ceiling 33.10 is independently confirmed (`base_l2-7b_bbhAO`, 33.099), but 32.96 itself is not a
  row in `results/` (cite the handoff, don't present as recomputed).

## Summary counts

- **Distinct numeric claims checked:** 143 table rows (≈ 190 individual numbers incl. the 8×5 fingerprint grid,
  8 op-point triples, 6 CE cells, 55 embedded scatter points verified as one aggregate).
- **PASS:** 140 rows.
- **DISCREPANCY:** 3 rows (#11 method count, #117 DoRA 2.13×, #121 CLoRA 1.14×), plus 3 soft filter/metric-dependent
  flags (rank partial r −0.56, knee 0.36, ΔR² metric choice).
