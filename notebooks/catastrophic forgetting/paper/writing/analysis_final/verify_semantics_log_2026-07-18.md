# Semantic + data-quality audit — catastrophic-forgetting pipeline (2026-07-18)

Auditor: subagent (semantics/data-quality). Raw tool output: `audit_out.txt` (same dir);
scripts: `audit.py`, `audit2.py`, `audit3.py`. All paths below relative to
`/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting/` unless absolute.

## Item 1 — Retention definition consistency: PASS

- Sampled 40 summary.json (lrsw/lrswm/qwsw/qwswm/frc/frm/e1/b4), then full-scan of all
  1511 `results/*/summary.json`:
  - `headline.retention_mean == mean(headline.bbh, headline.mmlu_pro)` in **all** files that
    have the fields (0 mismatches, tol 0.011 for rounding).
  - `headline.retention_broad == mean(bbh, mmlu_pro, mmlu, arc_c, truthfulqa)`: 0 mismatches.
  - 53 summaries lack task fields entirely (CE-only/partial runs); they carry no retention_mean
    and fall out of the pool loader anyway.
- Units uniform: retention on 0–100 scale in every family (pool medians 21.7–43.4; e1/b4 same scale).
- fdelta semantics: top-level `fdelta` is a dict; `fdelta_token_weighted` == `headline.fdelta`
  exactly in all 20 sampled files (the earlier "mismatch" was dict-vs-float type, not value).
  log10(fdelta) medians per family: lrsw −0.36, lrswm −0.45, qwsw −0.62, qwswm −0.95,
  frc −0.27, frm −0.25 — same convention/order of magnitude across Llama and Qwen.

## Item 2 — CE store semantics: PASS on identity checks, CONCERN on protocol mixture

- `results/forgetting_merged.jsonl`: 1354 rows, 0 duplicate run_names.
- 12/12 random sidecar `results/<run>/forgetting.json` rows match merged **byte-exactly**
  (forgetting_ce, base_entropy, forgetting_kl, n_positions, n_blocks). All 1317 sidecars are in
  merged; 37 merged rows without sidecars = old rows kept by union semantics of
  `fleet/evac_merge.py` (per-run file wins, old merged kept when no sidecar — as documented).
- `forgetting_kl == forgetting_ce − base_entropy` holds to <1e-9 in **all 1354 rows**; no
  negative KL anywhere.
- **base_entropy has 4 distinct values, not 2** — but coherently: 2 models × 2 eval protocols:
  - Llama 40-block slice: 1.8520239241 (n_positions 40920 = 40×1023)
  - Llama full WikiText-103 test: 1.8100692117 (337590 = 330×1023)
  - Qwen 40-block slice: 1.8790178807 (40920; only 16 rows, 11 in pool)
  - Qwen full test: 1.9346918347 (297693 = 291×1023)
  n_positions/n_blocks take exactly these 3 shapes; nothing suspicious beyond the mixture itself.
- **CONCERN (protocol mixture inside the pool):** pool CE rows mix short-slice and full-test
  within families: frc 90/185 (40-block/full), frm 65/79, lrsw 76/104, lrswm 41/79,
  qwsw 7/86, qwswm 4/95. `fleet/derive_loop.sh` deliberately re-scored only adapters still
  present ("evacuated adapters keep whatever record they have"), so the mixture is by design and
  handoff/41 says to disclose the 40-block cap — but the ladder's `kl` regressor treats the two
  protocols as one variable. KL subtracts the matched base entropy, so the first-order corpus
  effect cancels, yet the two estimates are on different text amounts (40 vs 291/330 blocks).
  Not a bug; must stay disclosed. If CE ever becomes headline (it is corroboration only), split
  by protocol.

## Item 3 — Geometry store semantics: PASS

- `adapter_metrics_merged.jsonl`: 1470 rows, unique keys. Local `adapter_metrics.jsonl` (Llama,
  496 rows) and `adapter_metrics_qwen.jsonl` (149) are **strict subsets with identical values**:
  5+5 spot-checked runs → all 13 shared numeric fields equal, no extra fields in merged.
  The other 825 merged rows come from other nodes' per-run `results/<run>/geo.json` files via
  `fleet/evac_merge.py` (union; per-run file wins) — merged was *rebuilt from sidecars*, not
  recomputed with different settings.
- The premise "llama per-file lacks e_top" is **false**: e_top_w present 496/496 in the local
  llama file, 149/149 qwen, 1470/1470 merged.
- Per-model base-SVD handling confirmed in code:
  - `geo_drift_phase2.py` → `results/geo_drift/base_svd/` (Llama, from `geo_drift_phase1.py`,
    writes adapter_metrics.jsonl); `geo_drift_phase2_qwen.py` overrides
    `BASE_SVD=base_svd_qwen`, filters adapters by `adapter_config.base_model_name_or_path`
    containing "Qwen", writes adapter_metrics_qwen.jsonl ("kept separate ... never pooled by
    accident"). `fleet/derive_loop.sh` runs phase1 for both models per node before phase2.
  - Cross-model contamination is dimensionally impossible: a Qwen adapter (hidden 3584) hitting
    the Llama base SVD (4096×256) raises a shape error in `matrix_metrics` (refU.T @ dirs), which
    phase2 catches and logs as ERR — it cannot silently write a wrong row.
  - File purity: llama file families = {b4, clora, dora, frc, frm, lora, lrsw, lrswm, mem, mtx,
    mtxm, scl2, valfix}; qwen file = {qwsw, qwswm} only.
- Value plausibility: all e_top_w in [0,1] (0 violations, n=1470). Llama-family medians
  0.067–0.076 vs Qwen-family 0.112–0.118 — two clean per-model clusters, method ordering smooth
  and consistent across models (lora_null > sclora > lora/dora > milora/clora within each model).
  Consistent with two different (correct) base SVDs.

## Item 4 — DeepSeek 284B comparability (07's Spearman +0.86): CONCERN

Facts established (geo `r` field + training scripts, not just name parsing):

1. **Saved vs trained rank.** Residual-init methods (milora, sclora, lora_null) are saved as
   rank-2r W0-relative adapters (`train_cs.py` → `residual_save`, "converted ... to
   W0-relative rank-{2*r0} adapter"). So `r` in geo rows = saved factor rank. The DS 284B grid is
   trained-r16 for all 7 methods (dsv4 rows: r=16/α=32 for lora/lorawd/dora/clora; r=32/α=64,
   n_mat=255 for lora_null/milora/sclora — exactly the residual doubling; run names all `_r16_`).
   The "r16-only" claim is correct in trained-rank terms.
2. **The 7B pool is NOT rank-matched by design.** Trained-rank distribution per method
   (pool families, merged geo): lora {8:5,16:94,32:36,64:3}, lorawd {16:10(frc `lorawdr16` arm
   only), 32:234, 64:105}, dora {16:94,32:3,64:3}, sclora {16:1,32:115,64:6,128:3},
   clora {32:124,64:11}, milora {32:147,64:14}, lora_null {16:73,32:28,64:6}.
   → **clora, milora, sclora have zero trained-r16 runs at 7B; lorawd only the 10-run frc-only
   `lorawdr16` arm** (which 07's `method_of` excludes since token "lorawdr16" ∉ METHODS).
3. **r16-restricted recompute** (the mission's ask): only 4 methods computable
   (lora, dora, lora_null, lorawd-via-lorawdr16), and per-family medians survive in ≤3 methods
   per family (frm: none). Pooled r16 ordering: dora < lora < lorawd < lora_null on BOTH
   stable_rank (4.18 < 4.73 < 5.83 < 6.79) and eff_rank (10.57 < 11.52 < 12.16 < 15.93) —
   **identical to the all-ranks ordering restricted to those methods, and identical to the 284B
   ordering of those methods** (Spearman +1.00, but n=4 and lorawd rests on one frc arm).
4. **Matched trained-r32 sensitivity** (the only rank stratum where all 7 methods exist):
   Spearman vs 284B drops to **+0.32 (stable_rank) / +0.14 (eff_rank)** from +0.86/+0.75
   pooled. Caveat: the r32 stratum is compositionally skewed (dora n=3 total, lora absent from
   3 families, lora_null r32 only in frc), so this is itself not a clean test.
5. **Structural observation:** the 284B stable-rank fingerprint splits exactly on saved factor
   rank — saved-r16 methods 1.55–1.78 vs saved-r32 residual methods 4.21–4.89. Part of the
   "recurrence" is the residual-method rank-capacity dichotomy recurring, which is
   method-inherent (same at both scales) but is not "geometry style" independent of rank.

**Verdict:** the ordering the +0.86 rests on does NOT change within the subset where a matched
r16 comparison is possible (item 3 above), but a full matched-rank 7-method test is impossible
at 7B, and the one feasible all-7-method stratum (r32) yields much weaker correlations. 07
should state that 7B medians pool trained ranks r8–r128 and that lorawd/clora/milora/sclora were
never run at trained-r16 at 7B.

## Item 5 — Quarantine semantics: PASS

- `results/quarantine_diverged.txt`: 71 lines, 71 unique names.
- Pool rebuilt under the freeze convention (drop SMOKE/smoke/corda substrings, finite fd>0 &
  retention_mean, 6 family prefixes): current n=1042; minus the 7 STRAGGLERS = **1035** ✓.
- Quarantine ∩ primary pool = **32** ✓ (frm 8, lrsw 10, frc 8, qwswm 4, qwsw 2).
- The other 39 quarantine names fully accounted: 16 bad/absent fdelta, 14 corda/cordapp
  (substring-excluded), 8 mtx family, 1 scl2 family (non-pool families).
- Geometry-joined primary pool = 1034; quarantine-excluded V1 = **1002** ✓ (matches
  `ladder_output_2026-07-17.txt` line 63: "V1 ... (n=1002, 10 methods ...)").

## Item 6 — DS adapt-score provenance: CONCERN (single-source scores; diverged flag corroborated)

- `results/dsv4_adapt_n1000_logscores.jsonl`: 20 rows (21 − d016), internally consistent
  (medmcqa_acc == n_correct/n×100 exactly; unique node↔run map; d016 = dsv4_lorawd_r16_lr5e4_s42
  absent, matching handoff/41 "d016 DS adapt score — node gone before salvage").
- Diverged flag independently corroborated: handoff/41_EVACUATION_2026-07-17.md:63
  "`d032` (`dsv4_lora_null_r16_lr5e4_s44`) had diverged (adapt≈25.7% = chance)" — matches the
  file's 25.7/257/1000 row.
- **The claimed raw source (`relaunch_0717.log`) does not survive in the repo**: harvested
  `results/evac_logs/<node>/` contains only locks/pidlocks/gapfill scripts; git history shows the
  logscores file introduced whole in the evacuation harvest commit 49d7f6ce. So the 19 non-diverged
  values are single-source (transcribed once); no second artifact can confirm e.g. 78.1 or 79.0.
  Nothing contradicts them either. Flag as unverifiable-but-unchallenged.

## Item 7 — 123 vs 136 Qwen CE (reviewer A8): RESOLVED

Both numbers are counts of "pool Qwen runs missing CE", at different times; today's truth is 123:

- `jobs/ce_backfill_qwen.txt` = **136** run names, ALL inside the current pool (0 strays,
  0 smoke/corda, 0 without summary.json). This was the freeze-time (§18.6) missing set.
- **13 of those 136 have since received CE** — all seed-42, all scored at the FULL Qwen protocol
  (base_entropy 1.934692, n_blocks 291), sidecar mtimes 2026-07-17 06:34Z and 09:49Z
  (i.e., the final fleet sync ~06:34Z and a last harvest). 136 − 13 = **123 still missing**,
  which is what a live recount gives: Qwen pool current 321 / primary 315, missing CE = 123 in
  both (the 6 Qwen stragglers all have CE).
- §17.8's earlier "123" was the pre-freeze count — numerically equal to today's 123 by
  coincidence (13 new gaps appeared as more Qwen summaries synced at freeze, then 13 old gaps
  were filled after freeze).
- Handoff/41's "123 primary-seed Qwen CE cells permanently impossible" matches today's count.
- Outside the pool, only 2 more Qwen dirs lack CE (qwsw_corda_r16_lr2e5_s42,
  qwswm_SMOKE_lorawd_lr5e4_s42) — excluded by convention; total dirs missing = 125.
- Recommendation: key_numbers should say "136 at freeze; 13 backfilled 07-17; 123 final", and
  regenerate `ce_backfill_qwen.txt` or annotate the 13 stale entries.

## Item 8 — Qwen CE missingness (reviewer A8): systematic by SEED, ignorable w.r.t. regression variables

Primary Qwen pool (n=315): CE-present 192 vs CE-absent 123.

| stratum | var | present mean±sd (n) | absent mean±sd (n) | rank-sum z |
|---|---|---|---|---|
| qwsw | log10 fd | −0.574±0.369 (93) | −0.600±0.329 (58) | +0.19 |
| qwsw | retention | 31.46±12.28 | 31.47±12.38 | +0.78 |
| qwswm | log10 fd | −0.883±0.506 (99) | −0.827±0.531 (65) | −0.76 |
| qwswm | retention | 39.07±10.77 | 38.45±11.25 | +0.46 |
| pooled | log10 fd | −0.733±0.471 (192) | −0.720±0.462 (123) | −0.36 |
| pooled | retention | 35.38±12.14 | 35.16±12.30 | +0.44 |

- No detectable skew in either regression variable (all |z| < 0.8; means within 0.06 dex / 0.6 pts).
- Method-balanced: absent counts 14–24 across the 7 methods.
- **Strongly seed-structured**: absent = 107× s42, 9× s43, 7× s44; present = 13× s42, 87× s43,
  87× s44, 5× s45. Mechanism documented in handoff/41: primary-seed (s42) adapters were trained
  2026-07-03 on since-destroyed infra — CE unfillable. Since seed is a randomization device and
  the observed fd/retention distributions of missing rows are indistinguishable from present
  ones, missingness is plausibly ignorable (MAR-given-nothing) for the CE-corroboration
  regressions; but any per-seed CE contrast on Qwen (e.g., seed-42-only analyses) is impossible
  and should be barred in the text.

## Summary table

| # | Item | Verdict |
|---|---|---|
| 1 | Retention definition | PASS |
| 2 | CE store | PASS (identities) / CONCERN (40-block vs full-test mixture in pool) |
| 3 | Geometry store | PASS |
| 4 | DS 284B comparability | CONCERN (no rank-matched 7-method test possible; pooled +0.86 falls to +0.32/+0.14 at r32; r16-feasible subset ordering unchanged) |
| 5 | Quarantine | PASS (32 in pool; V1 n=1002 exact) |
| 6 | DS adapt provenance | CONCERN (scores single-source; diverged flag corroborated) |
| 7 | 123 vs 136 | RESOLVED (136 at freeze; 13 s42 backfilled 07-17; 123 now) |
| 8 | Qwen CE missingness | Systematic by seed (s42 lost infra), ignorable w.r.t. fd/retention |
