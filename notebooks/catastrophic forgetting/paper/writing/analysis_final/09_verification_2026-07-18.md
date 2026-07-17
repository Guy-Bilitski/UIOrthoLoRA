# 09 — INDEPENDENT VERIFICATION OF THE ANALYSIS LAYER (2026-07-18)

`[Two independent audit passes over the frozen numbers (§18), the addendum
analyses (§19, 06–07), and the underlying data stores. Pass 1: full numerical
re-derivation with numpy/scipy (QR/SVD least squares — a different linear-algebra
path from the stdlib normal-equations scripts) — verify_numerics_log_2026-07-18.md.
Pass 2: semantic/data-quality audit of what the fields actually measure —
verify_semantics_log_2026-07-18.md. Plus a same-day CE-protocol sensitivity run
(ce_protocol_check_output_2026-07-18.txt). No frozen number was changed.]`

## VERDICT

**Every committed number is arithmetically correct.** All ~90 values in
ladder/seed-stability/284B outputs and the §18.1 preflight were reproduced
exactly (≤ print-rounding) by an independent implementation. All joins are
collision-free; retention_mean = mean(BBH, MMLU-Pro) holds in all 1,511
summary.json files with zero mismatches; CE identity kl = ce − base_entropy holds
to <1e-9 in all 1,354 rows; the geometry merged store is a value-identical
superset of the per-model files with per-model base-SVD provenance confirmed;
the quarantine-32 claim is exact; V1 n=1002 confirmed.

**Qualifiers that change how numbers should be QUOTED (not their values):**

### Q1 — Within-cell seed correlation (affects F/t statistics, not R²)
Run-level rows contain 3–5 seeds per recipe cell; residual ICC = 0.78, design
effect ≈ 2.6 (effective n ≈ 401). The committed OLS F=1890 overstates evidence
~2.6×. Cluster-robust (343 cells): t(log F_Δ) −17.9 → **−7.9** (decisive);
e_top −5.1 → −3.5; stable_rank −7.4 → −3.5; **log spec_max +2.1 → +1.4, NOT
significant** — the one committed effect that dies under clustering (and the term
06 §5 already reclassified as magnitude-contaminated; it is now doubly dead).
Cell-level cluster bootstrap (B=2000, 95% CI): magnitude ΔR² +0.395
[+0.311, +0.482]; shape-geometry unique +0.016 [+0.006, +0.032];
magnitude-unique +0.296 [+0.203, +0.386]; β_std(log F_Δ) −0.744 [−0.894, −0.615].
**Ordering magnitude > geometry: 2000/2000 bootstrap replicates.**
→ Paper rule: quote ΔR² magnitudes + these CIs; never quote raw OLS F as evidence
strength; drop any positive-spec_max sentence.

### Q2 — CE protocol mixture (checked; benign)
forgetting_merged.jsonl mixes two CE protocols within families (40-block slice vs
full WikiText-103 test; 4 base_entropy values = 2 models × 2 protocols; pool split
283/628). Sensitivity: protocol dummy leaves the KL step unchanged (ΔR² +0.0049 →
+0.0049; interaction +0.0067); per-family r(KL, ret) nearly identical across
protocols (e.g., lrsw −0.856 full vs −0.876 short; frc −0.870 vs −0.871).
→ Disclose the mixture (one sentence); no re-analysis needed.

### Q3 — 284B recurrence carries a rank-pooling confound (07 amended)
Residual methods (milora/sclora/lora_null) save rank-2r adapters, so part of the
284B "fingerprint" (stable_rank 4.2–4.9 vs 1.6–1.8) is the saved-rank dichotomy
recurring by construction. At 7B, an r16-restricted comparison is possible for
only 4 methods (ordering identical, Spearman +1.00, n=4); the only all-7 stratum
(trained-r32) gives Spearman **+0.32/+0.14** vs the pooled +0.86/+0.75.
→ 07 §1 downgraded (see its amendment): the defensible claim is the design-family
dichotomy (residual vs direct) recurring + directional consistency (6/6 families
positive, sign test p≈0.03); within-stratum ordering evidence is weak. Do not
quote +0.86 unqualified.

### Q4 — Bookkeeping facts
- `frc_lorawdr16_wd0p3_lr3e4_c256_s42_reeval` duplicates its parent run inside the
  frozen n=1035 (byte-identical values). It enters no regression (no geometry row).
  Footnote if "1,035 distinct runs" is ever claimed.
- 123-vs-136 Qwen CE counts RESOLVED: 136 = freeze-time missing set
  (jobs/ce_backfill_qwen.txt); 13 of those were backfilled 07-17 (all s42, full
  protocol); 123 = currently missing. Backfill file refreshed accordingly.
- Qwen CE missingness is a seed-block deletion (107×s42 + 9×s43 + 7×s44; s42
  adapters destroyed 07-03), method-balanced; CE-present vs CE-absent runs are
  indistinguishable on log F_Δ and retention (all rank-sum |z| < 0.8). Ignorable
  for the CE-corroboration regressions; per-seed Qwen CE analyses are impossible
  and barred.
- DS 284B adapt scores (20 values) are single-source (relaunch log did not survive
  evacuation); the diverged run is independently corroborated (handoff/41).
  Disclose as single-source.

## STATUS OF REVIEWER ATTACKS AFTER VERIFICATION
A4 (ladder order-dependence): resolved (06 §5) and now cluster-bootstrap-backed.
A8 (Qwen CE count + missingness): RESOLVED (Q4). A11 (284B overstated): confirmed
and extended by Q3 — the sign-test framing is mandatory, plus rank-pooling
disclosure. All other attack dispositions in 08 §4 stand.
