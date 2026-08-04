# RQ Brief — Llama-2-7B x Math (lrswm: LR sweep, MetaMathQA-100k; frm: faithful-recipe grid, MetaMathQA-395k, c256)

`[2026-07-30. Source: frozen pool.csv (n=1035, quarantined-but-finite included). Two conventions,
stated per exhibit: (i) MAGNITUDE relation = full family, retention = ret (retention_mean), frozen
convention; (ii) METHOD adjudication = quarantine-excluded, frm c256-only, retention = BBH alone
(base Llama BBH 33.1) because the MMLU-Pro parser fails on MetaMath-tuned outputs. The frc _reeval
dedupe does not touch these families. All numbers recomputed from pool.csv this pass.]`

**Preflight vs frozen anchors (§18.1): PASS.** lrswm n=120, r(log10 F_Delta, ret) = -0.865
(anchor -0.865); frm n=144, r = -0.929 (anchor -0.929). On BBH: lrswm -0.779, frm -0.900.

## Operating points (best-adaptation cell per method; mean +/- SD over seeds; adjudication convention)

| Method | Cell (lr) | n | GSM8K | BBH | F_Delta | KL |
|---|---|---|---|---|---|---|
| LoRA+wd (wd0.3) | frm 2e-4 | 3 | **66.79 +/- 0.79** | **33.57 +/- 1.04** | 0.28 | 0.23 |
| LoRA (wd0) | frm 1e-4 | 3 | 63.99 +/- 0.87 | 31.29 +/- 0.35 | 0.44 | 0.52 |
| LoRA-Null | frm 1e-4 | 1 | 63.76 | 32.15 | 0.47 | 0.73 |
| MiLoRA | frm 1e-4 | 3 | 63.68 +/- 0.80 | 32.44 +/- 0.10 | 0.45 | 0.67 |
| CLoRA-k256 | frm 3e-4 | 3 | 60.65 +/- 0.40 | 28.63 +/- 0.11 | 1.01 | 1.40 |
| SC-LoRA | frm 1e-4 | 3 | 60.47 +/- 0.53 | 27.94 +/- 0.52 | 0.86 | 1.18 |
| CLoRA-k128 / k64 | frm 3e-4 | 3/4 | 60.00 +/- 0.52 / 59.38 +/- 1.05 | 28.17 / 28.08 | 1.09 / 1.13 | 1.52 / 1.62 |
| DoRA | frm 3e-4 | 3 | 59.19 +/- 0.50 | 28.09 +/- 1.07 | 2.85 | 2.42 |
| PiSSA (recipe-rate only) | frm 3e-4 | **1** | 49.66 | 7.23 | 2.21 | 4.46 |

Headline verified: LoRA+wd vs CLoRA-k256, paired per seed (42-44): dGSM8K = +6.14 (t=26.2,
p=0.0015), dBBH = +4.94 (t=9.1, p=0.012). LoRA+wd vs plain LoRA at each's best cell:
+2.81 GSM8K (p=0.038), +2.28 BBH (p=0.076). CLoRA's published numbers are faithful; the
question answered here is only whether LoRA+wd matches them at matched capacity in our harness.

## Verified findings (RQ1: does the magnitude relation hold; RQ2: method adjudication)

1. **Magnitude relation holds in both families**: r(log10 F_Delta, ret) = -0.865 (lrswm, n=120)
   and -0.929 (frm, n=144); on BBH alone -0.779 / -0.900. frm knee at log10 F_Delta ~ -0.50
   (F_Delta ~ 0.32, §18.2): LoRA+wd's op point (0.28) is the only best-adaptation cell below it;
   all non-wd op points sit above (0.44-2.85).
2. **wd moves the collapse point, on-curve** (frm grid): at wd=0, BBH decays 31.3 -> 17.0 -> 3.6
   across lr 1e-4 -> 5e-4 -> 7e-4; at wd=0.3, BBH >= 32.6 through 7e-4 with GSM8K 63.6-66.8. All
   8 frm quarantined runs are lr >= 7e-4 (wd0/0.3/0.5 at 1e-3 diverge, adapt ~0); lrswm has 0
   quarantined. lrswm safe-LR count (BBH within 2 pp of base): LoRA+wd 7/7 landed LRs, plain
   LoRA/MiLoRA/DoRA/CLoRA-k1024 5/7, SC-LoRA 2/7.
3. **Method offsets from the family curve are bounded and second-order, two exceptions flagged**
   (residual from the family ret-vs-logF line): frm PiSSA **-10.9 pp** below curve (n=1,
   collapse-driven; frozen dummy-OLS -11.4 +/- 2.1) and frm DoRA **+5.2 pp** above (frozen
   +4.6 +/- 1.4, one cell); all other methods within ~+/-2.7 pp.
4. **Geometry does not reorder the op points**: DoRA reaches spec_max ~700 (vs 23-38 for others)
   yet its BBH sits where its F_Delta predicts (above the line, +5 pp bounded); CLoRA k64->256
   moves adaptation and retention together via magnitude (F_Delta 1.13->1.01, KL 1.62->1.40,
   GSM8K 59.4->60.7, BBH 28.1->28.6) — the k knob acts through the magnitude channel here.
5. **CE/KL corroborates**: op-point KL spans ~20x (LoRA+wd 0.23 vs PiSSA 4.46), ordered with BBH.
   One disclosed exception to "F_Delta is the best single predictor": in **lrswm KL beats
   log F_Delta** (R2 0.856 vs 0.747, the only such family on the frozen pool); in frm F_Delta
   wins (0.863 vs 0.836). Quote per verification A3 wording.
6. **Seed variance is small on math**: median within-cell BBH SD 0.3-1.1 pp, GSM8K SD 0.4-1.0 pp
   across methods (unlike CS adaptation). Exception: frontier cells are seed-bimodal —
   frm_lorawd_wd0p2_lr7e4 seed 43 collapsed (BBH 4.4, F_Delta 1.22) while seeds 42/44 held ~31
   (cell BBH SD 15.2); lrswm SC-LoRA at 1e-3 BBH SD 7.6.
7. **SC-LoRA (as configured, nq_open calibration) shows a within-sweep adaptation edge on the
   100k sweep only**: lrswm GSM8K 58.5-59.1 at lr 5e-5/1e-4 (F_Delta 0.27-0.39) vs LoRA+wd's
   lrswm ceiling 50.7 — but n=2 seeds/cell, BBH already 1-2 pp lower at matched LR, and on the
   395k recipe grid SC-LoRA is mid-pack (60.5 vs 66.8). Echoes the qwswm pattern; attribute to
   the method-as-configured, not subspace geometry (no eval-matched calibration control here).

## Reviewer-facing caveats

- **PiSSA is recipe-rate-only and single-seed in frm** (1 run); its -10.9 pp offset is
  collapse-driven at lr3e-4 and should not be read as a swept operating point. LoRA-Null best
  cell is also n=1.
- **Retention metric**: math-family method comparisons use BBH alone; MMLU-Pro is excluded
  because its parser fails on MetaMath-tuned outputs. The pooled magnitude anchors use
  ret (retention_mean); both are reported above and agree in sign and ranking.
- lrswm's adaptation ceiling (~50.7 for LoRA+wd) reflects the 100k/older sweep; frm (395k,
  faithful recipe) supersedes it as the math headline (key_numbers §12/§18).
