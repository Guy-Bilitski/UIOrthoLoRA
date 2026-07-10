# Section 2 (Geometry analysis) — validation review

Reviewer: section-validator agent, 2026-07-10.
Data: `results/geo_drift/master_labeled.jsonl` (320 rows, 303 with outcomes; 217 CS runs with
retention+fdelta), `results/geo_drift/adapter_metrics.jsonl`, `results/geo_drift/permatrix/*.jsonl`,
scripts `geo_drift_phase1.py` / `geo_drift_phase2.py`; cross-checked against
`handoff/27_GEOMETRY_DRIFT_2026-07-09.md` and `paper/writing/data/key_numbers.md`.
All recomputation with `/home/guy/UIOrthoLoRA/.venv/bin/python`.

## A. Claim-by-claim verdict table

| # | Claim (artifact §2) | Verdict | Recomputed value |
|---|---|---|---|
| 1 | Fingerprint table, all 48 cells (8 methods × 6 columns) | **CONFIRMED** | Every cell reproduces exactly from master_labeled.jsonl (F²-weighted means; e.g. LoRA e_top 0.071, MiLoRA e_bot 0.115, LoRA-Null e_top 0.126, SC-LoRA ein_top 0.410 / srank 19.4 / lawres −5.66, CorDA ein_bot 0.494 / srank 13.0, DoRA +3.57, CLoRA +1.46, LoRA+wd +1.73) |
| 2 | Battery law used for residuals | **CONFIRMED** (with caveat #13) | Refit on all 217 CS runs: ret = 16.96 − 6.79·ln F_Δ (= −15.64/decade). Matches key_numbers exactly; note key_numbers' "logF" is **natural log** |
| 3 | Neutral baseline "~0.06 everywhere" (256/4096) | **CORRECTED (nuance)** | 256/4096 = 0.0625 exact for square matrices; but the table's numbers are F²-weighted over all 160 matrices, ~30–60 % of whose energy sits in MLP matrices with one 11008-dim side (neutral 256/11008 = 0.023). Effective pooled neutral ≈ **0.049 (e_top) / 0.052 (ein_top)**, not 0.06. Consequence: plain LoRA (e_top 0.071, ein_top 0.076) is a mild *principal* tilt (~1.4× neutral), not "at neutral". Signals ≥0.10 are unaffected |
| 4 | ΔW reconstruction "validated to <0.2 %" | **CORRECTED** | vs recorded dw_sv_max over 303 runs: median 0.005 %, p95 0.21 %, **max 0.51 %** (lrsw_clora_k1024_lr5e5_s42). Say "median <0.01 %, max 0.5 %" |
| 5 | 320 checkpoints, 160 matrices = 5 module types × 32 layers, one SVD per matrix | **CONFIRMED** | 320 rows; n_mat=160 in every record; TARGETS = (q,k,v,up,down)_proj in phase1; TOPK=BOTK=256 |
| 6 | Within-method size correlation "r −0.75 to −0.94" | **CONFIRMED** | CS: −0.746 (CorDA) … −0.944 (DoRA), all p<0.02. (Math runs are even stronger, −0.98…−0.99, so the quoted range is conservative) |
| 7 | Rank second-order: partial r ≈ −0.56; −0.5…−0.6 across rank measures | **CONFIRMED** | partial r(ret, ln r \| ln F) = −0.557 (p=4e-19); stable rank −0.553; eff rank −0.521; corr(ln stable_rank, ln r)=0.82 |
| 8 | "Alignment adds essentially nothing beyond size+method, ΔR² ≈ 0.0002" | **CORRECTED (overstated as blanket claim)** | Exact for amp_top (ΔR²=0.00025, p=0.526). But in the *same* specification e_top gives ΔR²=0.0067, **p=0.001**; and on the on-curve subset (SC-LoRA/CorDA dropped), e_bot / ein_bot give ΔR²=0.021–0.030 with **p<1e-4 even after controlling ln F + ln r + ln stable_rank + method**. All effects are small (≤3 % of variance vs 63–87 % for size) and none is a retention *benefit* (coefficients negative), so the ranking conclusion stands — but "ΔR²≈0.0002" quotes only the single most favorable metric. Restate as: "no alignment metric adds more than ~0.03 R² (most ≤0.007), versus 0.63 for size alone; where residual effects exist they are penalties, not protection" |
| 9 | Sign flip on outlier drop | **CONFIRMED (value differs slightly)** | partial r(ret, amp_top \| lnF): all CS −0.571 → **+0.307** dropping SC-LoRA (handoff says +0.25; same story) → −0.082 n.s. dropping SC-LoRA+CorDA (handoff: −0.08…−0.14 ✓) |
| 10 | SC-LoRA ein_top erosion 0.70→0.21, r=−0.96 | **CONFIRMED (and cleaner than advertised)** | The controlled lrsw series (fixed β=0.5, rank, seed 42, LR 2e-5→1e-3) gives ein_top 0.703→0.211 **monotonically**, r=−0.962. Attribution to LR is right: partial r(ein_top, log lr \| lnF) = −0.45 (p=0.002) while lnF given LR is n.s. (−0.22, p=0.16). Recommend stating it is a 7-point matched series |
| 11 | MiLoRA "only method with e_bot>e_top … retains slightly above the law" | **PART CONFIRMED / PART CORRECTED** | e_bot>e_top: only MiLoRA (0.115 vs 0.067) ✓. "Slightly above the law" (+1.6) is **fit-dependent**: canonical spline residual +1.04 (p=0.14, verdict "on the law" per key_numbers §5); on-curve-only battery refit gives −0.29. Soften to "on the law" |
| 12 | LoRA-Null highlighted cell e_top 0.126 = its "design signature" | **UNSUPPORTED as labeled** | LoRA-Null's advertised design is an **input-side activation null space**; its input-side metrics are flat (ein_top 0.080, ein_bot 0.054 ≈ plain LoRA). The elevated **output-principal** e_top 0.126 is real but is *not* the advertised geometry (the activation null space is not the base-W SVD bottom subspace — different reference basis, so the battery cannot verify this design). This also makes lead claim (1) "each method's advertised geometry is real" overbroad: verified for MiLoRA / SC-LoRA / CorDA (+PiSSA in math, n=1), **not verified for LoRA-Null**, and only weakly for CLoRA (e_top 0.060, lowest of all — plausibly its avoid-top constraint, but unhighlighted and unremarked) |
| 13 | Footnote †: −5.7 (battery refit) vs −4.15 (canonical n=49) | **CONFIRMED but internally inconsistent** | −5.66 reproduces. However the battery law is fit on **all 217 runs including SC-LoRA's own 44 (20 % of the fit) and CorDA's 20 uncalibrated runs** — while footnote * simultaneously declares CorDA's retention unusable for a residual. Sensitivity: fit excl. CorDA → SC-LoRA −5.11; fit on on-curve methods only → **−9.91**. So −5.7 is *conservative* (the outliers drag the line toward themselves), but the current choice is the least defensible of the three. Cheapest fix: refit excluding CorDA (residuals barely move; consistency restored) and note the on-curve-referenced value |
| 14 | Result (3) causal framing: placement "concentrates a large, mis-allocated update", "sets where on the magnitude axis a method starts" | **CONFIRMED — but currently asserted, not shown** | Two analyses in the existing data support it and appear nowhere in the artifact: (i) **matched-LR F_Δ**: at every one of the 7 swept LRs, SC-LoRA's update is 1.4–2.6× plain LoRA's (e.g. lr1e-4: 0.81 vs 0.31) and CorDA blows up (lr5e-4: 3.46; lr1e-3: 515.8 vs 1.42) while LoRA+wd is smallest — placement/regularization literally sets the magnitude-axis position at matched hyperparameters; (ii) adding ein_top to a size-only pooled model (no method dummies) lifts R² 0.630→0.753 and shrinks SC-LoRA's mean residual −5.66→−1.11 (coef −18.7/unit) — the input-principal concentration statistically *absorbs* most of the below-law offset. Caveat to keep: within SC-LoRA, ein_top does not predict retention beyond F (partial −0.15, n.s.; range restriction), so the arrow is cross-method, correlational |
| 15 | "Persists through three epochs" | CONFIRMED (weak sense) | All battery checkpoints are end-of-training (3 epochs) and still show the init signatures. No epoch-resolved trajectory exists in the data — don't imply one |
| 16 | amp_top formula & "why right candidates" description | **CONFIRMED** | Code matches ‖U_topᵀΔW V_top‖_F/‖ΔW‖_F exactly (phase2 `matrix_metrics`); e/ein energy fractions are exact projections via s-weighted singular directions |

## B. Does anything contradict the causal framing (task c)?

The framing survives, with one honest tension to disclose. Supporting: matched-LR magnitude
inflation (row 14i); ein_top absorbing the SC-LoRA offset (14ii); no alignment *benefit* anywhere;
within SC-LoRA, retention tracks F not ein_top. Tension: among on-curve methods, e_bot/ein_bot carry
a small residual **penalty** (ΔR² 0.02–0.03, p<1e-4, robust to rank+spread controls) — i.e. minor-
direction energy correlates with slightly *more* forgetting, the opposite of the minor-is-safe design
intuition. This does not rescue placement-as-protection (wrong sign for that) and is likely a
diffuseness/noise proxy, but §2's "adds essentially nothing measurable" should be weakened to the
max-over-battery statement of row 8, ideally with this sign twist in a footnote — it actually
*strengthens* the thesis (even the residual geometry effect points against the geometric-protection
hypothesis).

## C. Prioritized strengthening list

1. **Add the matched-LR F_Δ comparison (make the causal sentence load-bearing).** A 7-LR × 8-method
   strip (or small table: SC-LoRA/CorDA/LoRA/LoRA+wd rows) showing placement inflates ‖ΔW‖ at
   identical hyperparameters. Data fully present in master_labeled (`lrsw_*` runs). Cost: ~30 min
   analysis+figure. This is the single highest-value addition: it converts result (3) from assertion
   to measurement.
2. **Add the offset-absorption regression** (R² 0.630→0.753; SC-LoRA residual −5.66→−1.11 when
   ein_top joins size in the pooled model). One sentence + one number; ~20 min. Together with (1) it
   closes the "because of their placement" gap.
3. **Fix the residual-column inconsistency (row 13).** Refit the battery law excluding CorDA (SC-LoRA
   −5.1; MiLoRA +1.0; DoRA +3.5), keep CorDA n/a, and extend footnote † with the on-curve-referenced
   −9.9 as the upper bound. ~15 min, no new runs.
4. **Restate the stress test honestly (row 8):** "no alignment metric adds more than ~0.03 R²
   (amp_top 0.0002), vs 0.63 for size alone; residual effects are penalties, never protection."
   ~15 min.
5. **Fix LoRA-Null's highlight (row 12):** either un-highlight e_top and add a footnote that its
   activation-null-space design is not measurable in the base-SVD basis, or measure it properly
   (project ΔW·A_null onto the saved calibration null space — the init artifacts exist; ~2–3 h).
   Bonus insight already in the data: LoRA-Null's e_top erodes with LR exactly like SC-LoRA's ein_top
   (0.176@2e-5 → 0.079@1e-3) — "init-imposed alignment erodes with LR" generalizes across both
   init-constrained methods.
6. **Surface the per-module fingerprints** (permatrix data, no new runs; ~1 h for a small heatmap):
   CorDA puts ~97 % of its energy in MLP (52 % up + 45 % down) with ~4 % in attention; SC-LoRA's
   constraint is essentially total on attention inputs (q/k ein_top 0.86–0.89, all 32 layers ≥0.78
   at low LR); MiLoRA's minor-direction signature lives almost entirely in down_proj (e_bot 0.19 vs
   0.05–0.06 elsewhere); CLoRA/DoRA shift energy toward MLP (83 %/72 % vs LoRA's 57 %). This is the
   "allocation" half of the framing, currently invisible.
7. **Caption fixes:** neutral baseline ≈0.05 pooled (row 3); reconstruction "median <0.01 %, max
   0.5 %" (row 4); MiLoRA "on the law" (row 11); note erosion r is a 7-point matched series (row 10).
   ~20 min total.
8. (Optional, costly) **Epoch-resolved persistence:** re-run 2–3 representative cells saving per-epoch
   checkpoints to turn "persists through 3 epochs" into a trajectory plot. ~3 GPU-h; only worth it if
   a reviewer pushes on claim (1).
