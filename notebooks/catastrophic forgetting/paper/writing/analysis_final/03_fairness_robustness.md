# 03 — Fairness & Robustness Dossier (FINAL, post-fleet-kill freeze)

*Compiled 2026-07-17 from the frozen dataset (`results/*/summary.json`, 1,661 dirs / 1,500 evaluated;
quarantine `results/quarantine_diverged.txt`, 71 runs). Sources: `key_numbers.md` §17–§18 (single
source of truth), `analysis_final/analyze_{full,adversarial,ebatch}_output.txt`,
`analysis_final/{e5_e6_salvage,final_census}.txt`, `adversarial_review_2026-07-16.md`,
`assessment_2026-07-17.md`, `REBUTTAL_PREP.md`, `paper/baseline_fidelity_audit.md`,
`EXPERIMENTAL_FAIRNESS_AUDIT.md`. Numbers re-verified against disk where marked [disk-verified].*

Guardrails observed throughout: CLoRA's published numbers are faithful and are used as
corroborating external evidence, never framed as artifacts. The E4 finding is a **fairness win for
the study design** (calibration-set choice matters), not a criticism of SC-LoRA the method.

---

## 1. SC-LoRA resolution (E4) — the paper's highest-leverage fairness result

**The question (REBUTTAL_PREP O3, the self-declared "kill-shot"):** SC-LoRA was the *only*
statistically significant ANCOVA deviator (−4.15 pp below the pooled curve, p=0.006, key_numbers
§5/§14) — but SC-LoRA calibrates its init on **nq_open** (factoid QA) while retention is measured
on BBH/MMLU-Pro (academic reasoning). The fairness audit flagged this calib↔eval mismatch as the
one asymmetry "that plausibly *creates* the SC-LoRA off-curve effect" and as biased AGAINST the
data-aware inits. All off-curve language was embargoed pending B4/E4.

**The experiment:** `b4_sclora_r32_*` — the full 7-LR ladder re-run with **eval-matched
calibration**, multi-seed. Final status: **20 of 24 cells evaluated** (unevaluated: lr1e3 ×2
[s42,s43], lr3e4_s46, lr5e4_s46 — lost with the fleet; the ladder's healthy range is fully
covered). [disk-verified against `analyze_ebatch_output.txt` + `results/b4_sclora_*` dirs]

**The result:**

| Calibration | mean residual vs the lrsw pooled curve | n |
|---|---|---|
| **eval-matched (E4/b4)** | **+0.92 pp ABOVE the curve** | 20 |
| nq_open (original lrsw sclora series) | **−3.39 pp below** | 24 |

Per-point detail (residuals vs `ret = 19.91 − 9.47·log10 F_Δ`, n=201 lrsw fit): the eval-matched
ladder sits +0.6…+3.0 above the curve through the entire healthy mid-range (lr5e-5…3e-4, 14 cells,
every seed), drifting to −0.5…−1.4 only at the lr5e-4 edge. It is not one lucky point — the whole
ladder moved.

**One footnote:** `b4_sclora_r32_lr2e5_s42` shows adapt=13.2 vs its seed twin s43=77.3 on the
identical recipe — an undertrained-format seed fluke at the lowest LR. Retention is unaffected
(27.53 vs 27.30) so the cell stays in the retention/residual analysis but is **excluded from
adaptation comparisons**. [disk-verified]

**What it means.** The single significant deviation from the magnitude relation **dissolves when
the calibration set is matched to the evaluation distribution**. The residual was a property of
the *calibration-set choice*, not of SC-LoRA's geometry — exactly the outcome the fairness audit
predicted as the "good case." Consequences:

1. The ANCOVA story reaches its cleanest form: **every adapter we can assess is consistent with
   one magnitude curve once calibration is fair.** The method-offset table (§18.4: sclora −2.5 to
   −3.7 pp in the nq_open-calibrated Llama families) is now *explained*, and the negative offsets
   are labeled as the E4 calibration artifact.
2. This retires O3 entirely, retires the "influential point" half of O9, and upgrades the paper's
   thesis rather than weakening it.
3. Framing (guardrail): this is **not** a deficiency of SC-LoRA — with eval-matched calibration
   SC-LoRA sits *above* the curve at healthy LRs with adapt ≈ 80–81. The finding is that
   **data-aware initializations inherit the distribution of their calibration data**, a fairness
   requirement for any cross-method forgetting benchmark. (SC-LoRA's genuine adaptation strength
   on Qwen-math, seed-replicated at 76.3±1.0, is reported positively elsewhere.)

---

## 2. CorDA status — withheld from all quantitative claims, and why

**Root cause (baseline_fidelity_audit Finding 1, HIGH):** our CorDA port calibrated its KPA
covariance on **wikitext-2**, but the CorDA paper/repo default is **nqopen** (QA knowledge). CorDA
KPA's entire mechanism is to freeze the directions most responsive to the calibration data;
calibrating on generic LM text freezes the *wrong subspace*, so the port — although its
decomposition math is faithful (audit Finding 8) — is **not the paper's CorDA** and likely
understates CorDA's retention. Findings 2–3 (activation-normalization `x.abs().max()` vs
`torch.max(x).abs()`; damping-norm choice) are LOW/negligible but are disclosed. This is our own
port deviation; it says nothing about CorDA's published results.

**What exists in the final dataset [disk-verified]:**
- 64 corda/cordapp-named entries under `results/`, all unusable for the law: the 25 `lrsw_corda`
  CS dirs are the contaminated wikitext-calibration runs (8 of them quarantined with exploded
  magnitude up to 54,741 / F_Δ=516); `mtx_corda_*` and `corda_kpa_*` (10 dirs) are all
  quarantined explosions; `qwsw_corda` low-LR cells are wikitext-era too.
- The planned clean re-run never landed: **zero corda cells evaluated after 07-11**.
  `b4_cordapp_r32_lr{1e4,3e4}_s42` (the nq_open/eval-matched CorDA++ arm) exist with
  `forgetting.json` + `geo.json` **only — no summary.json / no benchmark eval**; the
  `frc_cordapp` grid points that trained mostly diverged (lr≥5e-4 quarantined).
- 22 of the 71 quarantined runs are corda/cordapp — the quarantine's largest method block, all
  from a family already excluded from every claim.

**The honest coverage statement (final):** the study design has 8 adapters; **7 of 8 are
assessed** (LoRA, LoRA-Null, LoRA+wd, DoRA, MiLoRA, CLoRA, SC-LoRA). Pre-E4 wording: "six lie on
a single curve; SC-LoRA is the one provisional below-curve deviator." Post-E4 wording: **"all
seven assessed adapters are consistent with a single magnitude curve once initialization
calibration is eval-matched (E4); CorDA is withheld because our port mis-calibrated it on
wikitext-2 (a disclosed deviation from the reference implementation) and the clean re-run did not
complete."** CorDA's count is *not-yet-assessed*, never "off-curve." Do not cite any CorDA number
(incl. the old 77.9/19.9 or "−3.0 pp off-curve").

---

## 3. Quarantine effect — the 71 exclusions do not drive the relation

**Composition [disk-verified from `results/quarantine_diverged.txt`]:**

- **By reason:** every quarantined run is an unusable measurement — exploded magnitude
  (‖ΔW‖ 1,012…117,927), NaN magnitude/CE, and/or retention collapsed to ≈0. These are diverged
  trainings, not inconvenient data points.
- **By LR:** of the 60 LR-tagged runs, **54 sit at lr ≥ 5e-4** — lr1e-3: 37, lr7e-4: 8, lr2e-3: 6,
  lr5e-3: 3, lr5e-4: 5 (the lone lr3e-4 is a cordapp explosion). The 2e-3/5e-3 probes are the 11
  extreme-LR PERMANENT-FAIL cells already ledgered as "the expected divergent boundary of the LR
  grid, not missing data" (`writing_readiness_2026-07-16.md` §D). The 11 untagged runs are
  corda/mtx/scl2 legacy dirs.
- **By family:** frm 25 (almost entirely the wd-ladder lr1e-3/7e-4 boundary), lrsw 18, frc 11,
  mtx 8, qwswm 4, qwsw 2, corda_kpa 2, scl2 1. **By method:** corda/cordapp 22, lorawd 23 (the
  frm wd×LR ladder's divergent corner), milora 9, dora 7, sclora 5, lora_null 3, clora 1, lora 1.

**Why the exclusions cannot drive the relation:**

1. The quarantined runs have no finite (F_Δ, retention) coordinates to contribute — including them
   is not an analysis option, and where diverged-but-finite high-LR evals DO exist they are
   *retained* and anchor the high-F_Δ/low-retention end (readiness §D).
2. The headline numbers (§18.1: pooled r −0.830…−0.929 per family, n=1035) are computed
   quarantine-filtered; the pre-filter §13 numbers (n=1001–1018) are essentially identical
   (e.g. lrsw −0.886 both ways) — the filter moves nothing at the third decimal that matters.
3. The **A7 format-collapse control** (the aggressive version: additionally dropping *retained*
   cells with adapt<25 or any zero retention task) moves r by **≤0.03 in five of six families**;
   the exception is **qwswm −0.830 → −0.695** (9 cells dropped) — Qwen-math's pooled r is partly
   carried by collapsed cells, and the paper quotes the clean −0.70 alongside the pooled −0.83
   (pre-registered in §17.7/§18.6).
4. 22/71 quarantined runs are CorDA — a family excluded from every claim regardless.

---

## 4. Reviewer-objection scorecard — O1–O13 + the three-reviewer adversarial synthesis

| # | Objection (short) | Final status | The final numbers |
|---|---|---|---|
| **O1** | "8 adapters claimed, 6 shown; CorDA hidden" | **Partially resolved (honest-coverage)** | LoRA-Null half fully retired (split convention, §14; pooled law unchanged at the same 49→180 points). CorDA half stays a declared withholding: port mis-calibration disclosed (audit F1), clean re-run 0-evaluated. Coverage = 7 of 8 assessed; post-E4, all 7 consistent with the curve. |
| **O2** | "Single seed everywhere" | **Resolved** | 3-seed spine landed: 287 cells with n≥3 seeds (§13.2/§18.1); within-cell SD(ret) 0.33–1.00 pp Llama, 2.1–2.7 pp Qwen — 10–50× smaller than the 15–30 pp retention range. Math headline 66.79±0.79; all 7 CS operating points 3-seed. |
| **O3** | "Your one significant result is a calibration confound" | **Resolved (E4)** | Eval-matched SC-LoRA +0.92 pp above curve (n=20) vs nq_open −3.39 (n=24). §1 above. |
| **O4** | "LoRA+wd won by 2× rank + exclusive knob" | **Resolved** | Param-matched r16 control: `frc_lorawdr16_wd0p3_lr5e4` CS 81.04 / ret 26.27 / F_Δ 0.334 — reproduces the r32 point at half the params (§16). Rank ladder r8→r32: more capacity → more F_Δ → less retention, no benefit at controlled F_Δ. wd knob generalizes to MiLoRA (E6, +1.75/+2.36 pp above curve, adapt 80.2) — with the DoRA boundary disclosed (§5). |
| **O5** | "Near-circular law; in-sample R² bump" | **Resolved, upgraded to interventional** | E1: 15/15 pure rescales move adapters ALONG the curve (mean residual +1.29±2.07, within-set r=−0.732); 9/9 random-direction controls at matched F_Δ sit −3.05 pp below trained. LOMO RMSE ≤2.5 pp (§15). Within-cell micro-test: seed-level F_Δ fluctuations at fixed recipe predict retention fluctuations, r=−0.713 (n=954/290 cells, t=−31.3). |
| **O6** | "Incremental over CLoRA / scooped by Lee et al." | **Strengthened; residual is framing** | Breadth is now real: 2 models × 2 domains × 7 adapters × ≥3 seeds, n=1035 + full-FT anchor (E2) + off-recipe bridging (E7: brl r=−0.878, brq r=−0.995). CLoRA remains the honest external anchor: its F_Δ Table-4 diagnostic *is* our axis and its published numbers corroborate the relation from a competitor's own data (fairness audit Q2). CorDA absence and the lost 284B arm keep this partially open on breadth-of-methods/scale. |
| **O7** | "Pareto margin < noise floor" | **Resolved (bounded, verb unchanged)** | Error bars exist: lrsw within-cell SD 0.94 pp; frm LoRA+wd 66.79±0.79 vs SC-LoRA-math 60.9±0.4; retention SD ≤0.53 across the six 3-seed frm configs. Claim stays "matches or edges / reaches the frontier," now with CIs. |
| **O8** | "Qwen-math anti-replicates (r=+0.67)" | **Resolved** | Final qwswm: **r=−0.830, n=164, 3 seeds** (clean-subset −0.695 quoted alongside). The +0.67 was a broken-parser artifact on 10 early low-LR cells (§11→§16). E3 densification: bottom-half r = −0.03/−0.04 — flat below the knee, not positive. Anti-replication is dead; scope wording = "flat-then-falling with a knee." |
| **O9** | "ANCOVA underpowered, one influential point" | **Resolved** | n=1035; method offsets at matched F_Δ bounded ±1.2–4.6 pp (§18.4), Qwen offsets all n.s.; the one influential point (SC-LoRA) is explained by E4. Direction = real, bounded, second-order (partial r=+0.117, t=3.7). |
| **O10** | "Sweet-spot band post-hoc" | **Answered by framing (unchanged)** | Band stays labeled descriptive/annotation, `[EXTERNAL: design choice, not fitted]` (§6/§12). No new data needed. |
| **O11** | "Biderman et al.: wd doesn't work" | **Resolved at analysis level** | Reconciled in-text: their wd = 5e-5…1e-4 on full-FT at one LR (barely moves ‖ΔW‖ → law predicts their null); ours = 0.3 on adapters, verified to bound F_Δ (0.394). Residual: the direct full-FT+large-wd sweep remains out of scope, said gracefully. E2's full-FT anchor (monotone, family-specific level) supports the "same law, different point" reading. |
| **O12** | "Just compare published numbers" | **Resolved (structural)** | Only CLoRA reports our retention axis (BBH+MMLU-Pro); the others report QA-EM / WikiText loss / nothing (fairness audit Q2 table) — a common harness is a *precondition*. Scale validated: BoolQ 69.97 vs canonical 69.8; CS-8 79.1 between DoRA's 77.6 and CLoRA's 79.9. |
| **O13** | "DoRA's F_Δ omits the magnitude vector" | **Still deferred (disclosed lower bound)** | Checkpoints not retained; recompute needs a re-run. Sign argument contains it: DoRA sits ON/ABOVE the curve (final offsets +1.7…+4.6 pp, §18.4), and correcting its x rightward only increases its positive residual. Camera-ready TODO stands. |

**Adversarial-synthesis verdicts (2026-07-16 three-reviewer attack) — final disposition:**

- *Monotone magnitude→retention*: SURVIVED, then **upgraded to interventional** (E1) and
  off-recipe (E7). Per-seed, per-cell, within-cell, fixed-LR replication all held at final n.
- *Log-LINEAR "law"*: **conceded and repaired** — headline retitled "magnitude relation
  (flat-then-falling with a knee)" per the pre-registered rule; 2-segment beats linear in every
  family, normalized slopes do not converge (−0.33…−0.70) (§18.2).
- *"R² doubles vs LR" strawman*: **rewritten** (§2/§18.5) — the claim now rests on fixed-LR strata
  (r ≤ −0.7 at every LR ≥1e-4 in every family), partials (r(F_Δ|LR) −0.58…−0.91, |t|≥7.6 vs
  r(LR|F_Δ) ≤ |0.29|), and the decoupling grids (frc/frm: R² 0.86 vs LR-dummies 0.39/0.37).
- *"Magnitude, not direction"*: **softened with exact numbers** — direction is a real, bounded
  1–4 pp second-order effect (partial +0.117; E1 random-direction −3.05 pp; offsets ±1.2–4.6 pp);
  magnitude is first-order (R² 0.69–0.86). Sharper slogan adopted (§17.4): the retention cost of a
  unit of magnitude is near-universal; methods differ in the adaptation that unit buys
  (adapt method-offset spreads 4.8–16.0 pp vs retention spreads 3.4–7.7 pp).
- *CE "independent corroboration"*: **reframed within-family** — evidential link r(CE, ret)
  −0.631…−0.923; the mechanical r(F_Δ, CE) +0.81…+0.92 no longer sold as independent. Coverage
  hole disclosed: 136 Qwen runs lack CE, backfill unfillable without GPUs.
- *Replay demand (E5)*: **partially answered** — trained 4/4, benchmark evals 0/4 (lost). CE
  salvage: replay-5% CE lower than matched plain-LoRA twins in **4/4 cells**
  (lr3e4: 2.248/2.204 vs 2.307/2.254; lr5e4: 2.465/2.462 vs 2.551/2.526; ΔCE −0.05…−0.09, KL
  likewise lower). State as: replay gives a small consistent CE-forgetting reduction; the
  benchmark-retention comparison died with the fleet.
- *E8 284B LR ladder / DSV4 generalization*: **LOST** — 0/21 summaries synced; trains + geometry
  completed on the DeepSeek nodes but capped evals never synced before the kill. Ledger as
  designed-but-lost (spec preserved: `handoff/DEEPSEEK_GEN_EXPERIMENT.md`); the generalization
  section has NO data in this repo. E7's two 7B bridging arms are the surviving
  beyond-the-recipe evidence.

---

## 5. Anomaly / footnote ledger (everything a hostile reader could find, pre-disclosed)

1. **E1 measured-F_Δ offset.** Rescaled adapters measure F_Δ at ×1.06–1.09 of nominal
   (SC-LoRA ×1.3–1.6, e.g. nominal 0.15 → measured 0.240) — an environment/measurement-context
   offset. All analysis uses the **measured** F_Δ, so the axis convention is unaffected; disclose
   in the E1 methods note.
2. **Full-FT F_Δ undercount (E2).** The probe measures the adapter target modules;
   full FT also moves o_proj/embeddings/norms, so fft F_Δ under-counts dense mass — part of E2's
   −4.1…−8.6 pp below-family-curve gap may be unmeasured magnitude. Disclosed; E2 is quoted for
   monotonicity + "universal in form, family-specific in level," never for its absolute x.
3. **b4_sclora lr2e-5 s42 fluke.** adapt 13.2 vs seed-twin 77.3; retention unaffected
   (27.5/27.3). Undertrained-format seed fluke at the lowest LR; excluded from adaptation
   comparisons only. (§1.)
4. **MiLoRA/DoRA seed-accuracy fragility.** At their best-adapt LRs the 3-seed adaptation spread
   is large — MiLoRA lr3e-4: **57.69±22.67** (79.9/58.7/34.5); DoRA lr2e-4: **74.29±8.65**
   (78.3/80.2/64.4) — while retention is rock-stable (24.20±0.48 / 25.20±0.33). Retention claims
   are seed-robust; per-method *adaptation* rankings at these cells are not. Reinforces the
   "frame the relation, not a ranking" rule.
5. **r16 lr3e-4 collapse basin — now seed-scoped [disk-verified, updates §16 which had the
   replicate "queued"].** `frc_lorawdr16_wd0p3_lr3e4_c256_s42`: CS 13.53 (below-chance answer
   format) with INTACT retention 26.84 and healthy training; re-eval reproduces exactly (13.54).
   The queued seed replicates LANDED: s43 71.95 / s44 61.33 / s45 80.67, retention 26.9–27.3,
   F_Δ 0.306–0.334 throughout. So the basin is **deterministic within seed 42 but
   seed-specific**, and not a magnitude effect. Report the lr5e-4 cell; footnote the basin (note
   s44's 61.3 shows the cell is genuinely high-variance for adaptation).
6. **11 extreme-LR NaN cells** (lr 2e-3/5e-3 probes): PERMANENT-FAIL, diverge to NaN, no valid
   eval exists — the expected divergent boundary of the LR grid, not missing data. All in
   quarantine (§3).
7. **DoRA+wd degeneracy (E6, from CE salvage).** Naive AdamW wd0.3 on DoRA — which decays its
   magnitude vector too — breaks training: CE 20.83/10.37 vs DoRA twins 2.13/2.57, spec_max up to
   1183 (benchmark evals lost, but the CE/geometry verdict is unambiguous). Report as a boundary:
   **wd is not a universally free knob** — it transfers to MiLoRA (+1.75/+2.36 pp above curve),
   breaks DoRA as-implemented. This honesty strengthens, not weakens, the E6 claim.
8. **Qwen adaptation tax.** Qwen adapters plateau ~37–39 retention even below the knee vs its
   44.35 no-FT ceiling (≈6 pp fine-tuning offset before any magnitude effect; ≈0 on Llama) —
   model-dependent intercept, one disclosure paragraph next to the knee.
9. **Trained-not-evaluated tail** (fleet kill): qwsw 27 / qwswm 22 cells, E4 ×4, E5 ×4 benchmark,
   E6-DoRA ×2 benchmark, brq lr1e3, DSV4 21, base-ceiling dirs 18/22 — full ledger in §18.7;
   none of these gaps is silent in the paper.

---

## 6. Metrology — what the magnitude axis is (and is not)

- **F_Δ is CLoRA's Table-4 diagnostic (their Eq 3)**, not the Frobenius norm:
  mean over tokens of ‖ΔW·x‖/‖x‖ on real eval inputs, averaged over updated matrices. Our
  adoption of a competitor's own published diagnostic as the study's axis is a fairness feature —
  it makes our axis **directly comparable to CLoRA Table 4** (LoRA 0.79, LoRA-L2 0.29, k2048
  0.14), whose numbers we treat as faithful corroboration.
- **Measured per-run, never nominal**: every plotted x is the run's own measured F_Δ (this is what
  absorbs the E1 rescale offset, the scaling-asymmetry α=2r vs α=r across families, and rank
  differences — they move points along the axis, not off it).
- **Axis label (final, §17.9): "effective update magnitude on the adaptation distribution."**
  The data-dependence is load-bearing: R²(ret ~ log F_Δ) = 0.72 vs log ‖ΔW‖_F = 0.56 vs
  dw_sv_max = 0.58 — the gap IS the adaptation-distribution weighting. Alignment
  (F_Δ/‖ΔW‖_F) is not method-invariant (dora 2.71e-3 … clora 1.54e-3) but within-method spread
  is as large as the between-method gap, so F_Δ is not a method-relabeling in disguise.
- **Known measurement caveats, both disclosed with sign arguments**: DoRA's x is a lower bound
  (O13; correction only raises its positive residual); full-FT's x under-counts dense mass (E2;
  quoted for form, not level).

---

## 7. Figure-ready statements

1. **(E4 / hero-ANCOVA caption)** "With eval-matched calibration, SC-LoRA's full learning-rate
   ladder sits +0.92 pp above the pooled magnitude curve (20 runs, 5 seeds); with its default
   nq_open calibration the same recipe sits −3.39 pp below (24 runs). The study's only significant
   ANCOVA deviation was a property of the calibration-set choice, not of the method's geometry."
2. **(Coverage)** "Of the eight adapters in the study design we assess seven; all seven are
   consistent with a single magnitude curve once initialization calibration is eval-matched.
   CorDA is withheld: our port calibrated it on wikitext-2 rather than the reference nq_open —
   a disclosed deviation — and the corrected re-run did not complete."
3. **(Quarantine)** "71 of 1,661 runs are quarantined as diverged (non-finite or collapsed
   measurements); 54 of the 60 learning-rate-tagged exclusions sit at lr ≥ 5e-4 and 22 are CorDA
   runs already excluded by design. Additionally dropping every retained cell with degenerate
   adaptation or a zeroed benchmark moves the pooled correlation by at most 0.03 in five of six
   families; in Qwen-math it moves −0.830 → −0.695, and we quote both."
4. **(Interventional)** "Rescaling a trained adapter — changing nothing but its magnitude — moves
   it along the retention curve (15 rescales, mean residual +1.29 ± 2.07 pp); a random direction
   at the same magnitude pays a −3.05 pp penalty. Magnitude is first-order; direction is a real,
   bounded second-order effect."
5. **(Seeds)** "287 recipe cells carry ≥3 seeds; within-cell retention SD is 0.3–1.0 pp on Llama
   and 2.1–2.7 pp on Qwen — one to two orders of magnitude below the 15–30 pp range the relation
   spans."
6. **(Qwen-math)** "Qwen-math replicates the relation at r = −0.830 (n=164, 3 seeds; −0.70 after
   excluding collapsed cells — we quote both). The earlier positive correlation was a parser
   artifact on ten early low-learning-rate cells and does not reproduce."
7. **(wd boundary, E6)** "Weight decay transfers to MiLoRA (+1.8…+2.4 pp above the curve at
   adapt 80.2) but breaks DoRA as-implemented (it decays DoRA's magnitude vector; CE 10–21 vs
   2.1–2.6 for its unregularized twins): a magnitude knob, not a universally free one."
8. **(Replay, E5)** "Replay (5%) lowers CE-forgetting in 4 of 4 cells (ΔCE −0.05…−0.09; KL
   likewise); its benchmark-retention comparison was lost with the compute fleet and is reported
   as an open item."
9. **(Axis)** "The x-axis is F_Δ — CLoRA's own published forgetting diagnostic (Eq 3): the
   effective update magnitude on the adaptation distribution, measured per run. It outpredicts
   the Frobenius norm (R² 0.72 vs 0.56) precisely because it weights the update by where the
   adaptation data lives."
10. **(Lost arms, limitations)** "The 284-B generalization arm (21 cells) trained to completion
    but its evaluations never synced before the fleet was decommissioned; it is reported as
    designed-but-lost. The two 7-B bridging arms (MedMCQA, attention-only) that did land
    reproduce the relation at r = −0.878 and −0.995."
