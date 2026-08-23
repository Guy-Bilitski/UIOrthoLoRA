# Artifact gap map — "Who Affects What" data report vs. paper exhibits

Compared: artifact HTML `artifact-5c46636f-1784276194-17a3.html` ("Adapter Dynamics —
Understanding the Data", computed 2026-07-17, 1,004 usable runs; geometry 986; CE/KL 857)
against `paper/.overleaf-git/main.tex` (branch ortho_new) and its `tables/` + `figures/`.
Report only; nothing in the paper was touched.

Artifact provenance (as stated in its footer): merged run-level data from
`results/*/summary.json` + `geo.json` + `forgetting.json` + `results/geo_drift/adapter_metrics_merged.jsonl`;
raw analysis outputs in `paper/writing/analysis_final/dyn1_structure.txt`, `dyn2_mediation.txt`,
`dyn3_exchange.txt`, `dyn4_geometry.txt`; synthesis `analysis_final/05_adapter_dynamics.md`;
frozen headline stats `paper/writing/data/key_numbers.md` §18.

---

## A. INVENTORY — artifact sections and their exhibits

**Header block** — run census callouts: 1,004 usable of 1,500 evaluated, 71 quarantined,
CorDA fingerprint-only; 986 with geometry, 857 with CE/KL drift; fixed per-method color legend.

**S1 — "The raw phenomenon — one falling curve per family"**
- Fig S1 (6 panels): retention vs log F_delta scatter, methods as colors, binned-median trend,
  dashed base ceiling; one panel per run family (Llama-2 CS sweep, Llama-2 math sweep,
  Qwen2.5 CS, Qwen2.5 math, Llama CS grid, Llama math grid). Point: colors interleave; no
  per-method curves. Source: master run table (observatory `master_runs.csv` / `m2_master.csv`).
- Table S1: per family n, Pearson r, rank-r, knee (F_delta), two-segment slopes below/above knee
  (pp per decade); Qwen-math quoted with its collapse-clean twin (-0.830 / -0.695) per the
  pre-registered rule; pooled row n=1035, r=-0.847, rank-r=-0.923. Source: `dyn1_structure.txt`,
  frozen in `key_numbers.md` §18.1.

**S2 — "The dose-benefit split: adaptation saturates, the cost doesn't"**
- Fig S2a (2 panels): mean adaptation per F_delta bin (saturates by F_delta ~0.3-0.4) vs mean
  retention per the same bins (keeps falling another decade); healthy Llama-CS runs
  (adapt >= 70, n=268).
- Callout (good) + Fig S2b (1 panel): the intervention exhibit. Grey = observational sweep;
  green = 15/15 trained adapters rescaled to a target F_delta land on the curve
  (mean residual +1.29 +/- 2.07 pp); red = random directions at matched F_delta lose only
  -3.05 pp retention but adaptation collapses (0.5-7.0 vs 13-80). Also: post-hoc shrinking beats
  training natively at the small size by +1.09 +/- 1.80 pp retention (15 matched pairs).
  Sources: `analysis_final/analyze_ebatch_output.txt`, `key_numbers.md` E1 block (~line 579),
  `01_law_final.md` (line 27, 133, 257 for the +1.09 pairs).

**S3 — "The drift axis: what forgetting is made of"** (n=857 KL subset)
- Fig S3 (2 panels): KL drift (WikiText-103) vs retention for Llama-2 CS sweep and Qwen2.5 math;
  claim: drift is the strongest single predictor of retention in 5 of 6 families.
- Table S3 (mediation): per family, path a = r(KL, logF); "via drift" = partial
  r(ret, KL | logF); "direct" = partial r(ret, logF | KL); R2(size)/R2(drift)/R2(both).
  Direct path collapses in the math and Qwen arms (-0.14, -0.08, +0.05) but survives in
  Llama-2 CS sweep (-0.63) and Llama CS grid (-0.68). Source: `dyn1_structure.txt` §2.
- Callout (warn), two-channel decomposition: the surviving drift-independent magnitude path
  goes almost entirely to MMLU-Pro (partial -0.67...-0.71 in Llama families) while BBH's direct
  path is ~0 in math/Qwen (+0.07...+0.12). Channel A = capability drift (KL-visible, hits
  BBH-style scoring); Channel B = format/instruction damage (KL-invisible, magnitude-direct,
  hits generative parsing; same channel as the accuracy-collapse seed basins). Micro-scale
  caveat: within a fixed recipe, seed fluctuations follow F_delta (partial -0.51) not KL (-0.10);
  KL from 40 text blocks is noisy at small deltas. Source: `dyn2_mediation.txt` §7-8,
  `05_adapter_dynamics.md` §2.

**S4 — "Geometry I: the designs are real — and training pressure dissolves them"**
- Table S4a (full fingerprint): per method x model medians of e_top / e_bot / ein_top / ein_bot /
  amp_top / stable rank (Llama and Qwen columns), neutral ~0.05-0.06, design-signature cells
  highlighted; PiSSA Llama-only (n=5), CorDA fingerprint-only rows. Source: `dyn4_geometry.txt`
  G1; observatory `m3_master.csv` via `30_m3_geometry.py`.
- Fig S4 (2 panels): MiLoRA ein_bot and SC-LoRA ein_top vs F_delta — the design signature decays
  toward neutral as the update grows.
- Table S4b (signature erosion): r(signature, log F_delta) within each method's sweep and
  low-F -> high-F medians: MiLoRA ein_bot 0.186 -> 0.059 (Llama CS, r=-0.83) and
  0.309 -> 0.063 (Qwen CS, r=-0.98); SC-LoRA ein_top halves (0.353 -> 0.181 / 0.260 -> 0.126);
  CLoRA stable rank spreads 2.8 -> 11.5 / 2.1 -> 17.6. Source: `dyn4_geometry.txt` (erosion
  block, e.g. line 27).
- Callout: mechanistic reading — protective placements are initialization-time constructs that
  gradient descent overwrites in proportion to update size; designed geometry is a low-magnitude
  phenomenon.

**S5 — "Geometry II: what placement does control — the exchange rates"**
- Table S5a (principal-touch tax): retention ~ logF + e_top per family; beta(e_top) = -1.4 to
  -4.2 pp per +0.1 energy; partial r(ret, e_top | F) up to -0.41; partial r(KL, e_top | F) up to
  +0.27 (same energy drifts more per unit). Qwen tax small (arms below the knee).
- Table S5b (slope steepening): split each family at median e_top; high-e_top half pays a
  1.2-1.9x steeper above-knee slope (Qwen math flat exception, 0.9x).
- Table S5c (adaptation exchange rate): per-method residuals against the family
  adaptation-vs-magnitude fit; SC-LoRA +8.4 (Llama math) / +5.4 (Qwen math); LoRA+wd +6.1
  (Llama CS sweep); explicitly flagged "qualitative reads".
- Table S5d (drift exchange rate): per-method residuals against the KL-vs-magnitude fit;
  PiSSA +0.61 (Llama CS grid), CLoRA/SC-LoRA slightly positive in sweeps, LoRA and DoRA negative.
- Sources: `dyn3_exchange.txt` (§11 has the adaptation residuals verbatim),
  `m3_residual_corr.csv`, `acl_analysis/insights/exchange_rate.csv`.

**S6 — "What we now understand (and what's still open)"** — narrative synthesis only, no
exhibits: dose chain (knobs -> magnitude -> drift -> forgetting), two channels, direction-adapts/
magnitude-forgets, saturation at 0.3-0.4, erosion, exchange rates, concentration protective at
matched dose (partial r(ret, spec_max | F) +0.40...+0.60 Llama CS), SC-LoRA coherence, and three
open questions (direct channel-B scoring, per-layer geometry, DoRA's negative drift residual).

---

## B. ALREADY IN PAPER — artifact exhibits with a paper equivalent

| Artifact exhibit | Paper equivalent | Coverage |
|---|---|---|
| S1 fig, Llama-CS panel | `fig:hero` (448 Llama-CS runs, knee ~0.4) | Full for that family; other five families have no figure |
| S1 table, pooled r and per-family range | RQ2 body prose (-0.847 pooled; -0.830...-0.929 per family) | Numbers in prose |
| S1 table, knees / two-segment slopes / F-tests | app:exhibits "The magnitude relation: functional form and constants" | Prose only, no table |
| S1 one-curve-per-knob idea | `fig:dose` (wd, CLoRA k, rank; delta-R2 <= 0.006) | Full |
| S2b rescale + random-direction numbers | RQ2 "not an artifact of the recipe" + app:exhibits "The rescaling intervention" (+1.29+/-2.07, -1.76+/-1.32, 3.05-pp gap) | Numbers in prose; no figure; shrinking result absent |
| S2a adaptation-saturation | app:exhibits "The adaptation side of the same relation" (concave, optimum band at the knee) | Prose only |
| S3 drift predicts retention; adds +0.005 beyond magnitude | RQ3 body; app:exhibits "The weaker accounting, disclosed"; `tab:league`; `fig:ceproxy` | Full for the monitor/accounting framing; mediation table absent |
| S3 micro-scale seed caveat (F beats KL within cell) | Partially: RQ2 within-cell check (r=-0.713) and app:exhibits "Predicting seed instability" | Related but the F-vs-KL seed-level contrast is not stated |
| S4a fingerprint | `tab:fingerprint` (a: stable rank per method; b: partial-r leverage) + RQ2 prose naming each method's subspace | Stable rank only; energy-fraction columns qualitative in prose |
| S5 residual geometry leverage | app:exhibits "The residual geometry effect" (partial r stable-rank -0.32...-0.67 Llama, ~0 Qwen); `tab:fingerprint`(b) | Stable-rank version present; e_top version absent |
| S5 SC-LoRA steeper-slope exception | app:exhibits "Knob effects on the retention-magnitude slope, and the SC-LoRA exception" | Method-level version present; e_top-split version absent |
| S3/S6 MMLU-Pro takes the method differences | app:exhibits "At matched update magnitude, method differences concentrate in MMLU-Pro" (31/37, Wilcoxon) | Adjacent claim (method-level, not channel decomposition) |
| S6 concentration protective (spec_max partial) | Not in main.tex; noted in `key_numbers.md` §19.2 with a seed-variability caveat | Absent from paper by prior decision; leave out |

Not in the artifact at all (so no gap either way): head-to-head battery, TOST/MDE, fragility
order (RQ4), Pareto/LR-artifact/LR-band figures, per-adapter LR sweeps, CLoRA cross-check,
284B arm.

---

## C. GAPS RANKED — worth porting to the appendix

**1. Signature-erosion exhibit (S4 fig panels + Table S4b).**
- Shows: each retention-aware design's measured placement (MiLoRA ein_bot, SC-LoRA ein_top,
  CLoRA stable-rank concentration) decaying toward neutral as F_delta grows; quantified as
  within-method correlations and low-F/high-F medians.
- Why it matters: it is the mechanistic companion to the paper's central RQ2 claim. Today the
  paper says geometry fingerprints methods but adds ~nothing to retention prediction; erosion
  states the observation behind it — the designed placement is present at low magnitude and is
  gone in exactly the regime where forgetting happens. It also explains, in one exhibit, why
  SVD-init methods look fine at low LR and ordinary at high LR (ties into fig:lrartifact).
- Regenerate from: `analysis_final/dyn4_geometry.txt` (erosion block), observatory
  `30_m3_geometry.py` + `m3_master.csv` (+ `m3_fig_*` siblings show the plotting stack exists);
  run pool `acl_analysis/insights/pool.csv`.
- Risk: the erosion correlations live in dyn4 raw output, not in the frozen `key_numbers.md`
  §18 — they would need a verification pass before quoting; per-method high-F cell counts are
  modest (MiLoRA Llama n=73, SC-LoRA smaller); phrase observationally ("the signature decays as
  the update grows"), not as a verdict on the method class.

**2. Mediation table (S3) with the MMLU-Pro/BBH direct-path split.**
- Shows: per family, how much of the magnitude-retention link is carried by KL drift; the
  direct path collapses in the math/Qwen arms and the surviving direct path concentrates in
  MMLU-Pro while BBH's is ~0.
- Why it matters: strengthens RQ3 twice over. It upgrades "KL drift tracks retention" to
  "drift carries the magnitude effect in four of six families", and it gives a data-level
  account of the monitor's disclosed blind spot (format collapse): the drift-invisible part of
  forgetting is precisely the part that lands on the generative-parsed benchmark. Dovetails
  with the existing "method differences concentrate in MMLU-Pro" paragraph.
- Regenerate from: `dyn1_structure.txt` §2 and `dyn2_mediation.txt` §7-8 (numbers verbatim);
  observatory `40_m4_cedrift.py` + `m4_master.csv` (n=857 KL subset).
- Risk: the "two channels" naming is interpretive — `05_adapter_dynamics.md` itself lists direct
  channel-B scoring as an open question. Port the partial-correlation table as observation and
  keep the channel story to one cautious sentence; include the seed-level caveat (F_delta partial
  -0.51 vs KL -0.10 within cell) or the table will overstate KL at micro scale. Numbers not in
  frozen §18; verify first.

**3. Intervention figure (S2b) plus the post-hoc shrinking pairs.**
- Shows: grey observational curve; 15/15 rescaled adapters landing on it; random directions at
  matched magnitude losing only ~3 pp retention while adaptation collapses; and rescale-vs-
  retrain twins (+1.09 +/- 1.80 pp, n=15) — direction carries the adaptation, dose carries the
  damage.
- Why it matters: this is the paper's strongest causal check and it currently exists only as
  prose (RQ2 + one appendix paragraph). A single panel makes it legible, and the shrinking
  result — nowhere in the paper today — is the natural constructive coda to the magnitude story.
- Regenerate from: `analysis_final/analyze_ebatch_output.txt`, `key_numbers.md` E1 freeze block,
  `01_law_final.md` (rescale > retrain rows); observatory `master_runs.csv` for the background
  sweep.
- Risk: one setting only (Llama-2 commonsense; the Limitations section already owns this — the
  figure caption must repeat it). Keep the shrinking framed as an observed comparison, not a
  recommendation; note the asymmetry caveat in the blueprint (upscaling -3.86 pp) if the
  exhibit implies rescaling is free in both directions.

**4. Principal-touch tax + slope-split tables (S5a, S5b).**
- Shows: at matched dose, energy in the base weights' top-256 output directions costs
  -1.4...-4.2 pp retention per +0.1 energy and adds drift per unit (partial r up to +0.27); and
  high-e_top updates pay a 1.2-1.9x steeper above-knee slope (Qwen math flat exception).
- Why it matters: this is the measured, continuous version of PiSSA's mechanism and the honest
  counterweight to the paper's null — it says exactly what geometry *does* control (the exchange
  rate and the slope, not the curve's existence). Preempts the reviewer objection "you claim
  geometry does nothing" while keeping the first-order claim intact.
- Regenerate from: `dyn3_exchange.txt`, `m3_master.csv` + `m3_residual_corr.csv`,
  `acl_analysis/insights/exchange_rate.csv` (`03_freelunch_exchange.py`).
- Risk: the paper already juggles two named geometry blocks (shape-only vs ladder block) and is
  careful to name which one each result uses; an e_top-only regression adds a third accounting
  and must be introduced with the same discipline. Qwen rows sit below the knee, so per-family
  heterogeneity needs stating. Numbers unverified against §18.

**5. Per-family fit-constants table (S1 table) and/or the six-panel family scatter (S1 fig).**
- Shows: n, r, rank-r, knee, below/above-knee slopes for all six families plus the pooled row —
  the numeric skeleton of fig:hero's claim across every family.
- Why it matters: the constants exist in the appendix as prose; a compact table is easier to
  audit and directly supports "one shape, per-family constants". The figure version already has
  a generator in the repo (`observatory/36_fig_family_scatter.py` -> `fig_family_scatter.png`)
  that is not referenced by main.tex, so the port cost is near zero.
- Regenerate from: `dyn1_structure.txt`, `key_numbers.md` §18.1 (frozen — lowest-risk gap),
  `20_m2_magnitude.py` + `m2_master.csv`; figure via `36_fig_family_scatter.py`.
- Risk: minimal, but two bookkeeping points: Qwen-math must carry its collapse-clean twin
  (-0.830 / -0.695) per the pre-registered rule, and the pooled row's n=1035 (paper's number)
  vs the artifact header's 1,004-usable census reflect different pool snapshots — reconcile the
  n before porting.

**6. Method exchange-rate residual tables (S5c adaptation, S5d drift).** Lower priority.
- Shows: per-method adaptation-per-dose and drift-per-dose residuals (SC-LoRA's math adaptation
  gain, LoRA+wd's CS gain, PiSSA +0.61 drift, DoRA drifting less than its size predicts).
- Why: texture for RQ1's "the one suggestive gain is an adaptation gain" and for the SC-LoRA
  calibration story; DoRA's negative drift residual is flagged in the artifact itself as an
  open question, not a result.
- Regenerate from: `dyn3_exchange.txt` §11-12. Risk: the artifact itself labels these
  "qualitative reads" (linear residuals on a saturating quantity); PiSSA cell is n=5; the drift
  residual signs invite prescriptive readings that the framing guardrails prohibit. Port only
  if S5a/S5b go in and a qualitative-read caveat rides along.

**7. Dose-benefit split panels (S2a).** Marginal.
- The concave-adaptation fact is already in the appendix prose with cluster-robust statistics;
  the binned two-panel version adds readability, not information. The healthy-run filter
  (adapt >= 70, n=268) is an analysis choice that would need disclosure. Skip unless the
  intervention figure (gap 3) is built, in which case the pair can share one figure block.

**Constraint noted (does not currently bind):** no artifact exhibit pools 7B-vs-284B, so nothing
above triggers the adjudication in `analysis_final/07_deepseek_284b_recurrence.md` §4. But if
any ported geometry exhibit (gaps 1, 4, 6) is later extended with 284B rows or a 7B-284B
rank-correlation, that comparison is APPENDIX-ONLY and must carry the rank-pooling disclosure
(residual methods — MiLoRA / SC-LoRA / LoRA-Null — save rank-2r adapters, so stable-rank splits
partly restate the design dichotomy; the within-stratum ordering evidence is weak, Spearman
+0.32/+0.14 at trained-r32). The superseded pooled "+0.86" must not be used as a headline.
