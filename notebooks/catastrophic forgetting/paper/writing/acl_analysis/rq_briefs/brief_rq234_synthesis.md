# RQ2 / RQ3 / RQ4 synthesis brief — candidate Results text with verified numbers

`[Synthesis agent, 2026-07-30. Every number below is traced to a frozen source file
(provenance table, section D). Where sources conflicted, the frozen-convention /
verified value was taken and the conflict is logged in section E. Guardrails:
observational framing, no verdict titles, no prescriptive rules; CLoRA's published
numbers are faithful; verified numbers quoted at full strength. No em-dashes in
the LaTeX candidates. Pool conventions: "frozen pool" = n=1035 (§18.1),
"geometry join" = n=1034, "CE join / same-sample" = n=911.]`

---

## A. One-sentence answers

**RQ2.** Update magnitude governs the adaptation-retention outcome: log F_delta
adds +0.395 R^2 over family effects (95% CI [+0.311, +0.482]) where update
geometry adds +0.017 and method identity +0.006, the ordering
magnitude > geometry holds in 2000/2000 cluster-bootstrap replicates, the
relation survives a within-cell seed-only micro-test (r = -0.713) and a direct
rescaling intervention (15/15 rescales on-curve, 9/9 random-direction controls,
direction penalty -3.05 pp), while behavioral drift is the same process read
downstream (+0.005 unique variance beyond magnitude) and geometry's reliable
role is to fingerprint methods, not to predict retention.

**RQ3.** Yes as a calibrated monitor on Llama and as a tripwire everywhere: one
forward pass of KL drift to the base model predicts held-out retention to
1.3-2.0 pp RMSE after per-family calibration on the four Llama families,
separates runs damaged by more than 5 pp at AUC >= 0.976 in all six families,
and damage onset was observed near ~0.3 nats of drift in four of six families;
on Qwen the quantitative mapping degrades to roughly 4-6 pp (tail-driven), so
there it is a tripwire, not a ruler.

**RQ4.** Yes: normalized by each benchmark's base ceiling, retention benchmarks
degrade in the identical order MMLU-Pro > BBH > MMLU > TruthfulQA in all six
families (Kendall W = 1.000, p = 4.4e-4), with the caveat that TruthfulQA
actually rises under forgetting on all four Llama families (r up to +0.79),
which attenuates the measured Llama broad-retention slope by 24-30% when it is
included in the aggregate.

---

## B. LaTeX-ready paragraph candidates

Conventions: `\dw` = effective update magnitude F_delta (as in
paper_conventional.tex); plain style; observational voice; no em-dashes.

### B1. RQ2(a) — the magnitude relation

```latex
\paragraph{Retention tracks update magnitude.} Pooled over the frozen set of
$1{,}035$ runs, retention correlates with $\log_{10}\dw$ at $r\!=\!-0.847$
(Spearman $-0.923$), and the six per-family correlations run $-0.830$ to
$-0.929$. In every family a two-segment fit beats a line: retention is flat to
mildly declining up to a per-family knee (in $\log_{10}\dw$, between $-0.91$
and $-0.02$) and falls steeply above it ($-7.5$ to $-40.8$ points per decade).
The normalized above-knee slopes do not converge ($-0.33$ to $-0.70$), so we
describe a magnitude relation with a knee, not a single universal law: the form
is shared, the knee location and steepness are family properties.
```

```latex
\paragraph{Not an artifact of recipe aggressiveness.} Larger updates partly
reflect more aggressive training, so we do not rest on the pooled correlation.
Within cells, with method, learning rate, and every other knob fixed and only
the seed varying, seed-level fluctuations in $\log\dw$ still predict seed-level
retention ($r\!=\!-0.713$, $n\!=\!954$ over $290$ cells, $t\!=\!-31.3$). A
direct intervention on Llama commonsense gives the reading causal footing in
the setting where it ran: rescaling a trained adapter's update with no
retraining lands it on the family curve in $15/15$ cases (mean residual
$+1.29\pm2.07$ points), while $9/9$ random-direction updates of matched
magnitude land below the curve ($-3.05$ points relative to the trained
rescales at matched magnitude) and buy essentially no adaptation. Direction
carries adaptation; magnitude tracks forgetting.
```

### B2. RQ2(b) — the variance ladder and its honest caveat

```latex
\paragraph{Geometry adds little once magnitude is known.}
Table~\ref{tab:ladder} decomposes retention variance on the $1{,}034$ runs
with geometry coverage. Family fixed effects give $R^2\!=\!0.390$; adding
$\log\dw$ raises it to $0.785$ ($\Delta R^2\!=\!+0.395$, cluster-bootstrap
$95\%$ CI $[+0.311, +0.482]$); adding update geometry
($e_{\mathrm{top}}$, $\log$ spectral norm, stable rank) raises it to $0.802$
($+0.017$, roughly $23\times$ less); method identity adds $+0.006$. The
ordering magnitude before geometry holds in all $2{,}000$ cluster-bootstrap
replicates, and the standardized coefficient of $\log\dw$ is $-0.744$
($[-0.894, -0.615]$) against $|\beta|\!\le\!0.138$ for every geometry term. A
commonality split on the two exogenous blocks (magnitude vs.\ a shape-only
geometry block of $e_{\mathrm{top}}$ and stable rank) attributes $+0.296$
($[+0.203, +0.386]$) uniquely to magnitude, $+0.016$ ($[+0.006, +0.032]$)
uniquely to geometry, and $+0.099$ shared.
```

```latex
\paragraph{The three-way accounting, disclosed in full.} Adding behavioral
drift (KL to base) as a third block on the common sample ($n\!=\!911$, with the
five-metric shape block) shrinks every unique share, because the blocks overlap
heavily: magnitude $+0.033$, geometry $+0.031$, drift $+0.009$ unique, with the
bulk shared (magnitude$\cap$drift $+0.181$, three-way $+0.154$); under this
accounting the strict ordering of magnitude's unique share above both others
holds in only $1{,}078$ of $2{,}000$ bootstrap replicates. We read this as
overlap, not weakness: on the same sample the single-block $\Delta R^2$ values
are magnitude $+0.420$, drift $+0.340$, geometry $+0.234$, and drift is a
downstream readout of the update rather than a knob, so its shared variance
with magnitude is expected rather than confounding.
```

### B3. RQ2(c) — the learning rate is a proxy

```latex
\paragraph{The learning rate acts through magnitude.} In a league table of
single predictors on the common sample ($\Delta R^2$ over family effects,
Table~\ref{tab:league}), the learning rate scores $+0.207$, below every
magnitude measure ($\log\dw$ $+0.420$, spectral norm $+0.349$, Frobenius norm
$+0.348$) and below behavioral drift ($+0.340$), above every geometry measure
($\le\!+0.116$). Per family, $\dw$ reaches $R^2\!=\!0.69$ to $0.86$ against
$0.22$ to $0.52$ for the rate, and the partials separate them cleanly:
$r(\log\dw, \mathrm{ret}\mid \mathrm{LR})\!=\!-0.58$ to $-0.91$
($|t|\!\ge\!7.6$) while $r(\mathrm{LR}, \mathrm{ret}\mid \log\dw)$ stays
within $-0.17$ to $+0.29$ ($|t|\!\le\!4$). At fixed rate the relation persists
($r\!\le\!-0.7$ at every rate $\ge\!10^{-4}$ in every family): the rate matters
exactly to the extent that it sets the magnitude.
```

### B4. RQ2(d) — geometry's real role, and the three roles

```latex
\paragraph{What geometry does reliably: identify methods.} The same
measurements that fail to predict retention cleanly fingerprint each design
from its trained weights. At matched $\dw$ the update's stable rank separates
the methods (from about $4.5$ for DoRA to about $18$ for PiSSA), and the
energy fractions recover each method's advertised subspace (MiLoRA minor,
SC-LoRA input-principal, LoRA-Null output-principal). Whatever residual
retention leverage geometry retains is second-order and model-dependent:
$r(\text{stable rank}, \mathrm{ret}\mid\log\dw)$ runs $-0.32$ to $-0.67$ on
the Llama families but is $\approx\!0$ on Qwen ($-0.004$ and $+0.073$). Across
seeds, the $e_{\mathrm{top}}$ and stable-rank partials keep their sign in
$4/4$ replicates, while the spectral-norm direction partial changes sign at
one seed and is not significant under cell clustering, so we do not quote it.
The three measurement families thus play three roles: update magnitude is the
controllable variable that tracks retention, behavioral drift is the same
process read in behavior space and serves as a monitor, and update geometry is
a method fingerprint.
```

### B5. RQ3 — behavioral drift as a practical retention monitor

```latex
\paragraph{A one-forward-pass monitor.} KL drift of the adapted model from its
base on held-out text costs a single forward pass over roughly $40$ WikiText
blocks (about $3$ seconds in our harness), against a full BBH and MMLU-Pro
retention evaluation, and requires no benchmark harness and no access to the
adapter weights. Calibrating a per-family knee-shaped map from KL to retention
once, then predicting held-out cells (leave-cells-out cross-validation),
recovers retention to RMSE $1.3$ to $2.0$ points (MAE $0.8$ to $1.2$) on the
four Llama families, roughly twice the retention benchmark's own seed noise
($0.4$ to $0.9$ points). Under the same cross-validation the KL calibration
gave lower error than the best $\log\dw$ calibration in all six families.
```

```latex
\paragraph{Damage detection and the observed drift threshold.} Used as a
screen rather than a ruler, the monitor is accurate everywhere we measured:
runs damaged by more than $5$ points below the family healthy ceiling are
separated from healthy runs at AUC $\ge 0.976$ in all six families
($0.976$ to $0.996$). The hinge fits place damage onset near $0.26$ to $0.30$
nats of KL drift in four of the six families, spanning both base models and
both task types; the Llama grid family puts it near $0.4$, and the Llama math
grid has no flat region at all (its below-knee slope is already $-8.3$ points
per decade, so its fitted knee at $1.69$ nats marks a slope change, not damage
onset). Across these families, damage onset was observed near $0.3$ nats of
drift. Where adapter weights are available, $\log\dw$ is an equally good
tripwire (AUC $0.98$ to $1.00$); the drift signal's distinct value is the
quantitative calibration and the fact that it needs only forward passes.
```

```latex
\paragraph{Scope of the monitor.} On the Qwen families the quantitative
mapping degrades several-fold (RMSE roughly $4$ to $6$ points, tail-driven and
fold-sensitive), so there the monitor functions as a tripwire, not a ruler.
Three qualifications apply throughout: drift is a downstream consequence of
the update, a monitor rather than a knob; it is measured on base-model text
and is therefore close in kind to retention itself; and it is blind to the
format-damage channel that benchmark evaluations see. On the frozen pool,
$\log\dw$ remains the stronger single within-family predictor in five of six
families (KL wins only Llama math sweep, $0.86$ vs.\ $0.75$); an earlier
analysis reporting the reverse used the pre-freeze quarantine-excluded pool,
where removing far-collapse runs favors the behavioral metric. Same-sample
pooled $\Delta R^2$ is $0.420$ for $\log\dw$ vs.\ $0.340$ for KL under either
convention.
```

### B6. RQ4 — the fragility ordering (compact)

```latex
\paragraph{Benchmarks degrade in a fixed order.} Normalizing each retention
benchmark's degradation slope by its base ceiling (fraction of base capability
lost per decade of $\dw$), the fragility ordering is identical in all six
families: MMLU-Pro $>$ BBH $>$ MMLU $>$ TruthfulQA (Kendall $W\!=\!1.000$,
$p\!=\!4.4\times10^{-4}$; the contaminated ARC-c is excluded, and including it
gives $W\!=\!0.906$). Normalized slopes run $-0.36$ to $-0.73$ of base per
decade for MMLU-Pro, $-0.29$ to $-0.64$ for BBH, $-0.15$ to $-0.49$ for MMLU,
and $+0.13$ to $-0.21$ for TruthfulQA; the pattern is consistent with
generative format-following degrading first and calibration-style benchmarks
last, though part of the ordering could reflect distance to the chance floor
(MMLU-Pro is 10-way). TruthfulQA is the caveat: it rises with forgetting on
all four Llama families (cell-level $r\!=\!+0.40$ to $+0.79$, $+2.2$ to
$+5.0$ points per decade, $p\!\le\!8\times10^{-3}$) while falling on both Qwen
families ($r\!=\!-0.87$ and $-0.90$). Because it sits inside the broad
retention aggregate, its inclusion attenuates the measured broad slope by
$24$ to $30\%$ on Llama ($12$ to $14\%$ on Qwen), so we report the aggregate
with and without it.
```

---

## C. Assembly notes (for the author)

- B1/B2/B3/B4 restructure and absorb paper_conventional.tex §5.1-§5.3 (lines
  424-566). Deltas vs the current text: (i) the ladder paragraph gains the
  bootstrap CIs and the honest 3-block caveat (currently absent); (ii) the CE
  monitor moves from one sentence inside §5.1 to a full RQ3 block (B5); (iii)
  the current §5.1 "One curve across methods" figure paragraph and the E-checks
  sentence (per-method slopes F-test, leave-one-method-out RMSE 2.5 vs 2.2) are
  kept as-is, not duplicated here; (iv) the current LR-artifact §5.3 second
  paragraph (rate-as-different-dose, safe-LR band 26/29) stays where it is,
  B3 replaces only its first paragraph.
- The "three knobs, one curve" sentence (wd, CLoRA-k, rank all move F_delta;
  residual Delta R^2 <= 0.006) already exists in §5.1 "It reproduces" and
  should stay there; it is RQ2-adjacent but belongs to the dose story.
- Unit hygiene for knees: §18.2 knees are in log10 F_delta (lrsw -0.02 means
  F_delta ~ 0.95), while the hero-figure caption's "knee near 0.4" is in raw
  dw units for the frc-style fit. Do not mix the two conventions in one
  sentence; B1 stays in log10 units throughout.
- League position of LR if a rank is quoted: LR is row 6 of 11 in
  league_table.md; CE and KL are near-duplicate rows under family FE, so
  "fifth of ten distinct predictors" (current paper wording) is defensible.
  B3 avoids the rank and quotes the position qualitatively plus the +0.207.

---

## D. Numbers table with provenance

All paths relative to `paper/writing/`. VR = acl_analysis/verification/verification_report.md.

| # | Number (as used above) | Value | Source | VR status |
|---|---|---|---|---|
| 1 | Pooled r(ret, log10 F_delta), frozen pool | -0.847 (Spearman -0.923), n=1035 | data/key_numbers.md §18.1 | Confirmed (A1) |
| 2 | Per-family r range | -0.830 to -0.929 | key_numbers.md §18.1 | Confirmed |
| 3 | Knees (log10 F_delta) | -0.91 to -0.02 | key_numbers.md §18.2 | frozen |
| 4 | Above-knee slopes | -7.5 to -40.8 pp/decade | key_numbers.md §18.2 | frozen |
| 5 | Normalized slopes (non-convergent) | -0.33 to -0.70 | key_numbers.md §18.2 | frozen |
| 6 | Within-cell demeaned micro-test | r=-0.713, n=954, 290 cells, t=-31.3 | key_numbers.md §18.6 | frozen |
| 7 | E1 trained rescales | 15/15 on-curve, mean residual +1.29±2.07 pp | key_numbers.md §18.3 | frozen |
| 8 | E1 random-direction controls | 9/9, mean -1.76±1.32 pp; direction penalty -3.05 pp vs trained | key_numbers.md §18.3 | frozen |
| 9 | Ladder R^2 steps | 0.390 -> 0.785 (+0.395) -> 0.802 (+0.017) -> 0.808 (+0.006), n=1034 | key_numbers.md §19.1 | Confirmed exact (B4) |
| 10 | Ladder magnitude CI | [+0.311, +0.482] (B=2000 cell bootstrap) | analysis_final/09_verification_2026-07-18.md Q1; tables/table_ladder.tex | verified source |
| 11 | Magnitude>geometry ordering | 2000/2000 bootstrap replicates | analysis_final/09_verification_2026-07-18.md Q1 | verified source |
| 12 | Standardized betas | log F_delta -0.744 [-0.894,-0.615]; geometry \|beta\| <= 0.138 | key_numbers.md §19.1; 09_verification Q1 | Confirmed |
| 13 | M-vs-G exogenous commonality (shape-only G: e_top, stable rank) | unique M +0.296 [+0.203,+0.386], unique G +0.016 [+0.006,+0.032], shared +0.099 | 09_verification Q1; VR B4 | Confirmed exact |
| 14 | 3-block commonality (5-metric G), n=911 | uM +0.033, uG +0.031, uC +0.009; M∩C +0.181, M∩G∩C +0.154 | acl_analysis/correlations/commonality.md | Confirmed exact (B4) |
| 15 | 3-block ordering caveat | uM > max(uG,uC) in 1078/2000 replicates | commonality.md | frozen |
| 16 | Single-block dR2 same-sample | M +0.420, C +0.340, G +0.234 | commonality.md | Confirmed (M, C) |
| 17 | Two-block M vs C | unique M +0.085, unique C +0.005, shared +0.335 | commonality.md; VR B4 | Confirmed exact |
| 18 | League table | F_delta +0.420 (t -12.0) > spec_max +0.349 ~ \|\|dW\|\|_F +0.348 > KL/CE +0.340 > LR +0.207 > stable rank +0.116 > rest <= +0.032 (n=911) | correlations/league_table.md | Confirmed exact (B4) |
| 19 | Per-family R^2: F_delta vs LR | 0.69-0.86 vs 0.22-0.52 | key_numbers.md §18.5 (0.689-0.863 vs 0.223-0.516) | frozen |
| 20 | Partials | r(F\|LR) -0.58..-0.91 (\|t\|>=7.6); r(LR\|F) -0.17..+0.29 (\|t\|<=4) | key_numbers.md §18.5 | frozen |
| 21 | Fixed-LR strata | r <= -0.7 at every LR >= 1e-4, every family | key_numbers.md §18.5 | frozen |
| 22 | Stable-rank fingerprint range | ~4.5 (DoRA) to ~18.1 (PiSSA) at matched dw | VR safe ledger (Observatory) | Confirmed |
| 23 | Stable-rank retention partial per family | -0.333, -0.323, -0.595, -0.666 Llama; -0.004, +0.073 Qwen | VR B8 | Confirmed exact |
| 24 | Geometry seed stability | e_top -0.218±0.128 (4/4), stable rank -0.313±0.184 (4/4), log spec_max +0.105±0.146 (3/4, crosses zero at s44) | key_numbers.md §19.2 | frozen |
| 25 | spec_max direction partial dies under clustering | t +2.1 -> +1.4, ns | 09_verification Q1 | verified source |
| 26 | KL monitor cost | 1 forward pass, ~40 WikiText blocks, ~3.4 s | correlations/ce_proxy.md header | frozen |
| 27 | Llama calibration error (leave-cells-out) | RMSE 1.3-2.0 pp, MAE 0.8-1.2 pp | ce_proxy.md (1.31-1.95 / 0.83-1.19); VR B5 widens to 1.30-1.98 | Confirmed |
| 28 | Seed-noise floor (Llama retention) | 0.43-0.94 pp (mean within-cell SD) | ce_proxy.md seed SD column; VR B8 | Confirmed |
| 29 | KL beats best logF calibration | 6/6 families | ce_proxy.md; VR B5 | Confirmed under independent CV |
| 30 | Damage AUC (>5 pp below healthy ceiling) | >= 0.976 in 6/6 (0.976-0.996); logF tripwire 0.98-1.00 | ce_proxy.md; VR B5 | Confirmed exactly |
| 31 | KL knee band | ~0.26-0.30 nats in 4/6; frc ~0.4; frm 1.69 = slope change (below-knee slope -8.3) | ce_proxy.md (0.259-0.399, frm 1.686); VR B5 widens band | Corrected wording (VR) |
| 32 | Qwen calibration error | ~4-6 pp, tail-driven, fold-sensitive | VR B5 correction (ce_proxy.md's 5.05/6.00 fold-sensitive) | Corrected |
| 33 | Within-family F vs KL count, frozen pool | F_delta better 5/6; KL wins lrswm only (0.856 vs 0.747) | VR A3 | Confirmed; binding convention |
| 34 | Fragility ordering | MMLU-Pro > BBH > MMLU > TQ, 6/6; Kendall W=1.000, chi2=18.0, p=4.4e-4; with ARC-c W=0.906 | insights/findings.md #1; VR B7 | Confirmed exactly |
| 35 | Normalized fragility slopes | MMLU-Pro -0.36..-0.73; BBH -0.29..-0.64; MMLU -0.15..-0.49; TQ +0.13..-0.21 | insights/findings.md #1; VR B7 | Confirmed |
| 36 | TQ inversion (Llama) | r +0.40..+0.79, +2.2..+5.0 pp/decade, p<=8e-3; Qwen -0.87/-0.90 | insights/findings.md #2; VR B7 | Confirmed exactly |
| 37 | Broad-slope attenuation from TQ | 24-30% Llama (24/26/27/30), 12-14% Qwen | insights/findings.md #2; VR B7 | Confirmed |
| 38 | Qwen CE coverage | 60-62% (ignorable missingness) | ce_proxy.md caveats | frozen |

---

## E. Self-check: conflicts found and resolutions

1. **KL vs F_delta within family (5/6 flip).** doc-05 era claimed KL better in
   5/6; frozen quarantine-included pool gives F_delta better in 5/6 (KL 1/6).
   Both reproduce (VR A3); the frozen convention is binding. B5 uses the VR A3
   safe wording including the pool disclosure.
2. **Knee band.** ce_proxy.md prints 0.259-0.290 (+frc 0.399, frm 1.686); VR B5
   shows grid sensitivity ±0.02-0.05 nats and mandates "~0.26-0.30" / "~0.3
   nats". B5 quotes the widened band and states the threshold observationally
   ("damage onset was observed near 0.3 nats"), never as an instruction.
3. **Qwen RMSE.** ce_proxy.md table says 6.00/5.05; VR B5 shows qwswm is
   3.6-5.1 depending on folds/grid. B5 quotes "roughly 4 to 6, tail-driven and
   fold-sensitive" per the VR correction; "5-6 pp" is not printed.
4. **Two geometry blocks in circulation.** unique(G)=+0.016 belongs to the
   2-metric shape block (e_top, stable rank); +0.031 to the 5-metric block.
   B2 names the block at each quote, per VR B4's required disclosure.
5. **Ladder magnitude dR2: +0.395 vs +0.420.** Not a conflict: +0.395 is the
   full geometry-join pool (n=1034, ladder), +0.420 is the CE-join same-sample
   league (n=911). Both quoted with their pools.
6. **Seed SD ranges.** §18.1 prints within-cell ret SD 0.33-1.00 (Llama);
   ce_proxy.md / VR B8 print 0.43-0.94 (mean within-cell SD, CE-join pool).
   B5's noise-floor comparison uses 0.4-0.9 (the CE-join numbers, same pool as
   the RMSE it is compared against).
7. **E1 direction control depth.** Current paper text says "about 1.8 points
   below the curve (3 points below trained rescales)"; §18.3 frozen values are
   -1.76±1.32 below curve and -3.05 pp vs trained. B1 quotes -3.05 (the
   sharper, frozen number) and +1.29±2.07 for rescales.
8. **spec_max positive direction effect (§18.4 +0.117).** Dies under cell
   clustering (t +1.4) and crosses zero at seed 44; per 09-Q1 rule it is not
   quoted anywhere above; B4 explains why.
9. **Cluster-robust statistics only.** All t values quoted are cluster-robust
   (league t -12.0, micro-test t -31.3 is the §18.6 frozen within-cell value);
   no naive OLS F is used as evidence strength.
10. **LR league rank.** league_table.md has LR at row 6 of 11; the current
    paper's "fifth of ten" counts CE/KL as one predictor. B3 avoids the
    ordinal; assembly note C flags the counting convention if a rank is kept.

---

## F. Deliberately left out (appendix candidates or discussion-only)

- **share_q second-order axis** (pooled partial -0.28 beyond magnitude, but
  present only in the CS LR-sweeps, absent in grid/math arms; insight 6):
  appendix with the family split disclosed; fingerprint-grade, not an axis.
- **Per-matrix method fingerprints** (CLoRA 82% of update energy in MLP, PiSSA
  most attention-heavy, U-shaped depth profiles, depth-centroid retention null;
  insight 7): appendix; not independently re-derived (VR B7), keep its own
  confidence labels. The depth-centroid null (+0.03, ns) is a useful
  "we checked layers" sentence if space allows.
- **Adaptation-side collapse ordering** (hellaswag/ARC first, social_i_qa last,
  Kendall W=0.738; insight 8): discussion-only; explains cs_avg mixture.
- **Free-lunch and exchange-rate tables** (99-100% of peak adaptation below
  the knee 6/6; negative-sum region beyond ~2x knee): appendix, RQ2-adjacent
  practitioner framing; quote LoRA+wd reachability as 12/26 or ~40-46%, never
  12/31 (VR B7 denominator correction).
- **CE-proxy protocol-sensitivity and task-transfer tables** (ce_proxy.md):
  appendix support for B5; the protocol mixture is benign (09-Q2) and gets one
  disclosure sentence.
- **Per-benchmark knees within a family** (e.g. frc BBH -0.18 vs MMLU-Pro
  +0.18 in log10 F_delta): appendix footnote to RQ4.
- **284B stable-rank recurrence** (§19.3, rank-r +0.86): design-family
  recurrence framing only; no retention/magnitude claims at 284B.
- **E5 replay CE salvage and E6 wd-boundary (MiLoRA yes, DoRA degenerate)**:
  limitations/appendix; not RQ2-4 material.
- **TQ "regression to indifference" mechanism**: labeled speculation only; the
  sign flip and attenuation are the quotable facts.
