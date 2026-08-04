# RQ1 synthesis brief (2026-07-30)

RQ1: when retention-aware adapters are compared under one protocol, swept over
tasks, learning rates, and seeds, are any significantly different (adaptation
or retention)?

All statistics below are traceable to a named file; the rq1_stats layer was
independently recomputed (verify_log.md: ALL OK). Style: observational, no
em-dashes, no verdict titles. CLoRA guardrail respected throughout: every
"worse" statement refers to our within-harness swept reproduction at matched
capacity, never to CLoRA's published numbers, which are faithful.

---

## (i) Headline RQ1 answer

Under a common protocol with per-method learning-rate sweeps, no
retention-aware adapter is significantly better than plain LoRA with adapter
weight decay on either axis: across 25 Holm-corrected paired comparisons at
best-adaptation operating points, zero methods win, two are significantly
worse on retention and five on adaptation. At matched update magnitude, seven
of nine methods are statistically equivalent to LoRA+wd within +/-3 pp (TOST,
cluster-robust), the two exceptions (PiSSA -7.1 pp, SC-LoRA -4.1 pp) falling
below, and method identity adds only Delta R^2 = +0.006 of retention variance
once magnitude is controlled. The design detects effects of roughly 2-5 pp,
so smaller differences remain open; the one apparent reversal, SC-LoRA's
+8.3 pp GSM8K adaptation gain on Qwen math, does not survive multiplicity
correction (Holm p = 0.46) and is reported as suggestive.

---

## (ii) Paragraph candidates (LaTeX-ready)

### Candidate A. The variance view: method identity is a +0.006 effect

```latex
Method identity carries almost none of the retention variance once the size
of the update is known. In the pooled nested regression over the frozen
multi-seed pool ($n\!=\!1034$ runs, family fixed effects), family alone gives
$R^2\!=\!0.390$; adding $\log_{10}\dw$ raises it to $0.785$
($\Delta R^2 = +0.395$, $F \approx 1890$); three geometry summaries add
$+0.017$; and method dummies, added last, contribute
$\Delta R^2 = +0.006$ ($F\!=\!3.5$). Whatever separates the adapters under
this protocol is a bounded second-order effect riding on the magnitude
relation, which frames the operating-point comparisons that follow.
```

[sources: key_numbers.md §19.1 (ladder 0.390 -> 0.785 (+0.395, F~1890) ->
0.802 (+0.017, F=29.5) -> 0.808 (+0.006, F=3.5), n=1034, family FE);
verification_report.md B4 confirms the ladder exactly]

### Candidate B. The operating-point view: Holm head-to-heads, with the frontier as description

```latex
At each method's best-adaptation operating point we ran $26$ method
$\times$ setting comparisons against LoRA{+}wd, $25$ of them testable with
paired per-seed $t$ tests over $2$--$4$ common seeds (PiSSA on Llama math is
one seed against three; its deltas, retention $-26.3$, adaptation $-17.1$,
are reported without a test). After Holm correction across all $25$, no
method is significantly better than LoRA{+}wd on either axis. Two
comparisons are significantly worse on retention (our swept CLoRA
reproduction on Llama commonsense, $-4.25$ pp, $p_{\text{Holm}}\!=\!0.035$,
and on Qwen math, $-7.56$ pp, $p_{\text{Holm}}\!=\!0.024$) and five on
adaptation (on Llama commonsense LoRA $-2.58$, LoRA-Null $-2.89$, CLoRA
$-3.46$, MiLoRA $-4.56$; on Llama math CLoRA $-6.14$). Descriptively,
LoRA{+}wd records lower retention in none of the $26$ comparisons, $14$
opponents are dominated on both axes outside seed noise, and a seed-resample
bootstrap places LoRA{+}wd on the observed retention--adaptation frontier
with probability $1.00$ in three of the four settings (SC-LoRA holds the
fourth, Qwen math, at $1.00$, with LoRA{+}wd at $0.53$). We read frontier
membership as a stability statement about the observed operating points, not
as a hypothesis test; the corrected battery above is the inferential claim.
The one apparent reversal, SC-LoRA's $+8.26$ pp GSM8K gain on Qwen math at
statistically tied BBH (retention delta $+0.17$, $p\!=\!0.63$), has raw
$p\!=\!0.035$ over three seed pairs but $p_{\text{Holm}}\!=\!0.21$ within
family and $0.46$ across the battery; we report it as suggestive, and note
that no calibration-corpus control exists for this setting.
```

[sources: head2head_corrected.md (all deltas, p, Holm values, summary);
findings.md §1; verification_report.md B6 CORRECTION 2 (14 dominated, not
17; "0/26 retention losses" confirmed) and OVERCLAIM 1 (SC-LoRA scope);
pareto_bootstrap.csv (llama_cs / llama_math / qwen_cs LoRA+wd
p_on_frontier=1.0; qwen_math SC-LoRA 1.0, LoRA+wd 0.5285)]

If a CLoRA-faithfulness reminder is wanted in-place (it already exists
elsewhere in the paper), append:

```latex
These are within-harness comparisons of swept reproductions at matched
capacity; they do not bear on the published CLoRA results, which our
faithful-recipe replication treats as the reference point.
```

### Candidate C. The equivalence and power view: what "no difference" can and cannot mean

```latex
Non-significance alone certifies nothing, so we state the positive bounds.
Regressing retention on $\log_{10}\dw$ with method dummies (reference
LoRA{+}wd; cluster-robust standard errors at the recipe-cell level,
$n\!=\!1034$, $G\!=\!343$ cells, family fixed effects), seven of nine
methods are statistically equivalent to LoRA{+}wd within $\pm 3$ pp of
retention at matched update magnitude (TOST, $90\%$ confidence interval
inside the margin), and three of nine already within $\pm 2$ pp; no method
reaches $\pm 1$ pp, a statement about power rather than about differences.
The two non-equivalences are one-sided and below the reference: at matched
magnitude PiSSA retains $7.1$ pp less ($90\%$ CI $[-9.0, -5.3]$) and SC-LoRA
$4.1$ pp less ($[-5.6, -2.6]$); SC-LoRA's deficit is consistent with the
calibration-corpus attribution established by the eval-matched control on
Llama commonsense, which moved it from $3.4$ pp below the curve to $0.9$ pp
above. On the power side, at the observed three to five common seeds and the
empirical spread of paired per-seed deltas, the head-to-head design detects
retention differences of roughly $2$--$5$ pp at typical cells (per-family
median minimum detectable effect $1.7$--$4.5$ pp), and far larger ones where
a method's best cell is seed-unstable (SC-LoRA on Qwen commonsense: $50$ pp).
Method differences below about $2$ pp are therefore outside what this
protocol can adjudicate, and the equivalence bounds above, not the
null results, are the quantitative content of ``no difference''.
```

[sources: tost_offsets.md (pooled table and summary; PiSSA -7.12
[-8.96, -5.28], SC-LoRA -4.08 [-5.60, -2.56]); power_notes.md (family MDE
table; SC-LoRA qwen_cs MDE 49.89); key_numbers.md §18.3 E4 (-3.39 -> +0.92);
findings.md §2-3]

### Candidate D. Scope limits (one sentence)

```latex
These conclusions are scoped to two 7B models, two task types, and
best-adaptation operating points under our swept protocol at three to five
seeds: differences smaller than about $2$ pp, other scales, and the
calibration-corpus sensitivity of data-aware initializations (no
eval-matched control exists on Qwen math) remain open.
```

[sources: power_notes.md; verification_report.md B6 OVERCLAIM 1]

---

## (iii) Numbers table (every statistic used, with provenance)

| statistic | value | source file |
|---|---|---|
| Comparisons vs LoRA+wd | 26 (25 testable) | rq1_stats/head2head_corrected.md, findings.md §1 |
| PiSSA llama_math untestable deltas | dRet -26.34, dAdapt -17.13 (welch 1v3, reported without test) | head2head_corrected.md |
| Retention, Holm across 25 | 0 better, 2 worse | head2head_corrected.md summary; verify_log.md V1 |
| CLoRA llama_cs retention | -4.25 pp, p_Holm(all)=0.0348 | head2head_corrected.md (use -4.25, not head2head.md's -4.26; see tension T3) |
| CLoRA qwen_math retention | -7.56 pp, p_Holm(all)=0.0236 | head2head_corrected.md |
| Adaptation, Holm across 25 | 0 better, 5 worse | head2head_corrected.md summary |
| Adaptation losers | LoRA -2.58, LoRA-Null -2.89 (file: -2.88 in raw table, -2.89 in findings; both round to -2.9), CLoRA -3.46, MiLoRA -4.56 (llama_cs); CLoRA -6.14 (llama_math) | head2head_corrected.md; findings.md §1 |
| SC-LoRA qwen_math adaptation | +8.26 pp, raw p=0.0352, Holm(fam)=0.2115, Holm(all)=0.4582; t=5.18 | head2head_corrected.md; verification_report.md B6 |
| SC-LoRA qwen_math retention delta | +0.17, p=0.6318 | head2head_corrected.md |
| Retention losses by LoRA+wd | 0 of 26 | adjudication/tables/head2head.md tally; verification_report.md B6 (confirmed) |
| Dominated opponents | 14 of 26 | verification_report.md B6 CORRECTION 2 (supersedes "17") |
| P(on frontier), LoRA+wd | 1.00 in llama_cs, llama_math, qwen_cs; 0.5285 in qwen_math | adjudication/tables/pareto_bootstrap.csv |
| P(on frontier), SC-LoRA qwen_math | 1.00 | pareto_bootstrap.csv |
| llama_cs frontier footnote | E6 MiLoRA+wd (1-cell arm) excluded from ranking | verification_report.md B6 CORRECTION 1 |
| Ladder Delta R^2 | family 0.390; +0.395 magnitude (F~1890); +0.017 geometry; +0.006 method (F=3.5); n=1034 | key_numbers.md §19.1; verification_report.md B4 |
| TOST pooled equivalence | 0/9 at +/-1 pp; 3/9 at +/-2 pp; 7/9 at +/-3 pp | tost_offsets.md summary; verify_log.md V2 |
| PiSSA pooled offset | -7.12 pp, 90% CI [-8.96, -5.28] | tost_offsets.md pooled table |
| SC-LoRA pooled offset | -4.08 pp, 90% CI [-5.60, -2.56] | tost_offsets.md pooled table |
| PiSSA per-family offsets | frc -5.88, frm -11.17 | tost_offsets.md (see tension T4 vs §18.4's -11.4) |
| SC-LoRA per-family offsets | frc -3.64, frm -2.90 | tost_offsets.md |
| Per-family equivalence at +/-2 pp | 14/39 offsets; all Qwen CIs too wide to bound | tost_offsets.md summary |
| TOST model spec | ret ~ log10 F_Delta + method dummies, CR1 at recipe-cell level, G=343, family FE | tost_offsets.md header |
| MDE, per family (median / max, pp) | llama_cs 2.67/4.97; llama_math 3.42/5.06; qwen_cs 1.71/49.89; qwen_math 4.49/33.73 | power_notes.md |
| MDE headline | design detects ~2-5 pp at typical cells | findings.md §3; power_notes.md |
| E4 calibration control | SC-LoRA -3.39 pp below curve (nq_open) vs +0.92 above (eval-matched), Llama CS only | key_numbers.md §18.3 E4 |
| §18.4 matched-F method offsets (context) | significant offsets bounded +/-1.2-4.6 pp on Llama grids (plus pissa frm -11.4 collapse-driven); Qwen arms n.s. | key_numbers.md §18.4 |
| Pool preflight | n=1035, r=-0.847 (frozen); deduped n=1034 for regression layers | findings.md header; key_numbers.md §18.1, §19.1 |
| Verification status of rq1_stats | ALL OK (25 checks; simulated power 0.796-0.801) | rq1_stats/verify_log.md |

---

## (iv) Tensions found and resolutions

T1. "LoRA+wd sole Pareto method in 3/4 settings" vs "no method significantly
different after Holm". Not a contradiction, but the two claims have
different epistemic status and the paper must not blur them. The bootstrap
frontier membership (pareto_bootstrap.csv) is a stability statement about
observed operating points under seed resampling; the Holm battery
(head2head_corrected.md) is the hypothesis test, and it is asymmetric:
nothing beats LoRA+wd, a few methods lose to it, most comparisons are n.s.
at 2-5 pp MDE. Candidate B states this reading explicitly ("frontier
membership as description, corrected battery as inference").

T2. SC-LoRA on Qwen math: frontier P=1.00 and raw p=0.035, yet
Holm(all)=0.46. The frozen adjudication layer and the current paper text
(paper_conventional.tex §5.4, "The one exception, scoped", quoting t=5.2)
present this as the one exception on the frontier; the corrected battery
demotes it to suggestive. Resolution: keep the descriptive numbers (+8.26,
t=5.18, P(frontier)=1.00) at full strength but attach "raw p=0.035, not
significant after Holm correction (p=0.46)" and the no-Qwen-math-
calibration-control scope from verification_report.md OVERCLAIM 1. The
existing §5.4 paragraph needs this one-clause update when Candidate B is
absorbed.

T3. CLoRA llama_cs retention delta: -4.26 in adjudication head2head.md vs
-4.25 in head2head_corrected.md. Rounding at the pipeline boundary; the
corrected file was independently recomputed to 3e-5 agreement
(verify_log.md), so quote -4.25. Same for LoRA-Null llama_cs adaptation
(-2.88 table row vs -2.89 findings summary; quote -2.9 or -2.89).

T4. PiSSA frm offset: -11.17 (tost_offsets.md, CR1, reference LoRA+wd) vs
-11.4 +/- 2.1 (key_numbers.md §18.4, OLS, reference CLoRA). Different model
specifications and reference categories, not a disagreement. Quote -11.17
in the TOST/equivalence context and reserve §18.4's -11.4 for the frozen
ladder context; never mix them in one sentence. (findings.md's "-11.2" is a
loose rounding of the TOST value; use -11.17 or "about -11 pp".)

T5. §18.4 "Qwen arms: no significant offsets" vs TOST "no Qwen offset
equivalent even at +/-3 pp". Both true and jointly instructive: Qwen CIs
are wide (power_notes.md, MDE up to 50 pp at seed-unstable cells), so
neither significance nor equivalence is established there. Candidate C
carries this via "every Qwen offset has a CI too wide to bound".

T6. Ladder Delta R^2 = +0.006 (method identity negligible pooled) vs real,
significant below-curve offsets for PiSSA and SC-LoRA (TOST) and the two
CLoRA operating-point losses (Holm). Both views are kept: method identity
explains almost no pooled variance, and the few real method effects are
bounded (1-7 pp) and all in the direction of retaining less than LoRA+wd,
never more. This is the structure of the headline answer, not a conflict.

T7. Dominated count: 17 (early adjudication findings prose) vs 14
(verification recount from the head2head CSV itself). Use 14 everywhere
(verification_report.md B6 CORRECTION 2 and DO-NOT-QUOTE item 3).

T8. Pool sizes n=1035 vs n=1034. The frozen preflight pool is 1035; the
regression layers (ladder, TOST) dedupe one byte-identical `_reeval` row to
1034. Disclose once, as verification_report.md A1 requires ("state, once,
which pool each exhibit uses").
