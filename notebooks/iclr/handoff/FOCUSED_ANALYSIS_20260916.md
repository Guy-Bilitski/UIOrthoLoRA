# Focused-study analysis and paper integration — 2026-09-16

> Superseded evidence status (2026-09-16): the bundle delivered at Overleaf
> `5510161` supplies the requested learned/module records, manifests and partial
> band export. See `PAPER_FINALIZATION_STATUS_20260916.md` and the independently
> generated `data/final_evidence_analysis.json`. Statements below about absent
> exports describe the earlier snapshot. The band block and a separately requested
> one-time held-aside evaluation remain pending.

Internal author record for project `6aa54397e58b10444b0fa2aa`; not anonymous
supplementary material. Analysis source: Overleaf revision `8729fe3`.

## Scope and reproducibility

Independently recomputed the 36 exported rows: 18 calibration/refinement and 18
confirmation, with three distinct seeds in each task/arm cell and 48 matching
modules per run. Checked the full dose frontier, selection rule, validation SHA
format, fraction sums (float32 tolerance 1e-6), equal-module norm reconstruction,
seed uniqueness, and all 36 learning-health flags (`ok`). Raw CSV/selection records
are unchanged. The extractor's mandatory invalidation-record checks are unchanged.

This is an audit of the exported snapshot, not independent reexecution of server
checkpoint reloads, tokenizer parity, or the reference probe. Server artifact
validation remains the upstream evidence contract; a SHA in the export alone does
not revalidate its target. No server/GPU process or shared research checkout was
modified. No BAND outcomes were present in the inspected snapshot.

Source SHA-256:

- `runs.csv`: `b401cc771f45cfb82fdd4d455c0d12a897d657b4e09ce59610383738896367a9`
- `selection_final.json`: `4c92bfead0c62bdac696d4abdec150351110ffe147609f0a0b036c8f0ba3d6ff`

`scripts/analyze_focused_confirmation.py --write` creates
`data/focused_norm/confirmation_analysis.json` and `figures/focused_module_norms.pdf`.
It uses NumPy, SciPy and Matplotlib. `--check` recomputes the JSON. The existing
`scripts/build_focused_tables.py --write/--check` now also covers the paired and
module tables. No extractor was run against absent server paths.

## Verified headline and matching correction

Pooled cross-block energy %, mean ± sample SD (n=3):

| Task | UNREG | MIX | NORM |
|---|---:|---:|---:|
| RTE | 43.6524 ± 0.0417 | 9.9520 ± 0.4965 | 46.1830 ± 0.0298 |
| MRPC | 42.3556 ± 0.0539 | 4.4576 ± 0.3394 | 46.2067 ± 0.0953 |

Calibration selections reproduce exactly: RTE beta=7.5, error 4.0966%; MRPC
beta=27, error 2.2819%. Confirmation errors must not inherit calibration status:

| Task | Seed 17 | Seed 42 | Seed 123 | Pairs within 5% |
|---|---:|---:|---:|---:|
| RTE | 7.9762% | 5.2216% | 7.5068% | 0/3 |
| MRPC | 5.5351% | 6.6889% | 0.0924% | 1/3 |

The source draft said three pairs exceeded tolerance; **five of six do**.
Its row errors were sorted as 17/42/123 while its caption listed 42/17/123.
Both are corrected. Errors are now explicit in the caption, in ascending seed
order, allowing readable condition labels and a larger main-table font.

The NORM calibration frontier spans rho=0.0211657–0.2765731 and cross=42.4128–
50.9134% (rounded range checked from the full data); the old prose's 0.36 norm
maximum and 43% lower cross bound were inaccurate. The figure no longer says
all points lie on one flat curve or have exactly the same magnitude.

## Paired differences and uncertainty

MIX minus NORM, percentage points:

| Task / outcome | Mean ± sample SD | Nominal paired t 95% interval |
|---|---:|---:|
| RTE cross | −36.2311 ± 0.5155 | [−37.5118, −34.9504] |
| MRPC cross | −41.7491 ± 0.2468 | [−42.3622, −41.1359] |
| RTE accuracy | −2.0750 ± 2.7337 | [−8.8658, 4.7159] |
| MRPC accuracy | −0.2725 ± 0.6243 | [−1.8234, 1.2784] |

Main text retains mean ± SD. The appendix gives all three seed differences for
all three pairwise arm contrasts, plus nominal paired Student-t intervals (df=2).
These assume independent, approximately normal seed differences, are unadjusted
for multiple contrasts, and do not describe uncertainty across tasks or held-out
examples. No significance stars, equivalence test, or module pseudo-replication.

“Statistically indistinguishable across arms” was removed. In particular, RTE
NORM−UNREG accuracy is positive for every seed, mean +2.2758 points, with nominal
interval [0.1990, 4.3525]. This does not support a blanket no-difference statement
or a confirmatory superiority claim. MIX−NORM intervals are wide.

The snapshot also supports normalized absolute cross energy rho² × cross-share.
Ratios of seed-mean energy are NORM/MIX=4.9221 and UNREG/MIX=61.1116 on RTE;
11.2243 and 186.9814 on MRPC. These ratios were removed from the main prose to
keep its focus on the paired contrast and limitations. They remain in the JSON.

## Module norm profiles: pooled matching conceals large differences

Only 1/0/0 RTE modules and 4/1/2 MRPC modules (seeds 17/42/123) meet a per-module
5% norm tolerance. After dividing each arm's module norms by its own pooled norm,
the corresponding counts are 0/0/0 and 2/1/2. This descriptive rescaling removes
the residual global norm ratio; it is not another training intervention.

RTE MIX has 14/15/12 module-relative norms below 1e-6; NORM has none. MIX module
medians are 0.0228/0.0209/0.0126 versus NORM 0.0788/0.0764/0.0746. MIX's largest
relative norms occur in output projections, notably layers 7 and 11. MRPC is less
extreme but still heterogeneous. The threshold is an exploratory display summary,
not a predeclared experimental gate. Individual near-zero module ratios are not
interpreted as meaningful enormous effect sizes.

Equal-module cross means are MIX/NORM=14.9479/46.2512% (RTE) and 5.4836/46.2110%
(MRPC): suppression survives a change in module weighting. This does not establish
that each module is magnitude matched or isolate within-module directional change
from redistribution. Per-module pretrained squared norms and signed deltas are
absent, so relative-norm profiles cannot be presented as reconstructed absolute
layer-energy shares or initialization-corrected geometry.

## Abstract, missing evidence, and presentation decisions

The current CSV has **no learned-since-insertion columns**, contrary to the
handoff's schema description. Removed the unsupported “under five points” and
“insertion accounts for almost none” statements pending the corresponding export.
The current paper explicitly describes total-update fractions. The original
backbone reference CE=2.0956 is supplied evidence, not recomputed in this session;
arm-mean trained CE exceeds it by 0.4717–0.8403 nats. No preservation claim follows.

The separate agent's `abstract_draft.md` was not available locally or in the
Overleaf snapshot. Requested its text and the missing learned-delta export from
the author. Meanwhile, integrated an independently drafted abstract using the
verified results. Preferred matching sentence: “Doses matched pooled norms within
5% at calibration; confirmation errors were 0.1–8.0%, with five of six pairs
outside that tolerance.” Giving only the range would obscure the failed rule.

`BAND_PRESENTATION_SPEC_20260916.md` was published via MCP in commit `410637f`
before inspecting any BAND confirmation outcomes. It fixes row order, endpoint,
all paired contrasts, uncertainty, incomplete-cell rules, geometry diagnostics,
and a seed-point figure. It acknowledges the already disclosed tail pilot and
does not claim to be a pre-pilot study registration. It is presentation guidance,
not an amendment to running jobs. No band superiority or pilot finding enters
the abstract; RTE and MRPC remain separate, and fresh scores stay separate from
the legacy broad-GLUE table.

## Paper validation

The original `8729fe3` PDF ended its main text on page 11. Supporting cross/norm
and legacy allocation figures and archived asymmetry detail moved to the appendix;
the revised draft ends on page 9 and retains the six-task GLUE table on page 8.
All 60 historical means and SDs are preserved. The fresh confirmation table is
Table 2 and the legacy broad-GLUE table is now Table 3 by automatic numbering;
these remain separate datasets. The focused figure was regenerated in grayscale
to satisfy the existing PDF audit, preserving every plotted observation.

Required checks: mixing-table regeneration, spectral algebra, review additions,
manuscript audit, focused-table check, independent focused JSON check, TeX compile,
and PDF audit. The rendered main confirmation table and appendix tables/heatmap
were inspected. The matching build log has no unresolved references, overfull
boxes or missing glyphs; normal underfull and Tectonic bibliography-rerun notices
remain. No new training, checkpoint reload, official held-out evaluation, or
band-result validation is claimed by these checks.
