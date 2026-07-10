# Section review — §8 "External review" + footer + whole-document coherence

Reviewer: section-validator. Date: 2026-07-10.
Artifact: `paper/writing/artifact_status_report.html` (timestamp 2026-07-10 00:52 IDT version).
Ground truth: `artifact_review_round_final.md`, `artifact_number_audit_final.md`,
`artifact_feedback_round2.md`, `fleet_findings.md`, `data/key_numbers.md`, current artifact text.

## Verdict

**§8 is numerically accurate but framed in a way that risks credibility ("External review" for
internal AI-agent passes), and its caption "All surviving findings were applied" is not literally
true. Footer: both lines verified correct. Whole document: coherent and strong, with one genuine
internal contradiction (the "ceiling" sentence) and a handful of small cross-section frictions.**

---

## (a) §8 claim-by-claim verification

| # | §8 claim | Verdict | Evidence |
|---|---|---|---|
| 1 | "pressure-tested by two independent reviewers … adversarial critic and a numeric auditor" | CONFIRMED (undersold) | Both docs exist (`artifact_review_round_final.md`, `artifact_number_audit_final.md`). A **third** pass (`artifact_feedback_round2.md`, verification of the 6 PI comments) also exists and its findings WERE integrated (§4 LR-asymmetry disclosure, §6 caption fix, §2 residual footnote, key_numbers §4 reconciliation) — §8 doesn't mention it. Not an error; a missed strengthening. |
| 2 | "7 pass · 5 minor · 2 major" | CONFIRMED | Review tally (line 51): PASS = 7 (B,D,F,G,H,K,M) · WEAK = 5 (C,E,I,J,N) · FAIL = 2 (A,L). §8 relabels WEAK→"minor", FAIL→"major" — defensible mapping. |
| 3 | "both majors fixed" | CONFIRMED | FAIL A (pending/not-yet/"honest boundary" language): grep of current artifact finds none — CorDA footnote now "not calibration-matched … reported only as a geometry fingerprint"; §3 callout retitled "Boundary condition". FAIL L: dek replaced with the reviewer's exact suggested sentence; seed disclosure added (gloss "seeds" entry, §3 footnote, §4 s43=65.88); Qwen qualified ("LoRA, commonsense: r = −0.88"). |
| 4 | "verdict *sound paper basis*" | CONFIRMED w/ nuance | Review's literal verdict was "**Yes, conditionally**" — "Do that [seed disclosure], remove the one 'pending' phrase, and fix the dek, and this is a sound paper basis." All three conditions were met, so quoting the verdict is fair; "conditionally — conditions met" would be airtight. |
| 5 | "140 / 143 reproduced exactly; 3 corrected" | CONFIRMED counts, wording overbroad | Audit summary: PASS 140 / DISCREPANCY 3 of 143 rows — exact match. But ~7 of the 140 are `[EXTERNAL]`/audit-sourced/qualitative, not recomputes (#8 wd-flag, #115 ~55 GB "not verifiable", #123 +22 GB "not verifiable", #124–125 CorDA paper facts, #135, #138–139 handoff-sourced). "Reproduced exactly" should read "verified (140/143); every recomputable number reproduces to the decimals shown." All 3 corrections verified applied: DoRA 2.13→**2.14×** (15504.7/7229.9 = 2.145), CLoRA 1.14→**1.17×** (8433.4/7229.9 = 1.167), adapter count reconciled (byline "9 (8 swept on commonsense + PiSSA on math)" now consistent with §0's nine). |
| 6 | Caption: "All surviving findings were applied" | **CORRECTED — overstated** | Not applied: FIX 7's overlay **figure** (only the "parallel law" wording landed), FIX 9 (per-layer drift inset / Qwen geometry row — still asserted, not shown), FIX 10 (CorDA appears in the LR panel + legend with no "context only" note while the tile says "7 assessed"), and FIX 8 only partially (see coherence #1). Soften to "all blocking and numeric findings were applied" or apply the stragglers. |
| 7 | Integrated-fixes list (framing, seeds, DoRA 2.14×, CLoRA 1.17×, SC-LoRA 512, counts) | CONFIRMED, one footnote needed | SC-LoRA **512** is right per `fleet_findings.md` line 143 ("SC-LoRA = 512 calib forwards (D+ 256 + D− 256) + eigh; LoRA-Null/CorDA = 256") — but the numeric auditor's row #122 explicitly ruled **256** correct (reading the per-dataset registry arg `sclora_calib_size=256`). The two cited reviewers disagree and §8 says "all verified against source" without noting the adjudication (512 total = 256/dataset × 2 datasets). One clause resolves it. |
| 8 | "Every number reproduces to the decimals shown" | MOSTLY CONFIRMED | True for the recomputable set (incl. the post-audit soft-flag fixes: knee now "≈ 0.37", rank partial now "≈ −0.56, −0.5 to −0.6 across measures"). Overbroad for the analytical memory figures (~55 GB, +22 GB — which §5 itself flags as not instrumented) and the handoff-sourced §7 values (32.96, Δ6e-9). Say "every recomputable number". |

## (b) Is "External review" credible as framed? — **No; reframe, keep the content.**

The reviewers are internal AI-agent passes (adversarial-critic and data-verifier subagents), not
external humans. To a supervisor, "External review" implies outside reviewers; one question
("reviewed by whom?") turns a genuine strength into a credibility hit. The content is genuinely
valuable — a claim-by-claim adversarial critique plus an independent 143-row recomputation from the
raw result files is more verification than most submissions get. Reframe as:

- Retitle: **"Independent verification"** or **"Adversarial audit & independent recomputation"** —
  drop "External".
- One honest sentence: "Before finalization the report went through two independent automated
  review passes against the raw result files (`results/*/summary.json`): an adversarial
  claim-by-claim critique under an area-chair rubric, and a full numeric recomputation."
- Consider merging §7 + §8 into a single "Correctness & verification" section — §7 is
  pre-registered checks, §8 is post-hoc recomputation; together they are one coherent
  "pre-registered checks + independent recomputation" story, which is the strongest honest framing.
- "integrated before publication" → "integrated before this report was finalized" (it is a status
  report, not a publication).

## (c) Footer verification

**Evidence-base line — CONFIRMED, item by item:** commonsense fair sweep 8 adapters × 7 LRs = 56
`lrsw_` runs (audit #141 PASS; note the LR *panel* plots only 55 points — CorDA shows 6 of 7 LRs —
worth a silent fix or ignore); rank/wd grids = `mtx_` 3-seed matrix ✓; faithful math reproduction =
`frm_` block, now reconciled into `key_numbers.md` (lines 126–131: frm_ supersedes the old 50.6) ✓;
Qwen2.5-7B replication ✓ (but unqualified here — tile/§1 correctly say LoRA-arm/CS-only; suggest
"Qwen2.5-7B replication (LoRA, CS)" for consistency); geometry battery 320 saved adapters ✓ (audit
#31/#142); efficiency audit ✓ (§5); CE-to-base ✓ (§6); 8 ports audited ✓ (audit #135 — 8 is right:
LoRA+wd is not a port, so 9 methods / 8 ports is internally consistent).

**Metrics line — CONFIRMED:** retention = BBH+MMLU-Pro base 26.0 (recomputed 26.03), math BBH-only
base 33.1 (33.10) — audit #143 PASS; F_Δ definition matches `key_numbers.md` §0 and the gloss;
published-vs-in-pipeline separation is honored throughout (§4 explicitly colors published vs
pipeline bars).

## (d) Whole-document coherence pass

Cross-section inconsistencies found (ordered by severity):

1. **The "ceiling" contradiction (the one real error).** Gloss (§glossary, line 104): retention
   "can nudge slightly above" the base reference under heavy regularization. §1 callout (line 141):
   "Retention **cannot exceed** the base model's ceiling (26.0)". The data sides with the gloss
   (`mtx_lorawd` retention 26.6–27.9 > 26.0). Fix the callout: "Retention **saturates near** the
   base-model reference (26.0), so the left of the curve flattens." Residual "ceiling" wording also
   survives at §4 lead ("base BBH ceiling"), §4 bar label ("retention ceiling"), §7 row
   ("Base-model retention ceiling") — FIX 8 was applied in the gloss/§3 but not swept through.
2. **8 lines vs "7 assessed".** Tile 1 and the DW-panel annotation say 7 adapters; the LR panel and
   legend show 8 (CorDA included, no annotation). The critic's FIX 10 asked for "(shown for
   context; excluded from the law)" in the legend — still missing.
3. **Two within-method correlation ranges.** §1: "r −0.86 to −0.97" (lrsw pool, 7 adapters); §2:
   "r −0.75 to −0.94 across the battery" (320-run battery). Both verified correct, but nothing
   tells a cold reader these are different pools — half a sentence in §2 ("wider grid, hence the
   wider range than §1's sweep-only −0.86…−0.97") removes the apparent conflict.
4. **§8 caption vs reality** — "All surviving findings were applied" (see (a) #6).
5. **§0 "All nine methods share the same form ΔW = A·B"** vs the DoRA description two paragraphs
   later (magnitude × direction decomposition). The critic's nit F, still present. "Eight of the
   nine share… DoRA additionally re-parametrizes magnitude" fixes it.
6. Minor: §6 note "F_Δ here is on the α=2r math scale — about 2× the α=r commonsense values in §3"
   is good, but §4 quotes math-scale F_Δ (0.28) with no such note; a reader comparing §4's 0.28 to
   §3's 0.39 may wrongly conclude the math update is smaller than the CS one. Move/duplicate the
   scale note into §4.

**Boring/filler check:** no section is filler. §0 earns its place; §5's caption is long but every
sentence answers a PI comment. The densest patch is §2's three metric-definition paragraphs
(~600 words before any result) — trimmable by ~30% without losing a definition. §8 as currently
framed is the weakest-value section per word; the reframe in (b) fixes that.

**Tone:** consistent, constructive, inside the PI guardrails throughout; no "geometry doesn't
matter", no accusations; boundaries (SC-LoRA −4.15/−5.7 dual-fit footnote, high-k CLoRA callout,
Qwen math absent from claims) all present.

## The ONE add (recommended over any cut)

**Add the cross-literature overlay figure** — our 49 (F_Δ, retention) sweep points with the fitted
line (r = −0.86, slope −14.8) overlaid with CLoRA Table 4's 10 published (F_Δ, BBH) points and
their fit (r = −0.98, slope −14.7), annotated "two pipelines, same slope; parallel offset ≈ 2× in
F_Δ level". This is the campaign's single strongest external validation and the PI explicitly asked
for it (comment E, graded WEAK partly for its absence); it currently lives as a text-only callout.
As an inline SVG next to the §1 panels it becomes the document's signature visual. Cost: data and
fits already exist in `fig_cross_literature.py`; ~1 h to render in the artifact's plot style.

(If a cut is wanted instead: merge §7 into §8 under "Correctness & verification" and retitle away
from "External review" — one section, pre-registered checks + independent recomputation.)
