# Section 4 review — Math: adaptation & retention

Reviewer: section-validator (sec4). Date: 2026-07-10.
Sources recomputed: `results/frm_*/summary.json` (55 cells), `paper/writing/data/key_numbers.md`
(§4 + [ADDED 2026-07-10] frm_ block), CLoRA PDF (`repro/CLoRA/Controlled Low-Rank Adaptation...pdf`,
Tables 3/4/6/7/8), MiLoRA PDF (`repro/MiLoRA/MiLoRA Harnessing...pdf`, Tables 2/8/9/10),
MiLoRA repo `scripts/run_train.sh`, `results/train_registry.jsonl`.

## (a) Verdict table — every bar in both charts

### Chart 1: GSM8K published-vs-ours

| Claim / bar | Artifact | Recomputed / source | Verdict |
|---|---|---|---|
| LoRA+wd (wd0.3) ours | 67.3, F_Δ 0.28, BBH 33.1 | 67.25 / 0.278 / 33.10 (`frm_lorawd_wd0p3_lr2e4_c256_s42`) | CONFIRMED |
| CLoRA-k128 published | 64.6 | 64.59 (CLoRA Table 3; their math best — k64 64.29, k256 63.45) | CONFIRMED |
| MiLoRA published | 63.5 | 63.53 (CLoRA T3) = 63.5 (MiLoRA's own Table 2) — cross-confirmed | CONFIRMED |
| LoRA published | 60.6 | 60.58 (CLoRA T3) = 60.6 (MiLoRA T2) | CONFIRMED |
| CLoRA-k128 our pipeline | 59.6 | 59.59 (`frm_clora_k128_lr3e4_c256_s42`) | CONFIRMED |
| PiSSA published | 58.2 | 58.23 (CLoRA T3) = 58.2 (MiLoRA T2); ours 49.66 ("collapses") | CONFIRMED |
| "our in-pipeline LoRA reproduces the published LoRA (60.2 vs 60.6), so the pipeline is sound" | — | 60.20 vs 60.58 — true for GSM8K. **Does NOT transfer to MATH** (see (e)): our LoRA MATH 13.56 vs published 16.88 | CONFIRMED but incomplete — scope to GSM8K |
| "same 256-token setting as the competitors" (lead) | — | True for in-pipeline bars only. Published recipes: CLoRA paper states NO cutoff (Table 6 silent); MiLoRA's own math script uses `model_max_length 2048` and α=r (64), vs our α=2r (128, consortium ruling). Published bars differ in recipe, not just harness | CORRECTED — reword: "all in-pipeline cells matched at 256 tokens" (the caption already says this correctly); delete "as the competitors" from the lead |
| Bar widths | — | all consistent with 0–75 axis (67.25/75=89.7% etc.) | CONFIRMED |

### Chart 2: BBH retention

| Claim / bar | Artifact | Recomputed | Verdict |
|---|---|---|---|
| Base reference | 33.1 | 33.10 BBH-AO (key_numbers §, `[EXTERNAL: h00#6, h05]`) | CONFIRMED |
| LoRA+wd wd0.3 lr2e4 | 33.1 / GSM8K 67.3 | 33.10 / 67.25 — exactly at base | CONFIRMED |
| MiLoRA lr3e4 | 30.2 / GSM8K 59.0 | 30.18 / 58.98 | CONFIRMED |
| LoRA lr3e4 | 29.1 / GSM8K 60.2 | 29.14 / 60.20 | CONFIRMED |
| CLoRA-k128 lr3e4 | 27.6 / GSM8K 59.6 | 27.55 / 59.59 | CONFIRMED |
| PiSSA lr3e4 | 7.2 "collapses" | 7.23 (mmlu_pro 0.0, gsm8k 49.66) | CONFIRMED |
| Matched-LR control: "at 3e-4 for everyone, LoRA+wd still leads (65.05, 33.47)" | — | `frm_lorawd_wd0p3_lr3e4_c256_s42`: gsm8k 65.05, bbh 33.47 | CONFIRMED |
| "MiLoRA's own best-retention cell (lr 1e-4) 32.38 at 62.85" | — | 32.38 / 62.85; and it IS the best MiLoRA cell — lr7e-4 and 1e-3 collapsed to 0.0 | CONFIRMED |
| "F_Δ 0.28 against ~1.1–2.2 for the rest" | — | CLoRA-k128 1.079, MiLoRA 1.257, LoRA 1.283, PiSSA 2.206 | CONFIRMED (1.08 → "~1.1" is at the rounding edge) |
| "s43 sibling is 65.88 — still edging 64.6" | — | 65.88 (BBH 34.77) | CONFIRMED |
| "3–6 pp below base" for competitors | — | base−{30.18,29.14,27.55} = 2.9/4.0/5.6 pp | CONFIRMED (2.9 rounds to "3") |

### Claim (c): "no published retention numbers exist for math models"

| Sub-claim | Verdict |
|---|---|
| CLoRA Table 3 reports only GSM8K + MATH accuracy; 18.38 is MATH-benchmark, not BBH | CONFIRMED (verified in PDF) |
| CLoRA's BBH/F↑ numbers (Table 4, appendix Tables 7/8) are commonsense-trained models only | CONFIRMED — Table 4 contains k512/k1024/k2048, which exist only in the CS setting (math swept k∈{64,128,256}); appendix T7/T8 are explicitly CS |
| CLoRA math table has no weight-decay baseline | CONFIRMED — LoRA-L2 appears in CS Tables 2/4 only, absent from Table 3 |
| "no out-of-domain / retention numbers **for any math-trained model**" | **CORRECTED — overbroad as written.** MiLoRA Table 8 publishes a forgetting measurement on math-trained LLaMA-2-7B (CE-to-base on WikiText-103: LoRA 3.24, PiSSA 6.07, MiLoRA 2.54) — the very numbers §6 of this artifact reproduces. Airtight version: "no *accuracy-based* out-of-domain retention numbers (BBH/MMLU-style) exist for any math-trained model; MiLoRA reports only a perplexity-style CE metric (its Table 8), which we reproduce in §6." As written, §4 contradicts §6 |

### Claim (d): cross-harness "edges/matches, not beats"

CONFIRMED — consistently applied: hero tile "edges" (l.98), §4 lead "edges" (l.216), explicit
disclaimer (l.239). The only "beats" instances are about curve fits (l.141) and inside the
disclaimer itself. One residual gap: the comparison is cross-**recipe** as well as cross-harness
(MiLoRA published at α=r & 2048 tokens; ours α=2r & 256) — the disclaimer mentions only
"evaluation code". Add "recipe (α, token cutoff) and" to the disclaimer sentence.

## (b) Seed-cell status (as of 2026-07-10)

**None of the 3 cells has landed.**

- `frm_lorawd_wd0p3_lr2e4_c256_s44` — training FINISHED 2026-07-10T01:01 (+03:00, registry;
  33,197 s runtime), **eval not run / no summary.json**. Cheapest possible strengthening: eval only.
- `frm_lorawd_wd0_lr1e4_c256_s43` / `_s44` — **not in train_registry at all** (not started).

Spreads available now:
- Headline cell wd0.3/lr2e-4: s42 67.25 (BBH 33.10), s43 65.88 (BBH 34.77) → mean 66.57, GSM8K
  spread 1.37 pp. Seed-mean still edges published 64.6 by ~2.0 pp.
- Cautionary existing triple wd0.2/lr1e-4: s42 67.40, s43 64.52, s44 64.29 → mean 65.40,
  spread 3.11 pp — **s42 is a +2.9 pp lucky seed at that cell, and every headline in §4 is s42.**
  Also note: 67.40 (wd0.2/lr1e-4/s42) > the 67.25 headline — the artifact's "best" cell is not the
  grid max at c256. The seed-mean defense works (66.57 n=2 vs 65.40 n=3, wd0.3/lr2e-4 wins on
  means), but it must be made explicit or a reviewer will call cherry-picking.

## (e) New insights from unused data

1. **MATH benchmark (headline.math) — must be disclosed.** Our c256 MATH scores (13.5–15.3)
   sit 2–3.5 pp BELOW every published MATH number (LoRA 16.9, MiLoRA 17.8, CLoRA-k128 18.38),
   even though GSM8K calibrates. Cause is identifiable in our own data: 9 paired c256↔c512 cells
   show MATH gains of +1.5 to +3.2 pp at c512 (e.g. wd0.1/lr3e-4: 15.02→18.24, reaching published
   range) at zero BBH cost — MATH solutions are longer and the 256 cutoff truncates them. Either
   add a MATH row with this caveat or state why MATH is omitted; a hostile reviewer will pull
   CLoRA Table 3 and run this comparison in five minutes.
2. **c512 is a strictly better headline waiting in the data:** `frm_lorawd_wd0p3_lr2e4_c512_s42`
   = GSM8K **69.52** / MATH 16.44 / BBH 33.57 (above base) / F_Δ 0.282. That is +4.9 pp over the
   published best, at a cutoff still shorter than MiLoRA's published 2048. Fair use requires either
   (i) framing as ours-vs-published only, or (ii) matched c512 competitor cells (see priorities).
3. **Multi-metric retention row — yes, add it.** On the frm_ math cells (n=49 with finite F_Δ) the
   law reproduces on every broad-knowledge metric: BBH Spearman −0.87 (Pearson on log F_Δ −0.92),
   MMLU −0.87 (−0.68), ARC-C −0.96 (−0.81). The mandate's r≈−0.66/−0.80 are the log-F_Δ Pearson
   values — label them as such; raw-x Pearson is misleadingly weak (−0.28/−0.36) because F_Δ spans
   decades with PiSSA/DoRA leverage. Caveat for the row: **TruthfulQA moves the opposite way**
   (Spearman +0.58 with F_Δ) — the known inverse-capability artifact; exclude it with a footnote.
   This row defuses the "you dropped MMLU-Pro and kept a single metric" objection cheaply.
4. Registry hygiene: frm_ math rows carry `task: "commonsense_170k"` while `data_path` is
   metamathqa_395k — vestigial label, worth fixing before data release.

## Prioritized strengthening list

1. **Finish the seed story (highest value/cost ratio).** Eval the already-trained
   wd0.3/lr2e-4/s44 (~1–2 GPU-h, eval only); launch wd0/lr1e-4 s43+s44 (~9.5 h train + eval each,
   fits one B200-day). Then headline the seed MEAN with range: "66.6 ± (n=3) vs 64.6 published,"
   and disclose the wd0.2/lr1e-4 seed triple (spread 3.1 pp) as the honesty anchor. This converts
   the section's weakest sentence ("single-seed point") into its strongest.
2. **Add the MATH column + cutoff analysis (zero new compute).** One extra bar-pair or table
   column: ours-c256, ours-c512, published; caption explains the truncation mechanism using the
   9 paired cells. Simultaneously upgrades honesty and shows the 69.5 GSM8K / 18.2 MATH c512 cells.
3. **Multi-metric retention strip (zero new compute).** Small row under the BBH chart: Spearman of
   {BBH, MMLU, ARC-C} vs F_Δ across the 49 math cells (−0.87/−0.87/−0.96), TruthfulQA footnoted.
4. **MiLoRA α=r control (~10 GPU-h).** One cell: MiLoRA, α=64, lr3e-4, c256, s42 — kills the "you
   ran MiLoRA at twice its published α, inflating its update" objection to the retention chart.
   If its BBH still tracks its F_Δ on the law curve, the section gets stronger, not weaker.
5. **c512 competitor pair (~20 GPU-h, optional).** CLoRA-k128 + MiLoRA at c512/lr3e-4/s42, enabling
   an in-pipeline matched-cutoff claim for the 69.5 headline.
6. **Three wording fixes (zero compute):** (i) scope the "no retention numbers" claim to
   accuracy-based metrics (MiLoRA Table 8 exists and §6 uses it); (ii) replace "same 256-token
   setting as the competitors" with "all in-pipeline cells matched at 256 tokens; published
   recipes' cutoffs are longer or unspecified (MiLoRA 2048)"; (iii) extend the cross-harness
   disclaimer to "cross-harness and cross-recipe (α, cutoff)".
