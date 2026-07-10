# Section 6 review — "Forgetting, measured the field's own way" (CE-to-base)

Reviewer: section-validator (sec6). Date: 2026-07-10.
Sources recomputed: `forgetting_ce.py` + `ce_batch.py` (line-by-line), `results/forgetting.jsonl`
(6 rows) + `results/forgetting_chunk1.jsonl` (**52 new rows — the frm_ CE chunk, run and landed
during this review**, done-marker `results/ce_chunk1/summary.json`, log
`logs/ce_chunk1_manual.log`), `results/frm_*/summary.json`, MiLoRA PDF
(`repro/MiLoRA/MiLoRA Harnessing Minor Singular Components for.pdf`, §5.4 + Tables 8/10),
`results/train_registry.jsonl`. Artifact: `artifact_status_report.html` §6 (lines 265–285) and the
§7 "REPRODUCED" row (line 298).

## Overall verdict

**Publishable and now substantially stronger than written.** Every number in the section
recomputes exactly; the metric description matches the code token-for-token (soft CE, base as
target, forward-KL direction correct). One mislabel (the "two additional LoRA+wd points" are
wd=0 plain-LoRA cells) and one undisclosed knob (the wd0.5 row is at lr 1e-4) need fixing. The
big news: the full frm_ CE batch landed — **n grows 6 → 49 and Spearman(CE, F_Δ) rises to
+0.976 (p ≈ 1e-32)**, robust to every cut we tried, and CE-vs-BBH residuals give the section a
genuinely new second insight. The MiLoRA-2.54 disclaimer, previously an assertion, is now
**provable from MiLoRA's own Table 10** (they train MiLoRA/PiSSA at α=r, LoRA at α=2r).

## (a) Claim-by-claim verdict table

| # | Artifact claim | Verdict | Recomputed / evidence |
|---|---|---|---|
| 1 | Metric = soft CE, CE_t = −Σ_v p_base(v)·log p_ft(v), averaged over positions | **CONFIRMED** | `forgetting_ce.py:110-117`: `ce = -(p_base * logp_ft).sum(-1)`; direction exactly as MiLoRA §5.4 ("target … replaced by the distribution predicted by the pre-trained base model") = H(p_base) + KL(p_base‖p_ft), forward KL. Artifact formula and prose ("training loss with the hard target replaced by the base distribution") are correct. |
| 2 | "40 blocks × 1,023 positions = 40,920 scored positions" | **CONFIRMED (scope note)** | All rows: `n_positions=40920`, `n_blocks=40`, `max_length=1024`. Note not stated: 40 blocks = **12.1 % of the WikiText-103 test set** (330 full 1024-token blocks, 338,533 tokens; recomputed with the Llama-2 tokenizer). Same fixed slice for every adapter, so rankings/correlations are internally exact; only the *absolute* comparison to MiLoRA's (presumably full-set) numbers inherits slice noise. |
| 3 | Base = disable_adapter on the same wrapped model = "exactly the pretrained weights the update was added to" | **CONFIRMED, now instrumented** | `model.disable_adapter()` (line 108); residual-init methods saved rank-2r W0-relative (`residual_save.py`). This review's batch ran with `--check_base`: **max\|disable_adapter − fresh base\| = 0.000e+00** (log line). Worth quoting — it's a free correctness sentence. |
| 4 | Table row LoRA+wd wd0.5: F_Δ 0.20, CE 2.00 | **CONFIRMED (disclosure gap)** | `frm_lorawd_wd0p5_lr1e4_c256_s42`: fdelta 0.1959, CE 1.9983. **Not stated: this cell is lr 1e-4**, while the other three rows are lr 3e-4 — "heavier weight decay trades a little GSM8K (67→66)" conflates wd 0.3→0.5 with lr 2e-4→1e-4. Now moot: see #10. |
| 5 | LoRA (math, 3e-4) F 1.28 / CE 3.57; MiLoRA F 1.26 / CE 3.66; PiSSA F 2.21 / CE 6.31 | **CONFIRMED** | 1.2831/3.5702; 1.2568/3.6594; 2.2058/6.3068. GSM8K 67.25→"67", 66.03→"66" also check out. |
| 6 | "Rank correlation of CE with update size = 0.94 (n=6 … the four shown plus two additional LoRA+wd points)" | **CONFIRMED number, CORRECTED label; now SUPERSEDED** | Spearman = 0.9429 (p=0.0048), exact. But the two extra cells are `frm_lorawd_wd0_lr5e4/7e4` — **wd = 0, i.e. plain LoRA at higher LRs**, not "LoRA+wd points" (they're from the lorawd sweep family, decay zero). A checking reviewer will catch this. Supersede with n=49 (§b). |
| 7 | "Reproduces MiLoRA's published Table 8 for LoRA (3.24↔3.57) and PiSSA (6.07↔6.31)" | **CONFIRMED with required caveat** | Table 8 verified in the PDF (LoRA 3.24, PiSSA 6.07, MiLoRA 2.54). LoRA is a genuine same-recipe reproduction (their α=128=2r matches ours). **PiSSA is not same-recipe: their Table 10 sets α(PiSSA)=64=r; ours ran α=128=2r** (train registry: `lora_alpha: 128`). See §c for how to characterize tolerance. |
| 8 | "MiLoRA's own published 2.54 … measured at its lower-magnitude published operating point" | **CONFIRMED — and now provable** | MiLoRA Table 10 (math): rank 64, **α of LoRA = 128, α of PiSSA/MiLoRA = 64**, same LR 3e-4. Their own protocol gives MiLoRA *half* the update scaling of their LoRA baseline. This upgrades the disclaimer from assertion to citation. Bonus retro-diction (footnote-grade, see §d5). |
| 9 | "At matched LR and rank, MiLoRA (3.66) forgets essentially the same as plain LoRA (3.57)" | **CONFIRMED** | Matched lr 3e-4, r 64, α 128; realized F_Δ 1.26 vs 1.28. Δ CE = 0.089 (2.5 % relative), direction *against* a MiLoRA advantage. Noise scale now measurable: same-config twin runs (`frm_lora_lr3e4` vs `frm_lorawd_wd0_lr3e4`) differ by 0.015 nats; wd0p3_lr2e4 seeds s42/s43/s44 = 2.081/2.088/2.079 (±0.005). "Essentially the same" is fair. |
| 10 | Parenthetical: wd0.5 shown because the §4 headline is wd0.3 (F 0.28) | **CONFIRMED values; now RETIRABLE** | wd0.3 headline `frm_lorawd_wd0p3_lr2e4_c256_s42`: F 0.278, GSM8K 67.25. Its CE **is now measured: 2.081** (s43: 2.088). Replace the wd0.5 row with the actual §4 headline model and delete the whole disambiguation. |
| 11 | §7 row "CE-forgetting metric vs MiLoRA Table 8 … REPRODUCED" | **CONFIRMED with same caveat as #7** | Suggest the cell read "LoRA 3.57↔3.24 (same recipe); PiSSA 6.31↔6.07 (ours α=2r vs their α=r)". |
| 12 | Lead: "evaluated on text neutral to both the adaptation task and the retention suites" | **CONFIRMED** | WikiText-103 test; training = MetaMathQA; retention = BBH/MMLU-Pro. No overlap by construction. |
| 13 | Caption: "F_Δ here is on the α=2r math scale — about 2× the α=r commonsense values in §3" | **PLAUSIBLE, not re-derived** | Consistent with the α=2r consortium ruling; cross-section scale factor not recomputed here (belongs to §3's reviewer). |

## (b) The frm_ chunk landed — expanded table and statistics (n = 49)

Run during this review as the exact `ce_chunk1` dispatcher line (concurrency-safe; done-marker
written, so the queued chunk1 will skip-done cleanly). 52 new rows in
`results/forgetting_chunk1.jsonl`; 49 rows have both CE and fdelta. Excluded: 5 diverged lr 1e-3
cells (NaN adapters → NaN CE — note for the table-builder: these lines contain bare `NaN`
tokens, valid for Python's json but not strict JSON), and 3 rows scored for CE whose summary.json
(hence fdelta) hasn't landed yet (`frm_milora_lr2e4` CE 2.95, `frm_milora_lr5e4` CE 7.46,
`frm_lorawd_wd0p3_lr2e4_s44` CE 2.079) — n grows to 52 when those evals land.

**Headline statistics (all recomputed):**

| Statistic | Value |
|---|---|
| Spearman(CE, F_Δ), all n=49 | **+0.976** (p = 9.9e-33) |
| … excluding 2 degenerate cells (GSM8K ≤ 5) | +0.973 (n=47) |
| … also excluding F_Δ > 5 leverage points | +0.971 (n=46) |
| … non-wd cells only (competitors + plain LoRA) | +0.950 (n=16) |
| … **cross-method at matched lr 3e-4/c256** | **+0.962 (n=13)** |
| Pearson(CE, log10 F_Δ) | +0.943 |
| Per-family: LoRA+wd n=33 | +0.977; plain LoRA n=6: +0.943 |
| Power law (non-degenerate, F<5, n=46) | KL ≈ 1.09 · F_Δ^1.20 (log-log r = 0.965) |

The cross-method matched-LR subset is the cleanest new exhibit — one LR, one recipe, seven
method families, and CE still tracks F_Δ almost perfectly:

| Run (all lr 3e-4, c256, s42) | F_Δ | CE |
|---|---|---|
| LoRA+wd 0.5 | 0.23 | 2.01 |
| LoRA+wd 0.3 | 0.33 | 2.12 |
| LoRA+wd 0.2 | 0.42 | 2.25 |
| LoRA+wd 0.1 | 0.65 | 2.53 |
| CLoRA k256 | 1.02 | 3.24 |
| CLoRA k128 | 1.08 | 3.37 |
| CLoRA k64 | 1.11 | 3.42 |
| MiLoRA | 1.26 | 3.66 |
| LoRA | 1.28 | 3.57 |
| LoRA (wd0 twin) | 1.29 | 3.55 |
| PiSSA | 2.21 | 6.31 |
| DoRA | 2.84 | 4.06 |
| CorDA++ | 4.12 | 5.06 |

New method families now scored (none existed in the 6-row table): **CLoRA (3), DoRA (1),
CorDA++ (3)** — all on the CE-magnitude curve, with two instructive residuals (PiSSA above,
DoRA below; §d). MiLoRA's own LR sweep is monotone: CE 2.50 (1e-4) → 2.95 (2e-4) → 3.66
(3e-4) → 7.46 (5e-4) — its forgetting moves with its magnitude, not its geometry.

## (c) Is "reproduces their published values" honest at ~10 %? How to characterize tolerance

Yes — provided the artifact says *what was reproduced*. These are **not their checkpoints
re-scored**; they are (i) the recipe re-trained from scratch (different seed/data order/harness),
(ii) the metric re-implemented from a two-sentence description, (iii) a 12.1 % fixed slice of the
test set vs their (unstated, presumably full) evaluation, and (iv) for PiSSA, α=2r vs their α=r.
Under all four sources of variance, LoRA +0.33 nats (+10.2 %) and PiSSA +0.24 nats (+4.0 %) is
about as close as the pipeline could land. The stronger, checkable invariants DO reproduce
tightly: the ordering (PiSSA ≫ LoRA), and the relative gap — **PiSSA/LoRA CE ratio 1.77 (ours)
vs 1.87 (theirs)**. Recommended wording: "re-trained and re-scored from scratch, our
implementation reproduces the published ordering and values to within 0.24–0.33 nats (4–10 %),
and the PiSSA-to-LoRA ratio to within 6 %." Do not present a bare "REPRODUCED" without the
same-recipe (LoRA) vs matched-magnitude-protocol (PiSSA, α=2r vs α=r) distinction — a referee
with the MiLoRA paper open will find Table 10.

## (d) New insights (all recomputed, ready to use)

1. **CE adds signal beyond the magnitude law, and it agrees with the benchmarks.** Residuals of
   CE and of BBH from their respective log-F_Δ fits correlate at **r = −0.81 (p = 6e-12, n=47
   non-degenerate)** — where CE says a model drifted more than its update size predicts, BBH
   retention is below the law too. The two forgetting axes share their *exceptions*, not just
   their trend. This turns §6 from "third confirmation" into a cross-validation of the law's
   residual structure.
2. **PiSSA dissociates distributional drift from capability loss** — the one adapter whose CE
   disagrees with its benchmark retention. CE-matched comparison: PiSSA CE 6.31 → BBH **7.2**;
   plain LoRA at CE 6.92 → BBH **17.7**. At essentially the same next-token drift, PiSSA loses
   ~10 BBH points more. CE measures drift; benchmarks measure usable capability; PiSSA shows they
   can come apart (consistent with §4's "PiSSA collapses"). Honest boundary: worth one sentence,
   plus the standing caveat that collapsed models may also fail answer-format parsing.
3. **DoRA sits below the CE curve** (F_Δ 2.84 → CE 4.06, vs ~5.6 predicted): its
   magnitude+direction reparametrization makes recorded F_Δ overstate functional drift. A useful
   nuance for the geometry section — the law is about *functional* update size, and DoRA is the
   one family whose parameter-space F_Δ is a biased proxy.
4. **Noise floor now quantified**: seed triplet 2.081/2.088/2.079 (±0.005 nats), same-config
   twins Δ0.015 nats at F≈1.3. Makes "MiLoRA ≈ LoRA" (Δ0.089) interpretable at a glance.
5. **Retro-diction footnote (optional, flag as suggestive)**: at MiLoRA's published α=r operating
   point (≈half the update scale, F_Δ ≈ 0.63), our fitted CE-magnitude law predicts CE ≈ 2.48 —
   their published 2.54, within 2.5 %. Caveat: assumes F_Δ scales ~linearly with α (holds for
   LoRA-init dynamics, demonstrably not for PiSSA), so footnote, not headline.
6. **The floor is visible**: CE is bounded below by the base entropy H_base = 1.852 (identical to
   10 decimals across all 58 scored rows — a free internal-consistency check). LoRA+wd's 2.00 is
   0.15 nats above "zero drift". Add a "base model (no adapter): CE 1.85, KL 0" row to the table
   so readers see 2.00 for what it is.

## Top improvements, prioritized (cost estimates)

1. **Rewrite §6 around the n=49 result (no new compute — data landed today).** Replace "0.94,
   n=6" with Spearman +0.976 (n=49, p≈1e-32), add the 13-cell matched-LR cross-method table (or a
   CE-vs-F_Δ scatter colored by method — `fig_ce_vs_magnitude.py` exists), swap the wd0.5 row for
   the wd0.3 §4 headline model (CE 2.08), add the base-model floor row, fix the "two additional
   LoRA+wd points" label, and add the residual-correlation sentence (d1). Cost: writing only.
2. **Cite MiLoRA Table 10 for the 2.54 disclaimer** (α_MiLoRA = r = 64 vs α_LoRA = 2r = 128) and
   scope the PiSSA "reproduction" accordingly (§c wording). Cost: zero — strictly strengthens; it
   converts the section's only assertion into a citation. Optional gold-plating: train
   `frm_milora_alpha64_lr3e4_c256_s42` (one training cell, ~5 GPU-h + 6 s CE) — if it lands at
   CE ≈ 2.5 / F ≈ 0.6 it *reproduces their 2.54 on our own law line*, closing the loop entirely.
3. **Full-test-set CE for the headline rows** (`--max_blocks 0`, 330 blocks): ~45 s/adapter, so
   ~10 min for the 13 matched-LR cells or ~40 min for all 49 on one GPU. Removes the 12 %-slice
   asterisk from the MiLoRA comparison and tests slice stability (expect <0.05 nats movement).
   Do this before quoting absolute CE against Table 8 in the camera-ready.
4. **Seed error bars for the load-bearing pair**: CE on s43/s44 siblings of the lr3e-4 LoRA and
   MiLoRA cells (adapters exist? if not, 2 training cells ~10 GPU-h + seconds of CE). Turns
   "MiLoRA ≈ LoRA (Δ0.089)" into a mean±sd statement; the ±0.005–0.015 internal noise suggests
   the conclusion is safe but currently rests on one pair.
5. **CS-adapter CE (chunks 2–8, queued)**: when they land, check the CE-magnitude law transfers
   to the commonsense sweep (~250 adapters, ~35 min GPU total). If Spearman holds there too, §6
   stops being math-only and the caption's "math models are the right anchor" claim becomes a
   choice, not a limitation. Also enables the CE-vs-BBH residual test at n≈300.
