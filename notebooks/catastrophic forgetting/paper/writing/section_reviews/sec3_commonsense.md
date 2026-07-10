# Section 3 review — "Commonsense — adaptation & retention" (artifact_status_report.html)

Reviewer: section-validator (sec3), 2026-07-10.
Data: `results/lrsw_*_s42/summary.json` (7 series x 7 LRs), `results/mtx_*_s4{2,3,4}`,
`results/base_l2-7b*/retention_agg.json`, CLoRA PDF (`repro/CLoRA/`), key_numbers §0/§3/§14.
Recompute scripts: scratchpad `sec3_recompute.py`, `mtx_seed.py`, `sec3_bands.py`.

## (a) Verdict table — claim by claim

| # | Claim (artifact §3) | Recomputed | Verdict |
|---|---|---|---|
| 1 | Base retention 26.0, F_Δ 0 | BBH-AO 33.10, MMLU-Pro 18.96 → (33.10+18.96)/2 = 26.03 | CONFIRMED |
| 2 | LoRA+wd: 5e-4 / 81.6 / 25.6 / 0.39 / 6/7 | 81.62 / 25.55 / 0.394 / 6/7 | CONFIRMED |
| 3 | SC-LoRA: 5e-5 / 80.1 / 22.5 / 0.56 / 1/7 | 80.14 / 22.47 / 0.559 / 1/7 | CONFIRMED |
| 4 | MiLoRA: 3e-4 / 79.9 / 24.7 / 0.54 / 5/7 | 79.86 / 24.72 / 0.543 / 5/7 | CONFIRMED |
| 5 | LoRA (plain): 3e-4 / 79.1 / 24.4 / 0.62 / 5/7 | 79.11 / 24.42 / 0.623 / 5/7 | CONFIRMED |
| 6 | LoRA-Null: 5e-4 / 78.9 / 23.6 / 0.70 / 5/7 | 78.93 / 23.64 / 0.696 / 5/7 | CONFIRMED (= key_numbers §14) |
| 7 | CLoRA: 5e-4 / 78.4 / 21.9 / 0.64 / 5/7 | 78.36 / 21.88 / 0.643 / 5/7 | CONFIRMED |
| 8 | DoRA: 2e-4 / 78.3 / 24.8 / 0.45 / 4/7 | 78.27 / 24.84 / 0.445 / 4/7 | CONFIRMED |
| 9 | "tight 78–82 accuracy band" | best-LR CS range 78.27–81.62 | CONFIRMED |
| 10 | "highest mean retention across all rates" (LoRA+wd) | mean over 7 LRs: wd 26.30; next best LoRA-Null 24.59; MiLoRA 24.27; CLoRA 24.12; LoRA 23.98; DoRA 23.21; SC-LoRA 12.93 | CONFIRMED (large margin, +1.7pp) |
| 11 | Caption: 7 LRs swept, 2e-5 … 1e-3 | 2e-5,5e-5,1e-4,2e-4,3e-4,5e-4,1e-3 for every series | CONFIRMED |
| 12 | Footnote: MiLoRA cs 79.9→58.7→34.6 across seeds | mtx_milora_r32 s42/43/44: 79.86/58.72/34.62 | CONFIRMED |
| 13 | Footnote: "retention is stable (seed-SD ≈ 0.3)" | median ret seed-SD across the 34 3-seed mtx cells = **0.44** (mean 1.16). ≈0.3 holds only for low-F_Δ / on-law cells (wd0p3 0.09, dora_r16 0.10, milora_r16 0.12 … lora_r16 0.36, milora_r32 0.48). **SC-LoRA cells: 2.0–4.8 pp**; corda_r64 2.5; wd0p05 4.8 (one collapsed run) | **CORRECTED** — say "≤0.5 for on-law cells; retention SD grows with F_Δ (steep part of the law); SC-LoRA retention itself seed-fragile (SD 2–5 pp)" |
| 14 | Callout: CLoRA published k1024/k2048 = 82.6/83.7 CS, BBH 36.5/38.7 | CLoRA Table 2: 82.6 / 83.7; BBH 36.49 / 38.67 | CONFIRMED |
| 15 | Callout: "their Table 4 shows k2048 at the smallest F_Δ in their study" | Table 4 F_Δ: LoRA 0.79, MiLoRA 0.92, LoRA-r16 1.03, LoRA-r8 0.95, LoRA-L2 0.29, k128 0.36, k256 0.34, k512 0.27, k1024 0.21, **k2048 0.14** | CONFIRMED |
| 16 | Callout: "k2048 even exceeds the base model out-of-domain" | their reference F 34.91 vs k2048 38.67 | CONFIRMED (in *their* harness; see (d)) |
| 17 | "essentially no measurable forgetting" (wd 25.6 vs 26.0) | Δ=0.45pp < matrix seed-SD ~0.5 | CONFIRMED as worded ("within noise") |

Every printed number reproduces exactly at the displayed precision. The one correction is the
seed-SD footnote (#13).

## (b) Seed-3 error-bar status (checked 2026-07-10)

**No `lrsw_*_s43/s44` eval summaries exist yet.** Dispatch locks are placed for exactly the 8
error-bar cells (`results/dispatch_locks/`): clora_k1024_lr5e4_s43, dora_r16_lr2e4_s43+s44,
lora_null_r16_lr5e4_s43, lora_r16_lr3e4_s43, lorawd_wd0p3_lr5e4_s43, milora_r32_lr3e4_s43,
sclora_r32_lr5e5_s43. Only **lrsw_dora_r16_lr2e4_s43** has finished training
(adapter saved in /scratch/cf_models, train_runtime 15609 s); none are evaluated.

**However, two §3 operating points already have 3 seeds** via the mtx matrix (identical config,
lr = 3e-4, verified in train_registry):

| Operating point | CS acc (mean ± SD, seeds) | Retention (mean ± SD) | F_Δ (mean ± SD) |
|---|---|---|---|
| **MiLoRA r32 @ 3e-4 (its best LR)** | **57.7 ± 22.6** (79.9 / 58.7 / 34.6) | **24.19 ± 0.48** | 0.556 ± 0.011 |
| **LoRA r16 @ 3e-4 (its best LR)** | **79.0 ± 1.5** (80.2 / 79.5 / 77.3) | **24.32 ± 0.36** | 0.612 ± 0.010 |

These can go into the table *today* as the first mean±SD rows. Two caveats to disclose:
(i) the MiLoRA row's printed 79.9 is the **best of three seeds** — its #3 accuracy rank is a
seed-42 artifact (the footnote already half-says this; the table should carry the ±);
(ii) mtx_lora_r16_s42 (cs 80.2) is an independent retrain of the *same seed/config* as
lrsw_lora_r16_lr3e4_s42 (79.1) → **run-to-run nondeterminism ≈ 1 pp** on cs at fixed seed. Worse
for DoRA: lrsw dora@3e-4 s42 = 64.4 vs mtx_dora_r16_s42 (same config/seed) = 79.4 — a 15 pp
same-seed rerun gap. Error bars from seeds alone understate run variance for the unstable methods.

## (c) Safe-band metric — definition & sensitivity

Definition is well-posed (count of 7 LRs with retention_core ≥ 24) and the caption's disclosure
that Retention is at best-LR while the band spans all LRs is good. Two findings:

**Threshold sensitivity** (recomputed at 23.5 / 24.0 / 24.5):

| series | ≥23.5 | ≥24.0 (published) | ≥24.5 |
|---|---|---|---|
| LoRA+wd | 6/7 | **6/7** | 6/7 |
| LoRA-Null | **6/7 (ties wd)** | 5/7 | 5/7 |
| MiLoRA | 5/7 | 5/7 | 5/7 |
| LoRA | 5/7 | 5/7 | 4/7 |
| CLoRA | 5/7 | 5/7 | 4/7 |
| DoRA | 5/7 | 4/7 | 4/7 |
| SC-LoRA | 1/7 | 1/7 | 1/7 |

The headline (LoRA+wd widest, SC-LoRA 1/7) is threshold-robust; LoRA+wd is *uniquely* widest at
24.0 and 24.5 and tied with LoRA-Null at 23.5. The middle ordering (DoRA vs CLoRA vs LoRA)
shuffles — do not rank those verbally.

**Definitional weakness (reviewer attack surface):** the band counts LRs where the model
*failed to adapt* as "safe". E.g. LoRA+wd@2e-5 (cs 23.5) and @2e-4 (cs 40.7) both count toward
6/7; LoRA-Null@3e-4 (cs 31.6) counts toward its 5/7. A run that learns nothing trivially
forgets nothing. A joint "useful band" (cs ≥ 70 **and** ret ≥ 24) gives: **LoRA+wd 3/7, LoRA 3/7,
MiLoRA 3/7, DoRA 1/7, LoRA-Null 1/7, SC-LoRA 1/7, CLoRA 0/7**. LoRA+wd stays tied-widest (and is
the only one whose 3 joint-safe LRs span 1e-4→5e-4 with cs ≥ 77 and ret ≥ 25.5), and CLoRA drops
to 0/7 — so disclosing the joint band *strengthens* the section while closing the hole. At
minimum, grey out non-adapted cells in the count or footnote it. Also temper the caption's
"mis-pick the learning rate by an order of magnitude and still not forget": for LoRA+wd the
*adapted* safe range is 1e-4→5e-4 (5x), the full 6/7 range includes non-adapted LRs.

## (d) CLoRA boundary callout — fairness both ways

Fair to CLoRA: yes — numbers are quoted correctly (Table 2/Table 4 verified against the PDF), and
crediting "adaptation-efficiency per unit of update" is supported by their own table (k512
F_Δ 0.27 / F 34.32 vs LoRA-L2 F_Δ 0.29 / F 32.93 — +1.4 pp at matched F_Δ).

**Not quite fair to us — two omissions:**
1. It compares their *published-harness* 82.6/83.7 against *our* 81.6 with no commensurability
   caveat, while §4 of the same artifact documents a ~5 pp harness gap for CLoRA-k128 on GSM8K
   (64.6 published vs 59.6 ours). One clause ("different harness; not directly commensurate with
   our 81.6") is owed.
2. We *have* in-pipeline high-k CS data that the callout ignores:
   - `clora_cs_k1024` (their recipe): cs 79.85, ret 24.82, F_Δ 0.457 — below LoRA+wd on **both**
     axes in our harness, and on our law.
   - `clora_cs_k2048`: cs **65.4** (adaptation collapse), ret 25.65, **BBH 34.05 > base 33.10** —
     partially confirms their above-base OOD claim, but at collapsed adaptation.
   - `mtx_clora_k2048` 3-seed: cs 79.8 / 80.0 / **23.0** — a 1-in-3 seed-collapse basin at high k;
     `mtx_clora_k1024`: 80.3 / 61.8 / 76.5 (SD 9.8). High-k CLoRA's accuracy is seed-fragile in
     our hands.
   Adding one sentence with these keeps the boundary honest in both directions and *supports* the
   thesis (high-k retention gain rides on smaller F_Δ, exactly as the law predicts; the un-replicated
   part is the accuracy, not the retention).

## (e) Unsurfaced per-task insights (from per_dataset / headline fields)

1. **MMLU-Pro drives forgetting first.** At the 7 assessed operating points, mean deficit vs base
   is **BBH −0.95 pp vs MMLU-Pro −3.24 pp**. Extreme case: LoRA-Null@5e-4 holds BBH *at* base
   (+0.12) while losing 4.90 pp of MMLU-Pro; SC-LoRA@5e-5 loses 1.11 BBH vs 6.01 MMLU-Pro. Deeper
   in the sweep MMLU-Pro hits literal 0.00 (SC-LoRA 5e-4) while answer-only BBH floors near ~25:
   generative/CoT format-following is the first casualty; likelihood-scored MC (and TruthfulQA,
   slope −0.5/decade, key_numbers §7) barely move. A small stacked BBH/MMLU-Pro deficit bar per
   op point would make "what forgetting actually is" concrete.
2. **The base "ceiling" is soft — small updates show positive transfer.** At low LR, 5 of 7
   series *exceed* base retention (LoRA+wd@5e-5 27.80, MiLoRA@5e-5 27.57, LoRA-Null@1e-4 27.42,
   LoRA@5e-5 27.29, DoRA@5e-5 27.30 vs base 26.03; up to +1.8 pp). One sentence turns an apparent
   anomaly into a selling point (CS training mildly improves held-out reasoning until the update
   grows) and matters for §1's censored/saturating-fit framing.
3. **SC-LoRA's op point is at a 4–10x smaller LR (5e-5) than every other method** — its init
   front-loads task fit, which is exactly why its band is structurally narrow. Worth one clause;
   currently the table leaves the odd best-LR unexplained.
4. **Same-seed rerun variance** (b, caveat ii) — 1 pp for LoRA, 15 pp for DoRA@3e-4 — should be
   mentioned wherever error bars are introduced.

## Prioritized strengthening list (with costs)

1. **Insert the two available mean±SD rows now** (LoRA@3e-4, MiLoRA@3e-4 from mtx; table above).
   Cost: zero — data exists. Then finish the 8 dispatched lrsw s43/s44 cells (7 need training
   ~4.5 GPU-h each + eval ~1–2 h; dora s43 needs eval only) → every row gets ±SD.
2. **Add the joint "useful band" (cs≥70 & ret≥24) as a second column or footnote** (numbers in
   (c)). Cost: zero. Closes the "non-adapted runs count as safe" attack; CLoRA 0/7 vs LoRA+wd 3/7
   actually sharpens the message.
3. **Per-benchmark deficit mini-figure** (BBH vs MMLU-Pro at op points, insight e1). Cost: ~1 h
   figure work, data exists.
4. **One-sentence balance in the CLoRA callout**: harness-commensurability caveat + our
   in-pipeline k1024/k2048 numbers and the 1-in-3 k2048 seed collapse. Cost: zero (text only).
5. **Fix the seed-SD footnote**: "≈0.3" → "≤0.5 for on-law cells; grows with F_Δ; SC-LoRA
   retention itself varies 2–5 pp across seeds". Cost: zero.
