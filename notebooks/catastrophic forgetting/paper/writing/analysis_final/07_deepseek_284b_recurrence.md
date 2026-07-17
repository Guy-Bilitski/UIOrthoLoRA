# 07 — DEEPSEEK-284B ARM: WHAT SURVIVED, AND WHAT IT SHOWS (post-freeze addendum)

`[WRITTEN 2026-07-17 — addendum analyst. Sources:
results/geo_drift/adapter_metrics_deepseek.jsonl (21 rows),
results/dsv4_adapt_n1000_logscores.jsonl (20 rows),
results/geo_drift/adapter_metrics_merged.jsonl (7B reference),
results/ds_adapters_evac/ (raw adapters). Script: ds284b_recurrence_2026-07-17.py;
output: ds284b_recurrence_output_2026-07-17.txt. Framing per handoff/41 TODO-2:
recurrence + adaptation, explicitly NOT a retention slope. §18 unchanged.]`

## Ledger correction (additive to §18.3/§18.7)

§18 records "DSV4 284B — ABSENT (0/21 synced) … NO data in this repo." That was
true at freeze time for the *retention evals*, which remain lost. But the
evacuation salvage DID land two artifacts post-freeze:
**20/21 MedMCQA adapt scores** (from relaunch_0717.log, n=1000 each; missing:
dsv4_lorawd_r16_lr5e4_s42) and **21/21 factor-only geometry rows** (+ 21 permatrix
files). Raw adapters: 21/21 tar sets integrity-verified 2026-07-17
(ds_evac_verify_output_2026-07-17.txt; 63-part SHA256SUMS written). The arm is
designed-lost for retention/CE/F_Δ (GPU re-eval required) but is NOT data-free.

Grid: DeepSeek-V4-Flash (284B, MLA attn targets), MedMCQA, 7 methods × 3 seeds,
r16/α32, per-method fixed LR. One diverged run (dsv4_lora_null_r16_lr5e4_s44,
adapt 25.7 ≈ chance), flagged throughout.

## 1. GEOMETRY RECURRENCE — the method fingerprint survives a 40× scale jump

Spearman rank-correlation of the 7-method ordering, 284B (per-method mean of 3
seeds) vs 7B (per-method median within family):

| metric | lrsw | lrswm | qwsw | qwswm | frc | frm | POOLED |
|---|---|---|---|---|---|---|---|
| stable_rank | +0.61 | +0.54 | +0.79 | +0.71 | +0.50 | +0.43 | **+0.86** |
| eff_rank | +0.61 | +0.60 | +0.86 | +0.86 | +0.14 | +0.50 | **+0.75** |
| log spec_max | −0.21 | +0.20 | −0.64 | −0.68 | +0.18 | +0.04 | −0.43 |
| log fro_total | −0.86 | −0.14 | −0.64 | −0.39 | +0.14 | −0.43 | −0.79 |

**Reading.** The *shape* metrics — where a method spreads its update — recur:
stable-rank ordering is positive in all 6 families individually and +0.86 pooled.
The same two clusters found at 7B reproduce at 284B: {sclora, milora, lora_null}
high-spread (284B stable_rank 4.2–4.9) vs {lora, lorawd, dora, clora} concentrated
(1.6–1.8). Method identity determines update *shape* as an architectural signature
that transfers across base model (Llama/Qwen → DeepSeek), attention family
(MHA/GQA → MLA), scale (7B → 284B), and domain (CS/math → medical). The *scale*
metrics (spec, fro) do not recur — expected and disclosed: at 284B LR was fixed
per method, so scale ordering conflates method with LR. Do not quote spec/fro
recurrence numbers.

This strengthens, not weakens, the title claim: geometry is a stable *property of
the method* — yet at 7B, where retention exists, that stable property buys only
ΔR² +0.017 (06). Shape is real and portable; it is still second-order for
forgetting.

## 2. ADAPTATION — no magnitude ordering in the surviving regime

medmcqa_acc vs log10 fro_total (fro is the only surviving magnitude proxy;
F_Δ lost): primary n=19 (diverged excluded) pearson r=+0.20, spearman −0.10;
adapt range 53.1–80.0. **No detectable relation.** Consistent with the 7B
below-knee picture (adaptation saturates while magnitude varies) but with n=19,
one LR per method, and no F_Δ, treat strictly as descriptive. Method means (excl.
diverged): dora 79.4, sclora 78.4, lora_null 77.4, lora 76.2, lorawd 71.7,
milora 71.2, clora 65.7.

## 3. WHAT THE PAPER MAY AND MAY NOT SAY

MAY: "a designed 284B arm (7 methods × 3 seeds) lost its retention evaluations to
infrastructure failure; the surviving artifacts show the per-method update-shape
fingerprint recurs at 284B/MLA (pooled rank-r +0.86) and adapters reach 53–80%
MedMCQA" + limitation line. This is an honest generalization-of-geometry point
and evidence the pipeline ran at 284B.

MAY NOT: any 284B retention, forgetting, knee, or magnitude-relation claim; any
CE claim; any use of the fro/spec orderings as method comparisons (LR confound).

Recovery path (GPU-gated): restore from results/ds_adapters_evac/ (verified),
run scripts/deepseek/eval_deepseek.py + ce_deepseek.py on 8×B200-class nodes;
spec: handoff/DEEPSEEK_GEN_EXPERIMENT.md.
