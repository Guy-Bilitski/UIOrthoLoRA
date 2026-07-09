# handoff/30 — 3-seed retention error bars for the 7 §3 headline cells (research-planner, 2026-07-09)

Goal (PI-approved): put 3-seed retention error bars on the 7 commonsense headline operating points in
the supervisor artifact §3. This doc: exact seed inventory, family decision, deduplicated missing-run
list, and what was queued.

`retention_cs = mean(headline.bbh, headline.mmlu_pro)` per run's `results/<run>/summary.json`
(== the `retention_mean` field; base ceiling 26.0). Adaptation = `headline.cs_avg`. F_Δ =
`fdelta.fdelta_token_weighted`.

## 1. Cell identity — AUTHORITATIVE (from paper/writing/artifact_number_audit_final.md rows 92–98)

The §3 table shows the **lrsw_ family, s42**. The data-verifier's audit maps each §3 triple to an exact
run_name; my independent recompute of each reproduces the audited retention/F_Δ, confirming identity
(notably CLoRA = k1024_lr5e4 → 21.88, NOT the collapsed k-variant at 3.6 the coordinator warned about):

| # | §3 config | exact s42 run_name | s42 recipe (method flags · r/α · LR · cutoff) | s42 retention_cs | audit row |
|---|-----------|--------------------|-----------------------------------------------|------------------|-----------|
| 1 | LoRA+wd wd0.3 @5e-4 | `lrsw_lorawd_wd0p3_lr5e4_s42`   | `--method lora --lora_r 32 --lora_alpha 64 --weight_decay 0.3 --learning_rate 5e-4` c256 | 25.55 | 92 |
| 2 | SC-LoRA @5e-5       | `lrsw_sclora_r32_lr5e5_s42`     | `--method lora --sclora 1 --sclora_beta 0.5 --lora_r 32 --lora_alpha 32 --learning_rate 5e-5` c256 | 22.47 | 93 |
| 3 | MiLoRA @3e-4        | `lrsw_milora_r32_lr3e4_s42`     | `--method lora --milora 1 --lora_r 32 --lora_alpha 32 --learning_rate 3e-4` c256 | 24.72 | 94 |
| 4 | LoRA (plain) @3e-4  | `lrsw_lora_r16_lr3e4_s42`       | `--method lora --lora_r 16 --lora_alpha 32 --learning_rate 3e-4` c256 | 24.42 | 95 |
| 5 | LoRA-Null @5e-4     | `lrsw_lora_null_r16_lr5e4_s42`  | `--method lora --lora_null 1 --lora_r 16 --lora_alpha 16 --learning_rate 5e-4` c256 | 23.64 | 96 |
| 6 | CLoRA @5e-4         | `lrsw_clora_k1024_lr5e4_s42`    | `--method clora --clora_k 1024 --lora_r 32 --lora_alpha 64 --learning_rate 5e-4` c256 | 21.88 | 97 |
| 7 | DoRA @2e-4          | `lrsw_dora_r16_lr2e4_s42`       | `--method lora --use_dora 1 --lora_r 16 --lora_alpha 32 --learning_rate 2e-4` c256 | 24.84 | 98 |

(cutoff_len = default 256 for every lrsw_ cell; s42 recipes verified verbatim in
`archive/jobs/lr_sweep.txt` + `archive/jobs/_m_lrsw.txt`.)

## 2. Seed inventory (lrsw_ family) — what exists NOW

| config | s42 | s43 | s44 | n now | mean±SD now |
|--------|-----|-----|-----|-------|-------------|
| LoRA+wd wd0.3 @5e-4 | **25.55** | missing | missing | 1 | n/a (n=1) |
| SC-LoRA @5e-5       | **22.47** | missing | missing | 1 | n/a |
| MiLoRA @3e-4        | **24.72** | missing | missing | 1 | n/a |
| LoRA (plain) @3e-4  | **24.42** | missing | missing | 1 | n/a |
| LoRA-Null @5e-4     | **23.64** | missing | missing | 1 | n/a |
| CLoRA @5e-4         | **21.88** | missing | missing | 1 | n/a |
| DoRA @2e-4          | **24.84** | missing | missing | 1 | n/a |

**No lrsw_ headline cell has >1 seed today** (`ls results/lrsw_*_s43 results/lrsw_*_s44` = 0). All 7 need
s43 + s44 → **14 runs**, all genuinely missing (skip-done verified against `results/<run>/summary.json`).

### Cross-source note (why we do NOT reuse mtx_ for the extra seeds)
The 3-seed `mtx_` grid uses the **default LR (3e-4)**, so it only lines up with the *3e-4* operating
points (MiLoRA, plain LoRA). Two blockers:
- **Different code commit.** mtx_ = `f41c78b906`/`2464793134` (Jun 21–24); lrsw_ = `15666c94fa`/`21517195c2`
  (Jun 24–29). MiLoRA s42 happens to be *bit-identical* across both (deterministic, unchanged path), but
  plain-LoRA s42 **differs** (lrsw 24.42 vs mtx 23.99) → the campaigns are not interchangeable in general.
- Mixing `mtx_` s43/s44 into an `lrsw_` s42 row = apples-to-oranges (the family-consistency trap).
mtx_ values (MiLoRA s43 23.79 / s44 24.07; LoRA s43 24.28 / s44 24.70) may be cited as an *external
robustness cross-check*, never merged into the §3 error bars.

## 3. Family decision — run lrsw_ s43/s44 for all 7 (option a). Justification

- §3 currently shows **lrsw_ s42** numbers (audit rows 92–98; foot row 141 "56 lrsw_ CS runs"). Error-bar
  seeds MUST come from the same family or the mean is apples-to-oranges.
- The alternative (switch §3 to the `frc_` c256 reservoir family) is **not viable**: (i) `frc_` covers only
  3 of the 7 configs — there is no frc_ SC-LoRA / LoRA-Null / CLoRA / DoRA cell; (ii) **every frc_ CS cell
  (even the s42 anchors) is MISSING and not queued** (0 done — the reservoir hasn't started, it is Node-B
  work); (iii) frc_ standardizes on r32/α64, which **differs** from lrsw method-native ranks for plain-LoRA
  (lrsw r16/α32) and MiLoRA (lrsw α32). Only LoRA+wd0.3 is config-identical between the families.

## 4. Deduplication vs handoff/28's already-planned "7 new cells"

handoff/28 lists 7 new cells: `frc_lorawd_wd0p3_lr5e4_c256 {s43,s44}`, `frc_lorawd_wd0_lr3e4_c256 {s43,s44}`,
`frc_milora_lr3e4_c256 {s43,s44}`, `frm_lorawd_wd0p3_lr2e4_c2048_s42`. Checked: **all 7 MISSING, none in
master_dispatch.** They are the `frc_`/`frm_` families (Node-B reservoir + math anchor) → **zero run_name
overlap** with my 14 `lrsw_` runs. I added **no** frc_/frm_ cells.

One config identity to flag for coordination (avoid double compute):
- `lrsw_lorawd_wd0p3_lr5e4_{s43,s44}` (mine) and handoff/28's `frc_lorawd_wd0p3_lr5e4_c256_{s43,s44}` are the
  **same training config** (r32/α64/wd0.3/lr5e4/c256/Llama-2/CS-170K). Whichever family runs first, the
  other 2 seeds can be reused (config-identical). The other 5 configs have no frc_ counterpart, and
  frc_ plain-LoRA / MiLoRA use r32/α64 ≠ lrsw → not interchangeable.

## 5. What was queued, where, and how

- **Wrote** `jobs/seed3_headline.txt` — the 14 command lines (each replicates its s42 recipe exactly; only
  seed 43/44 + run_name change). Ordered: all s43 first (every cell reaches n≥2 asap), then all s44;
  DoRA first in each group (slowest, ~4.3 h train).
- **Appended** the same 14 lines to `jobs/master_dispatch.txt` (append-only, guarded against dup) →
  file now **117 job lines** (103 + 14), under a `# seed3_headline` comment header. auto_dispatch skips
  `#`/blank lines and only launches lines with a valid `--run_name`, so the header is safe.
- **NOT live yet:** the running `disp` auto_dispatch (pid 1468389) read its queue once at startup (117-line
  file will be seen only after a restart). I attempted the documented safe restart (idempotent; skip-done;
  skips pool-owned GPUs; disp has **no children** — it has launched nothing, all 8 GPUs are pool-owned) but
  the permission system **blocked killing the shared dispatcher** (correct guardrail — I did not create it).
  → The 14 runs are **queued in the canonical file** and will enter the live dispatcher at the **next
  consolidated restart** (already planned in handoff/29 to also pick up the 2 milora + handoff/28 cells).

### Operator action to make them live (single-dispatcher; no race)
Run the standard consolidated restart (this is the handoff/29 plan; disp has no children so nothing is lost):
```
cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting"
kill $(pgrep -f "auto_dispatch.py --jobs jobs/master_dispatch.txt")
setsid nice -n 5 /home/guy/UIOrthoLoRA/.venv/bin/python auto_dispatch.py \
  --jobs jobs/master_dispatch.txt --gpus 0,1,2,3,4,5,6,7 --tag disp \
  >> logs/auto_dispatch.log 2>&1 < /dev/null &
```
Do **not** launch a second dispatcher on `jobs/seed3_headline.txt` over the same GPUs — two auto_dispatch
instances share no per-GPU lock and can collide on a freshly-freed GPU (OOM). Keep one dispatcher.

## 6. Cost + ETA

- **~75 GPU-h** total (≈ 5 GPU-h/cell = ~2 h train + ~3 h broad eval; DoRA ~7.3 h/cell). Train part:
  33.4 GPU-h; eval part: ~42 GPU-h. ≈ 8–9 % of the 883-GPU-h two-node campaign.
- **Wall-clock:** 0 GPUs free now (all pool-owned). Position in queue = end (Priority #3 sits behind the
  frc_ CS spine #1 + recovery cells). Left as-is they finish within the campaign window but near the Sun
  deadline. **Recommendation:** at the next consolidated restart, move the 14-line `seed3_headline` block
  to run right after the frc_ CS spine and **ahead of the Qwen/c2048/CE tail** (Priority #6–8) — then the
  first n≥2 (all s43) lands within ~one wave (~5 h) of capacity freeing, and n=3 within ~2–3 waves
  (~15–20 h wall on 4–6 GPUs). On Node B splitting the spine, sooner.

## 7. Plan alignment

Fits handoff/26 (single-node) / handoff/28 (two-node) Priority #3 ("3-seed headlines"). Not a parallel
track: same canonical queue, same dispatcher, same lrsw_ family as §3. No live pool touched; setsid-only;
venv `/home/guy/UIOrthoLoRA/.venv/bin/python`.

### changed since last version
New doc. Establishes: lrsw_ family locked for §3 error bars (frc_ not viable — 4/7 configs uncovered, 0
frc_ cells done, rank mismatch); 14 missing runs written to jobs/seed3_headline.txt + appended to
master_dispatch (117 lines); mtx_/frc_ seeds explicitly excluded from the bars (commit + family
mismatch); ~75 GPU-h; live-queue entry pending the planned consolidated dispatcher restart (self-restart
blocked by permissions, as intended).
