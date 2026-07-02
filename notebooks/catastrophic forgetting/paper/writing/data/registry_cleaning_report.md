# Registry Cleaning Report (B2)

**Goal (blocker B2):** produce a clean results registry for the paper by removing (a) the CorDA
residual-save "explosion" rows and (b) duplicate `run_name` rows, so the "faithful port" claim is defensible.

- **Source (never modified):** `paper/writing/data/campaign_summary.jsonl` (a frozen copy of the
  live, still-being-appended `results/campaign_summary.jsonl`).
- **Output:** `paper/writing/data/campaign_summary_clean.jsonl`
- **Tool:** `/home/guy/UIOrthoLoRA/.venv/bin/python`

## Headline

| Stage | Rows |
|---|---:|
| Input (frozen copy) | **359** |
| After de-duplication (keep latest `evaluated_at` per `run_name`) | 343 |
| After CorDA-explosion exclusion | **320** |
| **Rows out** | **320** |

Removed: **16 duplicate rows** (across 16 run_names) + **23 CorDA run_names** = 39 rows removed.

## Exclusion rule 1 — De-duplicate by `run_name`

The registry is append-only, so re-evaluations create multiple rows with the same `run_name`.
**Rule:** for each `run_name`, keep the single row with the maximum `evaluated_at`; drop the rest.
This removed **16 rows** across these 16 duplicated names:

- `clora_k128_fast`, `clora_k256_fast` — identical re-append (same metrics, same values).
- `lora_wd0p01`, `lora_wd0p05`, `lora_wd0p05_full`, `lora_wd0p1`, `lora_wd0p1_full`,
  `lora_wd0p3`, `lora_wd0p3_full` — LoRA+weight-decay runs re-evaluated (metrics differ slightly
  between passes; latest kept).
- `lrsw_corda_r16_lr{2e5,5e5,1e4,2e4,3e4,5e4,1e3}_s42` — CorDA LR-sweep points that were
  re-evaluated on 2026-06-29/30 after the 2026-06-25 pass. (These 7 names are moot for the final
  registry because all CorDA rows are excluded by Rule 2, but they are counted here as the
  de-dup pass ran first.)

## Exclusion rule 2 — CorDA residual-save "explosion" rows

**Rule:** exclude **every** CorDA-family row (`run_name` contains `corda`, case-insensitive).
This removed **23 run_names** (post-dedup): all `mtx_corda_r{16,32,64,128}_s{42,43,44}`,
`corda_kpa_r{32,128}`, `lrsw_corda_r16_lr*_s42`, `valfix_corda_r16`, and `qwsw_corda_r16_lr2e5_s42`.

### Why exclude *all* CorDA, not just the obvious blow-ups

1. **CorDA is the only family that carries the documented residual-save-bug signature.**
   Per the project's own diagnosis (peft-residual-init-save-bug), a correct **post-fix** CorDA
   should have `fdelta_token_weighted` (`F`) in **0.04–0.15** with sane retention; the **pre-fix
   invalid** symptom is `F≈4.5`, `dw_sv_max≈3000+`, retention `≈0`.
2. **The registry's CorDA rows still match the INVALID pre-fix signature**, even after the
   `residual_save` conversion was supposed to run:
   - `corda_kpa_r128` / `mtx_corda_r128_*`: `F=4.55`, `dw_sv_max≈3100–3300`, retention `0.0–0.26`
     — an *exact* match to the documented pre-fix explosion.
   - `mtx_corda_r64_*`: `F≈2.3`, `dw_sv_max≈1700–1800`, retention `0.7–5.7`.
   - `lrsw_corda_r16_lr1e3_s42` (latest, 2026-06-30): `F=515.8`, `dw_sv_max=54741`, retention `0.0`
     — the single worst blow-up in the entire registry.
3. **The re-evaluation pass did not fix it.** The 2026-06-29/30 CorDA re-evals still show
   `dw_sv_max` 10–100× the matched LoRA/DoRA control and, in the `lr1e3` case, exploded *worse*
   than the original pass.
4. **Off-scale magnitude at matched rank/LR.** At r16 the plain-LoRA LR sweep tops out at
   `dw_sv_max≈200` (`lrsw_lora_r16_lr1e3_s42`); CorDA at the same rank/LR runs `dw_sv_max` 35→1394.
   This is the un-reversed `-B_init@A_init` init leaking into the evaluated ΔW, not a genuine
   magnitude effect.
5. **No CorDA row is simultaneously in the sane band AND actually adapted.** The one row inside
   the F=0.04–0.15 band, `valfix_corda_r16` (F=0.088), has `cs_avg=17.75` — i.e. it barely adapted
   (a validation/near-0-step artifact), so it is not a usable adaptation point either.
6. **Consistency with the paper's own decision.** The paper already reports the magnitude–budget
   law across 6 methods with CorDA dropped (blocker B1). Excluding all CorDA here makes the
   registry match the analysis that is actually published.

After exclusion, the **maximum `dw_sv_max` in the clean registry is 1073.6**
(`lrsw_dora_r16_lr1e3_s42`, F=3.73) — a genuine extreme-LR DoRA point, not a save-bug artifact.
The 54741 / 3300 / 1800 CorDA explosions are gone.

## Are any CURRENT (post-fix) CorDA rows trustworthy?

**No.** No CorDA row in this registry can be trusted for the paper:

- Every Llama CorDA row either explicitly explodes (`F` up to 515, `dw_sv_max` up to 54741) or
  carries the residual-save signature (`dw_sv_max` 10–100× the matched LoRA/DoRA control).
- The post-fix "sane" band (F=0.04–0.15) is reached by exactly one row (`valfix_corda_r16`), and
  that row failed to adapt (cs_avg 17.75), so it is not evidence of a working CorDA port.
- The newest row, `qwsw_corda_r16_lr2e5_s42` (2026-07-02), is a **Qwen** run (the Qwen study is
  still in progress) with a single LR point and `dw_sv_max=51` — still ~10× the matched Qwen-LoRA
  scale and unverified. It cannot validate the Llama CorDA port and is a single point regardless.

**A trustworthy CorDA result would require a fresh run with a passing 0-step self-check**
(dW→0 ⇒ base retention) confirming the rank-2r W0-relative conversion actually cancels the init.
That evidence does not exist in this registry, so CorDA must stay excluded.

## Impact on the "faithful port" claim (B2)

After cleaning, the registry is clean **for the 6 methods the paper actually uses**
(LoRA, LoRA+wd, DoRA, MiLoRA, SC-LoRA*, LoRA-Null, CLoRA — all with sane, non-exploded ΔW and no
duplicate rows). The explosion rows and duplicate re-appends that made the raw registry look
un-faithful are removed.

The **"faithful port" claim is clean only if it is scoped to the non-CorDA methods.** CorDA itself
is **not** faithfully reproduced in this registry — the residual-save conversion never produced a
verifiable, adapted CorDA point — so the paper must either (a) continue to exclude CorDA and say so,
or (b) rerun CorDA with a passing 0-step self-check before making any CorDA claim. It must not
present these CorDA rows as a faithful port.

> Note: SC-LoRA is retained in the clean registry (its ΔW is *not* exploded: dw_sv_max 27–59), but
> its knowledge-collapse-at-moderate-magnitude behavior is flagged UNRESOLVED elsewhere and is
> outside the scope of this de-dup/explosion cleaning pass.
