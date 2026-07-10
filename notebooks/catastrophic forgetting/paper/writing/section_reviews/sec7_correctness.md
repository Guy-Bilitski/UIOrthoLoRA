# Section 7 review — "Correctness assurance" (artifact_status_report.html)

Reviewer: section-validator. Date: 2026-07-10.
Scope: the 6-row checks table + intro sentence; overlap with §8 External review.
Verdict: **SOUND BASIS, PUBLISHABLE AFTER WORDING FIXES.** All 6 rows now trace to primary
evidence (two rows that the numeric audit had marked "audit-sourced" are hereby traced to raw
logs/JSON — an upgrade). Three rows quote a number less carefully than a correctness table should
(blanket bound, best-case value, omitted anchor). The table also under-sells the campaign: its two
strongest gates (zero-step loss preservation; found-and-fixed port deviations) have no row.

---

## (a) Claim-by-claim verdict table

| # | Row as stated | Verdict | Recomputed / traced evidence |
|---|---|---|---|
| 1 | "8 adapter ports vs released code — line-for-line audit — FAITHFUL" | **CONFIRMED, but under-specified** | Primary: `handoff/18_ADAPTER_AUDIT_2026-07-02.md` (8-arm per-adapter audit: lora, lorawd, dora, milora, corda, sclora, lora_null, clora — all *code*-faithful) + `handoff/23_REPO_VERIFICATION.md` (reference repos cloned at pinned commits, e.g. LoRA-Null `1e6808a`, MiLoRA `c3c9469`, SC-LoRA `b524394`, CorDA `baffb03`). Caveat the row hides: the audit **found a blocking deviation** — the original LoRA-Null calibration was NOT repo-faithful (256 short validation questions ≈2.5k tokens → degenerate rank-deficient covariance vs repo's 524k-token full-rank C; handoff/23 §1.2–1.4) — which was fixed and re-run. "FAITHFUL" is true post-fix; saying "audit found and fixed one blocking calibration deviation (LoRA-Null)" is *more* convincing to a skeptic than an unblemished all-clear, and honest. Also: 8 "ports" includes lora/lorawd (baselines, trivially faithful); the non-trivial ports are 6. |
| 2 | "Update reconstruction ΔW=(α/r)B·A matches recorded spectral norm (<0.2%)" | **CORRECTED (mild overstatement)** | Recomputed from `results/geo_drift/master_labeled.jsonl` (`spec_max`, 320 adapters) vs recorded `dw_sv_max` (campaign_summary.jsonl + per-run summary.json; 307 matched): **median rel. err 0.0046%, 94.1% < 0.2%, p95 = 0.21%, max = 0.51%** (worst: `frm_lorawd_wd0p3_lr1e3_c256_s42`, 3916.75 vs 3906.33). Consistent with bf16 rounding; matches the number-audit row 136. The blanket "<0.2%" is falsified by ~6% of adapters — write "median 0.005%, 94% within 0.2%, max 0.5% (bf16 rounding)". Also 13/320 recon adapters had no recorded `dw_sv_max` to compare — worth one sentence or a trivial backfill. |
| 3 | "LoRA reproduces its published GSM8K — 60.2 vs published 60.6" | **CONFIRMED** | 60.2 = `results/frm_lora_lr3e4_c256_s42/summary.json` → `headline/gsm8k = 60.2`. Published 60.58 = LoRA row of the faithful-repro reference Table 3 (`handoff/20_FAITHFUL_REPRO_SPEC.md`: LoRA 60.58/16.88) [EXTERNAL]. Under-used: `handoff/24` also records CS-side anchors (CS-8 ~80 vs published 79.9; BoolQ 69.97 vs published) — one anchor per campaign is stronger than one anchor total. |
| 4 | "Adapter save→reload round-trip merges losslessly (Δ 6e-9)" | **CORRECTED (best-case value quoted)** | Traced: Δ5.91e-9 is the **single best** gate result (cordapp dyn-rank, `logs/gate_residual_cordapp.log:15`). Full primary record: `logs/validation_gate.log` — corda 7.45e-9, milora 1.49e-8, lora_null 3.35e-8, sclora 2.38e-7, "VALIDATION GATE: PASS"; `logs/gate_residual_cordapp.log` (r64→2r conversion) — milora 2.79e-8, lora_null 1.06e-7, sclora 3.58e-7, pissa 4.43e-7. All PASS, all genuinely lossless — but a correctness table must quote the **worst case**: "max |Δ| ≤ 4.4e-7 across all 9 residual round-trip gates". Quoting the minimum invites the exact reviewer question this table exists to preempt. (Note: `logs/_archive/roundtrip*.log` FAIL verdicts are the *UIOrthoLoRA* reload bug, out of scope for this campaign's arms — worth a footnote only if the archive ships.) |
| 5 | "Base-model retention ceiling — BBH 32.96 vs registered 33.10" | **CONFIRMED — and upgraded to primary-verified** | 33.10 = `results/base_l2-7b_bbhAO/retention_agg.json` (recomputed macro mean over 27 subtasks = 33.099). 32.96 = full-set answer-only base BBH re-eval under the current (metric-fixed) harness: `results/retfix_diag/base_bbh_fullset_current_harness.json` (`bbh_fewshot_fullset: 32.96`) + `logs/gate_base_bbh.log` ("PASS if |score − 33.10| ≲ 1.0"). |Δ| = 0.14 pp ≪ 1.0 gate. This **resolves** `artifact_number_audit_final.md` rows 138–139's "audit-sourced, not a row in results/" flag — the file exists; cite it. The row is also more valuable than it looks: it is the pre-registered no-op proof that the 2026-07-01 BBH metric fix left the Llama-2 axis unchanged (handoff/22 §3) — name it as such. |
| 6 | "CE-forgetting metric vs MiLoRA Table 8 — LoRA 3.57↔3.24, PiSSA 6.31↔6.07 — REPRODUCED" | **CONFIRMED numerically; verdict word slightly generous** | Recomputed from `results/forgetting.jsonl`: LoRA (frm_lora_lr3e4) forgetting_ce = 3.5703; PiSSA (frm_pissa_lr3e4) = 6.3068; MiLoRA (frm_milora_lr3e4) = 3.6594. External anchors 3.24 / 6.07 / 2.54 = MiLoRA Table 8. Gaps: LoRA +10%, PiSSA +4%. Two cautions: (i) "REPRODUCED" at a 10% gap is generous — "CONSISTENT (same ordering, ≤10%, at our operating points)" is unattackable; (ii) the third Table-8 anchor (MiLoRA 2.54 vs our 3.66) is silently absent from this row. §6's caption explains it correctly (their 2.54 is a lower-‖ΔW‖ operating point; matched-magnitude 3.66 ≈ LoRA 3.57 is the thesis), but §7 read alone looks like anchor-picking — add "(MiLoRA anchor differs by design; see §6)". |

Intro sentence check: "a fixed set of pre-registered checks passed before any result was trusted" —
defensible: the base-BBH gate criterion was written before the run (handoff/22 §3) and handoff/24
states "No new-method cell dispatches before its gate passes." Cite one of these, or soften
"pre-registered" to "pre-committed gates".

## (b) Is this the right set of checks? Missing rows the campaign actually performed

1. **Zero-step loss-preservation gate for residual inits (STRONGEST missing row).**
   `logs/validation_gate.log` + `logs/gate_residual_cordapp.log`: 0-step CE-preservation error
   1.5e-3–3.9e-3 for corda/milora/sclora/lora_null/pissa/cordapp, all under the 1e-2 gate. This is
   the check that kills the entire "residual init silently changes the function / eval explodes" bug
   class (the campaign's own worst bug family, `residual_save.py`). Numbers already on disk; zero
   compute. Honest phrasing per handoff/18 §b: "loss-preserving to bf16 precision (≤3.9e-3), not
   bit-exact".
2. **Calibration↔eval confound: found, fixed, sensitivity pending.** The audit's biggest catch
   (FIX-1, handoff/18 BLOCK-1): three data-aware arms originally calibrated on nq_open while
   retention is academic-reasoning; plus the LoRA-Null covariance degeneracy (handoff/23). Both
   were caught by the campaign's own audits and re-run repo-faithfully. A skeptical reviewer WILL
   probe calibration for CorDA/SC-LoRA/LoRA-Null; a row that owns "confound found by our audit →
   re-run; nq_open-vs-eval-matched sensitivity control pending" converts a vulnerability into
   credibility and honestly discloses the SC-LoRA −4pp boundary's provisional status.
3. **Parameter-parity audit** (handoff/18 §c): r32 arms 56,098,816 trainable; DoRA r16 28,925,952
   incl. +876,544 magnitude vec, exact. One row, zero compute — preempts the param-count confound
   question for the CS sweep.
4. **LoRA-Null post-fix rank diagnostic** (handoff/23 §1.2: repo prints `(S_ > 0.1).sum()` per
   layer; full-rank C is what distinguishes real LoRA-Null from a random-null-space method). Not
   yet run on our re-port as far as the record shows — cheap (<1 GPU-h, init pass only) and it is
   the one check that proves the re-port fixed the degeneracy rather than merely matching the recipe.
5. **Per-arm 0-step smoke before camera-ready** — handoff/18 §b explicitly flags the end-to-end
   0-step gate as "reported PASSED in handoff/13 but not independently re-run"; one smoke per
   residual arm ≈ 1–2 GPU-h total closes the campaign's own open caveat.

## (c) Checks a reviewer would consider weak as stated

- **No evidence pointers anywhere in the table.** Every row is an assertion; every row has a file.
  Adding an evidence column (or footnote list) is the single highest-credibility-per-effort fix:
  row 1 → handoff/18 + handoff/23 (with repo commits); row 2 → master_labeled.jsonl vs recorded
  dw_sv_max; row 3 → frm_lora_lr3e4_c256_s42/summary.json + repro-spec Table 3; row 4 →
  validation_gate.log + gate_residual_cordapp.log; row 5 → retfix_diag/base_bbh_fullset_current_harness.json
  + gate_base_bbh.log; row 6 → forgetting.jsonl + MiLoRA Table 8.
- Row 1 "line-for-line audit" with no artifact and no mention that the audit *found and fixed* a
  blocking deviation (see (a)#1) — reads as self-certification.
- Row 2 blanket "<0.2%" (94.1% true; max 0.51%).
- Row 4 best-case Δ (5.91e-9) instead of worst-case (≤4.4e-7).
- Row 6 omits the third published anchor without a pointer to §6's (correct) explanation.

## (d) §7 vs §8 — merge or keep separate?

**Keep separate; cross-link.** They answer different reviewer questions: §7 = "was the *experiment*
right?" (pipeline gates, run before/independent of results); §8 = "is the *report* right?" (post-hoc
adversarial + numeric audit of the document). Merging would blur pre-committed gates with post-hoc
review and weaken both. Two required touch-ups at the seam:
1. §8's numeric audit itself flagged §7's "32.96" and "Δ 6e-9" as audit-sourced
   (`artifact_number_audit_final.md` lines 133–134, 167–168). Both are now traced to primary files
   (see (a)#4–5): update the audit doc's two rows to primary-verified with the paths above, so §7
   and §8 stop citing each other in a circle.
2. §8's "140/143 reproduced" includes §7's own rows — one clause ("including the §7 gate values")
   avoids the impression of double-counted assurance.

## Prioritized strengthening list (concrete, with cost)

| P | Action | Cost |
|---|---|---|
| 1 | Add evidence column with file paths (+ repo commits for row 1) to all 6 rows | 0 compute, ~30 min |
| 2 | Fix the three number statements: row 2 → "median 0.005%, 94% <0.2%, max 0.5%"; row 4 → "≤4.4e-7 worst-case over 9 gates"; row 6 → "consistent (≤10%)" + MiLoRA-anchor pointer to §6 | 0 compute |
| 3 | Add zero-step loss-preservation gate row (1.5e-3–3.9e-3 vs 1e-2 gate, bf16-bounded) — data already in logs | 0 compute |
| 4 | Add "confound found-and-fixed" row: LoRA-Null calibration deviation + FIX-1 quarantine/re-run, sensitivity control status | 0 compute (status text); the actual nq_open-vs-eval-matched control ≈ 12–24 GPU-h (already a campaign TODO) |
| 5 | Run LoRA-Null rank diagnostic on the re-ported init + one 0-step smoke per residual arm (closes handoff/18 §b and handoff/23 open caveats) | ≈ 2–3 GPU-h total |
| 6 | Broaden row 3 to "published anchors" (add CS-8 79.9→~80, BoolQ 69.97) and add param-parity row | 0 compute (numbers in handoff/24 & handoff/18 §c) |
| 7 | Upgrade artifact_number_audit_final.md rows 138–139 from "audit-sourced" to primary-verified; backfill the 13 recon-unmatched adapters' dw_sv_max | ~15 min + trivial script |
