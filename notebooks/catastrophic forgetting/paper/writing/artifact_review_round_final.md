# Final adversarial review — supervisor status-report artifact
Reviewer: adversarial ML reviewer (harsh). Date: 2026-07-09.
Source reviewed: `/tmp/claude-1000/-home-guy-UIOrthoLoRA/72dbcb26-a4ce-47e1-aaa6-792e52121dea/scratchpad/status_report.html`
(repo backup `paper/writing/artifact_status_report.html`). Line numbers below refer to the /tmp source.
Ground truth: `paper/writing/data/key_numbers.md`, `handoff/27`, `handoff/29`, `fleet_findings.md`, `integration_notes.md`,
and independent recompute from `results/campaign_summary.jsonl` (latest-`evaluated_at` dedup).

## Independent verification of the load-bearing numbers (all recomputed this pass)
- Magnitude law, Llama CS pool (7 adapters, CorDA excluded, n=49): **r = −0.858, R² = 0.736, slope −14.78 pp/dec** → artifact's −0.86 / 0.74 / −14.78 are CORRECT.
- LR as predictor (n=49): **R² = 0.321, r = −0.567** → artifact's 0.32 CORRECT.
- Within-adapter r: LoRA −0.95, LoRA+wd −0.89, MiLoRA −0.94, DoRA −0.97, LoRA-Null −0.86, CLoRA −0.90, SC-LoRA −0.97 (key_numbers §1a) → artifact's "−0.86 to −0.97" CORRECT.
- §3 best-op-point CS table: every cell matches `lrsw_*_s42` (e.g. `lrsw_lorawd_wd0p3_lr5e4_s42` = 81.62/25.55/fΔ0.3941). CORRECT — but SINGLE SEED (s42).
- §4 math 67.3 = `frm_lorawd_wd0p3_lr2e4_c256_s42` = GSM8K 67.25 / BBH 33.1 / fΔ 0.278. The "256-token" claim is HONEST (it really is a c256 cell), and *conservative*: its **c512 sibling is HIGHER (69.52)**, so c256 does not inflate. But single-seed: the **s43 sibling of the same cell = 65.88** (edges CLoRA 64.6 by only 1.3, not 2.7).
- CE table (§6) values confirmed in `results/forgetting.jsonl` (field `forgetting_ce`, base_entropy 1.852): LoRA+wd 2.00, LoRA 3.57, MiLoRA 3.66, PiSSA 6.31 — all exact. **BUT** the §6 "LoRA+wd(math)" row is **wd0.5** (fΔ 0.20 = `frm_lorawd_wd0p5_lr1e4_c256`), while the §4 GSM8K headline "LoRA+wd" is **wd0.3** (fΔ 0.28). Two different runs presented under the same name in adjacent sections — must be disclosed.
- Registry note: the live `campaign_summary.jsonl` is now 473→457 unique rows (git-modified), larger than the 359→343 file `key_numbers.md` was recomputed from (2026-07-02). The `lrsw_` CS pool is unchanged so claims 1–4 reproduce exactly, but frm_/mtx_ numbers should be treated as a moving target until the file is frozen.
- Geometry table (§2): every value matches `handoff/27` per-method CS signature table. CORRECT.
- Efficiency (§5): matches `fleet_findings` EXCEPT SC-LoRA init forward-pass count (see finding I).

## THE SEED PROBLEM (the crux — recomputed from the 3-seed `mtx_` matrix)
Retention_mean is seed-stable; **adaptation (cs_avg) is NOT.** The `mtx_` 3-seed matrix shows large seed-dependent collapses in cs_avg:
- `mtx_lorawd_wd0p5`: s42 **82.02** / s43 81.92 / **s44 50.85** (confirms the killed 82.0 cell; integration_notes B1).
- `mtx_clora_k2048`: s42 79.84 / s43 79.97 / **s44 23.03**.
- `mtx_milora_r32`: s42 79.86 / **s43 58.72** / **s44 34.62**.
- `mtx_dora_r8`: **s42 63.07** / s43 79.52 / **s44 21.57**.
- `mtx_lorawd_wd0p3`: s42 78.17 / s43 74.30 / s44 80.44 (≈6 pt spread).
- BUT retention is stable: `mtx_lorawd_wd0p3` ret = 26.59 / 26.64 / 26.77.

Implication: the **magnitude LAW (retention vs F_Δ) is genuinely seed-robust** (retention and F_Δ both stable across seeds) — this is defensible. The **single-seed adaptation rankings ("81.6, top accuracy, tops both axes") are NOT robust** and must be softened. The artifact discloses seeds NOWHERE.

---

## (1) Per-comment verdict table A–N

| # | PI comment | Verdict | Evidence / why |
|---|---|---|---|
| A | No honest-accounting / future / queued / buggy / TODO / pending language anywhere | **FAIL** | §2 footnote (line 177): "CorDA's retention is **not yet** calibration-matched to the others (**a fairness fix is pending**)". Direct "pending"/"not yet" hit — the PI will Ctrl-F this. One-line fix, but currently violates the absolute instruction. (Also flag: §3 callout title "**Honest** boundary", line 202, echoes the banned "honest accounting" wording — rename.) |
| B | §3 CS adaptation + retention present, complete, clear | **PASS** | §3 table (187–198) has best-LR, CS acc, retention, F_Δ, safe band for all 7; numbers verified. Clarity good. (Single-seed caveat → belongs under L, not B.) |
| C | Full LR sweep reported per adapter, with the actual numbers AND a per-adapter graph | **WEAK** | Per-adapter graph present (LR panel, 131 + script S{}), one line per adapter. BUT the 7 swept LRs are not enumerated in text (x-axis shows only 4 ticks: 2e-5/1e-4/3e-4/1e-3; actual set is 2e-5,5e-5,1e-4,2e-4,3e-4,5e-4,1e-3), and the exact per-LR×per-adapter retention numbers are only in an unreadable SVG — no numeric grid. A supervisor cannot read the swept values off the plot. |
| D | "Magnitude law, three ways" self-explanatory | **PASS** | Three independent confirmations present and cross-referenced: our sweep (§1), CLoRA Table 4 external (§1 callout 143–146), CE-to-base (§6), tied together in §6 line 263 ("the retention suite and CLoRA's own table … same conclusion"). Distributed rather than consolidated, but self-explanatory. |
| E | External replication (CLoRA Table 4 r=−0.98, cross-lit overlay) prominent, not buried | **WEAK** | The r=−0.98 external replication is a green callout in §1 (143–146) — not buried, good. BUT (a) no actual **overlay figure** (the "cross-literature overlay" the PI named) — it is text-only; (b) the single strongest point (their own table gives −0.98, cleaner than our −0.86) is NOT in the header tiles, which instead lead with our own runs. Elevate it. |
| F | Top adapters-overview section explaining each method | **PASS** | §0 (108–123) covers all nine (LoRA, LoRA+wd, DoRA, PiSSA, MiLoRA, CLoRA, SC-LoRA, LoRA-Null, CorDA++), grouped sensibly. §0 LoRA+wd blurb (113) is now accurate ("indirectly reduces … F_Δ … no geometric structure"). Nit: "all nine … ΔW = A·B" over-simplifies DoRA (magnitude×direction). |
| G | Geometry: WHAT analyzed, DEFINES every column, CorDA caveat | **PASS** | §2 "what we measured" (155); caption (162) defines e_top/e_bot/ein_top/ein_bot/stable rank/law residual + the 0.06 neutral baseline; CorDA caveat present (177). Content correct vs handoff/27. (The caveat's *wording* triggers comment A — fix there.) |
| H | CS safe-band explicitly defined | **PASS** | §3 caption (186): "of the 7 swept learning rates, how many keep retention ≥ 24 (within ~2 pts of the ceiling)". Verified: LoRA+wd 6/7, SC-LoRA 1/7, etc. |
| I | Efficiency: compute AND memory; INIT TIMES; CLoRA extra memory ADDITIONAL not total | **WEAK** | All three requested elements ARE present (structure fully addressed): wall-clock + extra-memory columns; init column with times ("160 SVDs ~minutes", "~few min", "~1 GPU-h"); caption + row + prose all state CLoRA memory is ADDITIONAL to ~55 GB. BUT a factual error: SC-LoRA init = **512** calib forwards (256 D+ + 256 D−, per fleet_findings/method recipe), the table (239) says **256** and lumps it with LoRA-Null (which is 256). Also ~55 GB baseline and CLoRA's +0.42–6.7 GB are analytical/incidental, not instrumented (fleet: "Peak GPU mem not instrumented") — presented as measured. |
| J | Geometry DRIFT per layer / model / adapter represented | **WEAK** | Per-adapter drift: yes (table). Temporal drift: mentioned ("persists through three epochs", 159) and SC-LoRA erosion with LR (0.70→0.21, 176). BUT per-**layer** drift is only asserted ("layer by layer") not shown, and per-**model** geometry (Qwen) is absent. handoff/27 has the per-layer data; none of it is represented visually. |
| K | "Last updated" timestamp in header | **PASS** | Byline (91): "Last updated · 2026-07-09 16:55 IDT". |
| L | Framing guardrails: no bold unprovable claims; no "geometry doesn't matter"; no "everyone reports one LR"; constructive magnitude→LoRA+wd thesis; nothing beyond 3-seed support | **FAIL** | Banned phrases correctly AVOIDED ("fingerprint, not a knob" instead of "geometry doesn't matter"; no "everyone reports one LR"). BUT multiple overclaims: (1) **dek (86)** "the popular geometric methods **do not beat** a tuned plain-LoRA baseline once that size is controlled" — directly contradicted by the artifact's OWN §3 callout (202–203): CLoRA k1024/k2048 (82.6/83.7) beat LoRA+wd. (2) **No seed disclosure**; single-seed peaks ("81.6 top accuracy", tile 97; "tops both axes", 183; "no adapter beats it on both", 97) presented as robust while the 3-seed matrix shows cs_avg collapses (82→51). (3) **tile 1 (96)** "on a second model" overstates Qwen (LoRA-only, CS-only; Qwen math does NOT replicate, r=+0.67 ns per key_numbers §11) — and that negative is omitted. |
| M | CE-to-base forgetting section present + explained | **PASS** | §6 (247–264) present; MiLoRA Table-8 reproduction stated; α=2r scale disclosed in caption (253); load-bearing "MiLoRA 3.66 ≈ LoRA 3.57 at matched magnitude" (263) is exactly the right point. F_Δ values (1.28/1.26/2.21/0.20) match raw math cells. |
| N | Whole document understandable, self-explanatory, backs the thesis | **WEAK** | Structure, prose, and the retention/efficiency thesis are clear and well-supported. Downgraded because "actually backs the thesis" is undermined by the unqualified single-seed framing and the dek overclaim — a supervisor who asks "how many seeds?" gets no answer, and the dek claim is refuted three lines into §3. |

**Tally: PASS = 7 (B, D, F, G, H, K, M) · WEAK = 5 (C, E, I, J, N) · FAIL = 2 (A, L).**

---

## (2) Ranked concrete fixes (most severe first)

### FIX 1 — [L, BLOCKER] Dek overclaim, contradicted internally. Location: header dek, line 86.
Current: "…the popular geometric methods **do not beat** a tuned plain-LoRA baseline once that size is controlled, and they cost more to run."
Problem: §3 callout (202–203) admits CLoRA k1024/k2048 published numbers (82.6/83.7, BBH 36.5/38.7) BEAT LoRA+wd. A reviewer reading top-to-bottom hits the contradiction in <1 minute.
Replace with (constructive, provable): "…the size of the weight update predicts forgetting; at matched update size the geometric methods land on the same curve as a tuned plain-LoRA baseline, and they cost more to run — so LoRA + weight-decay is the best *efficient* operating point."

### FIX 2 — [L, BLOCKER] Add seed disclosure and soften single-seed adaptation claims. Locations: tiles 96–99, §3 lead 183, §3 table 187–198, §3 para 200, §4 210–222, foot 285.
Problem: every headline number is single-seed (s42); the 3-seed `mtx_` matrix shows cs_avg collapses (`mtx_lorawd_wd0p5` s44 50.85; `mtx_clora_k2048` s44 23.03; `mtx_milora_r32` s43/s44 58.7/34.6). "81.6 top accuracy" and "tops both axes / no adapter beats it on both" are not seed-robust. Retention IS stable, so the LAW survives.
Fixes:
- Add one line to the byline/foot: "Mature single-model results are seed-42 point estimates; the magnitude law (retention vs F_Δ) additionally holds across a 3-seed rank/wd matrix (retention seed-SD ≈ 0.3), where task accuracy is the seed-sensitive axis."
- Tile 2 (97): change "top accuracy and retention" → "best retention at the smallest update; task accuracy statistically tied with the field (78–82)."
- §3 lead (183): keep "The field is flat on accuracy" (this is the honest, seed-robust framing) but change "at least as good as every adapter on both axes" → "indistinguishable from the field on accuracy and best on retention, at the smallest update and widest safe LR band."
- §3 para (200): after "tops both axes", add "(single-seed; on adaptation the seven methods are within seed noise, so the robust claims are retention, update size, and safe-band width)."
- §4 (222): note the 67.3 is single-seed and cross-harness (the s43 sibling is 65.88, still edging 64.6) — keep the "edges/matches" verb.

### FIX 3 — [A, BLOCKER] Remove "pending"/"not yet" forward-looking language. Location: §2 footnote, line 177.
Current: "* CorDA's retention is **not yet** calibration-matched to the others (**a fairness fix is pending**), so its law residual is shown only alongside its geometry signature…"
Replace: "* CorDA's retention is not calibration-matched to the other methods, so its value is reported only as a geometry fingerprint, not as an assessed on/off-law result." (States the boundary without any future/pending verb.)
Also rename §3 callout title (202) "Honest boundary — high-rank CLoRA" → "Boundary condition — high-rank CLoRA" to avoid the banned "honest accounting" echo.

### FIX 4 — [L, MAJOR] Qwen "second model" is overstated; the math-negative is omitted. Locations: tile 1 (96), §1 line 136.
Problem: Qwen replication is LoRA-only, commonsense-only; Qwen MATH does not replicate (r=+0.67, ns; key_numbers §11). "on a second model" (tile) reads as full replication.
Fixes: tile 1 (96): "…and on a second architecture (Qwen-2.5-7B, LoRA, commonsense: r = −0.88)." §1 (136): keep "Qwen2.5-7B commonsense, r = −0.88" and add "(LoRA arm; other adapters and the math axis are not part of the assessed set here)". Do NOT claim math replication on the second model.

### FIX 5 — [I, MAJOR] Efficiency: correct the SC-LoRA init count and flag analytical-vs-measured. Location: §5 table row 239, para 244.
Problem: SC-LoRA needs **512** calibration forwards (256 D+ + 256 D−), not 256; it is currently lumped with LoRA-Null (256). Also ~55 GB baseline and CLoRA +0.42–6.7 GB are analytical/incidental, not instrumented.
Fixes: split the row — "SC-LoRA: 512 calib fwd passes (256 preserve + 256 target) + eigendecomp"; "LoRA-Null: 256 calib fwd passes + eigendecomp". Add to caption: "Memory figures are analytical resident-size / incidental-trace estimates (peak GPU memory was not instrumented); wall-clock is from training logs."

### FIX 5b — [MAJOR, internal consistency] §4 and §6 both say "LoRA+wd(math)" but are different runs. Locations: §4 bar (214), §6 table (256).
§4 GSM8K "LoRA+wd" = **wd0.3** (fΔ 0.28); §6 CE "LoRA+wd (math, small update)" = **wd0.5** (fΔ 0.20). A reviewer cross-referencing the two adjacent math sections will see one name pointing at two runs. Fix: label each explicitly — "LoRA+wd (wd0.3)" in §4, "LoRA+wd (wd0.5)" in §6 — and add a half-sentence that heavier wd trades GSM8K (67→66) for a smaller update and lower CE.

### FIX 5c — [MAJOR] Flag the seed-fragile §3 operating points. Location: §3 table rows for MiLoRA and SC-LoRA (192, 191).
The 3-seed matrix shows the §3 MiLoRA (r32) config collapsing cs 79.9→58.7→34.6 (spread 45 pp) and SC-LoRA (r32) retention swinging up to 9.2 pp across seeds. Their §3 rank is a single lucky seed. Add a footnote: "MiLoRA and SC-LoRA operating points are single-seed; both are seed-fragile in the 3-seed rank matrix and their relative rank should not be over-read."

### FIX 6 — [C, MINOR] Make the LR sweep legible. Location: §1 panel (131), §3.
Add a one-line note under the LR panel listing the swept set ("7 LRs: 2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3, every adapter"), or add a small collapsible per-LR×per-adapter retention grid. Currently the numbers exist only inside the SVG.

### FIX 7 — [E, MINOR] Strengthen external replication. Locations: tiles, §1.
Add a real overlay figure (our (F_Δ, retention) points + CLoRA Table-4 (F_Δ, BBH) points, both fitted lines) and consider promoting "CLoRA's own table: r=−0.98, slope −14.7 vs our −14.8" to a header tile. Note the honest nuance: same *slope*, not one literal line (their F_Δ levels are ~2× lower) — change "one line" (145) to "the same slope / parallel law."

### FIX 8 — [N/gloss, MINOR] "Ceiling" is the wrong word. Location: gloss (104), §3, plots.
Several 3-seed LoRA+wd runs EXCEED 26.0 (e.g. `mtx_lorawd_wd0p3` ret 26.6–26.8; wd1.0 ~27.9). Calling the base score "the ceiling" invites the nitpick "your own runs beat your ceiling." Call it "the base-model reference (26.0)" and, where relevant, note retention can slightly exceed it at heavy regularization.

### FIX 9 — [J, MINOR] Represent per-layer / per-model geometry drift.
Add a per-layer concentration inset (SC-LoRA ein_top by layer, early 0.61–0.75 → the erosion story) and, if available, one Qwen geometry row, so "drift per layer/model/adapter" is actually shown, not asserted.

### FIX 10 — [minor consistency] CorDA appears in the LR panel + legend but is "not assessed."
The DW panel correctly drops CorDA; the LR panel and legend still show it (8 lines). Either drop CorDA from the LR panel too, or add "(shown for context; excluded from the law)" to the legend, so "7 assessed" and the visible 8 lines agree.

---

## (3) Is this a good basis for a paper? — yes/no + biggest gap

**Yes, conditionally.** The scientific core is real and independently reproduces: the magnitude law (retention vs F_Δ) is r=−0.86 pooled, −0.86…−0.97 within every adapter, corroborated by CLoRA's own Table 4 (r=−0.98) and by an independent CE-to-base metric (MiLoRA≈LoRA at matched magnitude), and — critically — the retention axis is seed-stable in the 3-seed matrix. The geometry-as-fingerprint framing and the efficiency angle are the right constructive story and stay inside the guardrails. That is a defensible contribution.

**The single biggest gap: seed coverage / disclosure.** Every headline point is a single seed (s42), the document never says so, and the sibling 3-seed matrix shows that *adaptation* accuracy collapses catastrophically on other seeds (82→51, 80→23, 80→35). A NeurIPS area chair's first question is "how many seeds, where are the error bars?" — and the honest answer today is "one, for the numbers you're quoting." The law itself survives this (retention is stable), so the fix is not more science but honest framing: report seeds, put error bars / seed-SD on retention, and restate the adaptation-side claims as "statistically tied within the flat 78–82 band" rather than "LoRA+wd has the top accuracy." Do that, remove the one "pending" phrase, and fix the dek, and this is a sound paper basis.
