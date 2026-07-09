# Artifact feedback — round 2 (verification of the 6 PI comments, commit 7de28eee)

Artifact reviewed: `paper/writing/artifact_status_report.html` (identical to the scratchpad copy).
Verification sources: `geo_drift_phase2.py`, `forgetting_ce.py`, `results/forgetting.jsonl`,
`results/frm_*/summary.json`, `results/geo_drift/{adapter_metrics,master_labeled}.jsonl`,
`paper/writing/data/key_numbers.md`, and the CLoRA PDF. Venv python used for all recomputes.

---

## Per-comment verdicts

### Comment 1 — §2 geometry metrics from first principles — RESOLVED (with 2 consistency flags)

Formulas all match `geo_drift_phase2.py` exactly:

- **e_top.** Artifact: "‖projection of ΔW onto the base's top-256 output directions‖²_F / ‖ΔW‖²_F".
  Code `energy_frac(Uw, U_top)` computes `‖U_top^T (Uw·diag s)‖²_F / Σs²`. Because
  ΔW = Uw·diag(s)·Vw^T with Vw orthonormal, this equals `‖U_top^T ΔW‖²_F / ‖ΔW‖²_F`, i.e. the
  squared-Frobenius energy fraction of ΔW projected onto span(U_top). **EXACT match.**
- **amp_top.** Artifact: "‖U_top^T·ΔW·V_top‖_F/‖ΔW‖_F". Code builds `left@right = U_top^T(Uw diag s)(Vw^T V_top) = U_top^T ΔW V_top`, divided by `fro`. **EXACT match.**
- **stable rank** = ‖ΔW‖²_F/σ²_max — code `(s*s).sum()/s[0]**2`. **EXACT.**
- **ΔW = (α/r)·B·A** — code `scaling=alpha/r; dW=scaling*B@A`. **EXACT.**
- **Neutral 0.06** = 256/4096 = 0.0625. **Correct** for the 4096-dim q/k/v matrices.
- **Table values reproduce** from `master_labeled.jsonl` (mean over CS runs; this is exactly the
  Panel-C aggregation in `fig_geometry_4panel.py`): LoRA 0.071/0.047/0.076/0.051/8.8,
  MiLoRA 0.067/0.115/0.077/0.115/7.7, LoRA-Null 0.126/0.035/0.080/0.054/6.7,
  CorDA 0.078/0.048/0.041/0.494/13.0 all match to the decimals shown; DoRA, CLoRA, SC-LoRA match
  within rounding (SC-LoRA ein_top 0.401 vs printed 0.410; sr 19.1 vs 19.4 — immaterial).
- **SC-LoRA erosion 0.70→0.21** matches `lrsw_sclora` ein_top at lr2e-5 (0.703) → lr1e-3 (0.211). ✓

Flags (not errors, but reduce rigor):

1. **Law-residual column uses a different fit than the single source of truth.** The table prints
   SC-LoRA −5.7, LoRA +2.3, DoRA +3.6 "(vs the geometry-battery law)". `key_numbers.md` §5 (authoritative)
   gives spline residuals SC-LoRA **−4.15**, LoRA +0.79, DoRA +1.37, and the PI guardrail is "SC-LoRA
   −4pp". §2 prose then says "SC-LoRA sits 5.7 pp below the law." Same sign/ordering, but two different
   magnitudes for the campaign's headline deviator. Reconcile to −4.15 (or footnote the two fits explicitly).
2. **CorDA −3.0\* residual is printed** even though `key_numbers.md` §8 says "do not report" any CorDA
   number incl. the old "−3.0pp off-curve." The asterisk + footnote ("fingerprint only, not an assessed
   result") mitigate it, but it technically contradicts the SSOT directive.
3. Minor: the "0.06 neutral everywhere" is exact only for the 4096-dim matrices; up_proj (out) and
   down_proj (in) have an 11008-dim side where the neutral is 256/11008 ≈ 0.023. The F-weighted aggregate
   neutral is therefore slightly below 0.06. Does not change any conclusion; a half-sentence would make it airtight.

### Comment 2 — §3 safe band as LR-robustness — RESOLVED

- Threshold ≥24 stated; rationale added ("practitioners rarely tune LR per method; a wide safe band is
  what makes a method deployable at a default LR"). Good.
- Counts vs `key_numbers.md` §14: LoRA+wd **6/7**, SC-LoRA **1/7**, MiLoRA/LoRA/LoRA-Null/CLoRA **5/7**,
  DoRA **4/7** — all match exactly. The distinct values the PI named (6/7, 1/7, 5/7, 4/7) are all present and correct.
- "within ~2 points of the base reference 26.0" is consistent with the ≥24 cutoff. ✓

### Comment 3 — §5 efficiency (B200 + transient memory) — RESOLVED

- "All timings measured on NVIDIA B200 GPUs … single GPU per run" added. ✓ (consistent with the
  workdir memory note about the B200 host).
- Transient buffer now clarified: "exists only during the calibration forward passes at initialization
  … freed before training starts … costs nothing at train or inference time." Table + prose consistent.
- New honest caveat: "Memory figures are analytical resident-size / incidental-trace estimates (peak GPU
  memory was not instrumented)." Good — this correctly hedges the +22 GB / +6.7 GB numbers.
- No new inconsistency introduced: 112M-param claim checks out (r=64 over the 160 q/k/v/up/down modules
  = 112.2M); CorDA 1/5 ≈ "20% on top of ~5 GPU-h"; top tile ("6.7 GB frozen … DoRA 2.1×") matches the
  table (2.14× / +6.7 GB). ✓

### Comment 4 — §4 new BBH math-retention chart — RESOLVED numerically, PARTIAL on fairness/transparency

Every bar verified against `results/frm_*/summary.json` (headline.bbh / gsm8k):

| bar | run | BBH | GSM8K |
|---|---|---|---|
| Base | (registry) | 33.1 | — |
| LoRA+wd (wd0.3) | frm_lorawd_wd0p3_lr2e4_c256_s42 | 33.10 ✓ | 67.25 → 67.3 ✓ |
| MiLoRA | frm_milora_lr3e4_c256_s42 | 30.18 → 30.2 ✓ | 58.98 → 59.0 ✓ |
| LoRA | frm_lora_lr3e4_c256_s42 | 29.14 → 29.1 ✓ | 60.20 ✓ |
| CLoRA-k128 | frm_clora_k128_lr3e4_c256_s42 | 27.55 → 27.6 ✓ | 59.59 → 59.6 ✓ |
| PiSSA | frm_pissa_lr3e4_c256_s42 | 7.23 → 7.2 ✓ | 49.66 (collapses) |

- s43 sibling **65.88** verified (frm_lorawd_wd0p3_lr2e4_c256_s43 gsm8k 65.88). ✓
- "No published math retention" claim **confirmed against the CLoRA PDF**: its Math setting "use test
  set of GSM8K and MATH for evaluation" — no out-of-domain/BBH numbers for math-trained models. So "18.38
  = MATH accuracy, not BBH" is right and the retention column is genuinely unmet in the literature.
- All bar widths internally consistent (GSM8K axis 0–75, BBH axis 0–36).

**Fairness caveat (the exact concern the PI raised).** LoRA+wd is shown at **lr2e-4** (its own best-GSM8K
cell, F_Δ 0.28), while every competitor is pinned at the recipe **lr3e-4**. Crucially, MiLoRA's lr3e-4 cell
(GSM8K 59.0, BBH 30.2, F_Δ 1.26) is **not** its best operating point: `frm_milora_lr1e4_c256_s42` gets
**GSM8K 62.85 AND BBH 32.38** (only 0.7 pp below base, F_Δ 0.45). So the chart displays MiLoRA at a
worse-than-necessary point, inflating its apparent forgetting from ~0.7 pp to 2.9 pp and supporting the
"3–6 pp below base" sentence more strongly than a best-for-best comparison would. The conclusion survives
either way — at the matched recipe lr3e-4 LoRA+wd is still `frm_lorawd_wd0p3_lr3e4` = GSM8K **65.05 > 64.6
published**, BBH **33.47 ≥ base** — so the honest fix is cheap: either (a) show LoRA+wd at lr3e-4 in both
charts (still wins both), or (b) disclose that LoRA+wd is at its best swept LR while competitors are
recipe-pinned, and add MiLoRA's lr1e-4 point.

**Wording.** The BBH-chart caption says "same recipe as the adaptation chart above." Not quite: the
adaptation chart's competitor bars are **published** numbers (MiLoRA 63.5, LoRA 60.6, CLoRA 64.6, PiSSA
58.2); the retention chart's competitor bars are **our lr3e-4 pipeline** (MiLoRA 59.0, etc.). Only the
LoRA+wd bar is literally identical across the two charts. Reword to "our in-pipeline reproductions at the
recipe LR."

**Traceability to the SSOT (important).** The §4 headline **67.3** is not in `key_numbers.md`, which is
declared to "override everything." `key_numbers.md` §4 lists LoRA+wd math **GSM8K = 50.6** — verified as
`lrswm_lorawd_wd0p3_lr5e4_s42` cs_avg 50.64 (the earlier `lrswm_` sweep, evidently a shorter generation
cutoff). The `frm_` faithful-recipe (256-token) block the artifact uses is newer and not yet reconciled
into the authoritative file. Anyone checking §4 against the SSOT will hit a conflicting 50.6. Update
`key_numbers.md` §4 with the `frm_` 256-token faithful-math cells (and note the 50.6↔67.3 gap is the
generation cutoff — which is itself the artifact's stated point).

### Comment 5 — §6 CE metric explanation — RESOLVED (one wording flag)

- Recipe: WikiText-103 test, non-overlapping 1024-token blocks, 40 blocks. `forgetting.jsonl` confirms
  `n_positions = 40920`, `n_blocks = 40`, `max_length = 1024`. 40×1023 = 40,920 predicted positions. ✓
- Soft CE formula CE_t = −Σ_v p_base·log p_ft matches `forgetting_ce.py` line 113 exactly; base obtained
  via `disable_adapter()` on the same wrapped model (code + `--check_base` gate) — matches the artifact prose. ✓
- Table values all match `forgetting.jsonl` exactly: LoRA+wd 2.00 (1.9983, F_Δ 0.20/0.1959), LoRA 3.57
  (3.5703, 1.28/1.2831), MiLoRA 3.66 (3.6594, 1.26/1.2568), PiSSA 6.31 (6.3068, 2.21/2.2058). ✓
- Kalajdzievski 2024 / MiLoRA Table 8 citation correct (matches the docstring).
- **Spearman 0.94 verified** — but over all **6** cells in `forgetting.jsonl` (ρ=0.943), not the 4 shown
  in the table (those 4 alone give ρ=0.80). State "n=6 / over all measured math cells" to avoid confusion.

**Flag (clarity).** The §6 caption says "Our numbers reproduce MiLoRA's published Table 8 (their LoRA 3.24,
PiSSA 6.07, **MiLoRA 2.54**)." We do **not** reproduce MiLoRA 2.54 — ours is 3.66, and that mismatch is the
entire point of the section (MiLoRA's advantage was its lower-magnitude operating point, not geometry).
Listing 2.54 under "our numbers reproduce" is self-undercutting. Reword: "we reproduce their LoRA/PiSSA
magnitudes (3.24→3.57, 6.07→6.31); at matched LR our MiLoRA is 3.66, essentially LoRA's 3.57 — not their
2.54." Also "40×1024-token blocks = 40,920 positions" reads like an arithmetic slip (40×1024 = 40,960);
say "40,920 predicted positions (40×1023)."

### Comment 6 — §8 status table removed — RESOLVED

- The diff confirms the entire "Status update" table was deleted, including every "running now / running
  on node B, all 8 GPUs / prioritized onto GPUs as they free" row. Section retitled "External review."
- Whole-document grep for status/future/running language: the only hits are false positives inside prose
  ("designed **to do**", "**running** at a lower-magnitude operating point"). The one borderline term is
  "**mature** single-model operating points" in the gloss — a data-quality descriptor, not campaign status;
  optional to soften. No progress/roadmap/future language remains.

---

## Top remaining improvements (broader pass, prioritized)

1. **§4 operating-point asymmetry (highest).** Recipe-match everyone at lr3e-4 (LoRA+wd still wins:
   65.05 GSM8K / 33.47 BBH) OR disclose LoRA+wd's best-LR advantage and add MiLoRA's best cell
   (lr1e-4: 62.85 GSM8K / 32.38 BBH). Fix "same recipe as the adaptation chart above" (competitors are
   our-pipeline, not the published bars). A reviewer will notice LoRA+wd got LR-tuned while MiLoRA didn't.

2. **§4 traceability to `key_numbers.md`.** The SSOT holds a conflicting LoRA+wd math GSM8K = 50.6
   (`lrswm_`, short cutoff) and lacks the `frm_` 256-token block the artifact headlines (67.3). Reconcile
   so every §4 number resolves against the authoritative file.

3. **§6 caption.** Stop listing "MiLoRA 2.54" under "our numbers reproduce" — we get 3.66, which is the
   section's evidence. Add n=6 to the 0.94 Spearman and clean the "40×1024 = 40,920" phrasing.

4. **§2 law-residual consistency.** Reconcile the geometry-battery residuals (SC-LoRA −5.7, DoRA +3.6)
   with the authoritative spline residuals (SC-LoRA −4.15) / the "−4pp" guardrail — pick one baseline or
   footnote both. Reconsider printing CorDA −3.0 given the SSOT's "do not report."

5. **Two-baseline retention clarity.** The doc carries two "base reference" numbers on different scales
   (26.0 for CS = BBH+MMLU-Pro; 33.1 for math = BBH only). Both are flagged, but they co-occur in tiles/
   foot and can confuse a cold reader; one sentence ("math retention is BBH-only, so its ceiling differs
   from the CS retention scale") would remove the friction. Optional: the §2 "0.06 neutral" up/down_proj nuance.
