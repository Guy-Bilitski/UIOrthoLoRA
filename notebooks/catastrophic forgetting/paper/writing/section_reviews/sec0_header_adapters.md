# Section review — HEADER (title, dek, byline, tiles, gloss) + §0 "The adapters"

Reviewer: section-validator (header/§0 instance), 2026-07-10.
Artifact: `paper/writing/artifact_status_report.html` (lines 83–124).
Ground truth: `paper/writing/data/key_numbers.md` (authoritative), `results/campaign_summary.jsonl`
(481 rows → latest-`evaluated_at` dedup), port code (`train_cs.py`, `sclora_init.py`,
`lora_null_init.py`, `cordapp_init.py`, `milora_init.py`), method PDFs in `repro/<Method>/`.
All recomputes with `/home/guy/UIOrthoLoRA/.venv/bin/python` on the live registry.

**Overall verdict: NEEDS-WORK** — every number reproduces (no numeric errors found), but the header
carries one stale claim (Qwen), one dek overclaim that contradicts the body (SC-LoRA), and §0's lead
sentence is factually wrong for DoRA and imprecise for the SVD-init family. All fixable in prose;
no re-runs required for correctness (one small config table recommended).

---

## (a) Claim-by-claim verdict table

### Header (title / dek / byline)

| # | Claim | Recomputed / checked | Verdict |
|---|---|---|---|
| H1 | Title: "Update Size Is the First-Order Predictor of Forgetting in PEFT" | R²(F_Δ)=0.736 vs R²(logLR)=0.32; F_Δ uniquely predictive among axes (dw_sv_mean 0.36, dw_sv_max 0.33 — key_numbers §0). "First-order predictor" supported. | **CONFIRMED**, but "in PEFT" overscopes: evidence covers LoRA-family adapters on two 7B decoder models. Suggest "…of Forgetting in LoRA-Family PEFT" or a dek qualifier. |
| H2 | Dek: "nine LoRA-family adapters" | LoRA, LoRA+wd, DoRA, PiSSA, MiLoRA, CLoRA, SC-LoRA, LoRA-Null, CorDA++ = 9 ✓; matches byline "9 (8 swept on commonsense + PiSSA on math)" ✓ (8 CS-swept incl. CorDA, which is later excluded). | **CONFIRMED** |
| H3 | Dek: "at matched update size the geometric methods land on the same curve as a tuned plain-LoRA baseline" | Contradicted by the artifact's own §2 ("the two methods that fall below the size law") and key_numbers §5/§14: SC-LoRA −4.15 pp below the law, p=0.006 — the campaign's own honest boundary. | **CORRECTED** — must read "all but one land on the same curve (SC-LoRA falls below it)". As written it repeats the exact class of overclaim the previous review cycle flagged in the old dek. |
| H4 | Dek: "and they cost more to run" | §5's own table: MiLoRA/PiSSA/SC-LoRA/LoRA-Null train at 1.00×, no extra train memory; their only cost is minutes of init. | **CORRECTED (wording)** — "cost more to run" is false for MiLoRA at train time. Safer: "and none of them is cheaper." |
| H5 | Byline: Models Llama-2-7B · Qwen2.5-7B; "Last updated 2026-07-10" | ✓ | **CONFIRMED** |

### Tiles

| # | Claim | Recomputed | Verdict |
|---|---|---|---|
| T1a | Magnitude law r = −0.86 | Pooled Llama-2 CS n=49: r=−0.858, R²=0.736, slope −14.78 pp/dec — exact match. | **CONFIRMED** |
| T1b | "holds within every one of the 7 assessed adapters (r −0.86 to −0.97)" | Within-method (split convention): CLoRA −0.903, DoRA −0.969, LoRA −0.954, LoRA-Null −0.864, LoRA+wd −0.887, MiLoRA −0.938, SC-LoRA −0.972. Range −0.86…−0.97 ✓ (7 adapters, CorDA excluded). | **CONFIRMED** |
| T1c | "on a second architecture (Qwen-2.5-7B, LoRA, commonsense: r = −0.88)" | Qwen LoRA-only: r=−0.883 n=7 ✓ — but **STALE**: the full Qwen CS sweep is complete (key_numbers §11, 07-10): 7 adapters × 7 LRs = 49 cells, pooled core **r=−0.857, R²=0.735** (recomputed, matches), broad r=−0.94. The tile undersells the paper's strongest replication by a factor of 7 in coverage. | **CORRECTED (stale)** — upgrade to "replicates on Qwen2.5-7B: 7 adapters, n=49, r = −0.86 (broad −0.94)". The mirrored §1 sentence "other adapters … are not part of the assessed set on this model" is now false and must go too. |
| T2 | Best CS point 81.6 / 25.6, "best retention at the smallest update", "78–82 band" | `lrsw_lorawd_wd0p3_lr5e4_s42`: cs 81.62, ret 25.55, F_Δ 0.3941, safe band 6/7 ✓. Among the 8 best-LR rows in §3: highest retention (25.6) and lowest F_Δ (0.39) ✓; accuracy band 78.3–81.6 ✓. | **CONFIRMED**. Nit: "statistically tied with the field" — no test was run; the basis is seed-fragility of accuracy (3-seed matrix). Say "within the observed 78–82 band (accuracy ordering not seed-stable)". |
| T3 | Best math 67.3 vs 64.6, "at a far smaller update" | `frm_lorawd_wd0p3_lr2e4_c256_s42` GSM8K 67.25, BBH 33.1, F_Δ 0.278 ✓; s43 sibling 65.88 ✓ (disclosed in §4). 64.6 = CLoRA Table 4 published 64.59 [EXTERNAL, audited]. | **CONFIRMED**, with a precision caveat: "far smaller update" juxtaposes OUR F_Δ (0.28 vs 1.1–2.2 in-pipeline) with THEIR accuracy; their published F_Δ scale runs ~2× lower (per §1 callout), and math F_Δ is on the α=2r scale. Fine if read as in-pipeline; a one-word hedge ("in our pipeline") would close it. |
| T4 | Efficiency: 0 init cost; CLoRA up to 6.7 GB; DoRA 2.1× | Matches §5 exactly (k2048 +6.7 GB; DoRA 2.14× median over 7 runs). Internally consistent. Note 6.7 GB is CLoRA's published-best k2048, not our swept k1024 (~+3.4 GB) — "up to" keeps it honest. | **CONFIRMED** |

### Gloss

| # | Claim | Recomputed | Verdict |
|---|---|---|---|
| G1 | F_Δ definition (mean ‖ΔW·x‖/‖x‖ on real inputs; same metric as CLoRA Table 4) | Matches key_numbers §0 correction (CLoRA Eq 3; the old "token-weighted Frobenius" label is correctly gone). | **CONFIRMED** — this was the #1 reviewer trap and it is fixed. |
| G2 | Retention = BBH+MMLU-Pro, base 26.0; math BBH-only, base 33.1; parser rationale | ✓ (BBH-AO 33.10, MMLU-Pro 18.96, core 26.03 [EXTERNAL h00/h05]); parser rationale matches key_numbers §11. | **CONFIRMED** |
| G3 | "under heavy regularization retention can nudge slightly above it" | Max sweep retention = 27.80 (`lrsw_lorawd_wd0p3_lr5e5_s42`), i.e. +1.8 pp over base; ~⅓ of low-LR cells sit above 26.0. | **CONFIRMED**, though "+1.8 pp" stretches "slightly". Worth one honest phrase ("up to ~+1.8 pp at low LR — within the suites' eval noise band") because it also weakens the §1 "hard ceiling at 26.0" framing. |
| G4 | "retention seed-SD ≈ 0.3" on the 3-seed rank/wd matrix | Recomputed over the 34 `mtx_*` 3-seed configs: **median SD = 0.35** (0.34 excl. CorDA) — "≈ 0.3" is defensible as a median. But the **mean is 0.95**, driven by SC-LoRA (SD up to 3.9 across its β/rank cells) and `mtx_lorawd_wd0p05` (3.91). | **CORRECTED (precision)** — say "median seed-SD ≈ 0.3–0.4; SC-LoRA and near-zero-wd cells are the seed-fragile exceptions (SD up to ~4)". As written, a reviewer who computes the mean gets 1.0 and calls it a misstatement. Also note: SC-LoRA's retention being seed-fragile slightly undercuts the blanket "retention is the seed-stable axis" — and independently corroborates SC-LoRA's below-law/fragile status. |

### §0 "The adapters"

| # | Claim | Checked against | Verdict |
|---|---|---|---|
| A1 | Lead: "All nine methods share the same form — ΔW = A·B … What distinguishes them is only how that update is initialized or constrained, not the A·B form itself." | DoRA paper + our port (peft `use_dora=True`, `train_cs.py:144`): DoRA reparametrizes W′ = m · (W₀+BA)/‖W₀+BA‖_col — a trainable magnitude vector plus column normalization. Its weight change is NOT of the form W₀ + A·B. §0's own DoRA sentence ("splits each weight into a magnitude and a direction… learning the magnitude separately") contradicts the lead two paragraphs later. | **CORRECTED** — needs a DoRA carve-out: "eight of the nine share…; DoRA additionally learns a per-column magnitude on top of the low-rank direction update." This is the exact inconsistency the mandate flagged. |
| A2 | Lead: "added on top of the frozen pretrained weights" | For the SVD-init family (PiSSA, MiLoRA, SC-LoRA, LoRA-Null, CorDA) the frozen matrix is the **residual** W_res = W₀ − B·A, not W₀ (loss-preserving init; cf. `residual_save.py` machinery). Total starts at W₀; the frozen part does not. | **CORRECTED (precision)** — "…so that the model starts exactly at the pretrained weights (for the SVD-initialized methods, the frozen part is the residual after carving out the initialization)". |
| A3 | Factor convention: §0 writes "ΔW = A·B … A random, B zero" | §2 and §7 write ΔW = (α/r)·B·A (standard LoRA order: B ∈ R^{d×r} zero, A ∈ R^{r×k} random). §0 flips the product order and under the flipped order "A random, B zero" is ambiguous. | **CORRECTED (cosmetic)** — use B·A everywhere, one convention, and note the α/r scale once. |
| A4 | LoRA description ("A random, B zero, update starts at zero, grows freely; size governed by LR, rank, training length") | LoRA paper + port. | **CONFIRMED** |
| A5 | LoRA+wd: "AdamW weight decay on the adapter factors A, B … shrinks the L2 norm of the factors; indirectly reduces measured F_Δ; does not penalize ΔW directly; no geometric structure; one scalar knob, zero extra cost" | `train_cs.py:181,400` — `weight_decay` → HF `TrainingArguments` → decoupled AdamW decay on all trainable params (= adapter factors only; bias/LN excluded by Trainer defaults). Description is mechanistically exact, including the honest "indirect" framing (decay on factors ≠ penalty on ΔW — the distinction CLoRA's own LoRA-L2 baseline makes). | **CONFIRMED** — this description is the best-written of the nine. |
| A6 | DoRA description | DoRA paper (magnitude/direction decomposition; motivation = close FT gap). | **CONFIRMED** (and correct that DoRA never targeted forgetting). |
| A7 | PiSSA: principal-directions init, "updates tend to be large and, in our study, forget the most" | Paper ✓; data: `frm_pissa_lr3e4_c256_s42` ret-core 3.62, BBH 7.2, F_Δ 2.21 — worst in study ✓. | **CONFIRMED** |
| A8 | MiLoRA: minor-singular-directions init, premise = leave dominant knowledge undisturbed | Paper + `milora_init.py` ✓. | **CONFIRMED** |
| A9 | CLoRA under heading "Data-aware / constrained"; "orthogonality penalty… pushes the update into a null space of size k" | `train_cs.py:29-49` (ports github.com/sutakori/CLoRA exactly): penalty λ(½‖A·P_v‖² + ½‖Bᵀ·P_u‖²) with **frozen RANDOM-orthonormal** P_u, P_v. (1) CLoRA is **not data-aware** — the heading mis-files it; it uses no calibration data at all. (2) Direction of the sentence is inverted: the penalty pushes k random directions into the update's row/column **null spaces** (ΔW annihilates them), not "the update into a null space". | **CORRECTED** — split the heading ("Constrained (data-free)" vs "Calibration-based") or say "a randomly chosen subspace". The random-ness is not a nitpick — it is thesis-supporting (see (e)). |
| A10 | SC-LoRA: "initializes inside a subspace computed from calibration data to preserve target knowledge, with a scalar β trading preservation against freedom; constraint set once at init and (per its paper) erodes during training" | `sclora_init.py` (faithful port, verified vs repo): subspace = top-r eigvecs of (1−β)·Cov_out(D+ = task) − β·Cov_out(D− = knowledge-to-preserve). "to preserve target knowledge" garbles the two datasets — the subspace **aligns with the target task while avoiding** preserved-knowledge output directions; β trades task-alignment against preservation. Erosion claim ✓ (paper limitation; our §2 measures 0.70→0.21). | **CORRECTED (wording)** — e.g. "initializes in output directions that serve the fine-tuning task while avoiding those that carry the knowledge to preserve (balance set by β)". |
| A11 | LoRA-Null: "initializes in the null space of calibration activations, so the update's outputs start orthogonal to the directions that carry preserved knowledge" | `lora_null_init.py`: A's rows lie in the null space of knowledge-**input** activations ⇒ ΔW·x ≈ 0 on preserved-knowledge inputs. The update's output is (near-)**zero** on those inputs — not "orthogonal to knowledge directions", and the mechanism is input-side, not output-side. | **CORRECTED (wording)** — "so the update initially does nothing (ΔW·x ≈ 0) to inputs drawn from the preserved-knowledge distribution". Optional honesty footnote: the paper's best-preservation variant freezes A; we train A and B like every other arm (head-to-head fairness) — flag exists in the port docstring. |
| A12 | CorDA++: "context-oriented SVD weighted by calibration-activation covariance, plus dynamic per-layer rank; can run in a knowledge-preserving mode that adapts the least data-relevant directions" | `cordapp_init.py` (KPM = bottom-r context-oriented directions; dynamic covariance selection + dynamic rank allocation). | **CONFIRMED** (omits dynamic covariance selection; immaterial at this altitude). |

**Numeric bottom line: 0 wrong numbers.** All header numbers reproduce exactly from the registry
(pooled r −0.858/R² 0.736/slope −14.78; within-method −0.864…−0.972; 81.62/25.55/0.394/6-7;
67.25/65.88/F_Δ 0.278/BBH 33.1; Qwen LoRA −0.883; Qwen pooled −0.857 available). The problems are
staleness, scope, and mechanism-wording — not arithmetic.

## (b) Logic / hidden-assumption notes

- **Header coverage arithmetic (9 → 8 → 7)** is never explained above the fold: dek says nine, byline
  "8 swept", tile 1 "7 assessed". The CorDA exclusion (calibration-fairness, key_numbers §8) first
  appears in a §2 footnote. One parenthetical in the byline — "(CorDA withheld: calibration fairness)"
  — pre-empts the supervisor's first counting question.
- **"Assessed" load-bearing word in tile 1** quietly includes SC-LoRA, which is *on the within-method
  law* (r −0.97) but *below the pooled curve* (−4.15 pp). That is fine — within-method slope and
  pooled intercept are different claims — but the dek (H3) collapses the distinction; only the tile
  survives scrutiny.
- **Ceiling/censoring**: the within-method r's in tile 1 are computed on retention values partially
  pinned at the ~26 base level for low-LR cells; linear r on a saturating curve understates nothing
  here (r's are already −0.86…−0.97) and §1's Spearman/spline callout covers the pooled fit, so no
  statistical change needed for the header — but G3's "+1.8 above base" should be acknowledged once
  (eval noise), since "ceiling" language appears four times before §1.
- **§0 grouping = implicit taxonomy claim.** The three headings (Baselines / SVD-initialized /
  Data-aware-constrained) are an analytical claim about mechanism classes, and it is wrong for CLoRA
  (random, data-free). The cleanest taxonomy the data supports: (i) unconstrained baselines
  (LoRA, LoRA+wd, DoRA), (ii) weight-spectrum inits (PiSSA, MiLoRA), (iii) data-free constraint
  (CLoRA), (iv) calibration-based inits (SC-LoRA, LoRA-Null, CorDA++). This matches §2's fingerprint
  table structure almost exactly — the artifact already proves this taxonomy in §2 and then mislabels
  it in §0.

## (c) Tile ↔ body consistency after edits

- T1 ↔ §1: internally consistent with each other — and both stale on Qwen (see T1c).
- T2 ↔ §3: consistent (81.6/25.6/0.39/6-7; 78–82 band; base row 26.0).
- T3 ↔ §4: consistent (67.3, 64.6, 0.28, s43 65.88 disclosed; "edges" verb kept).
- T4 ↔ §5: consistent (2.1×↔2.14×, 6.7 GB, zero init).
- Gloss retention ↔ §4/§6 BBH-only convention: consistent.
- Dek ↔ §2: **inconsistent** (H3).
- Dek "cost more to run" ↔ §5 table: **inconsistent** for MiLoRA/PiSSA (H4).

## (d) What is MISSING that belongs in header/§0

1. **The Qwen full replication** — the strongest evidence added since the last edit, currently
   invisible (tile shows the obsolete LoRA-only arm). Zero cost: numbers are in key_numbers §11 and
   reproduce from the registry.
2. **A per-adapter configuration table in §0** (rank r, α, k/β, calibration set, trainable params,
   init type, "claims forgetting-prevention? Y/N"). key_numbers §13's "ranks NOT matched"
   (LoRA/DoRA/LoRA-Null r16; MiLoRA/SC-LoRA r32; CLoRA k1024) appears nowhere in the artifact — a
   supervisor's second question after seeds. Frames the law (not a method ranking) exactly as the
   guardrails require. Cost: ~10 lines of HTML, data already known.
3. **One honest-boundary clause in the dek/header** (SC-LoRA below the law; high-k CLoRA boundary;
   Qwen-math pending). The body earns credibility with these; the header currently hides all three.
4. **A geometry tile** — §2 is the paper's most novel section and has no header representation.
   Candidate: "Geometry: a fingerprint, not a shield — placement adds ΔR² ≈ 0.0002 beyond size."
5. (Optional) the CLoRA-Table-4 external replication (their slope −14.7 vs ours −14.8) is
   tile-strength evidence sitting in a §1 callout.

## (e) New insights from the raw data this section should surface

1. **CLoRA's constraint subspace is RANDOM — and that is thesis-gold.** Its retention gain cannot
   come from targeting knowledge directions (there is no data in the loop); it must come from what
   the constraint does to the update — and our own k-series shows exactly that:
   `clora_cs_k128→k2048`: F_Δ 0.60 → 0.58 → 0.51 → 0.46 → 0.34 monotonically, retention 22.5 → 25.7.
   One §0 sentence ("the null space is drawn at random — k is, in effect, a magnitude knob, which is
   how we will find it behaves in §1") turns an adapter description into foreshadowing of the thesis.
2. **Seed-fragility is method-structured** (from the 40 three-seed configs): retention seed-SD
   median 0.35, but SC-LoRA's cells reach SD 3.2–3.9 while every LoRA/LoRA+wd/CLoRA/DoRA cell with
   wd≥0.1 sits ≤0.5. The gloss can quote this precisely and gain a small extra argument for both the
   "law is seed-stable" claim and SC-LoRA's fragility.
3. **LoRA-Null trains A** (paper's best variant freezes it) — a one-line fairness disclosure in §0
   that pre-empts a LoRA-Null author's rebuttal, and possibly part of why its e_top fingerprint in §2
   drifts to the output side.

## (f) Prioritized strengthening list (with costs)

| P | Action | Cost |
|---|---|---|
| 1 | Fix dek H3 ("all but one") + H4 ("none is cheaper"); update tile T1c and the §1 mirror sentence to the full Qwen replication (7 adapters, n=49, r=−0.86 / broad −0.94). | 15 min editing; numbers already verified. |
| 2 | Rewrite §0 lead with the DoRA carve-out (A1) + residual precision (A2) + one factor convention B·A (A3). | 15 min. |
| 3 | Fix the three mechanism mis-wordings: CLoRA random/data-free + heading split (A9), SC-LoRA β wording (A10), LoRA-Null ΔW·x≈0 wording (A11); add the CLoRA random-subspace foreshadowing sentence (e1). | 30 min. |
| 4 | Add the §0 per-adapter config table (d2) incl. ranks-not-matched disclosure and "claims forgetting-prevention?" column. | ~1 h (data known; no runs). |
| 5 | Gloss G4: "median seed-SD ≈ 0.3–0.4; SC-LoRA/near-zero-wd exceptions up to ~4" with the recomputed numbers. | 10 min. |
| 6 | Optional: geometry tile (d4) and/or CLoRA-Table-4 tile (d5); byline "(CorDA withheld: calibration fairness)". | 20 min. |
| 7 | Optional run (only if reviewers push on H1 scope): one non-LoRA-family control (e.g. IA³ or full-FT low-LR point) to license "in PEFT" unqualified — otherwise re-title. | 1–2 GPU cells (~5 GPU-h each) vs free re-title. Recommend re-title. |

No corrective re-runs are required for this section: every number is right; the work is prose surgery
plus one config table.
