# NODE HANDOVER — 2026-07-13 (drain in progress; final numbers appended at shutdown)

PI ordered nodes returned for reprovisioning (~13:00, superseding the Tuesday plan).
This doc is the authoritative ledger of: what is preserved where, what dies with the
hosts, every piece of WANTED-BUT-NOT-RUN work (with pointers to its exact runnable
commands), and the resume recipe.

## 1. Preservation map (everything here is in git, branch ortho_new)
- results/*/summary.json + run_config.json — every landed cell, BOTH nodes (synced
  each check; final syncback at shutdown).
- results/train_registry.jsonl — MERGED A+B (B's 93 rows pulled 2026-07-13, includes
  peak_mem_init_gb/peak_mem_train_gb for instrumented runs).
- results/forgetting*.jsonl — CE-to-base union (415+ scored runs; ce_chunk9_final
  catch-all runs as A's last job).
- results/geo_drift/ — Llama battery (adapter_metrics.jsonl, 478+ rows incl. the
  2026-07-13 extension) + Qwen battery (adapter_metrics_qwen.jsonl, 100 rows,
  base_svd_qwen/, permatrix_qwen/) + permatrix/ per-matrix data.
- paper/writing/artifact_status_report.html — artifact source of truth; live URL
  https://claude.ai/code/artifact/5c46636f-036a-4fae-919f-43be8e07639c (publish ONLY
  with url:, favicon 📉).
- handoff/34 (state), 35 (plan validation), 36 (completeness audit), this doc.
- jobs/*.txt — full provenance; CANCELLED-FOR-HANDOVER comment lines preserve the
  exact command of every cancelled cell.
- portable_parity_pack/ — standalone train+eval bundle (see §3.1).

## 2. LOST at reprovisioning (accepted by PI 2026-07-13)
- /scratch/cf_models adapters, both nodes (~573 on A incl. 44 pulled from B).
  All derived data extracted first: eval summaries, CE scores, geometry fingerprints.
  Every adapter re-trainable from its run_config args + seed.
- HF model/dataset caches (re-downloadable; Llama-2 is gated — needs licensed token).
- TRAINING DATA JSONs were never in git (too big): commonsense_170k.json (from
  LLM-Adapters repo), metamathqa_395k.json (sha256[:32]=13c5920ac97dc4afa9d4420701533f5b,
  reconstructor: portable_parity_pack/fetch_data.py), metamathqa_100k.json (subset of
  the 395k used for Qwen math cells — first-100k slice; verify against run behavior).

## 3. OPEN WORK LEDGER — wanted but not run (ranked by paper value)
1. **Llama math parity block (37 cells, ~205 GPU-h)** — 3-seed best points for all 8
   math competitors + CORE-4 LR fills. THE top reviewer-defense item (fixes the
   single-seed, LoRA+wd-dominated math table). PACKAGED: portable_parity_pack/
   (README, pinned reqs incl. peft-fork install, resumable runner). Run [P1] first.
2. **Qwen operating-point seeds, 15 cancelled cells** — s43/s44 pairs for CS
   dora/clora/lora/lora_null and math milora/clora/dora/lora/lora_null. Commands: the
   CANCELLED lines in jobs/qwen3seed_B.txt (B's trimmed copy; repo copy holds the full
   original 29-cell plan). Landed before cancel: base_qwen25_noft, math lorawd pair,
   CS lorawd/milora/sclora/lora/lora_null pairs (see results/), math sclora pair (kept
   in drain). ~85 GPU-h remaining.
3. **Collapse-basin seed pairs (4 cells)** — r16/lr3e4 + wd0.3/lr1e-4 accuracy-collapse
   cells at s43/s44 (turns "scattered basin collapses" into a seeded phenomenon).
   Commands: CANCELLED section [8] in jobs/master_dispatch.txt. ~30 GPU-h.
4. **b4 confound extension + remaining wd-grid/reservoir** — CANCELLED lines in
   master_dispatch (b4_lora_null/b4_cordapp retries, frc reservoir). ~60 GPU-h.
5. **c2048 cutoff arm** — currently IMPOSSIBLE at recipe defaults: ANY method OOMs a
   178 GB B200 at cutoff 2048 / micro_batch 16 (LoRA and CLoRA alike, verified).
   Recipe fix: --micro_batch_size 4 (grad-accum handles the rest), ~12 GPU-h/cell.
   Not queued anywhere; would add a sequence-length axis to the law.
6. **Third-architecture replication (never queued; biggest scientific upside)** — e.g.
   Mistral-7B: LoRA+wd + plain LoRA + 1-2 geometry methods × 7 LRs ≈ 21-28 cells
   ≈ 120-160 GPU-h for a CS-arm law replication. Would make the law 3-model.
7. **SC-LoRA adaptation-edge follow-ups** (if drain verdicts confirm the edge):
   β-sweep on Qwen math (isolate Cov+ task-alignment vs Cov- preservation), and an
   eval-matched-calibration arm like the CS b4 study. ~40-60 GPU-h.
8. **Artifact/doc work (zero GPU)**: section reorder (results before analyses; needs
   full §-cross-reference renumbering) + explicit 2×2 per-model/per-arm law-stats
   table (PI-requested, scheduled for the final pass); W7 geometry-pooling
   documentation (verify the fingerprint pooling method in the geo analysis before
   writing it); math-law n unification (W1, final numbers at freeze).
9. **paper.tex — PAUSED by PI.** claims_coverage_audit_sat.md blockers B2-B5 unfixed
   (B1 partial, 042c37db); handoff/36's artifact fixes supersede several audit items —
   re-run the audit against the FINAL artifact before writing.

## 4. Resume recipe (fresh environment)
1. Clone repo, checkout ortho_new. `python3.12 -m venv .venv`, `pip install -r
   portable_parity_pack/requirements.txt`, then `pip install -e .` (repo root = the
   peft FORK — stock peft breaks residual-init adapters and lacks CLoRA/DoRA hooks).
2. HF login (Llama-2 gated; Qwen public). First eval self-patches lm-eval BBH metric
   (bbh_metric_fix.py).
3. Rebuild data JSONs (§2). Calib sets (nq_open) auto-download.
4. Infra: auto_dispatch.py (reads jobs file ONCE at startup — restart after queue
   edits), gpu_watchdog.sh (10-min heal loop), launch via setsid only, kill+relaunch
   as separate calls. Full operational rules: handoff/34 §OPERATIONAL RULES.
5. Un-cancel queue lines by deleting the `# CANCELLED-FOR-HANDOVER 2026-07-13: ` prefix.

## 5. Final drain results — APPENDED AT SHUTDOWN
(to fill: r32 control verdict, ep6 verdict, sclora math seed verdict, CS seed rows,
ce_chunk9_final/geo final counts, shutdown checklist confirmation)
