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

## 5. FINAL LEDGER — completed 2026-07-14 06:35 (night window through Tue 07:00)

**Night-window verdicts (all folded into the artifact, final publish 06:30):**
- SC-LoRA Qwen-math adaptation edge: CONFIRMED across every control — 3-seed 76.3±1.0
  GSM8K at F_Δ 0.181–0.183 / retention 46.6±0.7; size-matched plain-r32 control plateaus
  58–65 (diverged at lr1e-3, F 20.9); epoch controls closed both directions (plain 6ep
  regresses 64.1→62.4 as F grows; lorawd 6ep 71.7 at F 0.10→0.11 — wd caps magnitude);
  geometry mechanism: 33–39% update energy in input-principal dirs vs 5–7% others;
  Llama-math contrast negative (edge is base-model-dependent).
- SC-LoRA Qwen-CS 9.4-retention outlier RESOLVED: seed-level magnitude event
  (ret 9.4/36.2/37.9 tracks F 0.44/0.30/0.30; accuracy stable 87.2±0.2).
- CLoRA boundary both k-points 3-seed: k1024 ret 24.39±0.13 (acc basin 59.7 in s43),
  k2048 ret 25.17±0.13 / acc 70.0±5.1. Qwen math lorawd 3-seed 69.0±3.3 / 47.5±0.4.
- Qwen CS rows 3-seed: lorawd 86.9±0.6/40.7±0.3, milora 87.4±0.1/38.8±0.9,
  lora 85.6±2.6/38.3±0.5, sclora 87.2±0.2 (ret = magnitude event above); lora_null 2-seed.
- Qwen math milora 3-seed 61.9±10.0/44.7±0.9 (accuracy basin s43=50.4 at stable F/ret).
- Llama math P1: plain-LoRA triplet 59.6±1.5 / 28.2±0.9 folded; sclora/lora_null/
  pissa/cordapp pairs were mid-eval at freeze — summaries that landed after the final
  publish are IN GIT but not folded into tables (fold from results/ on resume).
- Final artifact structure: results-first section order (CS results §1, math §2, law §3,
  geometry §4; all § cross-refs renumbered and verified), 2×2 per-model/per-arm law-stats
  table (Llama CS 49/−0.86/−14.8/ρ−0.90; Llama math 39/−0.82/−14.3/ρ−0.89; Qwen CS
  49/−0.86/−32.0/ρ−0.78; Qwen math 48/−0.69/−14.8/ρ−0.71 — single-seed sweeps, diverged
  excluded, definitions in caption), W7 pooling documented (‖ΔW‖_F-weighted means —
  verified from geo_drift_phase2.py, NOT energy-squared).
- Final extraction: CE catch-all chunk10 ran 05:53 (+8 late adapters; union 438 records /
  385 finite pairs); geometry batteries final: Llama+9, Qwen+14 (Qwen battery 125 adapters).

**Shutdown checklist:**
- [x] Dispatchers + watchdogs killed BOTH nodes (05:50) — no new launches
- [x] All A background loops killed (sync_d002 loop, nodeB_syncback, results_book_loop,
      evacuate) 06:33; B loops killed 05:50
- [x] B-only adapters pulled to A (final round 05:50); B registry deltas merged (+11)
- [x] B logs pulled to A logs/nodeB_final/ (host-only; cited evidence logs in git
      at paper/evidence/)
- [x] Final CE + geometry extraction over all adapters
- [x] Final artifact published (2026-07-14 06:30 stamp) + mirrored to repo
- [ ] LAST results sync at ~07:20 (running evals may still land; sweep before hosts go)
- Data that will finish after the last sync is LOST (disclosed; re-trainable from configs)

**RELEASE 2026-07-14 ~09:05: all compute killed on BOTH nodes (GPUs 0%/0MiB verified); mid-eval tail cells terminated (lora_null/dora P1 partials on A; pissa/cordapp/qwen tail on B) — training adapters were already geometry/CE-extracted; their eval summaries do not exist. Nodes clean for reprovisioning.**

**FINAL AUDIT APPLIED 2026-07-14 10:45** — data-verifier full sweep (edit list A1-A9: CE seed spreads for PiSSA 6.53±0.22 / CorDA++ 5.03±0.79, CS CE union 282 cells rho 0.97, math union 106, faithful-math rho 0.974, MiLoRA llama-math 3-seed 63.7±0.8, minor consistency fixes) + cross-architecture geometry table (Qwen fingerprints: every design signature replicates) + slope-family chart. Verified-clean list covers every other displayed number. Filter-dependent counts flagged in the audit (C1-C5) for camera-ready.
