# Resume prompt — paste verbatim when compute returns (written 2026-07-14, campaign frozen at 5660d62e)

## The prompt (paste this, filling the one blank)

You are resuming the UIOrthoLoRA magnitude-law campaign as autonomous supervisor after
the July-2026 node handover. Compute available now: <DESCRIBE: hosts/ssh, GPU count+type,
disk, until when>. FIRST ACTIONS, in order: (1) read
handoff/37_NODE_HANDOVER_2026-07-13.md end-to-end — it is the final ledger (what exists,
what was lost, the ranked open-work list §3 with runnable commands, the resume recipe §4)
— then handoff/38 (this file) §Notes; (2) verify/rebuild the environment per handoff/37
§4 (repo IS the peft fork — pip install -e repo root; reconstruct the training-data JSONs;
HF login, Llama-2 gated; clone the public reference repos into repro/ if port work is
needed); (3) BEFORE queueing anything, audit results/ for cells that landed after the
final artifact publish (2026-07-14 06:30) and fold them (several P1 seed pairs were
mid-eval at termination — their run_names are in handoff/37 §5 release note; any with
summaries in git are fold-ready); (4) then work the open-work ledger handoff/37 §3 in
rank order, sized to the compute you actually have — rank 1 (math parity remainder:
diff jobs/qwen3seed_B.txt + master_dispatch CANCELLED/MOVED lines against landed
summaries to get the exact remaining cells; portable_parity_pack/ has the same cells
standalone), rank 2 (Qwen seed remainder), rank 3 (basin seeds), rank 6 (third
architecture — biggest upside if compute is plentiful). Rebuild the infra per
handoff/34 §OPERATIONAL RULES (dispatcher reads jobs file ONCE at startup; setsid only;
kill+relaunch separate calls; kill remote procs by PID never pkill -f). Recreate the
2-hourly campaign-check cron (CronList first — session crons die) using the check spec
in handoff/34_NEXT_AGENT_PROMPT.md step 1, adapted to the new hosts. Artifact CRITICAL:
publish only with url: https://claude.ai/code/artifact/5c46636f-036a-4fae-919f-43be8e07639c
(else you mint a duplicate); source of truth paper/writing/artifact_status_report.html;
scratchpad-copy publish procedure per handoff/34 step 4; favicon 📉; PI guardrails:
constructive framing, no "geometry doesn't matter", bold claims need 3 seeds, no
pending/roadmap content. paper.tex remains PAUSED — ask the PI before touching it
(when unpaused: re-run the claims audit against the FINAL artifact first; the writing
should mostly transcribe the artifact + handoff/36/37 verified numbers). Standing
orders: all GPUs busy per plan, sync results to GitHub at every wake-up and milestone,
nothing lives only on hosts, report honestly including failures.

## Notes (context the prompt references)

- **State freeze:** branch ortho_new @ 5660d62e (2026-07-14 ~09:05). Artifact final
  publish stamp 2026-07-14 08:52. ~730 result cells in git; CE union 438 records;
  geometry batteries: Llama 500+ / Qwen 125 adapters (metrics + permatrix in git;
  base-SVD tensors recomputable via geo_drift_phase1*.py).
- **Lost with the hosts:** all trained adapters, HF caches, training-data JSONs
  (reconstruction: portable_parity_pack/fetch_data.py + handoff/37 §2 checksums),
  base-SVD tensors. Every run re-trainable from run_config args + seed.
- **Mid-eval kills at release (no summaries):** frm_lora_null_lr1e4 s43/s44,
  frm_dora_lr3e4 s43, frm_pissa_lr3e4 s43/s44 (B), frm_cordapp_lr3e4 s43/s44 (B),
  qwswm_clora_k1024_lr3e4 s43/s44, qwswm_lora_null_r16_lr1e3 s44, qwswm_lora_r16_lr1e3
  s43 — these are the first cells to re-run (they complete P1 + Qwen seed rows).
- **Where the science stands:** the 2×2 law table + all verdicts are in the artifact
  (results-first structure). Confirmed-and-scoped: SC-LoRA adaptation-efficiency
  boundary (Qwen math +9.5 / Llama math −6.3 vs LoRA+wd, both 3-seed, mechanism =
  input-principal placement); accuracy is the seed-fragile axis (basins at stable F_Δ
  and retention: milora math, clora k1024, r16 lr3e4); retention never moved without
  F_Δ moving, anywhere.
- **Memory index** (auto-loaded) points here via qwen-3seed-plan.md; the reference
  repos in repro/ are public clones (CLoRA, CorDA, LLM-Adapters, LoRA-Null, MiLoRA,
  SC-LoRA) — re-clone only if porting work resumes.
