# 16-GPU two-node plan (research-planner, 2026-07-09)

Node A (this host, d001-source): 8×B200, 4 live pools + auto_dispatch (owns the 147GB of adapters + all
analysis). Node B (d002 / test-gpu02): 8×B200 idle, 42GB /home, no /scratch, bare — being provisioned
deliver-from-A (repo + PEFT fork + Qwen weights + datasets rsync'd from A; gated Llama stays on A only).

## Deadline math
~88 wall-h to Sun EOD. Demand ≈ 883 GPU-h (199 queued cells). Supply: A 704 + B ~640 = 1344 GPU-h →
~35% headroom. 8-GPU counterfactual: 704 vs 883 = ~180 GPU-h short = the ENTIRE Qwen block dropped.
**The 2nd node SAVES Qwen** (2nd-model+2nd-task law replication) + full 3-seed robustness + full CE batch.

## Priority order
1. CS reservoir spine (65 frc_ cells, 0 done — the paper's spine) — SPLIT across A+B → <1.5 days.
2. main5 (CLoRA k-grid s42 + frc_lora_l2 + 48 competitor math) — stays on A, don't kill (mid-flight adapters).
3. 3-seed headlines — ADD 6 new CS cells (see below) + finish queued seeds.
4. inject (7) + b4 (8) — on A, completes SC-LoRA/CorDA++ boundaries.
5. CE-to-base full batch over ~390 adapters — A ONLY (reads local /scratch adapters), eval-only.
6. Qwen block (57 cells: 51 math + 6 CS-CorDA) — NODE B workhorse; run qwswm_lorawd_wd0p3 math LR-sweep
   (incl 5e-4/1e-3) FIRST → converts Qwen-math anti-replication (r=+0.67, high-LR unrun) into positive
   2nd-model replication. Front-loaded so a short B window still secures the claim.
7. c2048 math anchors (MATH-offset attribution) — 3 queued in main5 + 1 new winner cell.
8. 2nd-model (Qwen) geometry-drift + CE — CPU on B after Qwen adapters land; sync labeled JSONL to A.
9. Writing (no GPU): ceiling-aware stats, efficiency table, fdelta→F_Δ relabel, cross-literature overlay.

## Sharding (locality-driven): adapters never leave A; B trains fresh, syncs summary JSON only (hourly rsync B→A).
- A: keep 4 pools draining; CE-batch + geometry on A GPUs as they free; dispatcher pulls CS spine.
- B: 8 GPUs → CS spine split → 3-seed headlines → Qwen (lorawd arm first) → c2048.
- COORDINATION: once B confirms it owns Qwen, REMOVE the 57 Qwen lines from A's master_dispatch (avoid
  duplicate work across non-shared filesystems; skip-done + sync partially guards but explicit is safer).

## 7 genuinely-new cells (everything else already queued):
1-2. frc_lorawd_wd0p3_lr5e4_c256 {s43,s44}   (CS LoRA+wd winner seeds)
3-4. frc_lorawd_wd0_lr3e4_c256 {s43,s44}      (CS plain-LoRA best seeds)
5-6. frc_milora_lr3e4_c256 {s43,s44}          (CS structured competitor seeds)
7.   frm_lorawd_wd0p3_lr2e4_c2048_s42          (winner MATH-offset anchor)
(Optional Tier-B: qwswm_lorawd_wd0p3_<bestLR> s43/s44 if Qwen elevated to co-headline.)

## Node-B bootstrap checklist: repo (download/rsync ortho_new) · venv (torch cu13x/transformers 5.10.2/
trl 1.5.1 + editable PEFT fork) · Qwen weights + nq_open/mmlu/wikitext-103 (deliver from A) · /scratch→
/home/ubuntu/cf_models symlink · smoke gate one frc_/qwsw cell (no rc=127, no nq_open→wikitext fallback,
base BBH≈33, residual Δ≈0) · hourly summary-JSON sync B→A · setsid dispatcher.

Full plan: tasks/afcf6f68742e543a5.output.
