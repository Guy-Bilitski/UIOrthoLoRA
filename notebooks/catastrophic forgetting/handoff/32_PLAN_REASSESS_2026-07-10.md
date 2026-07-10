# handoff/32 — FRESH-EYES PLAN REASSESSMENT (2026-07-10, PI order)

Research-planner deliverable. Every number below was **recomputed this pass from
`results/*/summary.json`** (not copied from key_numbers/handoffs); queue states verified live on
both nodes. Thesis under test: *retention after PEFT is governed by F_Δ; geometry acts through the
size/allocation of the update, not as an independent lever; LoRA+wd is the best efficient operating
point; published adapter wins are LR/recipe artifacts on the same law.*

Verified state at reassessment time:
- **Node A**: 475 `summary.json`; dispatcher live (pid 1965351, 146 jobs, **4 done / 142 pending**);
  8/8 GPUs busy on lrsw_ s43/s44 seeds + `frm_lorawd_wd0p3_lr2e4_c256_s44` + `b4_sclora_r32_lr5e4`.
  **0 of 14 lrsw_ seed cells landed yet** (all mid-flight). **frc_ spine: 0 of 77 done.**
- **Node B (d002)**: qwenB dispatcher live, 8/8 busy; **9 of 44 Qwen-math cells done, 35 pending**
  (~175 GPU-h ≈ 22 h wall → drains ~Saturday morning).
- Budget: ~2.5 days × 16 B200 ≈ 960 GPU-h total; A's pending queue alone ≈ 710 GPU-h → **~45% of
  A's queue cannot run** without re-prioritization. Data freeze Saturday EOD.

---

## (A) THESIS vs DATA — independent recompute

### A.1 The law itself: VERIFIED, exactly as published
- **Llama-2 CS canonical (lrsw_, s42, CorDA excluded): n=49, r=−0.858, R²=0.736, slope −14.78,
  Spearman −0.896.** Matches key_numbers §1/§14 to the 3rd decimal.
- **Qwen CS: n=49, core r=−0.857 (slope −31.98), broad r=−0.937 (slope −26.10).** Matches §11.
- **mtx_ 3-seed matrix: n=102, r=−0.900, Spearman −0.960.** Median per-cell retention seed spread
  **0.79 pp** (34 multi-seed cells); worst spreads are all extreme-knob cells (SC-LoRA b0.5 9.2pp,
  wd0.05 8.4pp).
- No crack found in the central claim. The thesis as framed is supported.

### A.2 NEW since the last plan — three story-relevant changes

**1. Qwen math is no longer flat — but the current "law" hangs on ONE point.**
With the 5 newly-landed high-LR cells (verified on A, union n=15):
- clean fit (excl. SMOKE + the wd lr1e-3 explosion): **n=13, r=−0.733 (BBH-only), slope −18.7**;
- **excluding the single diverged `qwswm_lora_r16_lr1e3` point (BBH 15.96), r collapses to −0.39.**
- Also: **`qwswm_lorawd_wd0p3_lr1e3` EXPLODED** (F_Δ=15.76, BBH 0, GSM8K 0) — on Qwen math, wd0.3
  did *not* prevent divergence at lr1e-3.
Verdict: keep reporting Qwen math as **pending** (current framing is correct). The 35 in-flight
Node-B cells (6 adapters × 7 LRs incl. 5e-4/1e-3) are exactly the discriminating data — they remain
compute priority #1 overall. Also note **11 of 13 clean cells sit at the Qwen BBH ceiling
(44.3–48.0)** — a linear fit is the wrong instrument here; the censored/ceiling-aware fit (PI
workstream 3) is what will make this arm interpretable.

**2. b4 eval-matched cells (3 done, 1 in-flight): the SC-LoRA off-curve deviation looks like a
calibration artifact — the thesis's one exception dissolves.** Residuals vs the pooled Llama law
(ret = 17.85 − 14.78·log10 F_Δ), same LRs:

| LR | b4 (eval-matched calib) resid | original lrsw_ resid |
|---|---|---|
| 5e-5 | **+0.05** | +0.89 |
| 1e-4 | **+2.70** (ret 27.0 vs 16.4!) | −2.80 |
| 3e-4 | **+1.79** | −6.25 |

Eval-matched calibration halves F_Δ at the same LR and puts SC-LoRA **on (slightly above) the law**,
at unchanged adaptation (cs 80.1↔80.1). If `b4_sclora_r32_lr5e4` (running now) confirms, the paper
can upgrade "6 of 7 on-curve + 1 provisional deviator" to **"all assessed adapters on the law; the
one apparent deviation was a calibration-distribution confound"** — a thesis-completing result that
was sitting un-analyzed. (Caveat honestly: eval-matched calib is an idealized condition; keep the
nq_open arm as the realistic one.)

**3. The frm_ faithful-math block is itself a second, unclaimed law.** BBH-only vs log F_Δ:
**n=49, r=−0.916, slope −13.9** (−0.830 with the 2 diverged cells removed; slope unchanged). This
supersedes the sparse old `lrswm_` n=14 row as the Llama-math law and is nearly as strong as the CS
law. Zero new compute; just claim it.

### A.3 Over-claims / anomalies to fix in the current framing
1. **"LoRA 64.97 beats CLoRA published 64.59" is within seed noise — do not headline it as a win.**
   The only 3-seed frm_ math cell (`wd0p2_lr1e4`) has GSM8K spread **3.1 pp** (67.4/64.52/64.29);
   the claimed margin is 0.38 pp. Correct framing: **plain LoRA at a tuned LR *matches* published
   CLoRA (64.97 vs 64.59); LoRA+wd0.3 *exceeds* it (67.25 s42 / 65.88 s43, both > 64.59)**. The
   LoRA+wd claim survives seed noise; the plain-LoRA "win" does not. s43/s44 for wd0 lr1e4 are
   queued and will settle it — pre-register this fallback framing NOW.
2. **wd is not a free lunch at the stability boundary**: Llama math lr7e-4+wd0.5 exploded (F=57)
   and lr1e-3 exploded at every wd; Qwen math wd0.3 exploded at lr1e-3. Claim must read "wd bounds
   F_Δ **within the stable LR regime**", with boundaries stated. (Supports, not contradicts, the
   magnitude thesis — divergence = unbounded F_Δ — but the "for free" corollary needs the caveat.)
3. **Pre-register an explosion-exclusion rule** before the B cells land: cells with adaptation
   ≈ 0 (diverged training) are excluded from law fits and counted/disclosed. Currently the frm_
   n=49 fit *includes* 2 diverged cells (they inflate r from −0.83 to −0.92); report both numbers.
4. Known-and-already-handled: MMLU-Pro math parser (BBH-only rule stands — verified frm_ ret_core
   fit is *steeper* than BBH-only, consistent with the broken column adding spurious signal);
   CorDA excluded; LoRA-Null labeling bug (§10/§14 conventions verified).
5. **Seed-collapse basins recur in the new pairs**: `lora_wd1p0_s44` cs 53.1, `clora_k2048_s44`
   44.3, `dora_r8_s44` 54.0 — all extreme-knob cells. Keep the honesty box; median spread 0.79 pp
   elsewhere justifies single-seed interior cells.

### A.4 The wd-causal-intervention lever (handoff/31 M-lever): data verified, ready to write
frm_ c256 grid, per fixed LR (the controlled intervention — only wd moves):
- lr2e-4: wd0→0.5 ⇒ F_Δ 0.87→0.20 (monotone), BBH 26.8→33.2 (monotone ↑), GSM8K 61.3→65.8 (flat/↑).
- lr5e-4: F 2.68→0.26, BBH 17.7→32.9, GSM8K 50.1→65.4 — **retention recovered 15 pp with adaptation
  IMPROVED**. mtx_ CS wd grid (3-seed) shows the same monotone chain (F 0.73→0.17, ret 22.4→27.7)
  plus the adaptation cost past wd0.3 (cs 78.7→67.1) that fixes wd0.3 as the operating point.
This is the "correlation → controlled manipulation" upgrade at zero compute. Highest-priority
writing item.

---

## (B) COMPUTE RE-PRIORITIZATION (~2.5 days × 16 B200)

### The two structural findings
1. **Node B's disk constraint is FALSE.** d002's root partition has **526 GB free** (`/home` is the
   cramped 44 GB volume everyone planned around). B *can* host Llama-2 work (HF cache + adapters on
   `/`), so after Qwen drains (~Sat AM) B contributes **~230 GPU-h ≈ 46 Llama cells** that the
   current plan writes off. This single fix rescues the frc_ spine.
2. **A's queue is over-subscribed ~1.8×** (142 pending ≈ 710 GPU-h vs ~480 available). Without
   kills, the tail (frc_ reservoir, main5 remainder, CE chunks 2–8) silently starves — i.e. the
   *dispatcher order*, not a decision, would choose what the paper loses.

### Verdict on current order
The head of the queue is RIGHT (seeds → ce_chunk1 → triaged frc_). The tail is wrong: it spends
~150 GPU-h on frm_ competitor LR-grid tails that no claim needs, while the frc_ CS head-to-head
(claim-3 spine on the flagship domain) is left to luck.

### KEEP / PROMOTE
| Item | Cells | Why |
|---|---|---|
| lrsw_ s43/s44 seeds (running) | 14 | Error bars on §3 operating points — review gap #1. |
| frm_ math headline seeds (running/queued) | 3 | Settles the 64.97-vs-64.59 noise question (A.3.1). |
| **PROMOTE: b4 remainder** (lora_null ×3, cordapp ×2; sclora 5e-4 running) | 5 | Thesis-completing calibration-confound result (A.2.2); cheapest high-value cells on the board. They live in the draining frepro4b4 pool — **verify they run; if the pool is dead, splice into master_dispatch head.** |
| ce_chunk1 (frm_ CE batch) | 1 job | Sec.6 forgetting table; eval-only, 1 GPU. |
| frc_ triaged block (wd winner column + plain-LoRA anchor + CLoRA k-grid) | 37 | The faithful-CS boundary cells; keep on A right after seeds. |
| **PROMOTE: frc_ reservoir → Node B** after Qwen drains | 40 | Fits B's freed ~230 GPU-h; requires ~1 h provisioning (HF cache on `/`, output to `/`, stream-delete adapters, reuse tar-over-ssh sync). Decision point Sat AM: if provisioning fails, these die honestly. |
| Qwen math B_keep (running on B) | 35 | Priority #1 overall — the only data that can turn Qwen math from "pending" into a replication (A.2.1). Protect the 5e-4/1e-3 cells if the tail is cut; the 2e-5/5e-5 cells are ceiling-flat and sacrificial. |

### KILL (queued cells that no longer earn their GPU-h) — frees ≈ 150–170 GPU-h on A
| Kill | Cells | Justification |
|---|---|---|
| frm_sclora LR-grid tail | 9 of 11 | Math table needs ONE SC-LoRA row (best-LR pair). The frm_ law is already n=49 / r=−0.92; more low-F grid points add nothing a reviewer asks for. |
| frm_milora / frm_lora / frm_clora LR-grid tails | ~18 of 24 | Same: each competitor already has its best-LR math-table row (MiLoRA 62.85, CLoRA-k256 60.80, LoRA 64.97). Keep ±1 LR neighbor per method for the table; kill the rest. |
| frm_cordapp / frc_cordapp trims | ~6 of 11 | CorDA++ is Tier-B, excluded from law claims by policy; keep 2–3 cells for a labeled table row at most. |
| frm2_pissa extra | 1 | PiSSA already has its collapsed data point (49.66/3.62); consortium ruling stands. |
| CE chunks 2–8 | demote, don't kill | Eval-only; run in GPU-free windows / overnight tail. If squeezed, coverage of frm_/frc_ table cells first, reservoir adapters last. |

### Explicit ranking of the marginal GPU-h
frc_ triaged (A) ≈ Qwen-math high-LR (B) > seeds (already running) > b4 remainder > frc_ reservoir
(on B only) > ce chunks > frm_ competitor tails (kill) > anything CorDA++ (trim). The 40-cell
reservoir is NOT worth Node-A hours against the triaged block or seeds — it is worth Node-B hours
that otherwise idle.

---

## (C) ANALYSES ON EXISTING DATA (no new GPU) — ranked

1. **b4 calibration-confound reanalysis (A.2.2).** Recompute the §5 spline residual / ANCOVA with
   the eval-matched SC-LoRA arm; write the "deviator returns to the law" subsection. Strengthens:
   the geometry-is-second-order claim by explaining its only exception. (CPU, ~2 h.)
2. **wd-as-controlled-intervention section + figure (A.4).** Per-LR wd→F_Δ→retention monotone
   chains (frm_ grid + mtx_ 3-seed grid), stability boundary stated. Strengthens: causal depth —
   the single cheapest A* lever (handoff/31 C.3 #1), data verified ready. (CPU/writing, ~half day.)
3. **Claim the frm_ n=49 faithful-math law (A.2.3)**: BBH r=−0.916 (−0.83 excl. diverged), slope
   −13.9 ≈ CS slope −14.8. Upgrades the weakest §1 row (old n=14) into a full second-task law and
   feeds the cross-literature overlay figure. (CPU, ~2 h + key_numbers/§1 rewrite.)
4. **Ceiling-aware statistics across all four arms** (PI workstream 3): Spearman + censored/
   saturating fits, below-ceiling slopes; decisive for interpreting Qwen math where 11/13 points sit
   at the BBH ceiling — turns "flat so far" into a *prediction* that the in-flight high-F cells will
   fall off the cliff. (CPU, ~half day.)
5. **Seed-noise audit + headline re-framing**: mtx_ median spread 0.79 pp, collapse-basin catalog
   (all at extreme knobs), GSM8K spread 3.1 pp on the 3-seed cell ⇒ rewrite the 64.97-vs-64.59
   sentence per A.3.1 and attach spread-based uncertainty to every n=1 headline cell. Removes the
   most attackable overclaim. (CPU, ~2 h.)

(Also standing: efficiency table needs the ~8 GPU-h instrumented-memory runs — kept from handoff/31
B#8; geometry-drift workstream is complete per handoff/27 and needs only figure consolidation.)

---

## (D) BIGGEST PUBLISHABILITY RISK

**The faithful-CS head-to-head (frc_) is still at 0/77 with ~60 h to freeze, and it is the paper's
only direct evidence for claim 3 ("published wins are recipe artifacts") on the domain CLoRA's own
headline table uses.** The law (claims 1–2) is safe — n=49×2 models + n=49 math + mtx n=102 — but a
reviewer who accepts the law can still say "you never re-ran the competitors' flagship comparison
faithfully." Compounding it: the queue as ordered would burn frc_'s hours on frm_ tails, and the
old (false) "B can't host Llama" belief wrote off the only spare capacity.

**Mitigation (concrete):**
1. Tonight: apply the KILL list (one dispatcher restart, idempotent) so the 37 triaged frc_ cells
   start ~12 h earlier.
2. Sat AM when Qwen drains: provision B's root partition (526 GB free) and point it at the frc_
   reservoir; ~1 h of setup buys ~40 cells.
3. Freeze-day fallback (pre-committed): any frc_ row that hasn't landed is reported as CLoRA
   **published** + mature-proxy, explicitly labeled — never presented as the faithful recipe. The
   paper is defensible with a partial frc_ table; it is not defensible with a mislabeled one.

Runner-up risk: the Qwen-math replication resting on one diverged point if B's tail is cut —
mitigated by protecting the 5e-4/1e-3 cells first (B section) and by the ceiling-aware framing (C.4).

---

## THE PLAN (updated numbered list — supersedes handoff/31 snapshot)

1. **[NOW, 0 GPU] Fix A.3 overclaims + pre-register exclusion rule.** WHAT: rewrite 64.97-vs-64.59
   as tie/LoRA+wd-win; wd stability-boundary caveat; explosion-exclusion rule; report frm_ law both
   ways. WHY: removes the most attackable claims before more data lands. COST: writing. STATUS: ready.
2. **[NOW, 0 GPU] Queue surgery on A.** WHAT: apply KILL list (~28–34 cells), splice b4 remainder
   to head if the frepro4b4 pool is dead, one consolidated dispatcher restart. WHY: frc_ starts
   ~12 h earlier; b4 completes the confound result. COST: 15 min ops. STATUS: ready — needs PI nod.
3. **[NOW, CPU] Analyses C.1–C.3** (b4 reanalysis, wd-causal section, frm_ math law claim). WHY:
   the three highest-leverage thesis upgrades, all on existing data. COST: ~1 day CPU/writing.
   STATUS: data verified this pass; ready to write.
4. **[running, B] Qwen math 35 cells.** WHY: the replication decider (A.2.1). COST: sunk (~175
   GPU-h). STATUS: live; protect high-LR cells if tail-cut; recompute on landing (BBH-only,
   exclusion rule pre-registered).
5. **[running, A] 14 lrsw_ seeds + 3 frm_ math seeds.** WHY: error bars on every headline cell.
   COST: sunk (~85 GPU-h). STATUS: mid-flight, 0 landed; on landing recompute §3 table ± spread.
6. **[next on A] 37 triaged frc_ + ce_chunk1.** WHY: faithful-CS spine + Sec.6 CE table. COST:
   ~190 GPU-h. STATUS: queued behind seeds; starts tonight if item 2 executes.
7. **[Sat AM, B] Provision d002 root partition → frc_ reservoir (40 cells).** WHY: rescues the
   remainder of the CS spine with otherwise-idle capacity. COST: ~1 h ops + ~200 GPU-h on B.
   STATUS: NEW — depends on Qwen drain + provisioning success; fallback = labeled published/proxy.
8. **[opportunistic, A] Instrumented memory (~8 GPU-h) + CE chunks 2–8 in GPU-free windows.** WHY:
   closes the efficiency-measurement review flag; independent-metric law at scale. STATUS: standing.
9. **[CPU, days 1–2] Ceiling-aware stats (C.4) + seed-noise audit (C.5) + figure set** (cross-lit
   overlay BBH↔BBH, Qwen 7-adapter panel, consolidated geometry, efficiency table). WHY: M2/M4 of
   the A* roadmap. STATUS: partially done; finish after seeds land.
10. **[Sat EOD] Data freeze**: dated registry snapshot; key_numbers full re-derive (incl. new §
    for b4, frm_ law, Qwen math outcome); then red-team round (adversarial-critic + data-verifier)
    on the frozen paper. STATUS: scheduled.
11. **[POST-deadline] Mistral arm, Qwen 3-seed, CorDA++ full wiring.** STATUS: deferred (unchanged).

---

## EXECUTION LOG — queue surgery applied 2026-07-10 (coordinator order)

**KILL-LIST APPLIED** to `jobs/master_dispatch.txt` (safety-checked: every removed run_name verified
to have NO `results/<rn>/summary.json` and NO `results/dispatch_locks/<rn>.lock` before removal;
dispatcher NOT restarted — coordinator does the consolidated restart, which re-reads the file).

- **Jobs before: 159 → after: 125 (34 removed).** ~150–170 GPU-h freed for the frc_ CS spine.
- **Removed (34), all `frm_` competitor math-table LR-grid tails:**
  - CLoRA (5): `frm_clora_k128_lr{1e4,2e4,5e4,7e4,1e3}_c256_s42` — best-LR row covered by the DONE
    k-grid (k64/k128/k256 @lr3e4); kept `frm_clora_k128_lr3e4_c2048_s42` (cutoff diagnostic).
  - LoRA-Null (5): `frm_lora_null_lr{2e5,3e4,5e4,7e4,1e3}_c256_s42` — kept best-LR pair
    `frm_lora_null_lr{1e4,2e4}_c256_s42` (no prior LoRA-Null math results existed).
  - MiLoRA (10): `frm_milora_lr{2e5,2e4,5e4}_c256_s42`, `frm_milora_lr3e4_c2048_s42`,
    `frm_milora_a1r_lr{1e4,2e4,3e4,5e4,7e4,1e3}_c256_s42` — best-LR pair (lr1e4 62.85 + lr3e4)
    already DONE; a1r is an off-table alpha sub-study.
  - SC-LoRA (9): `frm_sclora_lr{3e4,5e4,7e4,1e3}_c256_s42`, `frm_sclora_b0p5_lr2e5_c256_s42`,
    `frm_sclora_b0p8_lr{2e5,1e4,3e4}_c256_s42`, `frm_sclora_b0p9_lr2e5_c256_s42` — kept best-LR pair
    `frm_sclora_lr{1e4,2e4}_c256_s42`; beta-variant sweep is off-table.
  - PiSSA (1): `frm2_pissa_lr3e4_c256_s42` — the extra; DONE `frm_pissa_lr3e4_c256_s42` (49.66/3.62)
    serves the collapsed-PiSSA row.
  - CorDA++ (4): `frm_cordapp_lr{5e4,7e4,1e3,2e5}_c256_s42` — trimmed to the 3 DONE labeled cells
    (`frm_cordapp_lr{1e4,2e4,3e4}_c256_s42`).
- **NOT touched (protected):** all 14 lrsw_ seeds, 3 frm_ math-headline seeds, 84 frc_ CS-spine
  lines, 6 b4 cells, 11 ce_chunk lines, every locked/running job. `frc_cordapp` (CS spine) left in
  place — it is deep in A's queue (dispatcher won't reach it pre-freeze) so killing frees no
  critical-path hours; leave it rather than risk the protected spine.

**b4 COVERAGE: CONFIRMED.** SC-LoRA eval-matched arm = 3 DONE (`b4_sclora_r32_lr{5e5,1e4,3e4}_s42`)
+ `b4_sclora_r32_lr5e4_s42` running in the live GPU4 pool (`gpu_pool.py --tag frepro4b4`) and also
queued as backstop. The 5 flagged remainder cells are all in `master_dispatch` (queue backstop):
`b4_lora_null_r16_lr{2e5,1e4,3e4}_s42` + `b4_cordapp_r32_lr{1e4,3e4}_s42`. Inject re-queue cell
`frc_lorawd_wd0_lr2e5_c256_s42` verified present. **No b4 cell is uncovered.**

**Node-B reservoir file pre-built:** `jobs/frc_reservoir_B.txt` = **36 frc_ competitor baseline
cells** (milora/sclora/lora_null/cordapp + a1r/em variants) for the Saturday offload below.

---

## SATURDAY-MORNING NODE-B OFFLOAD RUNBOOK (execute only after Qwen drains; ~10 min)

**Facts verified 2026-07-10:** d002 has **526 GB free on `/`** (12 GB on `/home` — too small for
Llama weights), **passwordless sudo**, repo + `/home/guy/UIOrthoLoRA/.venv` present, HF cache holds
only Qwen weights, no `/scratch`, no rsync (tar-over-ssh). Node A Llama-2 weights live at
`/home/ubuntu/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf` (13 GB).

**Precondition:** `jobs/frepro4_qwen_B_keep.txt` drained (or at least the high-LR 5e-4/1e-3 cells
done). Run `./sync_d002.sh` once to pull final Qwen JSON; confirm the qwenB dispatcher is idle.

**Step 1 — prep B disk (one-time, ~1 min):**
```
ssh ubuntu@d002 'sudo mkdir -p /data/hf /data/cf_models /scratch && \
  sudo chown -R ubuntu:ubuntu /data && \
  sudo ln -sfn /data/cf_models /scratch/cf_models && sudo chown -h ubuntu:ubuntu /scratch/cf_models'
```
(`/scratch/cf_models` → `/data/cf_models` on the 526 GB disk, so the jobs file's `/scratch/...`
paths work unchanged; adapters land on big disk.)

**Step 2 — ship Llama-2 weights A→B (~13 GB, few min on internal net), symlink into HF cache:**
```
cd /home/ubuntu/.cache/huggingface/hub && \
tar cf - models--meta-llama--Llama-2-7b-hf | \
  ssh ubuntu@d002 'tar xf - -C /data/hf'
ssh ubuntu@d002 'ln -sfn /data/hf/models--meta-llama--Llama-2-7b-hf \
  ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf'
```
(Eval datasets — bbh/mmlu_pro/etc — are already in B's `/home` HF cache; leave them.)

**Step 3 — dedup + sync repo:** remove the 36 reservoir run_names from A's `master_dispatch` so they
can't double-run (no shared FS), commit, then pull on B:
```
# on A:
grep -vFf <(grep -oP '(?<=--run_name )\S+' jobs/frc_reservoir_B.txt) jobs/master_dispatch.txt \
  > /tmp/md.new && mv /tmp/md.new jobs/master_dispatch.txt
git add jobs/master_dispatch.txt jobs/frc_reservoir_B.txt && git commit -m "B-offload dedup" && git push
ssh ubuntu@d002 'cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting" && git pull'
```
(A's consolidated dispatcher restart then picks up the reduced queue.)

**Step 4 — launch reservoir dispatcher on B (detached):**
```
ssh ubuntu@d002 'cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting" && \
  nohup setsid nice -n 5 /home/guy/UIOrthoLoRA/.venv/bin/python auto_dispatch.py \
  --jobs jobs/frc_reservoir_B.txt --gpus 0,1,2,3,4,5,6,7 --tag frcB \
  >> logs/frcB_dispatch.log 2>&1 < /dev/null &'
sleep 8; ssh ubuntu@d002 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
```

**Step 5 — sync-back (extend the existing loop to pull frc_ too):** edit `sync_d002.sh` line ~15,
change the find pattern from `-path './qw*'` to `\( -path './qw*' -o -path './frc_*' \)` and the two
`results/qw*` git-add/status lines to also match `results/frc_*`. Or one-off manual pull:
```
ssh ubuntu@d002 'cd "/home/guy/UIOrthoLoRA/notebooks/catastrophic forgetting" && \
  find . -path "./frc_*" \( -name "*.json" -o -name "*.jsonl" \) | tar czf - -T -' \
  | tar xzf - -C results/ && git add "results/frc_*" && git commit -m "B frc_ results" && git push
```
Adapters stay on B's `/data` (526 GB, no stream-delete needed for 36 cells ≈ 15 GB); only JSON
returns to A. **Freeze-day fallback unchanged:** any frc_ row that hasn't landed → CLoRA published +
labeled proxy, never passed off as the faithful recipe.

---

### Changed since handoff/31
- **Thesis-completing new result surfaced**: b4 eval-matched SC-LoRA is ON the law (+0.05/+1.8/+2.7
  residuals vs −2.8/−6.3 original) — §5's only deviator explained as a calibration confound.
- **Qwen math re-quantified**: n=15 on A; nominally r=−0.73 but one-point-dependent (−0.39 without
  the single diverged cell) → stays "pending"; wd0.3 itself exploded at lr1e-3 (new honesty item).
- **frm_ n=49 math law claimed** (r=−0.92/−0.83, slope −13.9) — replaces the n=14 row.
- **Overclaim flagged**: plain-LoRA 64.97 "win" over published 64.59 is within 3.1 pp seed noise —
  reframe as tie; LoRA+wd's win survives.
- **Node B disk constraint debunked** (526 GB free on `/`) → frc_ reservoir moves from "sacrificial
  on A" to "runs on B Sat AM"; A's freed hours go to the triaged frc_ block via the KILL list
  (~150–170 GPU-h reclaimed from frm_ competitor LR-grid tails, CorDA++ trims, PiSSA extra).
- Queue accounting made explicit: A pending 142 cells ≈ 1.8× available hours — kills are now a
  decision, not an accident.
