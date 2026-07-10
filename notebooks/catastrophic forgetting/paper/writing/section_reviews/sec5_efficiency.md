# Section 5 review — "Compute & memory per adapter"

Reviewer: section-validator (efficiency). Date: 2026-07-10.
Sources: `results/train_registry.jsonl` (recomputed), `train_cs.py`, `sclora_init.py`,
`lora_null_init.py`, `corda_init.py`, `cordapp_init.py`, `fleet_findings.md`,
`fig_efficiency.py`. Artifact section: `artifact_status_report.html` lines 244–262.

## Overall verdict

**Publishable basis with one CORRECTED cell and several caption-scope fixes.** The two headline
ratios (DoRA 2.14x, CLoRA 1.17x) recompute exactly under the stated methodology, are NOT
contention artifacts (verified by timestamp interleaving), and replicate at r=64 (2.12x, 1.14x).
The CorDA++ wall-clock cell is wrong: our own r=64 runs measure **1.10x, not 1.00x**. Memory
numbers are analytical but well-founded; the one cell likely to change under instrumentation is
DoRA's "small (norm buffers)".

## (a) Claim-by-claim verdict table

| # | Claim (artifact) | Verdict | Recomputed value / evidence |
|---|---|---|---|
| 1 | DoRA wall-clock **2.14x**, median over 7-LR sweep, n=7 | **CONFIRMED** | DoRA core-7 median 15506.1 s / LoRA-r16 core-7 median 7229.9 s = **2.1447x**. n=7 exact for both. Robust: including extended LRs (n=9–10) gives 2.144–2.145x. |
| 2 | CLoRA wall-clock **1.17x**, n=7 | **CONFIRMED (scope caveat)** | CLoRA-k1024 core-7 median 8428.0 / 7229.9 = **1.1657x**. Caveat: measured at **k=1024, r=32**; at r=64 with k=64/128/256 the measured ratios are 1.135/1.148/1.136 (frm_ runs) — the overhead is k-dependent (regularizer FLOPs scale with k). State "1.14–1.17x depending on k". |
| 3 | Not-a-confound check (implicit in "ratio isolates the method's own overhead") | **CONFIRMED, and stronger than stated** | (i) DoRA runs finished interleaved with normal-speed runs of other methods (2026-06-29 07:19: DoRA lr5e3 at 15475 s within seconds of MiLoRA lr5e3 at 7208 s) — not node contention. (ii) Cross-rank replication: at r=64, frm_dora 36508.1 / frm_lora 17184.5 = **2.124x** (n=1); frm_cordapp/frm_clora consistent. Worth adding one sentence — free robustness. |
| 4 | CorDA++ train wall-clock **1.00x** | **CORRECTED → 1.10x** | frm_cordapp median 18932.6 s (n=3: 18892.8/18932.6/18965.4) / frm_lora 17184.5 = **1.102x** (vs pooled lorawd r64 median ~17150: 1.104x). Plain CorDA-KPA at r16 is 1.00x (7230.8/7229.9), but the table row says CorDA++. Likely cause: heterogeneous per-layer rank_pattern. This *helps* the thesis — CorDA++ pays at init AND per step. |
| 5 | "r=64 = 112M trainable params for every method" | **CORRECTED (two ways)** | (i) The wall-clock ratios were measured at **r=16 (DoRA/LoRA baseline) and r=32 (CLoRA)**, not r=64 — the r=64 framing describes the headline campaign, not the timing runs. Cross-rank agreement (2.14 vs 2.12) fixes this cheaply: say so. (ii) DoRA at r=64 trains **113,074,176** params (registry), = 112,197,632 + 876,544 magnitude params (sum of out-dims, 27392×32). CorDA++ 112,205,824. "≈112M" is fine; "= 112M for every method" is not literally true. |
| 6 | SC-LoRA init = 512 fwd passes (256 D+ + 256 D−) | **CONFIRMED** | `train_cs.py`: `sclora_calib_size` default 256; dplus = 256 task prompts, dminus = 256 nq_open windows; `sclora_init.collect_sclora_M` forwards each once (bs=1, max_len 2048). |
| 7 | LoRA-Null init = 256 fwd passes | **CONFIRMED** | `lora_null_init.py` `calib_size=256` (nq_open windows), fp32 input covariance. |
| 8 | CorDA++ init = 1,280 fwd passes, 5 rounds | **CONFIRMED** | `train_cs.py` line ~276: `_csz = cordapp_calib_size or cordapp_n * 256`, consortium default N=5 → 5×256=1280 (2048-token windows, bs=1). Note `cordapp_init.py` has `DEFAULT_N=8` but train_cs passes 5. |
| 9 | ~22 GB transient covariance buffer (SC-LoRA/LoRA-Null) | **CONFIRMED (it's GiB)** | Per layer 4×4096² + 11008² = 188,284,928 fp32 accumulator entries ×32 layers ×4 B = 24.1 GB = **22.4 GiB**. Same for input-side (LoRA-Null/CorDA) and output-side (SC-LoRA). "~22 GB" is the GiB value; harmless, but the CLoRA figures use the same convention — say GiB once or keep silently consistent. |
| 10 | CLoRA frozen block +0.42 GB (k=128) → +6.7 GB (k=2048); +1.7 GB at k=512 | **CONFIRMED from our code** | `train_cs.py` CLoRARegularizer: P_u(out×k)+P_v(in×k) per module, cast to base dtype **bf16**; coefficient = 54,784 dims/layer ×32 = 1,753,088 floats per k ×2 B. k=128: 0.449 GB (0.418 GiB ≈ "0.42"); k=512: 1.795 GB; k=2048: 7.18 GB (6.69 GiB ≈ "6.7"). No optimizer state (detached buffers). NOTE: our headline CLoRA timing cell is **k=1024 → +3.6 GB** — worth stating since that is the run behind "1.17x". |
| 11 | "+1.7 GB … already 8× the trainable LoRA weights" | **FRAGILE** | 8× holds only if adapter counted in bf16 (112.2M×2 B = 224 MB). The code comment says PEFT keeps LoRA A/B **fp32** → 449 MB → **4×**. Either say "4–8× depending on adapter dtype" or drop the multiplier. |
| 12 | "All timings were measured on NVIDIA B200 GPUs" + CorDA++ "~1 GPU-h (paper Table IX)" | **INTERNALLY INCONSISTENT** | The CorDA precompute hour is cited from the CorDA++ paper (their hardware), inside a caption promising B200-measured timings. Either measure our own precompute (one run, ~0.5 GPU-h) or scope the bold claim to "all training wall-clock". |
| 13 | Baseline "~55 GB GPU memory (7B bf16 + activations + optimizer state)" | **PLAUSIBLE, SOFT** | Derivation: weights 13.6 GB (6.79 B×2 B) + adapter/optimizer ≈1.4–1.8 GB (fp32 A/B 0.45 + grads 0.45 + AdamW m,v 0.9) + activations (no gradient checkpointing in train_cs.py; bs=16×256 tok, SDPA) ≈ 20–27 GB + fp32 logits ≈ 0.5–1.5 GB → ~40–45 GB allocated, ~50–55 GB reserved with fragmentation. Provenance per fleet_findings is "incidental OOM traces", i.e. reserved/process footprint. Defensible as "~55 GB process footprint"; the honest "analytical, not instrumented" disclaimer is already present. Instrumented allocated-peak would likely read **lower** (~42–48 GB). |
| 14 | "a ~5 GPU-h train+eval run per cell" | **MIXED SCOPE** | LR-sweep cells (r16/r32): train 2.0 h + eval ≈ 5 GPU-h ✓. But r=64 headline cells train alone = 4.8 h (17.2 ks) → train+eval ≈ 7–8 GPU-h. Since the caption leads with r=64, say "~5 GPU-h (sweep cells) / ~8 GPU-h (r=64 cells)" or anchor the 20% CorDA figure to the r=64 train time (1 h / 4.8 h ≈ 20% — still works). |
| 15 | MiLoRA/PiSSA/SC-LoRA/LoRA-Null 1.00x rows | **CONFIRMED** | Medians vs LoRA-r16 7229.9: MiLoRA 7206.3 (1.00), SC-LoRA 7188.4 (0.99), LoRA-Null 7232.2 (1.00), CorDA-KPA 7230.8 (1.00). |
| 16 | CLoRA init "none" | **NIT** | Our port QR-factorizes 320 random matrices on GPU at startup — seconds, but not literally "none". "negligible (random-orthonormal QR, seconds)" is airtight. |

### Registry hygiene finding (affects reproducibility of #1–2)
The registry contains a **contended batch (2026-06-27 02:05–07:32)**: 11 runs at 1.65–2.2×
normal runtime (lora 11947.7/15529.6; lorawd 15112.5/15428.9; milora 15625.5–16031.1 ×5;
clora 18193.2/18714.4), all later re-run at normal speed on 06-29 **under the same run_name**
(duplicate keys in the registry). The published core-7 medians dodge this, and
`fig_efficiency.py`'s all-entries medians happen to be robust, but anyone recomputing with
means or naive dedup-by-name gets different numbers. Document the batch and a dedup rule
(e.g. keep latest finished_at) next to the registry.

## (b) Is ~55 GB defensible? — Yes, as a reserved-footprint estimate (see #13); label it
"process footprint (reserved)" and expect instrumented allocated-peak to come in ~10 GB lower.
All relative claims in the table survive either way.

## (c) Converting analytical memory → measured

**Do it, but it costs ~2 GPU-h, not 8.** No full runs needed: a `memory_probe.py` that loads
each method, runs init + 30 train steps, and logs `torch.cuda.max_memory_allocated/reserved`
at (init-peak, steady-train-peak). 8 configs × ~12 min on one B200 ≈ 1.6–2 GPU-h.
Numbers that would plausibly CHANGE:
1. **DoRA "small (norm buffers)"** — most likely to change qualitatively. PEFT DoRA
   materializes W + (α/r)BA per module to compute the column-norm each step and backprops
   through it; measured delta could be several GB transient, not "small". A change here
   *strengthens* the thesis (DoRA pays time AND memory).
2. **The 55 GB baseline** — likely lands 42–48 GB allocated (relative story unchanged).
3. CLoRA +3.6 GB at k=1024 will match analytics exactly (deterministic buffer) — cheap
   credibility: one measured point validating the whole k-scaling line.
4. Init transient peaks (22.4 GiB accumulator + weights + eigh workspace) — currently only
   analytical; a measured ~40 GB init-peak makes the "fits alongside training on one B200
   but not on a 24 GB card" point concrete for practitioners.

## (d) Filler check
No dead rows: the 1.00x/"—" rows carry the argument ("the inits cost minutes and buy
nothing"), and all are genuinely measured (#15). The lowest-information cell is MiLoRA/PiSSA
"~minutes"; fine as is. The CorDA++ row becomes MORE interesting once corrected to 1.10x.

## (e) New insight — efficiency × retention combined view (recommended, zero GPU cost)
`fig_efficiency.py` shows cost only; nothing links cost to the retention result. Add a small
**Pareto panel: x = relative train wall-clock (measured, 1.00–2.14x), y = best-cell retention
at matched CS accuracy (cells within ~1 pp of each method's best CS), bubble size = extra
resident GB (0 for most, 3.6 GB CLoRA-k1024).** LoRA+wd sits at (1.00x, top-retention, 0 GB);
DoRA at (2.14x, no better); CLoRA at (1.17x + memory, tied at best). One derived stat lands
the sentence: "matching LoRA+wd's retention costs DoRA +2.9 GPU-h per run and buys 0 pp."
Pure analysis of existing summary.json + registry data; ~1–2 h analyst time. This is the
single cheapest addition that converts the section from "cost table" into direct evidence for
"best *efficient* operating point".

Also: `key_numbers.md` currently has **no efficiency block** — the 2.14x/1.17x/1.10x medians,
the contended-batch exclusion rule, and the CLoRA byte coefficient (1,753,088 bf16 floats per
unit k) should be added there so the authoritative file covers Section 5.

## Top 3 improvements (prioritized)
1. **Fix CorDA++ wall-clock to 1.10x** (measured, n=3) + add the cross-rank robustness
   sentence (r=64: DoRA 2.12x, CLoRA-k≤256 1.14x) and the k=1024 scope of the 1.17x.
   Cost: text-only, 0 GPU-h.
2. **memory_probe.py instrumentation** (~2 GPU-h): measured peak memory for all 8 methods +
   baseline; expected to correct the DoRA memory cell and pin the 55 GB anchor, removing the
   section's only "analytical estimate" disclaimer.
3. **Efficiency×retention Pareto panel + "GPU-h per retention point" stat** (0 GPU-h,
   ~1–2 h analysis): the missing bridge from Section 5's costs to the paper's thesis.
