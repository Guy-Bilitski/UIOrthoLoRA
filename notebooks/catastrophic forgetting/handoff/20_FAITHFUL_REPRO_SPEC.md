# 20 — Faithful CLoRA Reproduction Spec (Table 2 commonsense + Table 3 math) + LoRA+wd sweep

**Status:** PLAN + VERIFY ONLY. No pipeline file was edited; no data downloaded; the live 8-GPU
`camp5` pool (`gpu_pool.py --gpus 8 --tag camp5 --jobs jobs/combined_nocorda.txt`, PID 167660, 118
queued jobs) was NOT touched. camp5 launches `train_cs.py` / `eval_one_gpu.py` **fresh per job**, so
every code file here is treated as immutable — all proposed code changes go into a git worktree/branch
and RUN only after camp5 drains (single-scheduler rule, handoff/19 §7.2).

Date: 2026-07-05. Ground truth = the CLoRA settings the user supplied (paper Table 2 / Table 3) plus
first-hand reads of `repro/CLoRA/clora.py`, `repro/LLM-Adapters/{finetune,evaluate,commonsense_evaluate,mathqa}.py`,
`train_cs.py`, `eval_one_gpu.py`, `eval_cs.py`, `run_lib.py`, the init modules, and `residual_save.py`.

**Verified environment:** 8× NVIDIA B200 (183 GB each); venv `/home/guy/UIOrthoLoRA/.venv/bin/python`;
model checkpoints go to `/scratch/cf_models`; results to `results/`. GSM8K test set already local
(`repro/LLM-Adapters/dataset/gsm8k/test.json`, 1319 rows); Hendrycks MATH test set is **absent** and
must be acquired. MetaMathQA present only as the 100K subset; full 395K must be built.

---

## 0. What "faithful" requires — the six gaps to close (summary)

| # | Gap | Current state | Fix (section) |
|---|---|---|---|
| G1 | **Math data + length** | `metamathqa_100k.json` (100K, `metamath_prep.py`), math cells run at `cutoff_len` 256/512 mixed | Build MetaMathQA **395K**; train at `cutoff_len 512` (CoT). §3, §6 |
| G2 | **Math recipe (r/α/k)** | math cells ran r16(LoRA/DoRA)/r32(others), CLoRA k=1024 | Table 3 = **r64/α128**, CLoRA k∈{64,128,256}. §1, §2 |
| G3 | **Math eval metric** | `eval_one_gpu.py:91-101` uses lm-eval `gsm8k` (5-shot Q:/A: strict-match) — train/eval template mismatch | Faithful **0-shot instruction-template + last-number** eval (GSM8K) + **new Hendrycks MATH** eval. §4 |
| G4 | **PiSSA arm missing** | `pissa_BAR` exists (`data_aware_init.py:63-71`) but NO `--pissa` branch in `train_cs.py` (verified: `grep pissa train_cs.py` → none) | Wire `--pissa` (mirror `--milora`), residual-save applies. §2.3 |
| G5 | **Commonsense rank-unfairness** | plain LoRA is r16 but CLoRA's main LoRA is r32; our LoRA+wd is r32 vs plain LoRA r16; CLoRA only at k=1024 | Add LoRA r32/α64, PiSSA r32, LoRA-r8, CLoRA k{128..2048}, LoRA-L2. §5 |
| G6 | **LoRA+wd single point** | lorawd fixed at wd=0.3 across LRs | Full **LR×wd sweep** to match/beat CLoRA. §2.4 |

---

## 1. Recipe tables — ours vs CLoRA (with file:line)

### 1.1 Shared hyperparameters (both tables)

| Hyperparameter | CLoRA (ground truth) | Ours (default) | file:line | Verdict |
|---|---|---|---|---|
| Base model | LLaMA-2-7B | `meta-llama/Llama-2-7b-hf` | train_cs.py:124 | **MATCH** |
| Optimizer | AdamW | `optim="adamw_torch"` | train_cs.py:296 | **MATCH** |
| LR | 3e-4 | `learning_rate` default 3e-4 | train_cs.py:128 | **MATCH** |
| Scheduler | linear | `lr_scheduler_type="linear"` | train_cs.py:296 | **MATCH** |
| Warmup | 100 | `warmup_steps` 100 | train_cs.py:133,293 | **MATCH** |
| Batch size (effective) | 16 | batch 16, micro 16 → grad_accum 1 | train_cs.py:131,132,174 | **MATCH** |
| Epochs | 3 | `num_epochs` 3 | train_cs.py:127 | **MATCH** |
| Dropout | 0.05 | `dropout` 0.05 | train_cs.py:135 | **MATCH** |
| Target modules | q,k,v,MLP-up,MLP-down | `q_proj,k_proj,v_proj,up_proj,down_proj` | train_cs.py:134 | **MATCH** |
| LoRA A / B init | A~Gaussian, B=0 | PEFT LoraConfig default | train_cs.py:92-94 | **MATCH** |
| CLoRA P init | random orthonormal (unit cols, mutually ⟂) | QR + sign-stabilize, seed 42 | train_cs.py:53-60,289 | **MATCH** (functionally = `nn.init.orthogonal_`, which is what official `clora.py:34-36` uses) |
| CLoRA λ | 1 | `clora_lambda` 1.0 | train_cs.py:156 | **MATCH** |
| CLoRA penalty form | ½‖A·Pv‖² + ½‖Bᵀ·Pu‖² summed | identical | train_cs.py:62-69 vs CLoRA clora.py:41-66 | **MATCH** (port verified faithful, handoff/18) |
| Precision | (paper: fp16) | bf16 | train_cs.py:294 | **CHANGE (benign)** — bf16 on B200; disclose. LLM-Adapters/CLoRA used fp16. |
| Checkpoint reported | LAST checkpoint | `save_strategy="no"` → only final saved | train_cs.py:297,310 | **MATCH** (we always eval the last/final state) |

### 1.2 Table 2 — Commonsense (Commonsense170K)

| Item | CLoRA | Ours | file:line | Verdict |
|---|---|---|---|---|
| Data | Commonsense170K | `commonsense_170k.json` (170,420 rows) | train_cs.py:85 | **MATCH** |
| cutoff_len | 256 (LLM-Adapters default) | 256 | train_cs.py:126 | **MATCH** |
| r / α | r=32, α=64 | defaults r32/α64 | train_cs.py:152,153 | **MATCH** |
| CLoRA k grid | {128,256,512,1024,2048} | only k=1024 run so far | make_campaign_jobs.py:21 | **CHANGE** (add k{128,256,512,2048}) |
| In-domain eval | BoolQ,PIQA,SIQA,HellaSwag,WinoGrande,ARC-e,ARC-c,OBQA (MC acc) | 8-task CS via `eval_cs.run_eval` | eval_one_gpu.py:102-108, uio_inprocess.CS_DATASETS | **MATCH** |
| Out-domain eval | BBH 34.91 + MMLU-Pro 18.56 (lm-eval) | `bbh_fewshot`+`mmlu_pro` (lm-eval, in-proc) | eval_one_gpu.py:124 | **MATCH** (retention; base numbers align with our snapshots) |

### 1.3 Table 3 — Math (MetaMathQA)

| Item | CLoRA | Ours (current) | file:line | Verdict |
|---|---|---|---|---|
| Data | MetaMathQA 395K (GSM8K+MATH train aug) | `metamathqa_100k.json` (100K) | metamath_prep.py:10,13 | **CHANGE** (build 395K) |
| cutoff_len | 256 (LLM-Adapters math default, `math_running_commands`) | 256/512 mixed | train_cs.py:126 | **CHANGE→512** (see risk R2: paper used 256; 256 truncates MetaMathQA CoT which is longer than math_10k — recommend 512, disclose) |
| r / α | r=64, α=128 | ran r16/r32 | make_campaign_jobs.py:17-25 | **CHANGE** (r64/α128) |
| CLoRA k grid | {64,128,256} | k=1024 | make_campaign_jobs.py:21 | **CHANGE** |
| Baselines shown | LoRA, PiSSA, MiLoRA, CLoRA | LoRA/DoRA/MiLoRA/CLoRA (no PiSSA) | — | **CHANGE** (add PiSSA; DoRA optional-extra) |
| Eval | GSM8K + MATH test acc, last ckpt | lm-eval gsm8k only | eval_one_gpu.py:91-101 | **CHANGE** (faithful GSM8K + add MATH) |

**Reference targets (Table 3 GSM8K / MATH):** LoRA 60.58/16.88 · PiSSA 58.23/15.84 · MiLoRA 63.53/17.76
· CLoRA-k64 64.29/17.52 · CLoRA-k128 64.59/18.38 · CLoRA-k256 63.45/17.58.

---

## 2. Arm list + exact per-arm config

### 2.1 Table 3 (math) reproduction arms — all r=64, α=128, LR=3e-4, seed 42 first

| Arm | `train_cs.py` flags | scaling | residual-save? |
|---|---|---|---|
| `lora`   | `--method lora --lora_r 64 --lora_alpha 128` | 2 | no |
| `pissa`  | `--method lora --pissa 1 --lora_r 64 --lora_alpha 128` (NEW branch, §2.3) | 2 | **yes** |
| `milora` | `--method lora --milora 1 --lora_r 64 --lora_alpha 128` | 2 | **yes** |
| `clora`  | `--method clora --clora_k {64,128,256} --lora_r 64 --lora_alpha 128` | 2 | no |
| `dora` (extra, not in Table 3) | `--method lora --use_dora 1 --lora_r 64 --lora_alpha 128` | 2 | no |
| `lorawd` (our arm) | `--method lora --lora_r 64 --lora_alpha 128 --weight_decay <wd>` | 2 | no |

> **BLOCKING mechanics conflict (residual arms at α≠r).** `residual_save.py:59` **asserts α==r**
> (scaling=1) and `milora_init.apply_milora`/`data_aware_init.inject_lora_init` fold scaling into B.
> At the faithful α=128,r=64 (scaling=2) MiLoRA and PiSSA will **abort at save**. Two clean options —
> pick one and disclose:
> - **(A, recommended) Generalize `residual_save.py` to preserve scaling.** The W0-relative update is
>   `dW = s·(B_tr·A_tr − B_init·A_init)`; stacking `A''=[A_tr;A_init]`, `B''=[B_tr,−B_init]` and setting
>   the saved config to `r'=2r, α'=2·α_old` keeps `scaling'=s` and gives `s·B''A''=dW`. Change:
>   `residual_save.py:59` drop the α==r assert; `:61-62` set `cfg["r"]=2*r_old`, `cfg["lora_alpha"]=2*cfg_old_alpha`.
>   Then PiSSA/MiLoRA run at the true α=128. (This is the faithful path — one small, well-defined diff.)
> - **(B, fallback) Run the SVD-init arms at α=64,r=64 (scaling=1)** — keeps `residual_save` untouched
>   but **deviates from CLoRA's α=128 for PiSSA/MiLoRA only**. Disclose in the table footnote.
> LoRA / DoRA / CLoRA / LoRA+wd are non-residual and run at α=128 with **no change**.

### 2.2 Table 2 (commonsense) reproduction arms — all r=32, α=64, LR=3e-4, seed 42 first

| Arm | flags | note |
|---|---|---|
| `lora_r32`  | `--method lora --lora_r 32 --lora_alpha 64` | the paper's MAIN LoRA baseline (fixes G5) |
| `dora_r32`  | `--method lora --use_dora 1 --lora_r 32 --lora_alpha 64` | |
| `pissa_r32` | `--method lora --pissa 1 --lora_r 32 --lora_alpha 64` | NEW branch; needs residual-save (scaling-generalized or α=r=32) |
| `milora_r32`| `--method lora --milora 1 --lora_r 32 --lora_alpha 64` | residual-save |
| `lora_r8`   | `--method lora --lora_r 8 --lora_alpha 16` | reduced-rank baseline |
| `lora_r16`  | `--method lora --lora_r 16 --lora_alpha 32` | reduced-rank baseline (this is our OLD "plain LoRA" — demote it to the reduced-rank row, do not call it the LoRA baseline) |
| `lora_l2`   | `--method lora --lora_r 32 --lora_alpha 64 --weight_decay 1e-5` | LoRA-L2 (see risk R6: AdamW wd≈L2; faithful L2 would be an explicit loss penalty) |
| `clora_k{128,256,512,1024,2048}` | `--method clora --clora_k <k> --lora_r 32 --lora_alpha 64` | full k grid (fixes G5) |
| `lorawd` (our arm) | `--method lora --lora_r 32 --lora_alpha 64 --weight_decay <wd>` | LR×wd sweep |

### 2.3 Adding PiSSA as a campaign arm (exact wiring — spec only)

`pissa_BAR(r)` (`data_aware_init.py:63-71`) computes the **top-r (major) SVD** split
`B=Uᵣ√Sᵣ, A=√Sᵣ·Vtᵣ, W_res=W0−B·A` — this is exactly PiSSA's "major-SVD init", loss-preserving
(`W_res+B·A=W0`). It is a **residual method** (like MiLoRA), so it needs `residual_save`. Minimal diff,
mirroring the `--milora` branch:

1. `train_cs.py` args: add `ap.add_argument("--pissa", type=int, default=0, ...)` next to `--milora` (line 143).
2. After the milora block (`train_cs.py:228-232`), add:
   ```python
   if getattr(args, "pissa", 0):
       import data_aware_init as Di
       # scaling handled by inject_lora_init (folds scaling into B); residual-save preserves it.
       err = Di.inject_lora_init(model, Di.pissa_BAR(args.lora_r))
       print(f"[pissa] major-SVD init applied; loss-preserving err={err:.2e}", flush=True)
   ```
   (If using fallback (B): also `assert args.lora_alpha == args.lora_r`.)
3. `train_cs.py:280-281` — add `getattr(args,"pissa",0)` to the `residual_method` OR clause so
   `capture_init_adapter` + `convert_saved_to_w0_relative` run (PiSSA overwrites `base.weight=W_res`
   and must be converted to a W0-relative adapter, exactly like MiLoRA — see `residual_save.py` header).
4. `make_campaign_jobs.py:ARMS` — add `"pissa_r64": "--method lora --pissa 1 --lora_r 64 --lora_alpha 128"`
   (math) / `"pissa_r32": "--method lora --pissa 1 --lora_r 32 --lora_alpha 64"` (CS).

**Verify:** run `validate_residual_zero_step.py` on a 0-step PiSSA run → post-reload ‖ΔW‖ < 1e-4
(the residual-save round-trip is correct only if this passes).

### 2.4 LoRA+wd LR × weight-decay sweep grid (our headline "beat CLoRA" arm)

`--weight_decay` (`train_cs.py:129`) feeds AdamW `weight_decay` (`train_cs.py:295`) applied to the
adapter A/B params — the subspace-free magnitude knob. Proposed grid (identical for both tables, at the
table's native r/α):

- **LR set (6):** `1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3`.
  *Justification:* brackets CLoRA's fixed 3e-4 with finer resolution ABOVE it (5e-4, 7e-4), where the
  existing 100K math sweep already shows the accuracy peak (`lorawd` gsm8k 48.98→**50.64** climbing
  1e-4→5e-4 in `campaign_summary.jsonl`), and includes the collapse edge (1e-3) to map the retention
  frontier. Drops 2e-5/5e-5 (verified clearly underfit: gsm8k ~33–41 in existing rows).
- **wd set (5):** `0.0, 0.1, 0.2, 0.3, 0.5`.
  *Justification:* wd=0.0 recovers plain LoRA (control anchor / dedup with the `lora` arm); 0.1–0.3 is
  the magnitude-budget sweet spot the matrix campaign found (MEMORY `matrix-campaign-results`:
  "LoRA+wd0.3 wins"); 0.5 probes over-regularization (adaptation loss vs retention gain).

**Full grid = 6 × 5 = 30 cells/seed/table** (LR=3e-4,wd=0.0 == the `lora` baseline → 29 unique).
**Depth-first core (run first): LR{2e-4,3e-4,5e-4} × wd{0.1,0.2,0.3} = 9 cells** — this brackets the
expected optimum; expand to the full 30 only if the core is promising.

---

## 3. Data acquisition — MetaMathQA 395K (exact steps)

**Confirmed via HF (`meta-math/MetaMathQA`):** split `train`, **395,000 rows**, fields
`type, query, original_question, response`. `type` ∈ {GSM_AnsAug, GSM_Rephrased, GSM_SV, GSM_FOBAR,
MATH_AnsAug, MATH_Rephrased, MATH_FOBAR, MATH_SV} — i.e. augmented from GSM8K + Hendrycks MATH **train**
("none of the augmented data from the testing set", per the dataset card).

**Format target:** `train_cs.py` (via `run_lib.train_prompt`, `run_lib.py:35-53`) expects rows with
`{instruction, input, output, answer}`. Verified `metamathqa_100k.json[0]` has exactly those keys, with
`instruction=query`, `input=""`, `output=response`, `answer=""` (`metamath_prep.py:14`). The `response`
CoT ends `"#### <n>\nThe answer is: <n>"` (GSM) or `"The answer is: <expr>"` (MATH) — verified.

**Steps (data-only; safe to do NOW, does not touch code or GPUs):**
1. Reuse `metamath_prep.py` but with `N=395000` (or `len(ds)`) and `OUT=".../metamathqa_395k.json"`.
   The line `ds.shuffle(seed=42).select(range(min(N,len(ds))))` (`metamath_prep.py:13`) already handles it.
2. **Keep `original_question` too** (add it to the row dict) so the contamination dedup (§8/R1) is
   possible — the current prep drops it. New filename so it never collides with the 100K camp5 file.
3. Result: `repro/LLM-Adapters/ft-training_set/metamathqa_395k.json` (~300 MB, ~4× the 100K).
4. Point the math jobs at it via `--data_path repro/LLM-Adapters/ft-training_set/metamathqa_395k.json`
   (already threaded through `make_campaign_jobs.py:47`).

**Training-prompt template match:** `run_lib.train_prompt` (`run_lib.py:35-53`) is byte-identical to
LLM-Adapters `finetune.py:generate_prompt` **except LLM-Adapters has 2 trailing spaces after
"request."** (verified `finetune.py:337` `"request.  "` vs `run_lib.py:47` `"request."`). This is a
benign whitespace-only deviation; what matters is our train template == our eval template (both from
`run_lib`, both no trailing space — verified `run_lib.py:47` and `:68`). **The faithful math eval MUST
therefore use `run_lib.eval_prompt`, NOT `evaluate.py:generate_prompt` (which has 1 trailing space,
`evaluate.py:154`) — see §4/R4.**

---

## 4. Eval protocol (faithful) — GSM8K via instruction template, + new Hendrycks MATH

### 4.1 Why the current gsm8k eval is wrong
`eval_one_gpu.py:91-101` runs lm-eval `gsm8k` = **5-shot, "Question:/Answer:" template, strict-match**.
Our models are trained on the **Alpaca instruction template** (`run_lib.train_prompt`). That is a
train/eval template mismatch — it understates accuracy and is NOT what CLoRA/LLM-Adapters report. The
faithful protocol (`repro/LLM-Adapters/evaluate.py`) is **0-shot, instruction template,
`extract_answer_number` (last number), abs-diff ≤ 1e-3**.

### 4.2 GSM8K faithful (data already local)
- Data: `repro/LLM-Adapters/dataset/gsm8k/test.json` (1319 rows; `answer` = float string e.g. "576.0").
- Prompt: `run_lib.eval_prompt(instruction)` (0-shot Alpaca, matches training). NOT evaluate.py's.
- Generate: greedy or `num_beams=4` (evaluate.py uses beam=4, `max_new_tokens=256`); batched (reuse
  `eval_cs.run_eval`'s batching, `eval_cs.py:61-88`). Split off `### Response:`.
- Score: port `extract_answer_number` (`evaluate.py:272-287`) — strip commas, `re.findall(r'-?\d+\.?\d*')`,
  take the LAST number; correct iff `abs(float(answer) − pred) ≤ 1e-3`.

### 4.3 Hendrycks MATH (must be acquired)
- **Data (confirmed):** `hendrycks/competition_math` (a.k.a. mirror `EleutherAI/hendrycks_math`), split
  `test`, **5000 problems**; fields `problem, level, type, solution`. Gold answer is the LaTeX inside
  `\boxed{...}` in `solution`. Build `repro/LLM-Adapters/dataset/MATH/test.json` in the same
  `{instruction=problem, input="", output="", answer=<gold>}` schema, where `<gold>` = `remove_boxed(last_boxed_only_string(solution))`.
- Prompt: same `run_lib.eval_prompt`; generate with a larger cap (MATH CoT is long — `max_new_tokens ≈ 512`).
- **Answer extraction:** the model, trained on MetaMathQA, emits `"The answer is: <expr>"` (and/or a
  `\boxed{}`). Extract the model answer = text after the last `"The answer is:"` OR the last `\boxed{}`;
  compare to gold via **latex-normalized equivalence** (`is_equiv`), NOT numeric last-token (MATH answers
  are fractions/expressions/sets). Port the canonical Hendrycks `util.py` helpers
  (`last_boxed_only_string`, `remove_boxed`, `_strip_string`, `is_equiv`) — the exact functions lm-eval's
  `minerva_math`/`hendrycks_math` task uses. Do **not** reuse `extract_answer_number` for MATH.

### 4.4 Wiring into the pipeline WITHOUT breaking retention (spec only)
- **New evaluator module** `math_eval.py` (new file, additive — does not touch camp5's code paths):
  `run_gsm8k_faithful(model, tokenizer, ...)` and `run_math_hendrycks(model, tokenizer, ...)`, both
  in-process (per uiortholora-phase1-gotchas: in-process eval avoids the PEFT reload bug), reusing the
  `eval_cs.run_eval` generation skeleton + `run_lib.eval_prompt`.
- **In `eval_one_gpu.py`:** extend the `--adapt_task` choices (`eval_one_gpu.py:47`) with
  `gsm8k_faithful` and `math`. Add branches next to the existing `gsm8k` branch (`:91-101`) that call
  `math_eval`. **Leave the existing `cs` and `gsm8k` branches untouched** so already-queued/running
  camp5 jobs (which pass `--adapt_task {cs,gsm8k}`) are byte-for-byte unaffected.
- **Retention UNCHANGED:** the BBH+MMLU-Pro (+broad) block (`eval_one_gpu.py:117-145`) is not modified.
  The adapt metric still lands in `cs_avg` so the headline schema stays uniform. For MATH the run writes
  both `gsm8k` and `math` accuracy into `per_dataset`/`headline` (small additive change to the summary
  dict at `:147-150`).
- **Optional:** a standalone `eval_math_faithful.py` (load adapter once, run both GSM8K+MATH) is a
  zero-risk alternative that never edits `eval_one_gpu.py` at all — recommended if minimizing camp5 risk
  matters more than a single combined pass.

---

## 5. Commonsense recipe AUDIT + fixes (vs CLoRA Table 2)

Confirmed defects and fixes (G5):
1. **Plain-LoRA rank mismatch.** Our "plain LoRA" ran at **r16/α32** (`make_campaign_jobs.py:18`), but
   CLoRA's MAIN LoRA baseline is **r32/α64**; r16 is CLoRA's SEPARATE reduced-rank baseline. **Fix:** add
   `lora_r32` (r32/α64) as the headline LoRA; relabel the existing r16 run as the `lora_r16` reduced-rank
   baseline; add `lora_r8` for completeness.
2. **Internal rank-unfairness.** Our `lorawd` is **r32/α64** (`make_campaign_jobs.py:22`) while plain
   LoRA was r16 — so any "LoRA+wd beats LoRA" read is confounded by rank. **Fix:** compare LoRA+wd (r32)
   against LoRA r32 (both r32/α64), the paper-faithful operating point. (This mirrors handoff/19's
   locked "one operating point" fairness rule.)
3. **CLoRA k grid incomplete.** Only k=1024 was run (`make_campaign_jobs.py:21`); Table 2 is
   **{128,256,512,1024,2048}**. **Fix:** add k{128,256,512,2048}.
4. **Missing PiSSA.** Table 2 lists PiSSA (major-SVD). **Fix:** add `pissa_r32` (§2.3).
5. **Missing LoRA-L2.** Table 2 baseline "LoRA-L2 (L2 weight 1e-5)". **Fix:** `lora_l2` = r32/α64 with
   `--weight_decay 1e-5` (caveat R6: AdamW decoupled wd ≈ L2, not an explicit loss penalty).
6. **Confirmed already-correct:** r/α defaults (r32/α64), cutoff 256, dropout 0.05, target modules, the
   8-task in-domain eval, and the BBH+MMLU-Pro retention out-domain eval all MATCH CLoRA (§1.2).

---

## 6. Cell count + 8-wide B200 wall-clock

**Per-cell timing basis (measured, `train_registry.jsonl`):** commonsense 170K @ cutoff256 train median
**7234 s ≈ 2.0 h** (n=93); math 100K @ cutoff256/512 train median **5065 s ≈ 1.4 h** (n=45). Broad eval
(BBH 5-shot + MMLU-Pro + MMLU/ARC/TruthfulQA) ≈ **1.5–2 h/cell** (handoff/19 §2.4). MATH 395K @ cutoff512
≈ **~4× the 100K per-cell train cost** (395/100 × 512/~384 ≈ 5.3× vs 100K, but conservatively **4×** per
the standing rule) ⇒ **~6–8 h train/cell**; faithful math eval adds GSM8K (1319, beam4) ~0.5 h + MATH
(5000, long-gen) ~2–3 h + retention ~1.5–2 h ⇒ **~4 h eval/cell**. **Lock ~10 h/cell for math (395K),
~4 h/cell for commonsense (170K)** conservative.

### 6.1 Math faithful repro (Table 3), seed 42 first

| Block | Arms | Cells (s42) |
|---|---|---|
| Reproduction baselines @ LR=3e-4 | lora, pissa, milora, dora + clora k{64,128,256} | 4 + 3 = **7** |
| LoRA+wd sweep | LR{1e-4,2e-4,3e-4,5e-4,7e-4,1e-3} × wd{0,0.1,0.2,0.3,0.5} | **30** (−1 dedup with lora baseline) |
| **Math seed-42 subtotal** | | **~36** |
| Seeds 43/44 for winners | 7 baselines + ~6 best LoRA+wd | 13 × 2 = **26** |
| **Math TOTAL (3-seed, partial)** | | **~62** |

- Seed-42 math (36 × 10 h / 8) ≈ **45 h ≈ 1.9 days**.
- + seeds 43/44 (26 × 10 h / 8) ≈ **33 h ≈ 1.4 days**.
- **Math faithful repro total ≈ 78 h ≈ 3.3 days** at 8-wide (conservative). Depth-first to a Table-3
  reproduction verdict = the **36 seed-42 cells ≈ 1.9 days**.

### 6.2 Commonsense re-audit (Table 2), seed 42 first

| Block | Arms | Cells (s42) |
|---|---|---|
| Baselines @ LR=3e-4 | lora_r32, dora_r32, pissa_r32, milora_r32, lora_r8, lora_r16, lora_l2 | **7** |
| CLoRA k grid | k{128,256,512,1024,2048} (k1024 may exist post-fix) | **5** |
| LoRA+wd sweep | 6 × 5 | **30** (−1 dedup) |
| **CS seed-42 subtotal** | | **~41** |

- CS seed-42 (41 × 4 h / 8) ≈ **21 h ≈ 0.9 days**. +seeds 43/44 for winners (~13 × 2 × 4 h / 8 ≈ 13 h).
- **CS re-audit total ≈ 34 h ≈ 1.4 days** at 8-wide.

### 6.3 Grand total
**Math (3.3 d) + Commonsense (1.4 d) ≈ 4.7 days at 8-wide**, seed-42-first depth (36+41 = 77 cells ≈
2.8 days) reaching both reproduction verdicts, with seed 43/44 breadth (~1.9 days) added on the winners.
Note the **395K math cell is the cost driver (~10 h vs ~4 h CS)** — the 4× data multiplier dominates.

---

## 7. Implementation-without-breaking-camp5 plan

**Constraint:** camp5 (`gpu_pool --tag camp5 --jobs jobs/combined_nocorda.txt`, 118 queued) spawns
`train_cs.py`/`eval_one_gpu.py` **fresh per job**. Editing those files in the live checkout would corrupt
every not-yet-started camp5 job. Single-scheduler rule (handoff/19 §7.2): never launch a 2nd 8-GPU pool
while camp5 is live.

**Sequencing:**
1. **NOW (no GPU, no code-path risk):** build `metamathqa_395k.json` (§3) and `dataset/MATH/test.json`
   (§4.3) in the main checkout — purely additive input files; camp5 reads only `metamathqa_100k.json` /
   `commonsense_170k.json`, so new files cannot affect it.
2. **NOW (isolated):** make ALL code changes in a **git worktree/branch** (e.g. `EnterWorktree` →
   branch `faithful-repro`), so the live checkout camp5 executes from is untouched. Develop + unit-test
   there (e.g. `milora_init.__main__` self-test, `validate_residual_zero_step.py`, a 5-example smoke of
   `math_eval`).
3. **AFTER camp5 drains** (poll `logs/*.log` / `pgrep -f 'tag camp5'`): merge the branch (or run the
   repro pool directly from the worktree), then launch ONE new 8-GPU pool with a fresh tag (e.g.
   `--tag frepro`) on the generated job file. `make_campaign_jobs.py:46` already skips cells with an
   existing `results/<run>/summary.json`, so it is resumable.

**Exact code changes needed (all on the branch/worktree):**
1. `train_cs.py`: add `--pissa` arg + init branch (§2.3.1–2.3.2); add `pissa` to the `residual_method`
   OR-clause (`:280-281`); bump default `--cutoff_len` handling is not needed (pass `--cutoff_len 512`
   per-job for math).
2. `residual_save.py`: **generalize to scaling≠1** (option A, §2.1) — drop α==r assert (`:59`), set
   `cfg["lora_alpha"]=2*cfg_old_alpha` (`:61-62`). (Skip if using fallback B.)
3. **New file** `math_eval.py`: `run_gsm8k_faithful` + `run_math_hendrycks` (§4.2–4.3), reusing
   `eval_cs.run_eval` batching + `run_lib.eval_prompt`; port `extract_answer_number` and Hendrycks
   `is_equiv`/`last_boxed_only_string`/`remove_boxed`.
4. `eval_one_gpu.py`: add `gsm8k_faithful` + `math` to `--adapt_task` choices (`:47`) and two new
   branches calling `math_eval`; extend the summary dict (`:147-150`) to hold both metrics. Existing
   `cs`/`gsm8k` branches unchanged. (Or the zero-touch standalone `eval_math_faithful.py`.)
5. `make_campaign_jobs.py`: new `ARMS` for the two tables (§2.1, §2.2), the LR set (§2.4), a `--weight_decay`
   axis for the lorawd sweep (the current generator has no wd axis — add a `WDS` loop for the lorawd arm),
   and math jobs emit `--adapt_task gsm8k_faithful` (or `math`) + `--cutoff_len 512`.
6. `metamath_prep.py`: `N=395000`, keep `original_question`, new OUT filename (§3).

**Data prep + MATH acquisition scripts** are additive new files; no existing file is mutated by them.

---

## 8. Risks / open questions

- **R1 (contamination).** MetaMathQA claims no test augmentation, but `*_Rephrased`/`*_SV`/`*_FOBAR`
  restate **train** GSM8K/MATH questions — must confirm none collide with GSM8K/MATH **test**. Mitigation:
  keep `original_question` (§3) and hash-dedup the 395K train against `dataset/gsm8k/test.json` +
  `dataset/MATH/test.json` before training; report the overlap count. (Same disjointness discipline as
  handoff/19 §4.2.)
- **R2 (cutoff_len for math).** The paper/LLM-Adapters math recipe uses cutoff 256 (`math_running_commands`),
  but MetaMathQA CoT `response` is longer than math_10k; 256 truncates the "The answer is:" tail, which
  hurts the last-checkpoint accuracy the table reports. Recommend **512** and disclose; if reproducing the
  paper's exact number matters more than headroom, run a 256-vs-512 sensitivity pair on `lora`@3e-4.
- **R3 (residual-save at scaling=2).** Option A generalization is small but changes a file the whole
  audit assumed is scaling==1. Gate with `validate_residual_zero_step.py` per residual arm (0-step
  post-reload ‖ΔW‖ < 1e-4) BEFORE trusting any PiSSA/MiLoRA math number. If risk-averse, use fallback B
  (α=r=64) and footnote the deviation.
- **R4 (prompt template).** `run_lib` templates (train==eval, no trailing space) differ from LLM-Adapters
  originals by trailing whitespace (`finetune.py:337` 2 spaces, `evaluate.py:154` 1 space). Our internal
  train==eval consistency is preserved and is what matters — but the faithful math eval MUST use
  `run_lib.eval_prompt`, never `evaluate.py:generate_prompt`, or it reintroduces a train/eval mismatch.
- **R5 (MATH extraction edge cases).** Boxed vs "The answer is:" tail; LaTeX normalization (fractions,
  `\dfrac`, degrees, `\%`, `\text{}`, matrices/sets). Must port the canonical Hendrycks `is_equiv`
  (identical to lm-eval `minerva_math`) rather than roll a custom normalizer, or MATH accuracy will be
  systematically low (parse failures score 0). Log parse-failure rate as a sanity gate.
- **R6 (LoRA-L2 semantics).** CLoRA's "LoRA-L2 (L2 weight 1e-5)" may be an explicit loss-term L2 penalty;
  our `--weight_decay 1e-5` is AdamW **decoupled** weight decay. At 1e-5 the difference is negligible, but
  if a reviewer probes it, either state the equivalence or add a tiny explicit-L2 penalty (mirroring the
  CLoRA regularizer hook) for exactness.
- **R7 (precision).** We train bf16 on B200; CLoRA/LLM-Adapters used fp16. Disclose; expect ≤~0.5 pp
  drift. Do NOT mix pre/post gen_cap+BBH-fix eval rows (handoff/18 §3d) — run one clean `frepro` eval pass.
- **R8 (MATH eval cost).** 5000 MATH problems × long CoT × beam4 is the eval-time driver. If wall-clock
  is tight, greedy decode (beam=1) or a fixed subset (e.g. MATH-500) is an option — but that deviates
  from "MATH test accuracy"; prefer full 5000 greedy over a subset with beam4.

---

### Appendix — key file:line index
train_cs.py: args 120-164 (r/α 152-153, clora_k 155, wd 129, targets 134, cutoff 126); build_adapter
88-116; residual-method init blocks 210-276 (milora 228-232); residual_method set 280-285; TrainingArguments
291-297; save+residual convert 310-316. · data_aware_init.py: pissa_BAR 63-71, inject_lora_init 37-59. ·
milora_init.py: apply_milora 26-43. · residual_save.py: α==r assert 59, rank-2r convert 37-64. ·
eval_one_gpu.py: adapt_task 47, gsm8k(lm-eval) 91-101, cs 102-108, retention 117-145, summary 147-156. ·
eval_cs.py: generate_prompt 23-42, extract_answer 45-58, run_eval 61-88. · run_lib.py: train_prompt 35-53,
eval_prompt 56-74. · repro/LLM-Adapters/evaluate.py: extract_answer_number 272-287, generate_prompt 141-160.
· repro/CLoRA/clora.py: P init 33-36, reg loss 41-66. · make_campaign_jobs.py: ARMS 17-26, LRS 30-31, SEEDS 34.
