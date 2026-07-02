# Project Narrative — Catastrophic Forgetting in PEFT: The Magnitude Law

*A comprehensive intellectual history of the project, reconstructed from handoff docs `00`–`16` + README and the auto-memory files (`MEMORY.md` and everything it indexes). Written 2026-07-02.*

This document is the single source of truth for **how the project got to where it is**: what we set out to
do, how the thesis mutated three times, every adapter we ported and audited, every bug we hit and fixed,
which findings are mature enough for the paper, and the rationale behind the load-bearing decisions. Every
non-obvious claim cites the handoff (`hN`) or memory file it comes from.

---

## The central thesis (the paper's spine)

Everything below serves one four-part claim. Keep this framing throughout:

1. **MECHANISM.** Catastrophic forgetting (CF) under PEFT is governed by the **weight-update magnitude
   `‖ΔW‖_F`** (token-weighted Frobenius norm; field name `fdelta_token_weighted`), **not** by the adapter's
   geometric structure. All adapters fall on **one** retention-vs-`‖ΔW‖` curve.
2. **CONSEQUENCE.** Plain **LoRA + weight decay** matches or **surpasses** the adaptation-vs-retention Pareto
   frontier of the elaborate geometric adapters (CorDA, MiLoRA, SC-LoRA, LoRA-Null, CLoRA, DoRA), because
   weight decay is the simplest way to control `‖ΔW‖`.
3. **DIAGNOSIS.** The fancy adapters' reported "wins" are largely an **LR / magnitude artifact**. Evaluated
   at a single learning rate they look better; **sweeping LR** collapses them onto the same magnitude curve.
   The **LR-sweep-per-method is the instrument** that exposes this.
4. **MESSAGE.** A wake-up call to a field that ships a new adapter every week — **control the magnitude, not
   the geometry.**

**Study design.** Llama-2-7B (primary, **complete**) + Qwen2.5-7B (replication, **in progress**) ×
{commonsense, math} × 8 adapters (lora r16, lora+wd r32/wd0.3, dora r16, corda r16 KPA, milora r32,
sclora r32, lora_null r16, clora k1024) × 7 LRs (2e-5…1e-3) × seed 42. Retention = BBH (`bbh_fewshot`) +
MMLU-Pro (**core**) + MMLU + ARC-c + TruthfulQA (**broad**). Adaptation = 8-task commonsense accuracy, or
GSM8K exact-match (math). Magnitude axis = `‖ΔW‖_F`. (`h13 §1–2`, `port-audit-and-lr-sweep-plan`.)

---

## (a) Problem & thesis evolution

The project began under a completely different banner and pivoted **three times** before arriving at the
magnitude-law wake-up call. This arc is the most important thing for a reader to understand.

### Thesis 0 — "Is UIOrthoLoRA an A\*-worthy CLoRA-beater?" (DEAD)

The original goal was a head-to-head reproduction proving a home-grown adapter, **UIOrthoLoRA**
(orthogonal-rotation LoRA in a truncated-SVD spectral-tail basis) and its linear cousin **UILinLoRA**, could
beat the strongest published forgetting-mitigation adapter, **CLoRA**, on the LLM-Adapters commonsense +
GSM8K/MATH + retention (BBH/MMLU-Pro) setting on Llama-2-7B (`reproduction-campaign`, `h02`).

This goal is **dead**. Even after we fixed a serious layer bug (the "major term", see below), corrected
UIOrthoLoRA only **ties** CLoRA and **loses the high-commonsense corner**: adaptation ceiling ~74 CS vs
CLoRA/LoRA ~79–80, and it is Pareto-dominated on both axes in Wave 1 (`h00` lines 10–16, `h01`,
`uio-publishability-plan`). Best corrected Pareto points were only `k410_v205_dE1` 71/25.7 and
`k410_v410_dE0` 69/26.3 — ties, not wins (`h12` ledger). The instrument is **gone from the active campaign**
(`h13 §1`).

### Thesis 1 — UIOrthoLoRA as a controllable *instrument*; the leakage thermometers

When the head-to-head died, we repurposed the corrected adapter as a **measurement instrument**: it is the
only adapter with explicit knobs to independently set the two candidate drivers of forgetting —
**directional preserved-subspace leakage** (`use_de`, `k_vec`, plus a continuous λ_E/λ_D penalty) and
**update magnitude** (learning rate, `initial_sigma`) (`h05`, `uio-publishability-plan`).

We built **orthogonality-leakage thermometers** (`leakage.py`): `μ_E = ‖U_Rᵀ·E·Ū_r‖₂` (left/output leak
into the preserved subspace), `ν_D = ‖V_Rᵀ·D·V̄_r‖₂` (right/input), plus `Leak11`, `OffTailF`, `RelPertF`,
`DriftU/V` (`h03`). The user's exciting framing was an **"optimal leakage budget"** between the low tail and
the major component: too little leakage → under-adapt, too much → forget (`h03`, `uiortholora-phase1-gotchas`).

**This thesis was overturned by its own data.** The B1 leakage map (`h04`, `make_leakage_map.py`, 16 runs):
retention tracks a **ΔW magnitude budget (`dw_sv_max`)**, **not** the directional leakage the thermometers
measure. Clean arm (`use_de=0`): `μ_E ≈ ν_D ≈ 0.003` for *every* config, yet retention swings 4→26,
`corr(ret, dw_max) = −0.86`. Leaky arm (`use_de=1`): `μ_E` 1.3–1.8 but the D/E gates brake `dw_max` to
7–9.5 so retention survives at 22–25. So the thermometers measure **direction**; preservation needs a
**magnitude** budget (`h04` headline). Weight-basis direction `μ_E` predicts retention at **r ≈ −0.09** —
i.e. **not at all** (`h06` D1 verdict, `h07 §7`).

### Thesis 2 — THE MAGNITUDE (FROBENIUS) LAW

The pivot: retention ≈ f(`‖ΔW‖_F`), method-independent. Within a single structure the correlation is
**−0.96 to −0.98** across LoRA / CLoRA / UIOrthoLoRA, with points interleaving on **one curve**
(`h00` line 19, `h08`). Weight-SVD **direction is irrelevant** (`μ_E` r=−0.09); a **rank sweep** confirmed
the conventional direction (rank↑ → `‖ΔW‖_F`↑ → forget↑: ret 25.4→8.5 as r 4→256) with **no independent
rank effect** and no "diffusion" (σ₁ grows with rank) (`h09` #5, `h12` ledger).

Two honesty checks tempered this from becoming trivial:
- **The r=−0.98 is near-circular** (bigger ΔW perturbs more) — do **not** headline it (`h08` review #1).
- The **basis reveal** (`h06 ★`): our `μ_E` is in the *static weight-SVD basis*; the field (CorDA/SC-LoRA)
  argues knowledge lives in the **data/activation-covariance basis**. So "direction doesn't matter" is
  really "the weight-SVD basis is the wrong basis." Crucially, **the basis reframe is NOT novel** — it *is*
  CorDA/SC-LoRA's central claim (`h06 ★` user pushback). Three angles in a row were pre-empted: magnitude
  (≈ CLoRA's F∆), weight-direction (≈ OPLoRA), data-basis (≈ CorDA/SC-LoRA).

### Thesis 3 (CURRENT) — the LoRA+wd wake-up call, framed *around the law*

The constructive turn came from the **D2-killer / #3 falsifier**: if forgetting is governed by `‖ΔW‖`, then
plain **LoRA + weight decay** — a subspace-free magnitude knob — should reproduce (or beat) the elaborate
adapters' frontier (`h05` D2-KILLER, `h09` #3). It did. The bold first read ("LoRA+wd **dominates** CLoRA")
**deflated at full scale to a TIE/edge**, and the strongest claim ("geometry is irrelevant, all methods
collapse to one curve") is **not** supported by a perfect collapse (`h11`, `h12` one-line status).

The **current, deliberate framing** (`h13 §1`, `port-audit-and-lr-sweep-plan` "PAPER FRAMING DECISION"):
lead with **THE LAW** — *retention is governed by `‖ΔW‖` regardless of method; the simplest magnitude
control (LoRA+wd) matches/beats elaborate structured inits* — and the **LR-sweep-per-method** as the
instrument exposing the artifact. We deliberately do **not** headline "LoRA is the best method" (that would
require fairness experiments — give every method the wd knob, param-match — which are **deferred**). Framing
around the law sidesteps the method-ranking fairness gaps. HERO figure = all-methods-on-one
retention-vs-`‖ΔW‖` curve + a residual test showing method identity adds ~nothing beyond `‖ΔW‖`.

---

## (b) Methods / adapters — what each is, how ported, fidelity status

**Architecture principle** (`method-port-recipes`): MiLoRA, LoRA-Null, SC-LoRA, PiSSA, CorDA are *just LoRA
with a different starting point* — extract their init routine → `(A_init, B_init, W_residual)` → load into a
standard PEFT LoRA adapter → train with **one shared trainer** (`train_cs.py`). Only **CLoRA** adds a loss
term (vanilla init). This keeps every run comparable. Integrity bar: *"we run other researchers' adapters,
so prove our port is faithful BEFORE claiming a method fails"* (`port-audit-and-lr-sweep-plan`).

**Phase-0 fidelity audit (2026-06-25) — ALL ports verified faithful** (`port-audit-and-lr-sweep-plan`, `h13 §6`):

| Arm | What it is | Port / recipe pointer | Fidelity status |
|---|---|---|---|
| **LoRA** r16 | vanilla baseline | `train_cs.py` (Gaussian A, zero B) | reference |
| **LoRA+wd** wd0.3 (r32) | AdamW weight decay on the adapter = direct magnitude knob | `train_cs.py --weight_decay`; touches ONLY lora_A/lora_B (base+LN frozen) `h09` #3 | the protagonist |
| **DoRA** r16 | magnitude-direction decomposed LoRA (constant per-column magnitude) | `train_cs.py --use_dora` | faithful |
| **CorDA** r16 (KPA/KPM) | data-driven: SVD of `W·Cov_input` over a calib set; freeze principal (knowledge) comps, adapt the **smallest-r** (KPA=KPM retention mode) | vendored PEFT `src/peft/tuners/lora/corda.py`; calib **nq_open** 256; `method-port-recipes`, `h14 §0` | **faithful** vs PEFT; large `‖ΔW‖` is **genuine KPA** (C_inv un-whitening makes A-norm ~49), 0-step→ΔW=0 (`port-audit`) |
| **MiLoRA** r32 | init from **bottom-r (minor)** singular triples of W₀; residual = principal | `method-port-recipes`; `corda`/`milora` init frozen Jun 18 | **faithful** (selects minor, not PiSSA) |
| **SC-LoRA** r32 (β0.5) | subspace-constrained: `M=(1−β)Cov₊ − β·Cov₋`, top-r eigvecs `Q_r`; `B=Q_r`, `A=Q_rᵀW₀`, residual `W₀−Q_rQ_rᵀW₀`; **β swept** | `sclora_init.py`, output-side Y; `method-port-recipes` | **math faithful**; OPEN nuance: per-sample norm `Y.max().abs()` (|max|) vs CorDA-family `Y.abs().max()` (max\|·\|); calib max_len 2048 vs repo 512/1024 |
| **LoRA-Null** r16 | input-cov **null-space** init: init `BA=W₀·U_null·U_nullᵀ` from smallest-SV left vectors; residual `W₀−BA` | **ported this session** as `lora_null_init.py` (unit-tested: recon 2.4e-7, silent-on-used-dirs 3.6e-7); calib nq_open | faithful; FLAG: `null_dim` default=r is an assumption; `freeze_a` off for head-to-head |
| **CLoRA** k1024 | loss-term (not init): `λ·Σ(½‖A·Pv‖²+½‖Bᵀ·Pu‖²)`, frozen **random-orthonormal** P width k, λ=1 | `CLoRARegularizer` in `train_cs.py` vs `repro/CLoRA/clora.py`; `method-port-recipes` | **faithful**; caveat: over-constrains at extreme k (k2048 CS→65 vs paper 83.7) `h00` #8 |
| ~~UIOrthoLoRA / UILinLoRA~~ | orthogonal-rotation / linear-scaling in spectral-tail basis (our home-grown) | `uio_inprocess.py`; **removed from active campaign** | DEAD as a method; served as the instrument |
| *(future)* CorDA++ | dynamic covariance selection + dynamic rank allocation under fixed param budget | plan in `h14`; blocker = fetch arXiv:2506.13187 (now cleared, algorithms in `h14 §8`) | not yet built |

**Related-work positioning** (`h07`, with strict `[MECH]/[OURS]/[VERIFY]/[INTERP]` provenance tags): the
single load-bearing axis is **basis** — static weight-SVD (OPLoRA) vs data/activation covariance
(CorDA/SC-LoRA/CorDA++) vs gradient/Fisher (Subspace-Geometry). **All arXiv IDs are `[VERIFY]`** and must be
confirmed before manuscript: `2603.02224` is almost certainly wrong (implausible March-2026 date);
`2510.13003` (OPLoRA), `2406.05223` (CorDA), `2505.23724`/OpenReview KAE9YDK0t8 (SC-LoRA), `2506.13187`
(CorDA++) all need confirmation (`h07 §9`, `h12 §GATE-2`).

---

## (c) The measurement apparatus

**Retention suite.**
- **CORE** = mean of `bbh_fewshot` (answer-only, NON-CoT, 3-shot) + `mmlu_pro` (5-shot CoT). Answer-only BBH
  (not CoT `bbh`) is what reproduces CLoRA's base 34.91 → we get 33.1; CoT gives 39.5 (`h00` #6,
  `uiortholora-phase1-gotchas`). Base MMLU-Pro 18.96 vs 18.56 target → near-exact.
- **BROAD** adds MMLU + ARC-c + TruthfulQA (`h13 §1`).
- **Per-benchmark finding:** MMLU forgets fastest (−23pp/decade of `‖ΔW‖`); TruthfulQA ~immune (`h13 §2`).
- **Base ceiling = 26.0** (BBH-AO 33.1 + MMLU-Pro 19.0). Strong configs essentially TIE it → forgetting <1pt
  is noise; the lever is **adaptation at the ceiling** (`h05` phase-1 facts). *Open gap:* base ceiling for
  MMLU/ARC/TruthfulQA not yet calibrated (`h13 §8.4`).

**Adaptation metrics.** 8-task commonsense accuracy (`eval_cs.py`, LLM-Adapters generation eval; LoRA r32
reproduces BoolQ 69.97% vs target ~69–70% — the step-1 validation, `reproduction-campaign`); or GSM8K
exact-match for math (`eval_one_gpu --adapt_task gsm8k`, math trained on metamathqa_100k).

**The magnitude axis — `‖ΔW‖_F` (`fdelta_token_weighted`).** THIS IS THE CANONICAL AXIS, **not**
`dw_sv_max`. σmax is confounded — CorDA's spiky spectrum inflates it, so a σmax-based law mis-ranks CorDA
(`h13 §2`, `port-audit-and-lr-sweep-plan`). *(Note the historical inconsistency: the earlier `mtx_` matrix
memory and `h04` leakage map used `dw_sv_max`; the LR-sweep era switched to `‖ΔW‖_F`. See §(e) inconsistencies.)*
Computed by `fdelta.py` / `forensics.py` (weight-basis `UᵀΔWV`); data-basis variant in `forensics_databasis.py`
(`data_resp = ‖ΔW·C^½‖²`, `--cov_source retain` = MMLU-Pro, the fixed default — was buggily using the
fine-tuning task covariance, `h09` methodology fix #2).

**In-process eval.** UIOrthoLoRA checkpoints **cannot round-trip through PEFT** (two bugs, §d), so all
UIOrthoLoRA runs trained + evaluated in **one process, no save/reload** (`uio_inprocess.py`); the trick is
`HFLM(pretrained=<in-mem peft model>, tokenizer=tok)` accepts an in-memory model (`h00` #1,
`uiortholora-phase1-gotchas`). `eval_one_gpu.py` does the same per-GPU for the reloadable LoRA-family arms.

**Residual-save.** PiSSA-family inits (CorDA, MiLoRA, SC-LoRA) overwrite `base.weight = W₀ − B_init@A_init`
in memory, but PEFT `save_pretrained` persists **only the adapter** → reload adds the trained adapter onto
the ORIGINAL W₀ and the cancellation is lost → evaluated `‖ΔW‖` explodes (CorDA r128: F=4.5, σmax=3311,
retention 0.02) despite healthy training loss. **Fix** (`residual_save.py`, wired into `train_cs.py` via
`residual_method`): require scaling==1, snapshot init adapter, then write a **rank-2r** W₀-relative adapter
`A''=[A_tr; A_init]`, `B''=[B_tr, −B_init]` so `B''@A'' = B_tr@A_tr − B_init@A_init`. Self-check: a 0-step
run converts to ΔW=0 → base retention (`peft-residual-init-save-bug`, `validate_residual_zero_step.py`
PASSED, `h13 §6`).

**Fast vs full retention (historical).** UIOrthoLoRA's slow forward made full MMLU-Pro CoT 4–16h/GPU, so we
used `--ret_limit 64 --ret_max_gen 512`; calibrated **FAST ≈ FULL + ~0.9** (LoRA full 21.66 vs fast 22.52)
(`h00` #3, `h01`). The LR-sweep campaign uses full-scale eval with the Qwen-era caps below.

**Scheduling.** `gpu_pool.py` runs one job/GPU across 8 B200s; `run_all_experiments.sh` is the resumable
orchestrator (phases L2-CS → Qwen-CS → L2-math → Qwen-math, smoke-gated); `make_campaign_jobs.py`
regenerates each phase's remaining cells (skips done summaries) so any interruption resumes by just
re-running. Salvage (`make_salvage_evals.py`) recovers trained-but-uneval adapters eval-only (`h13 §3`).

**Figures/tables.** CANONICAL = `paper_figs_v2.py` → `paper/figs_v2/` + `table_main_{cs,math}.tex`
(fig0_hero=the law, fig1=axis choice, fig2=fairness residuals, fig3=pareto, fig4=LR-sensitivity,
fig5=per-benchmark, fig6=structure, fig7=LR-is-proxy, fig8=budget). OLD `paper_assets.py` is **DEPRECATED**
(pooled matrix+sweep → stale collapse-inflated table; output redirected to `*_LEGACY`) (`h13 §5`).

---

## (d) Chronological log of bugs / fixes and hard-won gotchas

Ordered roughly by discovery date. Each is a "do not relearn the hard way" entry.

1. **UIOrthoLoRA cannot round-trip through PEFT** (early). (a) SVD basis U/S/Vᵀ dropped (no prefix) → FIXED
   by storing as frozen prefixed Parameters. (b) Orthogonal rotators still don't reload — PEFT's
   `_insert_adapter_name_into_state_dict` doubles `default` in the parametrization keys → unfixable cleanly.
   ⇒ **always eval UIOrthoLoRA in-process** (`h00` #1, `uiortholora-phase1-gotchas`).

2. **The LR mismatch (the big early one).** UIOrthoLoRA's sigma/D/E init at 0.1 → adapter output ~0.01 →
   grad_norm ~10× smaller than LoRA → at LoRA's LR 3e-4 it **under-adapts** (CS 47–62). Use **LR=1e-2**
   (loss 0.65 < LoRA 0.76). NEVER compare UIOrthoLoRA at 3e-4 — a fake result. This planted the seed of the
   entire **per-method LR** insight (`h00` #2, `h01`). *Foreshadows the LR-artifact thesis.*

3. **The "major-term" layer bug (finding #4).** Our ΔW carried an extra frozen term
   `E·U₁·diag(1)·V₁ᵀ·D` (leading band, unit singular values) NOT in the paper's
   `ΔW=E·Ū_r·Σ'_r·V̄_rᵀ·D`, perturbing the *preserved* subspace (unscaled at `use_de=0`). Thermometers are
   tail-only so they DON'T see it → a config could show `μ_E≈0` yet still forget. At init its Frobenius was
   ~17× the entire tail — it *was* the dominant clean-arm forgetting driver. **Fixed** by the `drop_major`
   flag (sets frozen major-band SVs to 0), validated by `test_a5_drop_major.py` (`h00` #4, `h05`, `h03`
   caveat).

4. **`use_de` gates break orthogonality** — with `use_de=1`, `diag(E)·M·diag(D)` leaves the tail subspace →
   high leakage/high F∆ (UILinLoRA F∆ 0.75 despite `‖ΔW‖` 9). `use_de=0` is the clean-orthogonality arm
   (`h00` #5).

5. **BBH config** — answer-only `bbh_fewshot` 3-shot (not CoT) reproduces base (`h00` #6).

6. **CLoRA build gotchas** — over-constrains at high k (k2048 CS→65); build P-matrices via **GPU QR** +
   thread caps (CPU QR × many jobs → load 267, stalls) (`h00` #8–9).

7. **rc=127 idle incident (~9h lost, 2026-06-12).** `gpu_pool.py` job files used bare `python`; a detached
   `setsid` shell had no venv on PATH → `command not found` → all jobs die in 0s, GPUs silently idle.
   **Fix: full venv python path in every `jobs/*.txt`.** Sanity: `grep rc=127 logs/<tag>_pool.log`
   (`h00` #10, `catastrophic-forgetting-workdir`).

8. **Monitoring reality.** In-session `sleep` heartbeats and ScheduleWakeup are **suspended while the
   session is idle** (a `sleep 3300` accrued 24min in 3h20 wall). Only detached `setsid` processes run on
   real wall-clock. ⇒ rely on detached auto-pipelines through dormancy (`h00` MONITORING REALITY,
   `catastrophic-forgetting-workdir`).

9. **PiSSA-family residual-save explosion** (caught 2026-06-21) — see apparatus §(c); the rank-2r
   W₀-relative conversion fix (`residual_save.py`) (`peft-residual-init-save-bug`).

10. **Data-integrity / checkpoint-collision bug (the false positive).** Duplicate run_names wrote to the
    SAME checkpoint path (`magctl` + a killed `phase2` pool both trained `lora_wd0p3`) → overwrite/corruption
    → `lora_wd0p3_full` CS=52 vs fast 80.7. This produced the spurious "LoRA+wd **dominates**" read; the
    `*_clean` reruns (unique names) corrected it to a TIE/edge. **LESSON: never reuse run_name/checkpoint
    paths across pools** (`h12 §DATA-INTEGRITY`).

11. **CorDA calibration bug: wikitext-2 → nq_open (external fidelity audit, 2026-06-29).** CorDA-KPA freezes
    the directions **most responsive to its calibration data**; we had calibrated on **wikitext-2** instead
    of the paper's **nq_open**, so we preserved the WRONG subspace, **likely understating CorDA's
    retention**. FIXED in `train_cs.py` (log now reads `[corda] KPA init (nq_open calib)`); the 7 wikitext
    CorDA runs archived to `results/_archive_wikitext_corda/`, re-queued. **The "CorDA off-curve" finding is
    PENDING this re-run.** The old 3-seed `mtx_corda_*` rows are ALSO wikitext-tainted. **Lesson: always
    check the calibration dataset, not just the init math** — the earlier audit verified CorDA's
    decomposition but MISSED the calib choice (`h13 §2, §10`, `port-audit-and-lr-sweep-plan` audit fix).

12. **The calibration↔eval distribution mismatch (deeper, OPEN — CorDA++ review).** nq_open (factoid QA)
    calibration protects the wrong directions for our academic/reasoning eval (BBH/MMLU/ARC…). So
    *"data-aware inits forget more than their budget"* — for **both** CorDA **and** SC-LoRA — may be an
    artifact of preserving the wrong knowledge, handicapping the calibration-using arms vs the
    calibration-free arms. Resolution = **eval-matched calibration** (MMLU/ARC aux-train, disjoint) + an
    nq_open-vs-eval-matched **sensitivity arm** for ALL calibration-using arms (`h13 §2`, `h14 §8` Fix 1).
    **This is the single biggest open fairness question in the project.**

13. **The four Qwen eval bugs (2026-07-01).** Bringing Qwen2.5-7B online exposed four Qwen-specific eval
    issues; **each proven a no-op for Llama-2**, so the 56 L2-CS + 14 L2-math cells stay valid (`h15`, `h16`):
    - **Bug 1 — rambling cap.** Qwen ships `generation_config.max_new_tokens=2048`, which overrides HFLM's
      `max_length`; ~17% of `mmlu_pro`/`bbh` questions ran to 2048 tokens without a parseable answer (~8×
      slow). FIX: set `model.generation_config.max_new_tokens = gen_cap` on the base model pre-PEFT-wrap.
      Llama-2 A/B at cap 512 vs 2048 → **byte-identical retention**.
    - **Bug 2 — batch size 1.** HFLM's auto batch-sizer reserved memory for Qwen's native 32768 window →
      concluded only batch 1 fits (17GB/183GB, ~50% util, ~10× slow). FIX: pass `max_length=4096`
      (= Llama-2's window) → also improves cross-model comparability. Net: ~8h → ~30–45min/cell.
    - **Bug 3 — pad token = "!".** We hardcoded `pad_token_id=0`; Qwen's token 0 decodes to **"!"** (real
      pad is `<|endoftext|>`=151643) → trailing `!!!!` in responses. FIX: only fall back to 0 when the
      tokenizer has no declared pad. No-op for Llama-2 (its token 0 is `<unk>`).
    - **Bug 4 — bbh raw exact_match, no normalization.** `bbh_fewshot` (non-CoT) uses raw `exact_match` with
      no filter; Qwen emits a leading space (" -33" vs "-33") → even CORRECT answers scored 0 →
      **bbh=0.00 for Qwen**. FIX (`bbh_metric_fix.py`, applied idempotently at eval startup):
      `ignore_case: true` + `regexes_to_ignore: ["^\\s+","[\\s.]+$"]` (strip surrounding whitespace/trailing
      dot). **NOT** `ignore_punctuation` (that deletes minus signs and corrupts numeric answers). Measured:
      Qwen 0.00→0.54; **Llama-2 0.47→0.47 identical every subtask** (provable no-op).
    - **Cap decision:** finally chose **gen_cap=1024** (not 512/2048) — captures Qwen's longer genuine CoT
      without the ~2× slowdown of 2048; bbh needs <512 anyway (`h16`). Net eval config for both models:
      pad=declared-or-0 · gen_cap=1024 · max_len=4096.

14. **Divergence at extreme LR (2026-06-29).** At LR 2e-3 & 5e-3 ALL methods diverge to NaN weights → eval
    crashes (`CUDA device-side assert: probability tensor contains inf/nan`, corrupts CUDA ctx, job dies
    rc=1) → `gpu_pool` retries the deterministically-failing job forever (13× on one cell, ~26h wasted).
    FIX: **dropped 2e-3/5e-3** → 7-LR grid (2e-5…1e-3; collapse already captured at 1e-3). **Robustness gap
    remains: eval is not robust to NaN adapters** (`h13 §7`, `port-audit-and-lr-sweep-plan`).

15. **GPU-scheduler collisions (~45h lost).** TWO `gpu_pool.py` schedulers each grab all 8 GPUs → 2 runs/GPU
    → OOM + hung evals (orchestrator + a user-launched math pool collided). **RULE: only ONE GPU scheduler
    at a time.** Recovery: `make_salvage_evals.py` + `recover_and_resume.sh` (most collision compute was
    salvageable — training survived, only eval died) (`h13 §7`, `port-audit-and-lr-sweep-plan`).

16. **`pgrep -f` / `pkill -f` self-match.** The pattern matches your own command line — killed a shell twice
    via `pkill -f 'qwsw_batchtest'` / a plain `gpu_pool.py`. Always bracket: `[q]wsw`, `[g]pu_pool`, or kill
    by explicit PID (`h13 §7`, `h15`, `h16`).

17. **Math eval is heavy** — math-tuned models don't emit EOS → `ret_max_gen 512` crawls (~12h/job); use
    **256** for gsm8k retention (`h13 §7`, `port-audit-and-lr-sweep-plan`).

18. **`data_aware_init.py` injection bug** — toy reconstruction err 0.2 not ~0; flagged FIX-before-use
    (`h12 §NEW CODE`). Superseded by the audited `sclora_init.py`/`corda_init.py`/`lora_null_init.py` paths.

---

## (e) Findings — MATURE vs PROVISIONAL vs OPEN

### MATURE (Llama-2, seed 42 — the paper's evidence)

- **The magnitude law.** `r(retention, log‖ΔW‖_F) ≈ −0.87` pooled, **−0.93 on the 5 well-behaved methods**;
  slope ~ **−10pp/decade**. Within-method correlations −0.86 to −0.98 (LoRA −0.97, LoRA+wd −0.95, CLoRA
  −0.98, DoRA −0.86, CorDA −0.91, MiLoRA −0.96, SC-LoRA −0.88) (`h13 §2`, `summary.txt`).
  **Confirmed in BOTH domains:** math/gsm8k `r ≈ −0.93` (n=8) (`h13 §2`, `port-audit` results).
- **LR is a weaker proxy than magnitude.** retention~LR **R²=0.35** vs retention~`‖ΔW‖` **R²=0.75** — the
  mechanism is magnitude, and LR only matters because it sets magnitude (`h13 §2`).
- **LoRA+wd wins the Pareto plane.** wd bounds `‖ΔW‖_F` (~72 at lr1e-3 vs 200–1395 for others) → adapts
  ~81 CS **and** retains ~34. At each method's own best LR (fairness view): LoRA+wd **81.6 / 25.6**,
  SC-LoRA 80.1/22.5, MiLoRA 79.9/24.7, LoRA 79.1/24.4, CLoRA 78.4/21.9, DoRA 78.3/24.8, CorDA 77.9/19.9
  (CS-8 / ret-core) (`h13 §2`, `summary.txt` LR-fairness block). **Best LR is NOT the same across methods**
  ({5e-5, 1e-4, 2e-4, 3e-4, 5e-4}) — *this is the thesis*: a single fixed LR biases the comparison.
- **Budget:** `‖ΔW‖` buys adaptation (+21pp/decade) & costs retention (−16pp/decade); sweet spot
  `‖ΔW‖_F ≈ 0.31–0.62` (`h13 §2` fig8).
- **Per-benchmark:** MMLU forgets fastest (−23pp/decade); TruthfulQA ~immune (`h13 §2` fig5).
- **Mechanism (fig7):** data-aware inits transmit the same LR into a larger `‖ΔW‖` → why they fall off the
  frontier (`h13 §2`).
- **No independent rank effect** (LoRA r4→r256: ret 25.4→8.5, tracking `‖ΔW‖` up, `σ₁` grows — no diffusion)
  → the "rank surprisingly mitigates CF" surprise is DEAD; folds into the magnitude law (`h09` #5, `h12`).
- **Weight-SVD direction is causally irrelevant** (`μ_E` r=−0.09) (`h04`, `h06` D1, `h07 §7`).

### PROVISIONAL (real signal, needs a lock before publishing)

- **The ANCOVA off-curve nuance.** ANCOVA (p<0.001): the law is not *perfectly* method-free — **SC-LoRA
  (−3.3pp) and CorDA (−3.0pp) forget MORE than their `‖ΔW‖` budget predicts** (the data-aware inits); the
  other 5 straddle the curve → fair among them (`h13 §2`). **DO NOT publish this yet** — it is confounded by
  bugs 11 & 12 (CorDA wikitext taint re-running on nq_open; and the deeper calib↔eval mismatch for *both*).
  Needs **seeds 43/44 + a rank-matched control** to lock (`h13 §8.2`).
- **LoRA+wd "matches/beats" data-aware methods.** Full-scale clean: wd0.1 80.4/24.86 TIES CLoRA-k1024
  79.8/24.85; wd1.0 76.7/**26.87** beats CLoRA-k2048 65.4/25.7; DoRA-r8 79.8/25.38 also strong (`h12`
  ledger). The defensible claim is **"MATCHES"**, not "dominates"; frontier is noisy/non-monotone on single
  seed (wd0.3 anomalous) → **needs seeds** (`h11`, `h12`).
- **The universal collapse ("geometry irrelevant").** At matched `‖ΔW‖`, methods spread 3–5.6pp
  (LoRA+wd/UIO systematically ABOVE CLoRA) — **NOT a perfect collapse** → the strong "geometry irrelevant"
  claim is NOT supported; but the spread ≈ fast-eval noise so it's unresolved at full scale (`h11
  UNIVERSAL-CURVE`, `h12`). **Lead with "simple LoRA+wd ≥ elaborate adapters", not "geometry irrelevant".**
- **Directional norm `‖ΔW·C_retain^½‖` beats raw `‖ΔW‖_F`** — MARGINAL: −0.79 vs −0.77 (n=8), within noise
  (`h09` #2, `h12`). Was hoped to be the non-circular headline; currently too weak.
- **Qwen replication.** So far Qwen-CS `r(retention, log‖ΔW‖) = −0.92` (**the law replicates**); math sweep
  still filling in. As of `h13`, ~13 of ~112 Qwen cells done; ETA of full 2×2 seed-42 ~3–5 days
  (`h13 §2/§4`, `h15`). **Present as supporting / in-progress, not complete.**

### OPEN (unresolved questions)

1. **Finish the 2×2** (Qwen + math arms) — the main remaining work (`h13 §8.1`).
2. **Lock the off-curve claim** — seeds 43/44 on CorDA & SC-LoRA + rank-matched control (`h13 §8.2`).
3. **CorDA++** — replace vanilla CorDA with CorDA++ (dynamic covariance + dynamic rank under fixed budget);
   full plan + exact algorithms in `h14 §8` (compactness `π(C)=√(d_out·σ_max)/σ_min`; our budget =
   r16-equivalent 28,049,408, report *realized* param count). Execute AFTER the 2×2 (`h13 §8.2b`, `h14`).
4. **Fairness experiments** — give every method the wd knob; param-match ranks (currently mismatched:
   LoRA/DoRA/CorDA/LoRA-Null r16; MiLoRA/SC-LoRA r32; CLoRA k1024). DEFERRED; the law framing makes them
   optional but they are the honest prerequisite for any "LoRA is best method" claim (`h13 §8.3`,
   `port-audit` framing decision).
5. **Base-ceiling calibration** for MMLU/ARC/TruthfulQA (`h13 §8.4`).
6. **The calibration↔eval mismatch** (bug 12) — eval-matched calibration + sensitivity arm for all
   calibration-using arms (`h14 §8` Fix 1).
7. **SC-LoRA normalization** (|max Y| vs max|Y|) and **LoRA-Null `null_dim`=r** — low-risk fidelity flags to
   confirm vs raw repos if challenged (`h13 §8.5`, `port-audit`).
8. **The gated-magnitude adapter** (`h10`) — the *constructive* follow-up (input-conditional gate g(x)
   scaling only the adapter delta, dual loss with a preservation corpus DISJOINT from the eval set; oracle =
   (CS~79, ret~26) dominates the whole frontier). DEPRIORITIZED after LoRA+wd (the gate must now beat wd-LoRA,
   a higher bar) — future work, not the lead (`h10`, `h11` relation note).

### DEAD ends (kept for provenance)

UIOrthoLoRA/UILinLoRA as a method (dominated, ceiling ~74); the "optimal leakage budget" thesis (overturned
— magnitude, not direction, is the budget); the "rank mitigates CF" surprise (folds into magnitude);
the leakage-thermometer paper as headline (demoted to reference/diagnostic).

---

## (f) Decisions & rationale

- **Why the LAW framing (not "LoRA is best").** Framing around *retention = f(`‖ΔW‖`)* sidesteps the
  method-ranking fairness gaps (asymmetric wd knob — only LoRA has it; secondary knobs fixed; param
  mismatch). A "LoRA is best" headline would require the deferred fairness experiments; the law is
  defensible with what we have (`h13 §1`, `port-audit` PAPER FRAMING DECISION).
- **Why single-seed-first.** Full breadth (2 models × 2 domains × 8 arms × 9→7 LRs = ~288 runs seed-42) in
  ~9 days beats a narrow 3-seed slice, because the box may be reclaimed; add seeds 43/44 later **only where
  the result is interesting** (revert `SEEDS` in `make_campaign_jobs.py` → orchestrator regenerates missing
  cells). The **3-seed matrix taught us seeds matter** (single-seed "collapse basins": s44 collapsed
  clora_k2048→23, dora_r8→22, lorawd_wd0p5→51 — edge regimes) so seeds are on the roadmap, just deferred
  (`port-audit` DECISION, `matrix-campaign-results`). Single-seed is fine for **the law**; the **off-curve
  verdict** needs seeds.
- **Why `‖ΔW‖_F` not `dw_sv_max`.** σmax is confounded — CorDA's spiky spectrum inflates it and mis-ranks
  it; token-weighted Frobenius is the clean axis (`h13 §2`, `port-audit`). *(This is a genuine methodology
  change across the project's life — see inconsistency #1.)*
- **Why gen_cap=1024** (not 512 or 2048). 1024 captures Qwen's genuine longer CoT (mmlu_pro 43.6→47.0
  from 512→1024, then diminishing) without the ~2× slowdown of 2048; bbh needs <512 anyway. Consistent
  across all cells for comparability (`h16`).
- **Why max_len=4096 for both models.** Fixes Qwen's batch=1 problem AND matches Llama-2's window → better
  cross-model comparability (`h15`).
- **Why drop 2e-3/5e-3 LRs.** They diverge to NaN → eval crash → deterministic pool retry-storm; collapse is
  already captured at 1e-3 (`h13 §7`, `port-audit`).
- **Why LR-sweep-per-method is the instrument.** A single fixed LR flatters whichever method happens to be
  well-tuned there; each method's best LR differs ({5e-5…5e-4}), so a per-method sweep is the *only* fair
  comparison — and it is exactly what collapses the fancy adapters onto the magnitude curve (the artifact
  diagnosis) (`summary.txt` LR-fairness, `h13 §1`).
- **Why in-process eval / residual-save** — forced by the two round-trip bug classes (§d bugs 1, 9); both
  have 0-step self-checks (`h00` #1, `peft-residual-init-save-bug`).
- **Why the no-strawman discipline.** "We run other researchers' adapters, so prove the port is faithful
  before claiming a method fails." This is what caught the CorDA wikitext bug (a deterministic bug
  reproduces across seeds, so reproduction ≠ correctness) (`port-audit`, `matrix-campaign-results`).
- **Why provenance tags in related work.** `[MECH]/[OURS]/[VERIFY]/[INTERP]` prevent an autonomous agent
  from upgrading a `[VERIFY]` citation or `[INTERP]` hypothesis to asserted fact; all arXiv IDs are unverified
  (`h07 §0, §9`).

---

## Appendix — key numbers (Llama-2, seed 42, full-scale unless noted)

- BASE (no-FT) retention = **26.0** (BBH-AO 33.1 + MMLU-Pro 19.0) = ceiling.
- Magnitude law: pooled r≈−0.87 (R²=0.75); −0.93 on 5 well-behaved methods; math r≈−0.93.
- LR proxy: retention~LR R²=0.35 « retention~`‖ΔW‖` R²=0.75.
- Best-LR-per-method (CS-8 / ret-core / `‖ΔW‖_F`): LoRA+wd **81.6/25.6/0.39**, SC-LoRA 80.1/22.5/0.56,
  MiLoRA 79.9/24.7/0.54, LoRA 79.1/24.4/0.62, CLoRA 78.4/21.9/0.64, DoRA 78.3/24.8/0.45,
  CorDA 77.9/19.9/0.42 *(CorDA PENDING nq_open re-run)*.
- Full-scale clean wd: wd0.1 80.4/24.86 (TIES CLoRA-k1024 79.8/24.85); wd1.0 76.7/26.87.
- Qwen (in progress): Qwen-CS r(retention,log`‖ΔW‖`) = **−0.92** (law replicates); ~13/112 cells done.

*Numbers drawn from `h13 §2`, `paper/summary.txt`, `paper/table_main_*.txt`, `h12` ledger. Where the tables
show `dw_sv_max`/σmax columns, prefer the `‖ΔW‖_F` figures per the axis decision.*
