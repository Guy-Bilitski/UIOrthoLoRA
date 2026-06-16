# 10 — GATED-MAGNITUDE ADAPTER (input-conditional adapter scaling) — PLAN + REVIEW

Status: PROPOSAL (2026-06-16). The constructive "what do we DO about it" follow-up to the magnitude
finding. This is the ONLY mechanism in the project that can BREAK the adaptation–retention frontier
(DoRA / weight-decay / rank apply a fixed ΔW to every input ⇒ they only trace it).

## 1. Motivation (derived directly from our finding)
Established (controlled): retention is governed by update MAGNITUDE applied to inputs; weight-basis
DIRECTION is ~irrelevant (μ_E r≈−0.09). Forgetting on an out-domain input x is ‖ΔW·x‖. Fixed adapters
apply ΔW to ALL inputs, so they forget on out-domain x even though the adaptation is only needed on
in-domain x. **Implication:** don't shrink the update globally — APPLY it only where needed.

## 2. Mechanism
Input-conditional magnitude gate g(x)∈[0,1] scaling ONLY the adapter delta (not the base):
    out = base(x) + g(x) · (scaling · B A x)         # g=0 ⇒ out = base(x) exactly
g(x) = σ(MLP(summary(x))) — a tiny per-layer (or shared) gate on the layer input activation.
This is **input-conditional magnitude m(x)** — the natural generalization of DoRA's CONSTANT m.

### Oracle upper bound (≈ free to compute, do FIRST)
Perfect gate: g=1 on in-domain, g=0 on out-domain ⇒ CS = full-LoRA CS (~78), retention = base (~26).
That point ≈ **(CS 78, ret 26)** STRICTLY DOMINATES the whole measured frontier (CLoRA 80/25.6, etc.).
⇒ the ceiling is real; the entire question is **how close a LEARNED gate gets to it.**

## 3. THE core challenge (user's concern — correct) + the fix
Trained only on the downstream task, the gate sees only in-domain inputs ⇒ learns "always on" ⇒ no
muting ⇒ still forgets. FIX = DUAL objective:
  L = L_task(in-domain; gate free to fire)  +  β · L_preserve(general corpus; push g→0 / match base)
L_preserve options: (a) ‖g(x)·ΔW·x‖→0, (b) KL(model_gated(x) ‖ model_base(x))→0, (c) explicit g(x)→0.
The preservation term teaches the gate the "OFF" region it never sees from the task data.

## 4. ⚠️ THE MAKE-OR-BREAK CONTROL (must get right or it's worthless / cheating)
**Preservation corpus MUST be DISJOINT from the retention eval set.** If we push g→0 on MMLU-Pro and then
report retention on MMLU-Pro, that's training on the test distribution. ⇒ train L_preserve on a GENERIC
corpus (C4/wiki/general instructions), and EVALUATE muting-generalization on HELD-OUT BBH + MMLU-Pro the
gate never saw. The result is only real if the gate mutes on genuinely unseen out-domain. This is the
single point a reviewer will attack; design around it from day 1.

## 5. Design knobs (to sweep / decide)
- Granularity: per-token (works causally at gen time; noisier) vs per-prompt (cleaner but no full-prompt
  pooling during generation) vs per-layer gate vs single shared gate. DEFAULT: per-token, per-layer-shared
  small gate on the hidden state.
- Gate input: layer input activation (what ΔW acts on) — most direct.
- β (dual-loss balance): too small→always-on (forgets); too large→always-off (CS→base). Sweep.
- Preservation corpus: generic general text / instructions, ≠ eval benchmarks (see §4).
- Gate arch: σ(linear(mean-pooled hidden)) → scalar; keep tiny (few-K params).

## 6. Experiment plan
1. **ORACLE (free, FIRST):** eval a trained LoRA with adapter ON (CS) and OFF (retention=base) → plot the
   (78,26) oracle point; confirms frontier-break is achievable with perfect gating.
2. **Minimal learned gate:** gated-LoRA + dual loss, generic preservation corpus. Train, then place on
   the CS-vs-retention frontier vs LoRA / CLoRA / DoRA.
3. **Muting diagnostic:** log g(x) distribution on in-domain (commonsense) vs HELD-OUT out-domain
   (BBH/MMLU). Does it bimodally separate (fire vs mute)? This is the mechanism evidence.
4. **β sweep + ablations:** no-preserve (→always-on control), preserve-only (→always-off control),
   per-token vs per-prompt, shared vs per-layer gate.
5. **vs oracle:** how close does the learned gate get to (78,26)?

## 7. Pre-registration (predicted + meaning of BOTH branches)
- PREDICT: learned gate lands ABOVE the LoRA/CLoRA frontier (higher CS at matched retention), approaching
  the oracle — IF muting generalizes to held-out out-domain.
- If CONFIRMED (gate mutes on unseen BBH/MMLU, frontier broken): the constructive headline — "magnitude is
  governed per-input; gate it and you get adaptation without the forgetting tax." THE method contribution.
- If REFUTED (gate doesn't generalize muting / β forces a normal tradeoff): negative but informative —
  "the in/out boundary isn't learnable from a generic corpus" → bounds the approach; still a finding.

## 8. Failure modes / kill criteria
- Gate collapse (always-on or always-off) not fixable by β → kill.
- Muting does NOT generalize to held-out out-domain (only mutes on its training corpus) → core idea fails.
- Frontier NOT broken (gate just trades like everything else) → no method contribution; wrap.
- Joint adapter+gate instability → may need 2-stage (train adapter, then gate).

## 9. Novelty / related work (HONEST — lit-check required before claiming)
Architecturally this is a 2-expert gate (base, adapter) = a mixture-of-LoRA / conditional-adapter, which
EXISTS (MoLoRA / MoELoRA, conditional/dynamic adapters, continual-learning gating/masking). NOVELTY rests
on: (a) the forgetting-specific MOTIVATION derived from the magnitude law, (b) the preservation-loss
training that makes muting GENERALIZE to unseen out-domain, (c) the empirical frontier-break + the g(x)
in/out separation as mechanism. MUST search: "gated/conditional/dynamic LoRA forgetting", "input-dependent
LoRA scaling", MoELoRA-for-CF. If the exact method exists, contribution narrows to the analysis.

## 9b. REVIEW ADDITIONS (clear-mind pass, 2026-06-16) — things the first draft missed
- **Honest target reframe:** oracle = (CS 79.1, ret 26.0) = LoRA's in-domain CS at BASE retention.
  CS is BOUNDED by the underlying adapter (~79), so we DON'T exceed CLoRA's CS (79.9); the win is the
  RETENTION corner — full adaptation with ~zero forgetting. Success = APPROACH (79,26), not beat CS.
- **FIRING generalization (dual to muting):** gate must also fire on in-domain at test (preserve CS), and
  may OVER-mute on real downstream inputs that differ from the training format — note as a limitation.
- **Preservation-loss cost:** KL-to-base needs a 2nd (base) forward = 2× cost; ‖g·ΔW·x‖→0 on the general
  corpus is cheaper and directly pushes g→0. Default to ‖g·ΔW·x‖→0; KL as a variant.
- **Preservation corpus must be DIVERSE** (multi-format general text/instructions) so muting generalizes
  to BBH (format very different from commonsense), not just to its own training format.
- **Generation-time gating:** per-token gate on the current hidden state; prompt tokens (clearly
  out-domain) → mute. Check consistency across the generated sequence.
- **2-STAGE training as the MINIMAL FIRST EXPERIMENT + instability fallback:** freeze an ALREADY-trained
  LoRA, train ONLY the gate on top (task + preservation). Far fewer moving parts than joint training,
  directly answers "can a gate learn to mute on held-out out-domain?", and avoids adapter↔gate instability.
  Do this BEFORE joint training.
- **Mechanism evidence is mandatory:** report the g(x) distribution — high on commonsense, low on
  held-out BBH/MMLU. High retention WITHOUT demonstrated muting separation is not evidence.

## 10. Connection to the thesis (the paper arc, if it works)
finding (magnitude governs forgetting, direction doesn't) → implication (the cost is applying magnitude
off-task) → method (gate magnitude per-input, learn the off-region via a preservation loss) → result
(breaks the frontier, approaches the oracle). Clean finding→method arc. This is the upgrade from
"measurement paper" to "measurement + method".
