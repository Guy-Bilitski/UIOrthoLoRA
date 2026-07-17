# Thesis-Validation Plan — "Is LoRA+wd really doing the thing, at matched capacity?"

*Drafted 2026-07-04. A pre-registered, falsifiable protocol to test whether the magnitude thesis
holds under the two conditions a skeptic (and CLoRA's Table 2) will demand: **matched capacity** and
**faithful competitor ports**. Written so the falsification criterion is fixed before any run.*

---

## 1. The claim, stated so it can be killed

**Thesis (operational form).** At **matched trainable-parameter capacity**, under a **fair per-method
learning-rate sweep**, retention is governed by the update magnitude ‖ΔW‖ alone. Therefore plain
**LoRA + weight-decay can reach the same (adaptation, retention) operating point as every elaborate
geometric adapter — including a faithfully reproduced CLoRA** — because the geometry contributes
nothing beyond its effect on ‖ΔW‖.

**Pre-registered falsification criterion (fix this before running).** The thesis is **FALSE** if, at
the same rank and with ≥3 seeds, some geometric adapter attains an (adaptation, retention) point that
LoRA+wd **cannot reach at any (wd, LR)** — specifically, a point lying **> 2σ_seed above the LoRA+wd
achievable frontier** (equivalently: faithful CLoRA sits > 2σ_seed **above** the pooled
retention-vs-‖ΔW‖ curve). If CLoRA/DoRA/etc. all fall **on or below** the LoRA+wd frontier within the
seed band, the thesis is **supported**.

This is a two-sided test: it can confirm the paper or sink it. That is the point.

---

## 2. Design principles (each closes a specific objection raised so far)

1. **Matched capacity across a rank ladder (the headline requirement).** Compare methods **at equal
   rank**, and sweep a **rank ladder r ∈ {16, 32, 64, 128}** with **CLoRA and LoRA+wd both present at
   every rung** (not LoRA+wd-only). This is the disciplined form of the "just raise the rank" idea:
   raising LoRA+wd's rank only counts if CLoRA gets the same rank — LoRA+wd-r128 matching CLoRA-r32 would
   mean CLoRA is *more parameter-efficient* (geometry helping), not a win. The question the ladder answers
   is whether the LoRA+wd-vs-CLoRA gap **at matched rank** shrinks, persists, or widens with rank.
   CLoRA's `k` is a *frozen* projection (zero trainable params), so CLoRA at rank r = LoRA+wd at rank r in
   capacity. DoRA's magnitude vector adds a negligible per-column term (report it). **Match scaling α**:
   residual-init methods are forced to α=r; run LoRA/LoRA+wd/CLoRA/DoRA at both α=r and α=2r (or fix α=r
   everywhere) so α does not masquerade as a method effect.
   *Caveat tying this to our own law:* more rank at fixed wd raises ‖ΔW‖ → lowers retention (moving along
   the curve, not a free gain). So the rank ladder must be a **rank × wd grid** — raise rank AND raise wd
   to hold ‖ΔW‖ bounded — and the real test is whether extra rank buys adaptation *at the same ‖ΔW‖*. If
   our "no independent rank effect" finding holds, it will not; if higher-rank LoRA+wd shows a genuine
   both-axes gain, that is itself a finding (and a partial qualification of the law).
2. **Faithful ports before any negative claim (no-strawman).** For each geometric method, first
   reproduce **its own published number at its own config** in our harness. **CLoRA is the gate.**
   *Update 2026-07-05:* a line-for-line audit vs CLoRA's official `clora.py` + paper confirmed our
   **regularizer code is faithful** (random orthonormal P in both; matched penalty/λ/rank/α/targets/LR;
   their own SVD-informed variants don't beat random). So this is **not a code bug** — but our CLoRA is
   **under-tuned**: it scores 78.4, *below* our own faithful LoRA (79.1), whereas a faithful CLoRA
   *beats* LoRA by ~+2.7 (their Table 2), i.e. it should reach ≈82 in our harness. The k2048 collapse is
   an optimization-scale effect (un-normalized penalty:LM-loss ratio grows ~linearly in k; a low-k-optimal
   LR reused at high k; `--train_on_inputs`), not confinement. **Reproduction target: tune CLoRA
   (per-k LR, verify penalty:LM ratio, match `train_on_inputs`) until it reproduces its ≈+2.7 edge over
   LoRA (≈82) in our harness.** Until then our CLoRA number is not a valid comparison target — it settles
   nothing. If a method cannot be brought to its published relative advantage, that is flagged, not
   claimed as a loss.
3. **Fair instrument: per-method LR sweep.** Same 7-LR grid for every method; compare each at *its own*
   best LR. (This is what a single-LR table like CLoRA's cannot do.)
4. **Trace the LoRA+wd frontier, not one point.** Sweep **wd × LR** so we have LoRA+wd's *achievable*
   (adaptation, retention) frontier — the object the falsification test compares against.
5. **Both axes + magnitude, in one harness, with seeds.** Report (adaptation, retention_core, ‖ΔW‖_F,
   and the data-basis magnitude ‖ΔW·C½‖) for every cell; ≥3 seeds on the headline cells so a "2-point
   gap" is judged against noise, not asserted.

---

## 3. Experimental arms — the capacity-matched core

Fix **r\* = 32** (CLoRA's rank; our LoRA+wd is already r32). Base Llama-2-7B, commonsense-170k, one
shared trainer + eval, LR grid {2e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 1e-3}, seeds {42, 43, 44}.

| Arm (all r32 unless noted) | Role | Pre-req control |
|---|---|---|
| **LoRA r32** | capacity-matched plain baseline | — |
| **LoRA+wd r32**, wd ∈ {0.1, 0.3, 1.0} | protagonist; traces the frontier | run at α=r and α=2r |
| **CLoRA r32** (k512/k1024/k2048) | the decisive competitor | **fix port → reproduce 82.6/83.7 first** |
| **DoRA r32** | geometric | **corrected fdelta** (magnitude-vector term) |
| **MiLoRA r32** | weight-SVD init | — |
| **SC-LoRA r32** | data-aware init | **eval-matched calibration (B4)** |
| **CorDA r32** | data-aware init | nq_open + **eval-matched calibration (B4)** |
| **LoRA-Null r32** | data-aware init | — |
| *(optional) MiLoRA/DoRA/CorDA + wd* | "is wd the active ingredient?" | B5b-style |

**Rank ladder (core, not optional).** Run the head-to-head pair — **LoRA+wd and CLoRA** — at **r ∈ {16,
32, 64, 128}**, each as a **rank × wd (× k for CLoRA) grid**, so we can (a) compare at every matched rank
and (b) test whether extra rank buys adaptation at fixed ‖ΔW‖. The r=32 rung is the primary comparison
(CLoRA's reported rank); the others test capacity-sensitivity of the gap. This is the disciplined version
of "increase the rank."

Everything is compared **at equal rank = equal capacity**. Raising LoRA+wd's rank above CLoRA's is not a
valid match.

Repeat the **CS** protocol for **math (GSM8K / MetaMathQA)** on the reduced arm set that matters
(LoRA, LoRA+wd, CLoRA, DoRA) once CS is settled.

---

## 4. The four decisive tests

1. **Reproduction gate — establish CLoRA's operating point as the honest target.** CLoRA's published
   numbers are taken as faithful (its regularizer code matches ours; audit 2026-07-05). The goal is to
   reproduce CLoRA's faithful performance *in our harness* so LoRA+wd is chased against a real, correctly
   tuned CLoRA — not to interrogate CLoRA's numbers. If our reproduction lands below CLoRA's reported
   operating point, that is **our** tuning/harness (per-k LR, penalty:LM ratio, `train_on_inputs`), and we
   keep tuning until CLoRA reaches its faithful performance. That reproduced CLoRA point is the bar
   LoRA+wd must reach. (Note our retention is measured on our own axis, base BBH-AO 33.10; CLoRA's base is
   34.91 — a harness-config difference in *our* setup, not a property of CLoRA.)
2. **Frontier-domination test (the head-to-head the user wants).** Overlay every method's best-per-LR
   (adaptation, retention) point on LoRA+wd's wd×LR frontier, at matched r32. **Does LoRA+wd's frontier
   reach/cover CLoRA's (and every method's) joint point?** Judge gaps against 2σ_seed.
3. **Single-curve test.** Plot retention vs ‖ΔW‖_F for all r32 arms. Do they interleave on one curve?
   **Where does faithful CLoRA fall** — on it (thesis holds) or > 2σ above (geometry wins)?
4. **Basis fallback.** If CLoRA sits above the *scalar* ‖ΔW‖ curve, re-plot against the **data-basis**
   magnitude ‖ΔW·C½‖ (`forensics_databasis.py`, cov = retain set). If it falls **on** that curve, the
   honest conclusion becomes "*it's the magnitude in the right basis*" — still a magnitude law, not
   geometry per se. If it's off *both*, geometry genuinely helps.

---

## 5. Confounds to close first (else the test is not clean)

- **CLoRA tuning** (Step-2 gate) — the regularizer code is faithful (audited 2026-07-05); the fix is
  *tuning/config*, not code: per-k LR, confirm the penalty:LM-loss ratio isn't dominating at high k, and
  match `--train_on_inputs`. Target: CLoRA reproduces its ≈+2.7 edge over LoRA (≈82) in our harness
  before it enters any comparison.
- **DoRA magnitude in fdelta** — ‖ΔW‖ currently omits DoRA's magnitude vector (lower-bounds its x).
  Fix the measurement or exclude DoRA from the curve test.
- **Calibration mismatch (B4)** — SC-LoRA/CorDA/LoRA-Null on eval-matched calibration + an nq_open
  sensitivity arm; otherwise their "off-curve" is a calibration artifact, not geometry.
- **Scaling α** — matched as in §2.1.
- **Base-ceiling calibration** — measure *our* base model's BBH/MMLU-Pro under the same few-shot/format
  as the fine-tuned models, so our own retention axis has a correct ceiling and every method's numbers are
  interpretable in our units. (This is about calibrating our axis, not about CLoRA.)

---

## 6. Statistics

- ≥3 seeds {42,43,44} on all headline cells → report mean ± σ; σ_seed defines the 2σ falsification band.
- The comparison metric is the **joint** (adaptation, retention) point — never adaptation alone (a
  high-LR LoRA hits high adaptation with collapsed retention; that is not a match).
- Report ‖ΔW‖_F and ‖ΔW·C½‖ at every cell.

---

## 7. Cost (order-of-magnitude, single 8×B200 node, ~1.5–2.5 h/CS cell)

- Core matched sweep (r32): 8 arms × 7 LR × 3 seeds ≈ **168 CS cells**.
- LoRA+wd wd-frontier: 3 wd × 7 LR × 3 seeds ≈ **63 cells** (overlaps the core).
- CLoRA reproduction/tuning to its faithful ~82: **~10–20 cells**.
- **Rank ladder (core): LoRA+wd & CLoRA at r∈{16,64,128} × wd/k × 3 seeds ≈ 150–250 cells.**
- α-match controls + optional "+wd on fancy arms": **~60–120 cells**.
- Math subset: **~40–60 cells**.

Total ≈ **500–700 cells ≈ 5–8 GPU-days** on 8 GPUs (the rank ladder roughly doubles the earlier estimate —
this is the cost of answering the head-to-head *properly*, and it is why the existing `lrsw_` weeks are not
sufficient for it). The **load-bearing subset** — tune CLoRA to its faithful ~82 + LoRA+wd/CLoRA at r32
with the wd/k grid + 3 seeds ≈ 60–90 cells, ~1 GPU-day — answers the primary r32 comparison first; run it
before the ladder.

---

## 8. Decision tree — what each outcome means for the paper

- **LoRA+wd frontier reaches CLoRA's point; all r32 methods on one ‖ΔW‖ curve within 2σ** →
  thesis **confirmed at matched capacity**. Strongest paper: "the trivial knob reaches the elaborate
  method's operating point; geometry unnecessary." Ship the constructive framing.
- **CLoRA above the scalar curve but on the data-basis curve** → pivot to "**magnitude in the right
  basis**." Still a magnitude law and still novel; drop "geometry inert," keep the instrument + law.
- **CLoRA above both curves, LoRA+wd cannot reach its joint point at any (wd,LR)** → thesis **falsified**;
  CLoRA's geometry genuinely buys a real gain that a scalar magnitude knob cannot match. Report it
  honestly — CLoRA's numbers are faithful and it wins; per the user's framing, in that case the paper
  does not stand and we say so. The fair-measurement methodology would remain a valid smaller contribution.

---

## 9. Reuse (no new machinery)

Extend `make_campaign_jobs.py` (add r32 variants of every arm, the wd×LR grid, seeds 43/44); the
B4/B5a/B5c infrastructure in `07_experiment_plan.md` already covers calibration + param-match + seeds —
this plan **subsumes** them and adds the **CLoRA faithful-reproduction gate** and the **capacity-matched
frontier test**. Single 8-GPU scheduler; run the load-bearing subset (§7) first.
