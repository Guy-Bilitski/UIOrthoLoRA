# 05 — ADAPTER DYNAMICS: who affects what (geometry × magnitude × CE drift)

`[2026-07-17. Sources: merged run-level dataset built from results/*/summary.json + geo.json +
forgetting.json (1,004 usable runs; 1,002 with geometry; 857 with CE/KL; quarantine excluded).
Raw outputs: dyn1_structure.txt, dyn2_mediation.txt, dyn3_exchange.txt in this directory.
KL = forgetting_ce − base_entropy verified exactly (median |err| = 0.0000).]`

## The causal picture in one paragraph

Training pressure (LR × steps × wd × k × rank) sets the **magnitude** of the update (F_Δ);
placement (**geometry**) sets two *exchange rates* — how much adaptation and how much behavioral
drift a unit of magnitude buys; magnitude then drives **drift** of the model's next-token
behavior on neutral text (KL to base), and drift is what benchmark forgetting mostly *is* —
with one important exception: MMLU-Pro-type damage (generative format-following) has a direct
magnitude component that neutral-text drift cannot see. Adaptation rides on the *trained
direction* (a random direction at matched magnitude adapts ≈ 0); forgetting rides on the
*dose*. Adaptation saturates early (F_Δ ≈ 0.3–0.4) while the dose keeps costing — which is the
entire practical story: stop the magnitude where the benefit stops.

## 1. The three blocks are tightly coupled but not one thing

Pairwise correlations per family (dyn1 §1): log F_Δ ↔ log‖ΔW‖_F r = +0.85…+0.99;
log F_Δ ↔ KL r = +0.76…+0.93; **KL ↔ retention r = −0.89…−0.94 — uniformly stronger than
F_Δ ↔ retention (−0.71…−0.92)**.

R² for retention (dyn2 §10): KL alone beats log F_Δ alone in 5/6 families
(0.79–0.89 vs 0.57–0.85); both together reach **0.86–0.90** everywhere. Weight-space dose and
behavior-space drift are complementary measurements of the same damage process, each carrying
some signal the other misses.

## 2. Mediation: drift is the proximal damage variable — for one of the two channels

Family-level mediation (dyn1 §2): controlling KL, the direct magnitude→retention path collapses
in lrswm (−0.14), qwsw (−0.08), qwswm (+0.05) — in those arms, *once you know how far the
model's behavior drifted, the update size adds nothing*. But it survives in lrsw (−0.63) and
frc (−0.68).

The component split explains why (dyn2 §8): the surviving direct path is almost entirely
**MMLU-Pro** (direct −0.67…−0.71 in lrsw/frc/frm) while **BBH's direct path is ≈ 0 in the math
and Qwen families** (+0.07…+0.12). So forgetting has two channels:

- **Channel A — capability drift.** Tracked by KL on neutral text; hits BBH-style scoring;
  fully mediates magnitude in the math/Qwen arms.
- **Channel B — format / instruction-following damage.** Grows with magnitude but is invisible
  in next-token drift on WikiText; hits MMLU-Pro's generative parser; the same channel shows up
  on the adaptation side as the seed-level answer-format collapse basins (accuracy dies,
  retention intact).

Honest micro-scale caveat (dyn2 §7): within a fixed recipe, seed-level fluctuations are better
predicted by F_Δ (partial −0.51) than by KL (partial −0.10) — KL from 40 WikiText blocks is a
noisy readout at small deltas. F_Δ is the better *monitoring* variable; KL is the better
*damage* variable at operating-point scale.

## 3. What geometry actually does: sets the exchange rates, not the outcome

- **Drift per unit magnitude** (dyn2 §9, method residuals of KL ~ logF): PiSSA drifts far more
  than its size predicts (+0.61 in frc — principal output directions are where the base
  function lives); SC-LoRA (+0.18…+0.22) and CLoRA (+0.12…+0.15) drift slightly more per unit
  in the LR sweeps; plain LoRA and DoRA drift less (−0.12…−0.15). Spread updates drift more per
  unit in the math sweep (partial r(KL, stable_rank | F) = +0.48 in lrswm).
- **Adaptation per unit magnitude** (dyn3 §11 + A4): the biggest genuine method difference
  (spread 5–16 pp at matched magnitude). SC-LoRA converts magnitude to task accuracy best on
  the math arms (+5…+8 pp — its task-aligned init does what it advertises); LoRA+wd is high on
  the Llama CS sweep; dense full-FT is the extreme case (80.5 accuracy at F_Δ 0.08). Read these
  residuals qualitatively — adaptation is saturating, not linear in log F.
- **Alignment** — functional magnitude per unit weight norm (dyn1 §3) — is set by
  *concentration*, not by subspace choice: r(F_Δ/‖ΔW‖, stable rank) = −0.42, vs ≈ 0 for
  e_top/e_bot. Concentrated updates deliver more functional change per parameter norm.
- **At matched magnitude, retention offsets are bounded** (dyn1 §4 + §18.4): e_top costs
  retention (−0.15…−0.41 partial — touching principal output directions is bad, PiSSA's
  mechanism); concentration helps in the Llama sweeps (partial r(ret, log spec_max | F)
  +0.40…+0.59); everything is ≤ a few pp against a 15–30 pp magnitude range.
- **On the KL axis the same picture repeats** (dyn2 §10): methods collapse onto ret ~ KL with
  small offsets; SC-LoRA(nq_open) sits −1.5…−3.0 below even here — the calibration artifact is
  visible in both spaces, i.e. it is real behavioral damage, produced by where the calibration
  set pointed the update, not a benchmark quirk.

## 4. Direction adapts, magnitude forgets

E1 (interventional): random directions at matched F_Δ lose only −3.05 pp retention vs trained
directions — but gain **zero adaptation** (0.5–7.0 vs 13–80). r(adapt, logF) is inconsistent
across families (−0.85…+0.37) while method identity dominates the adaptation ANCOVA. The
useful information in an adapter is its direction; the damage is (mostly) its dose.

## 5. The saturation exhibit (the whole practical story in 5 rows)

Healthy Llama-CS runs binned by F_Δ (dyn3 §12):

| F_Δ | n | adaptation | retention |
|---|---|---|---|
| 0.10–0.25 | 8 | 72.8 | 26.70 |
| 0.25–0.40 | 19 | 79.0 | 26.17 |
| 0.40–0.60 | 19 | 79.7 | 24.87 |
| 0.60–0.90 | 24 | 78.6 | 21.74 |
| 0.90–1.50 | 17 | 75.0 | 17.46 |

Adaptation saturates by F_Δ ≈ 0.3–0.4; retention keeps falling ≈ monotonically. Every
magnitude-control result in the study (wd, k, rescaling, LoRA+wd's operating point at
F_Δ ≈ 0.38) is this table: **the dose keeps costing after the benefit stops, so cap the dose at
the saturation point.** Post-hoc rescaling works because the direction already contains the
adaptation; shrinking the dose keeps most of it (E1: 75.4 adapt at F_Δ 0.44) while recovering
retention.

## 6. What this changes about the story

- The "one curve" (retention vs F_Δ) is a *projection* of a two-channel process: it works
  because F_Δ is the common dose behind both channels; KL sharpens channel A; nothing measured
  on neutral text sees channel B — the paper should say so.
- Geometry is not "inert" and not "protection": it is the pair of exchange rates. The right
  method question is not "which init forgets least" (dose-controlled, they are within a few pp)
  but "which init buys the most adaptation per unit of dose" — where SC-LoRA-style task-aligned
  calibration genuinely shines and PiSSA is genuinely costly.
- The practical recipe drops out mechanically: pick any method whose direction adapts your
  task, train it, and control the dose (wd during training, or rescale after) to the
  saturation knee; monitor F_Δ, not LR.
