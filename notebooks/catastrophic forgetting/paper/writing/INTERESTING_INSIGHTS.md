# Interesting empirical insights (appendix-worthy)

Living doc of concrete, defensible findings surfaced during the faithful-reproduction campaign.
Each is stated with evidence + why it matters for the paper. Add as they land.

---

## 1. DoRA costs ~2× the training wall-clock of LoRA — for no retention/adapt benefit
**Finding.** At identical config (LLaMA-2-7B, r=64/α=128, MetaMathQA-395K, batch 16, 3 epochs =
74,064 optimizer steps), DoRA trains at **~2.0 steps/s vs LoRA's ~4.3 steps/s — a ~2.1× slowdown**
(DoRA cell: 68,354 steps in 9h22m; LoRA cell: 74,064 steps in ~4.8h). DoRA also runs at higher
peak memory. The cost comes from DoRA's weight decomposition — recomputing per-column magnitude
norms + direction renormalization every forward/backward.
**Why it matters.** DoRA's extra compute buys nothing on our axes: it sits on the *same*
retention-vs-‖ΔW‖ curve as LoRA, and at high LR its ‖ΔW‖ blows up (e.g. Llama-math lr1e-3:
DoRA ‖ΔW‖=2.19 vs LoRA 1.28 → DoRA retention 17.9 vs LoRA 19.5, and its adapt *collapses*,
gsm8k 31.9). So DoRA is a **2× training tax with no accuracy or retention payoff** — a clean
practical argument that geometric machinery isn't free and isn't earning its keep. (An honest
efficiency table — steps/s, peak mem, $/run — makes this concrete for practitioners.)

---

## 2. How you *evaluate* a MetaMath-tuned model changes GSM8K by ~20 points
**Finding.** The same trained adapter scores **46.55%** under lm-eval's default GSM8K (5-shot,
"Question:/Answer:" template, strict `#### x` match) but **60–66%** under the faithful protocol
(0-shot, the Alpaca instruction template the model was trained on, last-number extraction) —
a **~+19.5 pp** swing from the eval harness alone.
**Why it matters.** A large fraction of apparent method differences in this literature can be
eval-protocol artifacts. Reporting the train/eval-template match (and both numbers) is a
measurement-hygiene contribution in its own right; it's also why our LoRA reproduces CLoRA's
published 60.58 (we get 60.2) once the protocol is aligned.

---

## 3. The magnitude law spans the faithful reproduction — but is NOT resolvable within the competitor blob
**Finding (corrected 2026-07-06, consortium synthesis).** Across the 7 faithful math cells,
retention vs log‖ΔW‖_F gives r = −0.93 — but this is carried by two leverage points (collapsed
PiSSA at ‖ΔW‖=2.21→ret 3.6 and off-LR LoRA+wd0 at 0.43→22.6). Among the 5 same-LR competitors
(LoRA, MiLoRA, CLoRA×3; ‖ΔW‖∈[1.02,1.28]) the correlation is **r = +0.06 (flat)**, and MiLoRA
retains best (+1.8pp over LoRA) at near-highest ‖ΔW‖.
**Why it matters.** The law is visible across the *range* of update magnitudes but cannot be
claimed *within* the narrow competitor band on these 7 points. Per the consortium ruling (handoff/21),
Claim 1 rests on the n=49 CS sweep; the faithful math table is a *reproduction/dissolution*
exhibit, not a law exhibit. Do NOT cite r=−0.93 as an in-reproduction law without this caveat.

---

## 4. PiSSA's catastrophic forgetting is REAL — gate verdict 2026-07-08
**Finding.** PiSSA (principal/top-r SVD init) has the **largest ‖ΔW‖ (2.21, ~1.7× LoRA)** and
correspondingly the **worst retention** and worst adapt (GSM8K 49.7).
**Gate verdict (2026-07-08, retfix_retention_gate --mode pissa, 270 BBH generations inspected):**
**REAL-FORGETTING** — the correct target appears in only 37/270 generations (13.7%); exact-match
24/270; MetaMath-style contamination is negligible (3/270). The 2026-07-06 eval-artifact
hypothesis is REVISED: format breakdown contributes (60/270 empty generations, 22%) but the
dominant failure is that the model genuinely no longer produces the answers. Cite PiSSA's BBH
7.23 as real (severe) forgetting, with a one-line note that ~22% empty generations mean the
number is a mild *underestimate* of retained ability. The likelihood-MMLU parity with peers
(24.5) shows recognition survives while generation collapses — worth one sentence, not a caveat
that blocks the datapoint.
**Why it matters.** "Initialize from the *principal* directions" maximally perturbs the weights →
maximal forgetting — a clean contrast with MiLoRA (minor-init), and now a legitimate high-‖ΔW‖
law point (though it under-shoots the law's linear prediction; the wd0@5e-4 collapse anchor
remains the cleaner extreme point). PiSSA re-run frm2_pissa still queued (checks the 8.6pp
adaptation gap vs published 58.23).

---

## 5. CLoRA's advantage over LoRA does not reproduce in a controlled harness
**Finding.** In our identical pipeline, CLoRA retention (18–19) ≈ LoRA (18.0) at matched ‖ΔW‖, and
CLoRA GSM8K (58.5–60.8) ≈ LoRA (60.2) — i.e. the paper's ~+4 pp CLoRA-over-LoRA edge shrinks to
noise once LoRA is evaluated faithfully and both share a ruler.
**Why it matters.** Supports "the reported wins are fragile / setup-dependent." (Stated carefully:
we anchor on *reproducing LoRA* and beating CLoRA's *published* number, so the claim doesn't rest
on our possibly-conservative CLoRA reproduction.)

---

## 6. Plain LoRA at a well-chosen LR already beats CLoRA's best *published* number
**Finding.** LoRA (wd=0) at lr=1e-4 scores **GSM8K 64.97 > CLoRA-k128's published 64.59**, with
smaller ‖ΔW‖ (0.43) and higher retention (22.6). This is *before* adding weight decay.
**Why it matters.** The headline "wake-up call": the fancy method's edge is largely an
LR/magnitude-tuning artifact — plain LoRA, tuned, matches/beats it.

---

## 7. Use ‖ΔW‖_F, not σ_max, as the magnitude axis
**Finding.** σ_max (dw_sv_max) is confounded: MiLoRA/LoRA show huge σ_max (155–161) at moderate
‖ΔW‖_F (~1.26), while CLoRA shows small σ_max (~28) at similar ‖ΔW‖_F (~1.0). Retention tracks
‖ΔW‖_F (r=−0.86 pooled on the n=49 CS sweep; the 7-point faithful-math r=−0.93 is
leverage-point-driven, see insight 3), not σ_max.
**Why it matters.** Methodological: the *Frobenius* magnitude (total update energy) is what governs
forgetting; a single-direction spectral norm misleads. Prevents an easy reviewer misread.

---

## 8. CLoRA's own Table 4 IS the magnitude law — and they never swept the magnitude knob
Their capacity-forgetting analysis (Table 4) measures F_Δ = ‖ΔW·x‖/‖x‖ (output-change magnitude) vs
BBH: LoRA 0.79→26.7, LoRA-L2 0.29→32.9, CLoRA-k2048 0.14→38.7. **Retention is monotone in update
magnitude — stated in the CLoRA paper itself.** Independent support for the law. AND their LoRA-L2 is a
SINGLE point (weight 1e-5; "1e-4 too large") — the magnitude/weight-decay knob is **untuned** in their
comparison. Our LoRA+wd LR×wd sweep is exactly the fair baseline they omitted.

## 9. HONEST boundary — geometry is NOT useless on commonsense
Against CLoRA's full Table 2: LoRA+wd (wd0.3) matches CLoRA **up to k512** (adapt~81/ret~26 vs k512
82/25.7), but **CLoRA-k1024/k2048 beat it on BOTH axes** (83.7/29.6). Forcing LoRA+wd to that retention
**collapses its adapt** (→45), while CLoRA holds adapt 83.7 — so CLoRA's directional (null-space)
constraint buys real adapt-efficiency at high regularization that pure magnitude control does not. The
defensible claim is therefore NARROWER than "geometry is useless": the magnitude **law** governs retention
universally, and LoRA+wd matches fancy adapters on **math** and **mid-regularization CS** — but at **high-k
CS, geometry adds value.** (Open: full LoRA+wd wd-sweep on CS still running = definitive test; and
CLoRA-k2048's out-domain EXCEEDS base (BBH 38.7 > 34.9) — transfer, not just retention — which muddies
"forgetting" on CS.)

_Watch-items: MATH scorer/cutoff offset (~3pp low; 256-vs-512 sensitivity running); the CS LoRA+wd
wd-sweep verdict; CorDA++ faithfulness (N, rank-allocation direction, π operand) pending the CorDA++
paper (arXiv:2506.13187)._
