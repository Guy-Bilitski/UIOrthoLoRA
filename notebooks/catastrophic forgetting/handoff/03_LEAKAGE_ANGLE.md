> 📑 **REFERENCE (weight-basis leakage angle).** Superseded as the HEADLINE by the Frobenius-magnitude
> law (08/06) + the data-basis reframe (07). The weight-basis thermometers (μ_E/ν_D) turned out NOT to
> predict retention (r≈−0.09) — that's now part of the story, not the thesis. Kept for the diagnostics.

# LEAKAGE THERMOMETERS — the paper angle (user is excited about this)

> User's framing: *"how much leakage do we need between the low tail and major component"* — i.e.,
> there's an **optimal leakage budget**, and the thermometers let us measure and control it.

## The quantities (impl: `leakage.py`, validated)
For each adapted weight (SVD split: leading/major band `_R` = frozen top; tail `_r` = adapted bottom):
- **μ_E = ‖U_Rᵀ · E · Ū_r‖₂** (LEFT/output) — how much the E-scaled tail leaks into the preserved LEFT subspace.
- **ν_D = ‖V_Rᵀ · D · V̄_r‖₂** (RIGHT/input) — same on the right.
- Plus `Leak11` (‖C₁₁‖/‖ΔW‖ direct leading contamination), `OffTailF` (off-tail energy fraction), `RelPertF`, `DriftU/V` (sinΘ of leading subspaces, pre vs post; expensive 2nd SVD).
- **μ_E=ν_D=0 ⇒ update is exactly confined to the tail** (zero forgetting by construction). Scalers E,D "tilt" the tail into the preserved subspace = leakage.
- Norm switch: report with **operator norm ‖·‖₂**; penalize with **squared Frobenius ‖·‖_F²** (smooth).
- **Always log BOTH sides** — the key finding is the right-side escape route under one-sided penalties.

## The story (why it's interesting / publishable)
1. **Leakage is a real, measurable control knob**, not an analysis artifact. use_de=0 ⇒ μ_E=ν_D=0 (we confirmed).
   use_de=1 ⇒ leakage jumps (UILinLoRA: F∆ 0.75 despite ‖ΔW‖ 9 = pure leakage).
2. **Optimal leakage budget (the user's angle):** too little leakage (use_de=0, k_vec small) ⇒ under-adapts
   (low CS); too much (use_de=1, big k_vec) ⇒ forgets (low retention). The thermometers let us plot a
   **leakage→(CS, retention) curve** and locate the sweet spot. This reframes UIOrthoLoRA as a *tunable*
   adaptation/preservation tradeoff with a *measurable* mechanism — a stronger contribution than a single SOTA point.
3. **You must watch/penalize BOTH sides:** penalizing only left (λ_E>0, λ_D=0) drives μ_E down but reroutes
   the mixing through D ⇒ ν_D goes UP. Only penalizing both gives clean preservation. (Paper §4.1 / App B.1;
   reproduce on RoBERTa GLUE = exp B3.)

## ⚠️ Caveat that MUST be in the paper (impl finding #4)
Our ΔW carries an extra **frozen major term `E·U₁·diag(1)·V₁ᵀ·D`** (leading-band, unit singular values)
NOT in the paper's `ΔW=E·Ū_r·Σ'_r·V̄_rᵀ·D`. The thermometers are **tail-only** (per the brief) so they
DON'T capture this term's perturbation of the preserved subspace. So: a config can show **μ_E≈ν_D≈0 yet still
forget** via this term. ⇒ When reporting leakage-vs-retention, either (a) also report a *full-ΔW* leakage that
includes the major term, or (b) test the corrected layer (exp A5) so thermometers and actual retention agree.
This is exactly the kind of thing a reviewer will catch — get ahead of it.

## What's measured / what's left
- ✅ `leakage.py` implemented + validated (synthetic + real layer; sanity all pass).
- ✅ Wired into `uio_inprocess.py` → all UIOrthoLoRA runs from Wave 2 on log μ_E, ν_D, Leak11, OffTailF, RelPertF, DriftU, DriftV (mean over 160 modules), in `results/<run>/summary.json["leakage"]` + headline.
- ⬜ Wave 1 runs (already training) do NOT have leakage (pre-dated the wiring) — re-run the key points with leakage if needed, or rely on Wave 2's use_de on/off contrast.
- ⬜ Training-time penalty `R_mix` (λ_E,λ_D) — NOT yet wired (exp B2). Add a Trainer.compute_loss override computing M_E,M_D per UIOrthoLoRA layer IN the autograd graph (mirror `CLoRARegularizer` in train_cs.py).
- ⬜ RoBERTa-base GLUE A/B/C diagnostic (exp B3) — separate harness, not built.

## Reference: extraction mapping (this repo's 3-band layer → paper's 2-band)
leading `U_R`=`uiortholora_U1` (size rank−k_val); tail `Ū_r`=`[U2 | U3·R_U]` (size k_val); `Σ'_r`=`uiortholora_sigma`;
`E`=`uiortholora_E`, `D`=`uiortholora_D`; `R_U,R_V`=`uiortholora_{left,right}_unitary[adapter].weight` (materialized; None if k_vec=0).
Measure ΔW **tail-only**. See `leakage.py::uio_layer_leakage` and `model_leakage`.
