# Abstract draft — CONFIRMED NUMBERS (2026-09-16, 18/18 runs validated)

For author + Astra review. Every number below is from the seed-confirmed table
(n=3, hash-validated); nothing pending.

---

Parameter-efficient adaptation is usually evaluated by what it achieves, not by
where it acts. We analyze fine-tuning updates in the pretrained weights' own
singular frame, decomposing each update across 48 attention projections of
RoBERTa-base into leading, tail, and cross-band energies. Unconstrained
adaptation is spectrally diffuse: 42–44% of update energy lies in cross-band
interaction on both RTE and MRPC. A two-sided mixing penalty built from the
pretrained projectors suppresses this interaction to 10.0±0.5% (RTE) and
4.5±0.3% (MRPC) across three paired seeds, at statistically indistinguishable
task accuracy, and the surviving update occupies different bands on different
tasks — near block-diagonal on RTE, leading-dominated (70%) on MRPC. Critically,
a Frobenius penalty dose-matched to the same update magnitude (per-seed norm
errors 0.1–8.0%, reported for every pair) leaves the allocation at the
unregularized level across its entire dose frontier: within this family,
controlling magnitude does not reproduce the geometric effect, which is
attributable to the pretrained-frame penalty itself. All doses were selected on
update norms alone under registered protocols; we release the per-run spectral
diagnostics, full dose frontiers, and failed matches.

[Optional final sentence if the band study lands before the freeze:
Restricting adaptation to equal-sized leading, middle, or tail singular bands
further separates where adaptation can occur from how flexibly it acts there.]

---

Wording notes applied (Astra): question-form claim; block-diagonal not
tail-dominated; per-seed matching errors stated; no pretrained-preservation
claim (probe stays in the paper as a secondary measurement with the true
reference CE 2.10); "spectrally diffuse" replaces "promiscuous".
