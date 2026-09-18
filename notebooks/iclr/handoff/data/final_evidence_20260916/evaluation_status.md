# Held-aside evaluation status (2026-09-16)

**Superseding integration note (2026-09-17):** the original statement below
predates the one-time evaluation now exported in
`../locked_evaluation_20260916/held_aside_results.csv` for all 18 practical
confirmations. Those results have been joined to the frozen checkpoint request
and independently reaggregated from per-example records. The later 21-run band
block is fully trained and frozen; its held-aside evaluation is not in the export.

Original dated statement:

The official validation sets (RTE 277 / MRPC 408 examples, the `locked_evaluation` splits of preparation_20260915T0707Z) have NOT been evaluated for any clean-campaign recipe or checkpoint. All reported task scores are inner-selection-split scores; the training engine never receives the locked split. No checkpoint was chosen or tuned using it. A one-time locked-split evaluation of the frozen confirmation checkpoints can be run on request as a separate, clearly labeled export.
