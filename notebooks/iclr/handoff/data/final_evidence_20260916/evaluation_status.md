# Held-aside evaluation status (2026-09-16)

The official validation sets (RTE 277 / MRPC 408 examples, the `locked_evaluation` splits of preparation_20260915T0707Z) have NOT been evaluated for any clean-campaign recipe or checkpoint. All reported task scores are inner-selection-split scores; the training engine never receives the locked split. No checkpoint was chosen or tuned using it. A one-time locked-split evaluation of the frozen confirmation checkpoints can be run on request as a separate, clearly labeled export.
