# Section-by-section writing revision — 2026-09-16

Internal author record, not anonymous submission material.

The author requested a complete pass before reviewing: build the paper around
the completed interaction-control finding; state ideas, claims and results
directly; make every section easy to read. This supersedes the earlier narrative
that led with the unfinished strict-tail experiment.

The title is **Controlling Interaction in Spectral Fine-Tuning**.

| Section | Reader's question | Revised content |
| --- | --- | --- |
| Abstract | What did this study find? | Introduces our new adapter family, explains the two controls in plain language, and reports roughly halved prediction loss with average accuracy changes below half a point. |
| Introduction | What is the gap and why does it matter? | Separates choosing pretrained directions from controlling their interactions; explains the adapter family and main finding. |
| 2. Adapter family | What can each adapter change? | Coefficients, rotations and fixed support; exact approximation result distinguishes location from flexibility. |
| 3. Interaction control | What does the penalty do? | Explains ambient scaling, leading-plus-tail structure, block interactions and the uniform-scaler limit. |
| 4. Measurements | Why measure both behavior and weights? | Defines accuracy and prediction loss first, then interaction share, overall size, module allocation and learned displacement. |
| 5. Experiments | What happened? | Design, loss/accuracy, size sweep, module allocation, probe behavior, broader historical results. The new paired-seed figure makes the primary result visible. |
| 6. Related work | Where does this fit? | Organizes prior work by choosing directions, combining directions and other restrictions. |
| 7. Scope | What exactly does the experiment cover? | Collects calibration, seed precision, matching, mechanism and implementation cost in one place. |
| 8. Conclusion | What should a researcher remember? | Interaction control produces a different learned update and lower task loss than the tested size penalty; accuracy alone misses the loss difference. |
| Appendices | How can I check it? | Simplifies explanation of proofs, protocols, paired comparisons and allocation; preserves equations, numeric tables, provenance and all existing outcome rows. |

The strict-tail study is retained in Appendix B.19. Its pilot and confirmation
rows remain separate, with the missing head/rotation comparisons identified.
No experiments are implied by the writing changes. The GPU queue and coding-agent
handoff are unchanged.

The main empirical comparison is the complete regularization intervention.
The paper does not attribute lower loss uniquely to cross share, equate mean
accuracy with equivalence, or represent the practical adapter as strictly confined.
The unfavorable masked-token probe and archived score results remain explicit.

All numerical data are unchanged. The new outcome figure is generated directly
from the 18 frozen held-out rows, checks their manifest hash, shows every seed,
and uses the same accuracy scale for both tasks. The figure generator is included
in the standalone supplement and its audit. The revised main-table caption is
also changed in its generator so regeneration preserves the prose.

Validation build: `../build/interaction_story_20260916/neurips_2026.pdf`.
The source checks and exact delivery hashes are recorded in that directory.
Overleaf's own compiler remains the canonical PDF build.
