> **Current reviewed scope — 19 September:** Start with
> `handoff/DECODER_SUBSPACE_STUDY_20260919.md` and
> `handoff/CODING_AGENT_PROMPT_20260919.md`. Prepare the strict leading/middle/tail
> decoder study with DIAG/ROT128, balanced tuning and both accuracy/loss outcomes.
> LoRA/PiSSA, new penalty sweeps and CENTER are outside the active queue. Reuse
> the latest registration code, but replace its old matrix and selection rules.
> The plan fixes a measured two-GPU budget and DIAG-only fallback. Older launch
> commands/decision templates below and in the handoff are superseded.

> **Latest author priority:** modern decoder results first. Follow
> `handoff/DECODER_FIRST_PRIORITY_20260919.md`. Prioritize the prepared decoder
> pilot and full study; defer CENTER and additional encoder controls. The
> frozen-checkpoint analyses are already delivered. Two assigned GPUs only.

# ICLR research workspace — current entry point

Work on Overleaf project `6aa54397e58b10444b0fa2aa` only. Start with
`handoff/EXPERIMENT_PREPARATION_HANDOFF_20260919.md`, then read
`handoff/AGENTS.md` and `handoff/review_feedback/20260919/assessment.md`.
These September 19 instructions supersede older queue/completion summaries.

The author requested preparation of experiments following the ICLR system's LLM
feedback and reconfirmed two GPUs. All 18 practical and 21 RTE band/head
confirmations are complete. The immediate evidence work uses frozen checkpoints;
new CENTER/decoder training remains a proposal to prepare for discussion.
Do not relaunch old band queues or assume historical four-GPU access.

Use an isolated checkout, environment and output namespace. Preserve unrelated
projects, shared implementation changes, live jobs, immutable evidence and the
tokenizer-fault invalidation records. The actual assigned device IDs and storage
must be established from current authorization before GPU execution. CPU reads,
implementation and relevant tests are part of the authorized preparation.

The research branch is `ortho_new`; publish scoped commits on top of the latest
remote, without force-pushing or including unrelated local work. The live paper
is in the separate Overleaf repository. The `handoff/` directory is a portable
snapshot, not a Git bridge to Overleaf. Keep the approved abstract/introduction
intact and deliver evidence/code independently of proposed manuscript changes.

> **Coding-session report — 19 September 2026 (late):** frozen-checkpoint work is done (`EXPERIMENT_PREPARATION_STATUS_20260919.md`: temperature scaling of the 18 practical endpoints, held-aside evaluation of all 21 band/head endpoints, 39 run records and trajectories). CENTER (`CENTER_PROTOCOL_20260919.md`) and the decoder pilot (`DECODER_PILOT_DESIGN_20260919.md`) are implemented and CPU-tested but NOT launched; they wait for the author's two GPU IDs and scope approval.
