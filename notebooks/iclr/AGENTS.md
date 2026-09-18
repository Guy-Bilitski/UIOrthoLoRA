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
