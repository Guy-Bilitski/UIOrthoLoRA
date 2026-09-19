> **Latest author clarification — avoid LR sweeps:** Read
> `DECODER_SCOPE_REVIEW_20260919.md`. Keep the six subspace conditions and three
> seeds; use a reasonable fixed common recipe with brief learning checks.
> The old 18-run LR grid below is superseded for pending work. Preserve all
> completed work and reconcile the live ledger before changing the queue.

> **Current decoder plan — reviewed 19 September:** Read
> `DECODER_SUBSPACE_STUDY_20260919.md` and `CODING_AGENT_PROMPT_20260919.md`.
> Test leading/middle/tail support with our strict DIAG/ROT128 adapters; no
> LoRA/PiSSA comparison or new penalty sweep. The plan includes bounded balanced
> tuning, a two-GPU timing budget and a predeclared DIAG-only fallback. Older
> decoder commands, matrices and costs below are historical and superseded.

> **Latest author priority — modern decoder first:** Read
> `DECODER_FIRST_PRIORITY_20260919.md`. Prioritize the prepared decoder pilot,
> tuning/calibration and confirmation runs on the two assigned GPUs. Defer
> CENTER and further encoder controls. The coding agent has now delivered the
> temperature and band held-aside exports; verify/reuse them, do not rerun them.
> Older schedules below are superseded where they place controls first.

# ICLR spectral-adaptation research

Internal paper and experiment-coordination repository. Work only on the assigned Overleaf project `6aa54397e58b10444b0fa2aa`; another agent works on a different project.

For the coding session following the ICLR system's LLM feedback, **start with [EXPERIMENT_PREPARATION_HANDOFF_20260919.md](EXPERIMENT_PREPARATION_HANDOFF_20260919.md)** and the [full assessment](review_feedback/20260919/assessment.md). The author confirmed two GPUs and requested preparation of experiments for discussion. The repaired research runner already exists; all 18 practical confirmations and 21 RTE band/head confirmations are complete. Temperature scaling and held-aside band evaluation are still absent from the synchronized exports. Check the current server ledger before doing inference or launching anything.

The single active manuscript is [neurips_2026.tex](neurips_2026.tex), using the official ICLR 2027 style despite its legacy filename. The approved abstract and introduction are unchanged. The latest checked [PDF](review_feedback/20260919/paper.pdf) and [anonymous analysis supplement](supplement/analysis_artifact.zip) match the current scientific source. See the new handoff before reading dated status/queue notes in [GPU_HANDOFF.md](GPU_HANDOFF.md), [EXPERIMENTS_REQUIRED.md](EXPERIMENTS_REQUIRED.md) and the older review/audit documents. New proposed experiments are not manuscript findings.

No credentials belong in this repository. Do not upload internal planning, review or archival files wholesale as anonymous submission material. Direct author edits on Overleaf always take priority; fetch and reconcile before pushing, and never force-push.
