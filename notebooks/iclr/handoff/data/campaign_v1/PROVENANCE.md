# Prospective dataset version 1

This directory is distinct from all legacy archives. It contains immutable CPU
preflights and resource/provenance records. A separate persistent output root
now holds prepared public inputs, one interrupted GPU P0 attempt with a
step-0 checkpoint, and one completed, independently validated eight-step P0
retry. No magnitude-calibration or confirmation results exist yet. The initial
CSV ledger here is a historical empty startup snapshot, not the current run ledger.
Each preflight invocation uses a fresh
timestamp/UUID directory and records its source hashes, environment, commands,
exit statuses and actual CPU test outputs. Never overwrite an earlier bundle.

The author authorizes GPUs 2/3 through completion. The initial persistent output
allowance is 50 GiB and public downloads are allowed, with the interpretation of
the author's "Go ahead" recorded in `RESOURCE_AUTHORIZATION_20260914.json`.
Live scientific/resource ledgers reside in the assigned `campaign_outputs_v1/`,
not in legacy data. Full campaign status and continuation:
`../../CAMPAIGN_STATUS_20260914.md`. This small versioned Git dataset is not an
assumed persistent allocation for model checkpoints.
