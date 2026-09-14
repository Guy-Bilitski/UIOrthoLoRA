# Prospective dataset version 1

This directory is distinct from all legacy archives. Currently it contains only
immutable CPU preflight bundles and an empty model-run ledger; there are no new
task-training scores or checkpoints. Each preflight invocation uses a fresh
timestamp/UUID directory and records its source hashes, environment, commands,
exit statuses and actual CPU test outputs. Never overwrite an earlier bundle.

The author assigned GPUs 2 and 3, but training budget, output allocation and
model/dataset download policy are pending. Full campaign status and continuation:
`../../CAMPAIGN_STATUS_20260914.md`. This small versioned Git dataset is not an
assumed persistent allocation for model checkpoints.
