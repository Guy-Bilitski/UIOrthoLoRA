# Supplementary analysis archive — 19 September 2026

`analysis_artifact.zip` is the sanitized standalone analysis package described in
the paper. It includes a README, source/distributed hash manifest and CPU verifier.
Do not upload the entire private Overleaf project as supplementary material.
Attach this ZIP alongside the paper when delivering the revision to reviewers.
No public hosting or conference submission has been performed by preparing this file.

SHA256: `d45b985f7b8576b09da46f353d9d07b77a8484b2d54512a9d8ee0de139763d8a`.
Size: 1,479,828 bytes. Scientific manuscript source: Overleaf `932283e1`.
The ZIP includes the clean manuscript, all 21 band/head inner-selection outcomes
and the separate 18-run practical held-aside evaluation. It excludes private
handoffs and the new post-feedback calibration assessment. Temperature-scaling
logits and band held-aside scoring remain pending in the available exports.

Rebuild into a fresh output directory with
`python3 scripts/build_analysis_artifact.py --out /path/to/analysis_artifact`,
then verify the extracted ZIP before replacing this snapshot. All numerical data
remain unchanged by sanitization; private paths and infrastructure labels change,
so source and distributed hashes are distinguished explicitly.
