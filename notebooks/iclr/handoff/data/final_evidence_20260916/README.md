# Final evidence export (generated 2026-09-16T19:19:45.153058+00:00)

Response to FINAL_EVIDENCE_EXPORT_REQUEST_20260916.md. All rows come from ledger-validated runs with invalidation records enforced; per-run validation SHA256 values are included. The final block-B export contains all 21 registered RTE band/head confirmations and two separately labeled timing pilots. Zero-update fractions are exported as missing, never zero. Within-band off-diagonal energies are computed exactly from the compact checkpoint core (h and orthogonal-map originals), not from subtracting squared norms.

Paper-integration note, 2026-09-17: band outcomes in this directory use the
inner-selection split. The separate `../locked_evaluation_20260916/` directory
contains held-aside results for the original 18 practical confirmations and the
21-checkpoint freeze ledger for block B. Held-aside block-B results are not in
this export. The original evaluation-status note below is superseded for the
18 practical confirmations; it must not be interpreted as their current status.
