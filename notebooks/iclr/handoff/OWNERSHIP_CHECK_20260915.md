# Ownership / restart-safety verification — 2026-09-15, before unattended confirmation

Non-training orchestration checks requested before overnight execution. All
performed on the live campaign state (refusal paths have no side effects) or a
throwaway synthetic allocation root.

## Checks performed

1. **Duplicate-entry rejection (live).** The per-entry launcher was invoked for
   (a) an entry with a validated completion (`mrpc/P1_NORM/100`) and (b) an
   entry currently running (`rte/P1_NORM/5.5`). Both were REFUSED before any
   tmux session or ledger event was created. Retry launches require an explicit
   `retry_of` argument naming the terminal prior attempt.
2. **Same-GPU exclusive reservation (synthetic).** With an active lease on a
   GPU, a second `reserve` on the same GPU raises ("stale leases do not
   auto-expire"). Verified on a throwaway `AllocationLedger` root.
3. **No stale-lease release by strangers (synthetic).** `settle` with a wrong
   ownership token is refused; settlement with the recorded token releases the
   GPU for subsequent reservation.
4. **Restart reconciliation (queue semantics).** The fixed-lane confirmation
   queue skips an entry whose latest attempt is completed+hash-verified, blocks
   on any nonterminal attempt (never relaunching beside it), and retries only
   when every prior attempt failed terminally, with an explicit retry chain and
   a hard cap of two retries before requiring diagnosis.

## Supporting incident evidence (2026-09-15, ledger-recorded)

- A dispatcher double-booking was **blocked by the lease conflict** at worker
  startup (run `20260915T084710Z_14b74d5cc197`, failed fail-closed).
- A controller kill left an orphan worker whose lease was **not silently
  released**; it required manual settlement with the recorded ownership token
  after the worker's observed graceful exit (run `20260915T093911Z_61dd167b0558`,
  ledger note records the settlement).
- Dynamic dispatch (GPU-memory and reset-state variants) is retired; overnight
  confirmation uses fixed per-GPU queues with the admission checks above.
  GPU 0 remains excluded (ceded to a colleague).

## Probe reference (same session)

The genuine original-backbone probe reference was computed inference-only on
CPU and sealed at
`campaign_outputs_v1/inputs/preparation_20260915T0707Z/probe_reference_original_backbone.json`
(masked-token CE 2.0956 on the fixed 256-example set). The practical adapter's
step-0 state is not this reference and is never substituted for it.
