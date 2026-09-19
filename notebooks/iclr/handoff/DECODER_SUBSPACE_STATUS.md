# Decoder subspace confirmations — status snapshots

Append-only. Newest snapshot at the top of the log. Written by
`ops/subspace_status.py` from the ledger and the registered protocols only.
The remaining-time figure is an estimate from measured step medians plus a
decode allowance, not a measurement.

## Snapshots

### Snapshot 2026-09-19T15:44:21+00:00

- Population 18 entries; **4 completed and validated**, 1 running, 13 not started, 0 failed or interrupted.
- Running: LEAD_DIAG seed 17, step 632/842.
- Frozen full-test reference done: False.
- Remaining estimate: 11.25 GPU-hours, about 5.63 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |

### Snapshot 2026-09-19T14:59:12+00:00

- Population 18 entries; **2 completed and validated**, 2 running, 14 not started, 0 failed or interrupted.
- Running: LEAD_ROT128 seed 123, step 632/842.
- Running: MID_ROT128 seed 17, step 613/842.
- Frozen full-test reference done: False.
- Remaining estimate: 12.63 GPU-hours, about 6.32 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |

### Snapshot 2026-09-19T14:25:40+00:00

- Population 18 entries; **2 completed and validated**, 2 running, 14 not started, 0 failed or interrupted.
- Running: LEAD_ROT128 seed 123, step 43/842.
- Running: MID_ROT128 seed 17, step 43/842.
- Frozen full-test reference done: False.
- Remaining estimate: 13.68 GPU-hours, about 6.84 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |

### Snapshot 2026-09-19T14:14:05+00:00

- Population 18 entries; **0 completed and validated**, 2 running, 16 not started, 0 failed or interrupted.
- Running: LEAD_ROT128 seed 17, step 842/842.
- Running: LEAD_ROT128 seed 42, step 842/842.
- Frozen full-test reference done: False.
- Remaining estimate: 14.53 GPU-hours, about 7.27 hours wall on two GPUs.

### Snapshot 2026-09-19T13:37:48+00:00

- Population 18 entries; **0 completed and validated**, 2 running, 16 not started, 0 failed or interrupted.
- Running: LEAD_ROT128 seed 17, step 335/842.
- Running: LEAD_ROT128 seed 42, step 343/842.
- Frozen full-test reference done: False.
- Remaining estimate: 15.44 GPU-hours, about 7.72 hours wall on two GPUs.

### Snapshot 2026-09-19T13:28:58+00:00

- Population 18 entries; **0 completed and validated**, 2 running, 16 not started, 0 failed or interrupted.
- Running: LEAD_ROT128 seed 17, step 184/842.
- Running: LEAD_ROT128 seed 42, step 192/842.
- Frozen full-test reference done: False.
- Remaining estimate: 15.72 GPU-hours, about 7.86 hours wall on two GPUs.

### Snapshot 2026-09-19T13:28:18+00:00

- Population 18 entries; **0 completed and validated**, 2 running, 16 not started, 0 failed or interrupted.
- Running: LEAD_ROT128 seed 17, step 171/842.
- Running: LEAD_ROT128 seed 42, step 180/842.
- Frozen full-test reference done: False.
- Remaining estimate: 15.74 GPU-hours, about 7.87 hours wall on two GPUs.

### Snapshot 2026-09-19T13:27:31+00:00

- Population 18 entries; **0 completed and validated**, 2 running, 16 not started, 0 failed or interrupted.
- Running: LEAD_ROT128 seed 17, step 157/842.
- Running: LEAD_ROT128 seed 42, step 166/842.
- Frozen full-test reference done: False.
- Remaining estimate: 15.76 GPU-hours, about 7.88 hours wall on two GPUs.

