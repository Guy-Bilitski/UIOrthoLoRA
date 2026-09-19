# Decoder subspace confirmations — status snapshots

Append-only. Newest snapshot at the top of the log. Written by
`ops/subspace_status.py` from the ledger and the registered protocols only.
The remaining-time figure is an estimate from measured step medians plus a
decode allowance, not a measurement.

## Snapshots

### Snapshot 2026-09-19T18:44:51+00:00

- Population 18 entries; **12 completed and validated**, 2 running, 4 not started, 0 failed or interrupted.
- Running: TAIL_DIAG seed 42, step 842/842.
- Running: TAIL_ROT128 seed 42, step 584/842.
- Frozen full-test reference done: True.
- Remaining estimate: 5.07 GPU-hours, about 2.53 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_DIAG | 123 | 0.3820 | 0.5004 | 2.7e-12 | False |
| LEAD_DIAG | 17 | 0.3824 | 0.4943 | 2.7e-12 | False |
| LEAD_DIAG | 42 | 0.3821 | 0.4920 | 2.7e-12 | False |
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_DIAG | 123 | 0.3814 | 0.4852 | 3.2e-12 | False |
| MID_DIAG | 17 | 0.3816 | 0.4807 | 3.2e-12 | False |
| MID_DIAG | 42 | 0.3813 | 0.4913 | 3.2e-12 | False |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |
| TAIL_DIAG | 17 | 0.3773 | 0.5057 | 3.2e-12 | False |
| TAIL_ROT128 | 17 | 0.3622 | 0.4882 | 3.2e-12 | True |

### Snapshot 2026-09-19T17:59:45+00:00

- Population 18 entries; **10 completed and validated**, 2 running, 6 not started, 0 failed or interrupted.
- Running: TAIL_DIAG seed 17, step 842/842.
- Running: TAIL_ROT128 seed 17, step 684/842.
- Frozen full-test reference done: True.
- Remaining estimate: 6.76 GPU-hours, about 3.38 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_DIAG | 123 | 0.3820 | 0.5004 | 2.7e-12 | False |
| LEAD_DIAG | 17 | 0.3824 | 0.4943 | 2.7e-12 | False |
| LEAD_DIAG | 42 | 0.3821 | 0.4920 | 2.7e-12 | False |
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_DIAG | 123 | 0.3814 | 0.4852 | 3.2e-12 | False |
| MID_DIAG | 17 | 0.3816 | 0.4807 | 3.2e-12 | False |
| MID_DIAG | 42 | 0.3813 | 0.4913 | 3.2e-12 | False |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |

### Snapshot 2026-09-19T17:14:37+00:00

- Population 18 entries; **8 completed and validated**, 2 running, 8 not started, 0 failed or interrupted.
- Running: MID_DIAG seed 123, step 547/842.
- Running: MID_DIAG seed 42, step 842/842.
- Frozen full-test reference done: True.
- Remaining estimate: 8.47 GPU-hours, about 4.23 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_DIAG | 123 | 0.3820 | 0.5004 | 2.7e-12 | False |
| LEAD_DIAG | 17 | 0.3824 | 0.4943 | 2.7e-12 | False |
| LEAD_DIAG | 42 | 0.3821 | 0.4920 | 2.7e-12 | False |
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_DIAG | 17 | 0.3816 | 0.4807 | 3.2e-12 | False |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |

### Snapshot 2026-09-19T16:29:29+00:00

- Population 18 entries; **6 completed and validated**, 2 running, 10 not started, 0 failed or interrupted.
- Running: LEAD_DIAG seed 123, step 842/842.
- Running: MID_DIAG seed 17, step 81/842.
- Frozen full-test reference done: True.
- Remaining estimate: 9.89 GPU-hours, about 4.94 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_DIAG | 17 | 0.3824 | 0.4943 | 2.7e-12 | False |
| LEAD_DIAG | 42 | 0.3821 | 0.4920 | 2.7e-12 | False |
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |

### Snapshot 2026-09-19T16:29:19+00:00

- Population 18 entries; **6 completed and validated**, 2 running, 10 not started, 0 failed or interrupted.
- Running: LEAD_DIAG seed 123, step 842/842.
- Running: MID_DIAG seed 17, step 72/842.
- Frozen full-test reference done: True.
- Remaining estimate: 9.89 GPU-hours, about 4.95 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_DIAG | 17 | 0.3824 | 0.4943 | 2.7e-12 | False |
| LEAD_DIAG | 42 | 0.3821 | 0.4920 | 2.7e-12 | False |
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |

### Snapshot 2026-09-19T16:29:02+00:00

- Population 18 entries; **6 completed and validated**, 2 running, 10 not started, 0 failed or interrupted.
- Running: LEAD_DIAG seed 123, step 842/842.
- Running: MID_DIAG seed 17, step 58/842.
- Frozen full-test reference done: True.
- Remaining estimate: 9.9 GPU-hours, about 4.95 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_DIAG | 17 | 0.3824 | 0.4943 | 2.7e-12 | False |
| LEAD_DIAG | 42 | 0.3821 | 0.4920 | 2.7e-12 | False |
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |

### Snapshot 2026-09-19T15:45:38+00:00

- Population 18 entries; **4 completed and validated**, 1 running, 13 not started, 0 failed or interrupted.
- Running: LEAD_DIAG seed 17, step 680/842.
- Frozen full-test reference done: False.
- Remaining estimate: 11.24 GPU-hours, about 5.62 hours wall on two GPUs.

| Arm | Seed | Selection NLL | Exact match | Off-band | Rotation active |
|---|---:|---:|---:|---:|---|
| LEAD_ROT128 | 123 | 0.3635 | 0.4936 | 3.2e-12 | True |
| LEAD_ROT128 | 17 | 0.3636 | 0.4898 | 3.2e-12 | True |
| LEAD_ROT128 | 42 | 0.3636 | 0.4882 | 3.2e-12 | True |
| MID_ROT128 | 17 | 0.3625 | 0.4973 | 3.2e-12 | True |

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

