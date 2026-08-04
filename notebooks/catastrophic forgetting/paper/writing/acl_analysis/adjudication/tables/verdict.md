# VERDICT decision table (adjudication, 2026-07-18)

ret@matched-adapt = method retention at its best-adaptation cell minus the
retention LoRA+wd delivers at >= that adaptation (its LR is re-chosen; 'beyond' =
LoRA+wd's sweep never reaches that adaptation — only SC-LoRA on Qwen-math).
lr_band = LRs with retention within 2pp of base (qwen_cs: family-relative band),
summed over the 4 families. Divergence = quarantined/attempted runs at LR <= 1e-3
(the shared sweep band) over the 6 sweep+grid families; 2e-3/5e-3 probe cells are
excluded so methods are not penalized for having been probed at extreme LR.
Qwen-math LoRA+wd ceiling note: its sweep-max adaptation cell is lr1e-3 with n=1
(72.93; the two sibling seeds diverged) — SC-LoRA's 'beyond' flag and the qwen_math
adapt gaps are measured against that cell; against the 3-seed rule (3e-4, 68.97)
SC-LoRA's edge grows. Script: `06_verdict.py` (inputs: tables from 01-05).

| Method | ret@matched-adapt | adapt ceiling | LR band | seed SD(ret) | divergence | train | init | memory | deploy | beats LoRA+wd? |
|---|---|---|---|---|---|---|---|---|---|---|
| LoRA+wd | reference | reference | 26/29 | 0.43 | 6/146 (4.1%) | 1.00x | none (free AdamW flag) | 0 | rank-r | reference |
| LoRA | -2.2 pp | -4.0 pp | 10/29 | 0.83 | 7/179 (3.9%) | 1.00x | none | 0 | rank-r | 0/4 fam (dominated in 3) |
| CLoRA | -4.2 pp | -3.2 pp | 12/22 | 0.76 | 0/121 (0.0%) | 1.17x | k x d covariance/eigh on base weights (fast); frozen-P build | +3.34 | rank-r | 0/4 fam (dominated in 3) |
| MiLoRA | -2.8 pp | -4.2 pp | 13/31 | 0.65 | 9/162 (5.6%) | 1.00x | 160 base-weight SVDs (no forwards) | 0 | rank-2r | 0/4 fam (dominated in 2) |
| LoRA-Null | -3.2 pp | -2.5 pp | 13/24 | 0.72 | 3/97 (3.1%) | 1.00x | 256 calibration forwards + eigh | 0 | rank-r | 0/4 fam (dominated in 2) |
| SC-LoRA | -4.6 pp (1 beyond) | -0.9 pp | 6/25 | 3.06 | 4/113 (3.5%) | 0.99x | 512 calibration forwards + eigh | 0 | rank-r | 1/4 fam (dominated in 1) |
| DoRA | -4.2 pp | -5.9 pp | 8/22 | 0.7 | 3/73 (4.1%) | 2.15x | none | 0 | rank-r | 0/4 fam (dominated in 2) |
| PiSSA | -26.3 pp | -17.1 pp | 0/1 | nan | 0/5 (0.0%) | — | 160 base-weight SVDs (no forwards) | 0 | rank-2r | 0/1 fam (dominated in 1) |

CorDA/CorDA++: WITHHELD (own port bug — divergence rows 28.6%/36.4% shown in
tables/divergence.csv for completeness, never ranked).

Per-family matched-adapt gaps:
- llama_cs SC-LoRA: ret gap -1.26 pp, adapt gap -1.14 pp
- llama_cs LoRA: ret gap -2.00 pp, adapt gap -2.58 pp
- llama_cs LoRA-Null: ret gap -4.10 pp, adapt gap -2.88 pp
- llama_cs CLoRA: ret gap -4.25 pp, adapt gap -3.46 pp
- llama_cs DoRA: ret gap -6.70 pp, adapt gap -5.52 pp
- llama_cs MiLoRA: ret gap -4.42 pp, adapt gap -4.56 pp
- llama_math LoRA: ret gap -2.28 pp, adapt gap -2.81 pp
- llama_math MiLoRA: ret gap -1.13 pp, adapt gap -3.11 pp
- llama_math CLoRA: ret gap -4.94 pp, adapt gap -6.14 pp
- llama_math SC-LoRA: ret gap -5.63 pp, adapt gap -6.32 pp
- llama_math DoRA: ret gap -5.49 pp, adapt gap -7.61 pp
- llama_math LoRA-Null: ret gap -4.68 pp, adapt gap -4.93 pp
- llama_math PiSSA: ret gap -26.34 pp, adapt gap -17.13 pp
- qwen_cs SC-LoRA: ret gap -12.22 pp, adapt gap -0.62 pp
- qwen_cs LoRA: ret gap -2.75 pp, adapt gap -1.34 pp
- qwen_cs LoRA-Null: ret gap -1.76 pp, adapt gap -1.54 pp
- qwen_cs CLoRA: ret gap -0.55 pp, adapt gap -0.75 pp
- qwen_cs DoRA: ret gap -2.66 pp, adapt gap -1.33 pp
- qwen_cs MiLoRA: ret gap -4.02 pp, adapt gap -1.38 pp
- qwen_math SC-LoRA: ret gap +0.73 pp [beyond LoRA+wd adaptation ceiling], adapt gap +4.30 pp
- qwen_math LoRA: ret gap -1.79 pp, adapt gap -9.45 pp
- qwen_math LoRA(r32): ret gap -5.21 pp, adapt gap -2.50 pp
- qwen_math LoRA-Null: ret gap -2.22 pp, adapt gap -0.60 pp
- qwen_math CLoRA: ret gap -7.00 pp, adapt gap -2.47 pp
- qwen_math DoRA: ret gap -1.80 pp, adapt gap -9.02 pp
- qwen_math MiLoRA: ret gap -1.73 pp, adapt gap -7.58 pp