#!/usr/bin/env python3
"""Create references/testset.txt for the E2E official scorer."""

import os
from collections import OrderedDict
from datasets import load_dataset

# 1. Load the E2E NLG test split
ds = load_dataset("tuetschek/e2e_nlg", trust_remote_code=True)

# 2. Make sure the folder exists
os.makedirs("references", exist_ok=True)

# 3. Group references by identical MR (preserve order of first appearance)
groups = OrderedDict()
for ex in ds["test"]:
    mr  = ex["meaning_representation"]
    ref = ex["human_reference"].strip()
    groups.setdefault(mr, []).append(ref)

assert len(groups) == 630, f"Expected 630 unique MRs, got {len(groups)}"

# 4. Write file: refs for each MR, blank line BETWEEN groups (not after last)
with open("references/testset.txt", "w", encoding="utf8") as fout:
    for idx, refs in enumerate(groups.values()):
        for ref in refs:
            fout.write(ref + "\n")
        if idx < len(groups) - 1:      # no trailing blank line
            fout.write("\n")

print("✅ Wrote references/testset.txt")
print("   MR groups :", len(groups))          # 630
print("   Total refs:", sum(len(r) for r in groups.values()))
