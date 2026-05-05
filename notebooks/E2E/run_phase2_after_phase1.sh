#!/usr/bin/env bash
set -euo pipefail

PHASE1_PID="${1:?usage: $0 PHASE1_PID}"
STAMP="${2:-$(date -u +%Y%m%d_%H%M)}"

PYTHON="${PYTHON:-/home/guy_bilitski_idcdsi_ongcp_org/UIOrthoLoRA/.venv/bin/python}"
LOG_DIR="outputs/logs"
SUMMARY="$LOG_DIR/e2e_opt_search_${STAMP}_summary.txt"
PHASE2_LOG="$LOG_DIR/e2e_phase2_size_lr005_${STAMP}.log"

mkdir -p "$LOG_DIR"

echo "[$(date -u '+%F %T')] waiting for phase1 PID ${PHASE1_PID}" | tee -a "$SUMMARY"
while kill -0 "$PHASE1_PID" 2>/dev/null; do
    sleep 300
done
echo "[$(date -u '+%F %T')] phase1 PID ${PHASE1_PID} exited" | tee -a "$SUMMARY"

LR_WINNER="$("$PYTHON" - <<'PY'
from pathlib import Path
import re
import statistics
import sys

root = Path("outputs/results")
seeds = [1054, 2021, 31415]
metric_re = re.compile(r"^(BLEU|NIST|METEOR|ROUGE_L|CIDEr):\s*([0-9.]+)", re.M)

def read_score(prefix, lr, seed):
    tag = f"{prefix}_{lr:g}_svalues_256_svectors_0_seed_{seed}_init_sigma_0.1_init_scaler_0.1"
    path = root / tag / "scores.txt"
    if not path.exists():
        raise FileNotFoundError(str(path))
    metrics = {k: float(v) for k, v in metric_re.findall(path.read_text(errors="replace"))}
    return metrics

try:
    lr003 = [read_score("phase1_lr", 0.03, seed)["CIDEr"] for seed in seeds]
    lr005 = [read_score("phase1_lr005", 0.05, seed)["CIDEr"] for seed in seeds]
except FileNotFoundError as exc:
    print(f"missing {exc}", file=sys.stderr)
    sys.exit(2)

mean003 = statistics.mean(lr003)
mean005 = statistics.mean(lr005)
print(f"lr=0.03 mean_CIDEr={mean003:.4f} seeds={lr003}", file=sys.stderr)
print(f"lr=0.05 mean_CIDEr={mean005:.4f} seeds={lr005}", file=sys.stderr)
if mean005 > mean003:
    print("0.05")
else:
    print("0.03")
PY
)" 2>>"$SUMMARY"

echo "[$(date -u '+%F %T')] phase1 winner lr=${LR_WINNER}" | tee -a "$SUMMARY"
if [ "$LR_WINNER" != "0.05" ]; then
    echo "Skipping phase2: lr=0.05 did not beat lr=0.03 on matched seeds." | tee -a "$SUMMARY"
    exit 0
fi

echo "[$(date -u '+%F %T')] starting phase2 adapter-size sweep" | tee -a "$SUMMARY"
"$PYTHON" -u training_manager.py \
    --cuda-visible-devices 0 \
    --run-prefix phase2_size_lr005 \
    --learning-rates 0.05 \
    --num-svalues 256 \
    --num-svectors 30,60 \
    --seeds 1054,2021,31415 \
    > "$PHASE2_LOG" 2>&1

echo "[$(date -u '+%F %T')] phase2 finished" | tee -a "$SUMMARY"
"$PYTHON" - <<'PY' >> "$SUMMARY"
from pathlib import Path
import re
import statistics

root = Path("outputs/results")
seeds = [1054, 2021, 31415]
metric_re = re.compile(r"^(BLEU|NIST|METEOR|ROUGE_L|CIDEr):\s*([0-9.]+)", re.M)

def find_score(lr, svalues, svectors, seed):
    suffix = f"{lr:g}_svalues_{svalues}_svectors_{svectors}_seed_{seed}_init_sigma_0.1_init_scaler_0.1"
    candidates = sorted(root.glob(f"*lr*_{suffix}/scores.txt"))
    if not candidates:
        return None
    text = candidates[-1].read_text(errors="replace")
    metrics = {k: float(v) for k, v in metric_re.findall(text)}
    return metrics

rows = []
for svectors in [0, 30, 60, 90]:
    vals = []
    for seed in seeds:
        score = find_score(0.05, 256, svectors, seed)
        if score:
            vals.append(score)
    if vals:
        rows.append((svectors, vals))

print("")
print("Final phase2 size summary, lr=0.05, svalues=256:")
best = None
for svectors, vals in rows:
    cider = [v["CIDEr"] for v in vals]
    bleu = [v["BLEU"] for v in vals]
    line = (
        f"svectors={svectors:>3} n={len(vals)} "
        f"mean_CIDEr={statistics.mean(cider):.4f} "
        f"std_CIDEr={(statistics.stdev(cider) if len(cider) > 1 else 0):.4f} "
        f"mean_BLEU={statistics.mean(bleu):.4f}"
    )
    print(line)
    key = (statistics.mean(cider), statistics.mean(bleu))
    if best is None or key > best[0]:
        best = (key, svectors, line)

if best:
    print("")
    print(f"WINNER: lr=0.05, svalues=256, svectors={best[1]}")
    print(best[2])
PY
