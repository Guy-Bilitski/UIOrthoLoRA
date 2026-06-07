#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
STAMP="${2:-$(date -u +%Y%m%d_%H%M)}"

PYTHON="${PYTHON:-python}"
LOG_DIR="outputs/logs"
SUMMARY="$LOG_DIR/e2e_phase2_lr003_${STAMP}_summary.txt"
PHASE2_LOG="$LOG_DIR/e2e_phase2_size_lr003_${STAMP}.log"

mkdir -p "$LOG_DIR"

if [ -n "$WAIT_PID" ]; then
    echo "[$(date -u '+%F %T')] waiting for PID ${WAIT_PID}" | tee -a "$SUMMARY"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
        sleep 300
    done
fi

echo "[$(date -u '+%F %T')] waiting for GPU to become idle" | tee -a "$SUMMARY"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; do
    sleep 300
done

echo "[$(date -u '+%F %T')] starting phase2 adapter-size sweep at lr=0.03" | tee -a "$SUMMARY"
"$PYTHON" -u training_manager.py \
    --cuda-visible-devices 0 \
    --run-prefix phase2_size_lr003 \
    --learning-rates 0.03 \
    --num-svalues 256 \
    --num-svectors 15,30,60,90 \
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
    metrics = {
        k: float(v)
        for k, v in metric_re.findall(candidates[-1].read_text(errors="replace"))
    }
    return metrics

rows = []
for svectors in [0, 15, 30, 60, 90]:
    vals = []
    for seed in seeds:
        score = find_score(0.03, 256, svectors, seed)
        if score:
            vals.append(score)
    if vals:
        rows.append((svectors, vals))

print("")
print("Final phase2 size summary, lr=0.03, svalues=256:")
best = None
for svectors, vals in rows:
    cider = [v["CIDEr"] for v in vals]
    bleu = [v["BLEU"] for v in vals]
    meteor = [v["METEOR"] for v in vals]
    line = (
        f"svectors={svectors:>3} n={len(vals)} "
        f"mean_CIDEr={statistics.mean(cider):.4f} "
        f"std_CIDEr={(statistics.stdev(cider) if len(cider) > 1 else 0):.4f} "
        f"mean_BLEU={statistics.mean(bleu):.4f} "
        f"mean_METEOR={statistics.mean(meteor):.4f}"
    )
    print(line)
    key = (statistics.mean(cider), statistics.mean(bleu), statistics.mean(meteor))
    if best is None or key > best[0]:
        best = (key, svectors, line)

if best:
    print("")
    print(f"WINNER: lr=0.03, svalues=256, svectors={best[1]}")
    print(best[2])
PY
