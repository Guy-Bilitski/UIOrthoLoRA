#!/bin/bash
set -e

###############################################################################
# process_all_results.sh
#
# Runs all analysis scripts and saves everything under ./adapters_results/
#
# Output structure:
#   adapters_results/
#     hotpotqa/
#       threshold_0.6/
#         gemma-12b.csv
#         llama-3b.csv
#       threshold_0.8/
#         gemma-12b.csv
#         llama-3b.csv
#       mmlu_summary.csv
#       bigbench_summary.csv
#     triviaqa/
#       ... (same structure)
###############################################################################

DATASETS=("hotpotqa" "triviaqa")

# ── 1. Intrinsic evaluation (knowledge shift analysis) ──────────────────────
echo "=== Processing intrinsic evaluation results ==="
python process_results.py

# ── 2. MMLU results ─────────────────────────────────────────────────────────
echo ""
echo "=== Processing MMLU results ==="
for DATASET in "${DATASETS[@]}"; do
    MMLU_INPUT="../results/mmlu/${DATASET}"
    MMLU_OUTPUT="adapters_results/${DATASET}/mmlu_summary.csv"

    if [ -d "$MMLU_INPUT" ]; then
        echo "  ${DATASET}: ${MMLU_INPUT} -> ${MMLU_OUTPUT}"
        mkdir -p "adapters_results/${DATASET}"
        python ../results/mmlu/summarize_mmlu_results.py \
            "$MMLU_INPUT" \
            "$MMLU_OUTPUT"
    else
        echo "  ${DATASET}: no MMLU results found at ${MMLU_INPUT}"
    fi
done

# ── 3. BigBench results ─────────────────────────────────────────────────────
echo ""
echo "=== Processing BigBench results ==="
for DATASET in "${DATASETS[@]}"; do
    BB_INPUT="../results/bigbench/${DATASET}"
    BB_OUTPUT="adapters_results/${DATASET}/bigbench_summary.csv"

    if [ -d "$BB_INPUT" ]; then
        echo "  ${DATASET}: ${BB_INPUT} -> ${BB_OUTPUT}"
        mkdir -p "adapters_results/${DATASET}"
        python ../results/bigbench/parse_bigbench_results.py \
            "$BB_INPUT" \
            "$BB_OUTPUT"
    else
        echo "  ${DATASET}: no BigBench results found at ${BB_INPUT}"
    fi
done

echo ""
echo "=== All results processed! ==="
echo "Output in ./adapters_results/"