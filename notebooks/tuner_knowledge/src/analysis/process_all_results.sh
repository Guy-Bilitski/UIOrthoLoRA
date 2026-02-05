#!/bin/bash

# Process intrinsic evaluation results (knowledge shift analysis)
python process_results.py

# Create output directories
mkdir -p adapters_results/triviaqa
mkdir -p adapters_results/hotpotqa

# Process MMLU results for each dataset
echo "=== Processing MMLU results ==="

# triviaqa
python ../results/mmlu/summarize_mmlu_results.py \
    ../results/mmlu/triviaqa \
    adapters_results/triviaqa/mmlu_summary.csv

# hotpotqa
python ../results/mmlu/summarize_mmlu_results.py \
    ../results/mmlu/hotpotqa \
    adapters_results/hotpotqa/mmlu_summary.csv

# Process BigBench results for each dataset
echo "=== Processing BigBench results ==="

# triviaqa
python ../results/bigbench/parse_bigbench_results.py \
    ../results/bigbench/triviaqa \
    adapters_results/triviaqa/bigbench_summary.csv

# hotpotqa
python ../results/bigbench/parse_bigbench_results.py \
    ../results/bigbench/hotpotqa \
    adapters_results/hotpotqa/bigbench_summary.csv

echo "All results processed!"