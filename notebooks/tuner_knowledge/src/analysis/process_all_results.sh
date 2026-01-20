#!/bin/bash

python process_results.py
python ../results/mmlu/summarize_mmlu_results.py
python ../results/bigbench/parse_bigbench_results.py ../results/bigbench ../results/bigbench/summary.csv