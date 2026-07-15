#!/bin/bash
# Start d001's OWN dispatcher + watchdog on its local 8 GPUs (no ssh; runs as root here).
# Uses the same detached pattern as the fleet. Shard: jobs/fleet/d001.txt.
set -u
cd "$(dirname "$0")/.."
export HF_HOME=/scratch/hf_cache HF_HUB_DISABLE_XET=1
V=/home/guy/UIOrthoLoRA/.venv/bin/python
[ -s jobs/fleet/d001.txt ] || { echo "no jobs/fleet/d001.txt"; exit 1; }
setsid nice -n 5 $V auto_dispatch.py --jobs jobs/fleet/d001.txt --gpus 0,1,2,3,4,5,6,7 --tag disp --hf_offline 1 > logs/disp.log 2>&1 < /dev/null &
echo "d001 dispatcher started ($(grep -c . jobs/fleet/d001.txt) cells)"
