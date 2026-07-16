#!/bin/bash
# Wait for the 8 freed nodes to finish staging, deploy fixed train_cs.py, then launch the
# final 8 DeepSeek cells (dora s42/s44, clora s43/s44, sclora s43/s44, lora_null s43/s44)
# to complete 3 seeds x 7 methods. Runs unattended (background).
cd "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting" || exit 1
NODES="d015 d024 d020 d007 d023 d017 d006 d032"

echo "[launch8] waiting for staging to complete ..."
for i in $(seq 1 240); do
  [ "$(pgrep -fc stage_node.sh)" -eq 0 ] && break
  sleep 30
done
echo "[launch8] stage_node procs remaining: $(pgrep -fc stage_node.sh)"

# node -> "method lr run_name flags"
launch_one() {
  local node="$1" method="$2" lr="$3" run="$4"; shift 4; local flags="$*"
  # deploy fixed train_cs.py (stage_node does not copy it)
  scp -q -o BatchMode=yes train_cs.py "ubuntu@$node:/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting/train_cs.py"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "ubuntu@$node" \
    "cd '/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting' && rm -rf __pycache__ scripts/deepseek/__pycache__ && ( setsid nohup bash scripts/deepseek/run_node.sh $method $lr $run $flags >logs/dsv4_${run}.log 2>&1 </dev/null & )"
  echo "[launch8] launched $node: $method $run $flags"
}

launch_one d015 dora      2e-4 dsv4_dora_r16_lr2e4_s44      --use_dora 1 --seed 44
launch_one d024 dora      2e-4 dsv4_dora_r16_lr2e4_s42      --use_dora 1
launch_one d020 clora     3e-4 dsv4_clora_r16_lr3e4_s43     --seed 43
launch_one d007 clora     3e-4 dsv4_clora_r16_lr3e4_s44     --seed 44
launch_one d023 sclora    5e-5 dsv4_sclora_r16_lr5e5_s43    --sclora 1 --seed 43
launch_one d017 sclora    5e-5 dsv4_sclora_r16_lr5e5_s44    --sclora 1 --seed 44
launch_one d006 lora_null 5e-4 dsv4_lora_null_r16_lr5e4_s43 --lora_null 1 --seed 43
launch_one d032 lora_null 5e-4 dsv4_lora_null_r16_lr5e4_s44 --lora_null 1 --seed 44
echo "[launch8] DONE — all 8 launched"
