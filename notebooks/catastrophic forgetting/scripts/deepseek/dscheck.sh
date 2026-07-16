#!/bin/bash
# Clean per-node DeepSeek status: train proc alive + live run_name + last progress/error line.
for n in d012 d003 d009 d002 d005 d010 d016 d018 d019 d008 d013 d014 d029; do
  ssh -o BatchMode=yes -o ConnectTimeout=6 ubuntu@"$n" bash -s <<'EOF' 2>/dev/null
cd "/home/guyb/UIOrthoLoRA/notebooks/catastrophic forgetting" 2>/dev/null || exit 0
hn=$(hostname); t=$(pgrep -fc train_deepseek.py); rn=""
for p in $(pgrep -f train_deepseek.py); do
  rn=$(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | grep -oP -- '--run_name \K\S+' | head -1)
done
last=""
[ -n "$rn" ] && last=$(tail -n2 "logs/dsv4_${rn}.log" 2>/dev/null | tr '\r' '\n' | \
  grep -oE '[0-9]+/1875 \[[0-9:]+<[0-9:]+|Traceback|RuntimeError|cuda:0 and cuda:1|conversion of the weights|registered [0-9]+|err=[0-9.e+-]+|maxerr' | tail -1)
printf '  %s: train=%s run=%s | %s\n' "$hn" "$t" "${rn#dsv4_}" "$last"
EOF
done
