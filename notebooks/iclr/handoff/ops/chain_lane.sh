#!/bin/bash
# Chain a band lane behind a predecessor that already owns the GPU.
#
# Waits until (a) an optional predecessor lane-driver PID has exited, (b) an
# optional predecessor tmux session has exited, and (c) no session of ours is
# still bound to this GPU; then refuses if a colleague holds the GPU and
# otherwise hands off to band_lane.sh.
#
# The PID wait is what prevents double-booking: band_lane.sh sleeps between
# entries, so "no tmux session on this GPU" alone is true in the gap between a
# predecessor's entries and is NOT sufficient evidence the GPU is free.
#
# Usage: chain_lane.sh <gpu> <wait_pid|-> <wait_session|-> <entry_id> [<entry_id> ...]
set -uo pipefail
CAMP=/media/eimtest/data/guyb/UIOrthoLoRA/notebooks/iclr-campaign-20260914
S="$CAMP/notebooks/iclr/handoff/ops"
SOCK=iclr_6aa54397_20260914
GPU=${1:?gpu}
WPID=${2:?predecessor pid or -}
WSESS=${3:?predecessor session or -}
shift 3
[[ $# -ge 1 ]] || { echo "CHAIN ABORT: no entries given for gpu $GPU"; exit 2; }
cd "$CAMP" || exit 2

echo "CHAIN gpu$GPU: waiting on pid=$WPID session=$WSESS; queue: $*"

if [[ "$WPID" != "-" ]]; then
  while kill -0 "$WPID" 2>/dev/null; do sleep 60; done
  echo "CHAIN gpu$GPU: predecessor pid $WPID exited"
fi

if [[ "$WSESS" != "-" ]]; then
  while tmux -L "$SOCK" has-session -t "=$WSESS" 2>/dev/null; do sleep 60; done
  echo "CHAIN gpu$GPU: predecessor session $WSESS exited"
fi

while tmux -L "$SOCK" ls -F '#{session_name}' 2>/dev/null | grep -q "_gpu${GPU}\$"; do
  echo "CHAIN gpu$GPU: a session still occupies this GPU; waiting"
  sleep 60
done

# Never launch on a GPU a colleague is using (Goody / danielf share the box).
uuid=$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader)
if [[ -z "$uuid" ]]; then echo "CHAIN ABORT gpu$GPU: cannot read GPU uuid"; exit 1; fi
foreign=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
  | awk -F', ' -v u="$uuid" '$2 == u {print $1}' \
  | while read -r p; do [[ -n "$p" ]] && ps -o user= -p "$p" 2>/dev/null; done \
  | tr -d ' ' | grep -vx "$(id -un)" | head -1)
if [[ -n "$foreign" ]]; then
  echo "CHAIN ABORT gpu$GPU: foreign compute process (user $foreign) present; not launching"
  exit 1
fi

echo "CHAIN gpu$GPU: GPU free, starting lane"
exec bash "$S/band_lane.sh" "$GPU" "$@"
