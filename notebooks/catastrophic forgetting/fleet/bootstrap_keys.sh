#!/bin/bash
# One-time SSH key bootstrap for the fleet.
# Usage: SSHPASS='<password>' bash fleet/bootstrap_keys.sh
# Reads password ONLY from the SSHPASS env var (never stored on disk / in git).
# Installs d001's pubkey into ubuntu@<node>:~/.ssh/authorized_keys, then verifies key auth.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
NODES=$(grep -v '^#' "$HERE/nodes.txt")
USER=ubuntu
PUB=$(cat ~/.ssh/id_ed25519.pub)
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=$HOME/.ssh/known_hosts -o ConnectTimeout=8"

: "${SSHPASS:?set SSHPASS env with the node password}"
export SSHPASS

ok=0; fail=""
for n in $NODES; do
  # Append key if absent (idempotent), create ~/.ssh if needed.
  sshpass -e ssh $SSH_OPTS "$USER@$n" \
    "mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && grep -qF '$PUB' ~/.ssh/authorized_keys || echo '$PUB' >> ~/.ssh/authorized_keys" \
    >/dev/null 2>&1
  # Verify passwordless key auth (BatchMode disables password fallback).
  if ssh -o BatchMode=yes $SSH_OPTS "$USER@$n" 'echo ok' >/dev/null 2>&1; then
    ok=$((ok+1)); echo "  [ok]   $n"
  else
    fail="$fail $n"; echo "  [FAIL] $n"
  fi
done
echo "key auth OK on $ok/$(echo "$NODES" | wc -w) nodes"
[ -n "$fail" ] && echo "FAILED:$fail" && exit 1
exit 0
