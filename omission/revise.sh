#!/usr/bin/env bash
# Revision queue on one GPU: re-attack the trained monitors under the protocol (x8b), then train and attack
# further seeds. Logs the peak resident memory of the omission jobs every 10 s to logs/rss.out.
# usage: omission/revise.sh GPU
set -u
cd "$(dirname "$0")/.."
gpu=$1
mkdir -p logs
( while true; do
    kb=$(ps -u "$USER" -o rss=,args= | awk '/omission.x_aml/ && !/awk/ {s += $1} END {print s + 0}')
    echo "$(date '+%F %T') $kb" >> logs/rss.out
    sleep 10
  done ) &
sampler=$!
jobs=()
for s in 0 1 2; do jobs+=("views gin standard $s" "global gin standard $s"); done
for v in dropedge omission; do jobs+=("views gin $v 0" "global gin $v 0"); done
for s in 3 4; do jobs+=("views gin standard $s" "global gin standard $s"); done
for s in 1 2; do for v in dropedge omission; do jobs+=("views gin $v $s" "global gin $v $s"); done; done
omission/pipeline.sh "$gpu" 50 "${jobs[@]}"
kill "$sampler"
