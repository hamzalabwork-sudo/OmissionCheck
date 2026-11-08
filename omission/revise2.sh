#!/usr/bin/env bash
# Reordered revision queue: waits for the running attack, then re-attacks the monitors whose earlier attack was killed
# by the memory watchdog before training further seeds. usage: omission/revise2.sh GPU
set -u
cd "$(dirname "$0")/.."
gpu=$1
while pgrep -u "$USER" -f "omission.x_aml" > /dev/null; do sleep 30; done
jobs=()
for s in 0 2; do jobs+=("views gin standard $s" "global gin standard $s"); done
for v in dropedge omission; do jobs+=("views gin $v 0" "global gin $v 0"); done
for s in 3 4; do jobs+=("views gin standard $s" "global gin standard $s"); done
for s in 1 2; do for v in dropedge omission; do jobs+=("views gin $v $s" "global gin $v $s"); done; done
omission/pipeline.sh "$gpu" 50 "${jobs[@]}"
