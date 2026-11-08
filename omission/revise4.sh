#!/usr/bin/env bash
# Revision queue 4: revise3 capped at three seeds (0-2) for Multi-PNA+EU and for the hardened monitors. Each lane first
# waits for the job revise3 left running on its GPU, then works through its list; the hardened monitors of the global
# (pooled) setting, which the manuscript reports, come before those of the views setting. Jobs listed in both lanes
# run once (queue.sh holds a lock per monitor). usage: omission/revise4.sh A|B PID-TO-WAIT-FOR
set -u
cd "$(dirname "$0")/.."
while kill -0 "$2" 2>/dev/null; do sleep 60; done
hardened=()
for setting in global views; do
  for s in 1 2; do for v in dropedge omission; do hardened+=("full $setting gin $v $s"); done; done
done
if [ "$1" = A ]; then
  jobs=("full views pna standard 2" "full views gin standard 4")
  for s in 0 1; do jobs+=("thr views gin standard $s"); done
  for s in 0 1 2; do jobs+=("thr global gin standard $s"); done
  for v in dropedge omission; do jobs+=("thr global gin $v 0" "thr views gin $v 0"); done
  jobs+=("${hardened[@]}" "full views pna standard 0" "full global pna standard 0")
  omission/lane.sh 0 8 50 "${jobs[@]}"
else
  jobs=("full views pna standard 1" "full views pna standard 0" "full global pna standard 0" "${hardened[@]}")
  omission/lane.sh 1 14 50 "${jobs[@]}"
fi
