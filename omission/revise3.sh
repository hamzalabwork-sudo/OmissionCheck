#!/usr/bin/env bash
# Revision queue 3, in two lanes. Lane A (GPU 0): five seeds of the standard Multi-GIN+EU monitors first, then the
# validation-threshold re-runs of monitors attacked earlier, then the hardened monitors. Lane B (GPU 1, starts a job
# only with 14 GB free): Multi-PNA+EU monitors, seeds 0-4. usage: omission/revise3.sh A|B
set -u
cd "$(dirname "$0")/.."
if [ "$1" = A ]; then
  # the training of global s3 started by revise2 is still running: wait until it releases its lock
  until flock -n logs/global_gin_standard_s3.lock true; do sleep 30; done
  jobs=("full global gin standard 3" "full views gin standard 2" "full views gin standard 3"
        "full global gin standard 4" "full views gin standard 4")
  # the second detector, three seeds, also here: lane B only starts when other users leave memory free
  for s in 0 1 2; do jobs+=("full global pna standard $s" "full views pna standard $s"); done
  for s in 0 1; do jobs+=("thr views gin standard $s"); done
  for s in 0 1 2; do jobs+=("thr global gin standard $s"); done
  for s in 1 2 3 4; do for v in dropedge omission; do jobs+=("full views gin $v $s" "full global gin $v $s"); done; done
  for v in dropedge omission; do jobs+=("thr views gin $v 0" "thr global gin $v 0"); done
  for s in 3 4; do jobs+=("full global pna standard $s" "full views pna standard $s"); done
  omission/lane.sh 0 8 50 "${jobs[@]}"
else
  jobs=()
  for s in 0 1 2 3 4; do jobs+=("full global pna standard $s" "full views pna standard $s"); done
  omission/lane.sh 1 14 50 "${jobs[@]}"
fi
