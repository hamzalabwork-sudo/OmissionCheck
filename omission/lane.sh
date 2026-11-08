#!/usr/bin/env bash
# One lane of X8 jobs on one GPU of the shared server. A job is "kind setting model variant seed": kind full trains
# the monitor (if needed) and attacks it (x8b); kind thr re-runs the clean and insider scenarios of a monitor attacked
# before the validation-threshold columns existed (x8t). A job starts only when MemAvailable is at least GATE GB;
# jobs the watchdog kills are retried in later passes (training resumes from its checkpoint, an attack from its last
# finished scenario). usage: omission/lane.sh GPU GATE EPOCHS "kind setting model variant seed" ...
set -u
cd "$(dirname "$0")/.."
gpu=$1; gate=$2; epochs=$3; shift 3
export AML_WORKERS=${AML_WORKERS:-0}
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
wait_mem() { while [ "$(avail)" -lt "$gate" ]; do sleep 60; done; }
out() { if [ "$1" = thr ]; then echo "results/paper2/aml/x8t_$2.csv"; else echo "results/paper2/aml/x8b_$2.csv"; fi; }
for pass in $(seq 1 30); do
  left=0
  for job in "$@"; do
    read -r kind setting model variant seed <<< "$job"
    name="${setting}_${model}_${variant}_s${seed}"
    [ -f "$(out "$kind" "$name")" ] && continue
    left=1
    if [ "$kind" = full ]; then
      wait_mem; omission/queue.sh train "$gpu" "$epochs" "$setting $model $variant $seed"
      wait_mem; omission/queue.sh attack "$gpu" "$epochs" "$setting $model $variant $seed"
    else
      wait_mem; omission/queue.sh thr "$gpu" "$epochs" "$setting $model $variant $seed"
    fi
    [ -f "$(out "$kind" "$name")" ] || sleep 120   # killed or locked: let memory recover
  done
  [ "$left" = 0 ] && break
done
echo "$(date '+%F %T') lane on GPU $gpu done after pass $pass" >> logs/queues.out
