#!/usr/bin/env bash
# One job at a time on a shared server: each argument "setting model variant seed" is trained (if needed)
# and then attacked. Jobs the memory watchdog kills are retried in later passes; training resumes from its
# per-epoch checkpoint. usage: omission/pipeline.sh GPU EPOCHS "setting model variant seed" ...
set -u
cd "$(dirname "$0")/.."
gpu=$1; epochs=$2; shift 2
export AML_WORKERS=${AML_WORKERS:-0}
for pass in 1 2 3 4 5 6 7 8; do
  left=0
  for job in "$@"; do
    read -r setting model variant seed <<< "$job"
    name="${setting}_${model}_${variant}_s${seed}"
    [ -f "results/paper2/aml/x8_${name}.csv" ] && continue
    left=1
    omission/queue.sh train "$gpu" "$epochs" "$job"
    omission/queue.sh attack "$gpu" "$epochs" "$job"
    [ -f "results/paper2/aml/x8_${name}.csv" ] || sleep 120   # killed: let memory recover
  done
  [ "$left" = 0 ] && break
done
echo "$(date '+%F %T') pipeline on GPU $gpu done after pass $pass" >> logs/queues.out
