#!/usr/bin/env bash
# Run X8 jobs one after another on one GPU. PHASE=train trains the models; PHASE=attack evaluates
# the attacks on trained models (24 GB peak RSS, so never alongside training on the shared server).
# A job holds logs/NAME.lock (flock) while it runs, so parallel pipelines never run the same job; the lock is
# released when the process dies.
# usage: omission/queue.sh PHASE GPU EPOCHS "setting model variant seed" ...
set -u
cd "$(dirname "$0")/.."
source ../venv/bin/activate
phase=$1; gpu=$2; epochs=$3; shift 3
mkdir -p logs
for job in "$@"; do
  read -r setting model variant seed <<< "$job"
  name="${setting}_${model}_${variant}_s${seed}"
  args=(--setting "$setting" --model "$model" --variant "$variant" --seed "$seed" --gpu "$gpu")
  if [ "$phase" = train ] && [ ! -f "results/paper2/aml/models/${name}.pt" ]; then
    flock -n "logs/${name}.lock" nice -n 5 python -W ignore -m omission.x_aml train "${args[@]}" --epochs "$epochs"       >> "logs/${name}.out" 2>&1
  elif [ "$phase" = attack ] && [ -f "results/paper2/aml/models/${name}.pt" ] && [ ! -f "results/paper2/aml/x8_${name}.csv" ]; then
    flock -n "logs/${name}.lock" nice -n 5 python -W ignore -m omission.x_aml attack "${args[@]}"       >> "logs/${name}.out" 2>&1
  fi
done
echo "$(date '+%F %T') $phase queue on GPU $gpu done" >> logs/queues.out
