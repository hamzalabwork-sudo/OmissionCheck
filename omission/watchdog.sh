#!/usr/bin/env bash
# Shared-server guard: if free memory falls below a floor (GB), kill this user's youngest omission job, wait for the
# memory to settle, and check again (so an older job survives when stopping the newest is enough).
# A new watchdog replaces the previous one (pid file), so no pkill pattern can match the launching shell.
floor=${1:-16}
logs="$(cd "$(dirname "$0")/.." && pwd)/logs"
mkdir -p "$logs"
[ -f "$logs/watchdog.pid" ] && kill "$(cat "$logs/watchdog.pid")" 2>/dev/null
echo $$ > "$logs/watchdog.pid"
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt "$floor" ]; then
    pid=$(pgrep -n -u "$USER" -f "python -W ignore -m omission.x_aml")
    if [ -n "$pid" ]; then
      echo "$(date '+%F %T') MemAvailable ${avail}G < ${floor}G: killing youngest job $pid ($(ps -o args= -p "$pid" | cut -c40-140))" >> "$logs/watchdog.out"
      kill "$pid"
      sleep 30
    fi
  fi
  sleep 5
done
