#!/usr/bin/env bash
# Shared-server guard: kill this user's omission jobs if free memory falls below a floor (GB).
# A new watchdog replaces the previous one (pid file), so no pkill pattern can match the launching shell.
floor=${1:-16}
logs="$(cd "$(dirname "$0")/.." && pwd)/logs"
mkdir -p "$logs"
[ -f "$logs/watchdog.pid" ] && kill "$(cat "$logs/watchdog.pid")" 2>/dev/null
echo $$ > "$logs/watchdog.pid"
while true; do
  avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
  if [ "$avail" -lt "$floor" ]; then
    echo "$(date '+%F %T') MemAvailable ${avail}G < ${floor}G: killing omission jobs" >> "$logs/watchdog.out"
    pkill -u "$USER" -f "omission.x_aml"; pkill -u "$USER" -f "omission/queue.sh"
  fi
  sleep 5
done
