#!/usr/bin/env bash
LOG="$1"
strip(){ sed -E 's/\x1b\[[0-9;]*m//g' "$LOG"; }
while true; do
  if ! pgrep -f "verl_entry.py" >/dev/null 2>&1; then
    cs="$(strip)"
    if echo "$cs" | grep -q "global_step_6"; then echo "RESULT=SUCCESS (reached global_step_6, process exited)"; 
    else echo "RESULT=DIED (process gone; last errors below)"; fi
    echo "---- last errors ----"
    echo "$cs" | grep -inE "Traceback|RuntimeError|CUDA error|has no attribute 'dim'|OutOfMemory|Permission denied|Killed|Error:" | tail -8
    echo "---- clean tail ----"
    echo "$cs" | grep -vE "^\s*$|RouterReplay|NPU not support|UserWarning|warnings.warn|file_system_monitor|Loading checkpoint shards" | tail -8
    exit 0
  fi
  cs="$(strip)"
  # early success: final checkpoint written
  if echo "$cs" | grep -q "global_step_6" && echo "$cs" | grep -qiE "saved|checkpoint.*global_step_6"; then
    echo "RESULT=SUCCESS (global_step_6 checkpoint written)"
    echo "$cs" | grep -iE "val-core|step:[0-9]" | tail -8
    exit 0
  fi
  sleep 30
done
