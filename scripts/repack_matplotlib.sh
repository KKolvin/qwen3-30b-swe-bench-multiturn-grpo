#!/usr/bin/env bash
# Repack every matplotlib SWE-bench image so it loads under rootless docker
# (see scripts/repack_image.py). Idempotent: skips images already present.
# Sequential on purpose — gentle on the throttled mirror and the /data1 scratch.
set -u
export DOCKER_HOST="unix:///run/user/$(id -u)/docker.sock"

LIST=/home/kewen.liu/agentic/logs/verified_all_images.txt
SUMMARY=/home/kewen.liu/agentic/logs/repack_matplotlib_summary.txt
MIN_FREE_GB=80          # /data1 floor; matches prepull's guard
: > "$SUMMARY"

mapfile -t IMGS < <(grep matplotlib "$LIST")
total=${#IMGS[@]}
i=0; ok=0; skip=0; fail=0
for name in "${IMGS[@]}"; do
  i=$((i+1))
  tag="swebench/${name}:latest"
  if docker image inspect "$tag" >/dev/null 2>&1; then
    echo "[$i/$total] SKIP (present)  $name"
    echo "SKIP  $name" >> "$SUMMARY"; skip=$((skip+1)); continue
  fi
  free=$(df --output=avail -BG /data1 | tail -1 | tr -dc '0-9')
  if [ "${free:-0}" -lt "$MIN_FREE_GB" ]; then
    echo "[$i/$total] STOP: /data1 free ${free}GB < ${MIN_FREE_GB}GB floor"
    echo "STOP-DISK at $name (free ${free}GB)" >> "$SUMMARY"; break
  fi
  echo "[$i/$total] repack  $name  (/data1 free ${free}GB)"
  if python3 /home/kewen.liu/agentic/scripts/repack_image.py "swebench/${name}"; then
    echo "OK    $name" >> "$SUMMARY"; ok=$((ok+1))
  else
    echo "FAIL  $name" >> "$SUMMARY"; fail=$((fail+1))
  fi
done
echo "[repack-mpl] DONE  ok=$ok skip=$skip fail=$fail  (of $total)"
echo "DONE ok=$ok skip=$skip fail=$fail total=$total" >> "$SUMMARY"
