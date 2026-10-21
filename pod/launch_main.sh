#!/usr/bin/env bash
# Acquire the main training pod (8x H200 or H200 NVL, falling back to 4x after FALLBACK_MIN minutes),
# upload the code/data bundles and Modal credentials, and start setup + run_all in the background.
# Usage: RUNPOD_KEY_FILE=... BUNDLE_DIR=... MODAL_ENV=... [NAME=graft-main] [RUN_ENV="ARMS='...' EPOCHS=1 ..."] bash pod/launch_main.sh
# RUN_ENV is prepended to run_all.sh's command line (its env settings: arms, stages, epochs, ...).
set -uo pipefail
cd "$(dirname "$0")/.."
R="python3 -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
[[ -n ${PYTHON:-} ]] && R="$PYTHON -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
FALLBACK_MIN=${FALLBACK_MIN:-30}
t0=$(date +%s); POD=""; N=8
while [[ -z $POD ]]; do
  el=$(( ($(date +%s) - t0) / 60 ))
  (( el >= FALLBACK_MIN )) && N=4
  for cloud in ${CLOUDS:-SECURE COMMUNITY}; do  # secure first: run 1's community hosts had broken CUDA
    if out=$($R create --name ${NAME:-graft-main} --gpu "${GPUS:-NVIDIA H200,NVIDIA H200 NVL}" --count $N --disk 60 --local-disk 250 --cloud $cloud 2>&1); then
      POD=$(echo "$out" | awk '/^pod /{print $2}'); echo "$(date +%T) got ${N}x H200 ($cloud): $out"; break
    fi
  done
  [[ -z $POD ]] && { echo "$(date +%T) no ${N}x H200 yet (${el} min)"; sleep 120; }
done
bash pod/push_bundle.sh $POD || { echo "ERROR: bundle upload to $POD failed"; exit 1; }
$R spawn $POD main "bash /workspace/graft/pod/setup.sh && NPROC=$N ${RUN_ENV:-} bash /workspace/graft/pod/run_all.sh"
echo "LAUNCHED pod=$POD gpus=$N"
