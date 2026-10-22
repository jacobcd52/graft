#!/usr/bin/env bash
# Acquire a 4x H200 pod for evals only, upload the bundles and Modal credentials, and start
# setup.sh + run_evals.sh on one eval plan (checkpoints are pulled from the Modal volume).
# Usage: RUNPOD_KEY_FILE=... BUNDLE_DIR=... MODAL_ENV=... bash pod/launch_evals.sh <name> <plan, relative to graft/>
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=$1; PLAN=$2
R="python3 -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
[[ -n ${PYTHON:-} ]] && R="$PYTHON -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
t0=$(date +%s); POD=""
while [[ -z $POD ]]; do
  for cloud in ${CLOUDS:-SECURE COMMUNITY}; do
    if out=$($R create --name $NAME --gpu "${GPUS:-NVIDIA H200,NVIDIA H200 NVL}" --count 4 --disk 60 --local-disk 250 --cloud $cloud 2>&1); then
      POD=$(echo "$out" | awk '/^pod /{print $2}'); echo "$(date +%T) got 4x H200 ($cloud): $out"; break
    fi
  done
  [[ -z $POD ]] && { echo "$(date +%T) no 4x H200 yet ($(( ($(date +%s) - t0) / 60 )) min)"; sleep 120; }
done
bash pod/push_bundle.sh $POD || { echo "ERROR: bundle upload to $POD failed"; exit 1; }
$R spawn $POD main "bash /workspace/graft/pod/setup.sh && bash /workspace/graft/pod/run_evals.sh /workspace/graft/$PLAN"
echo "LAUNCHED pod=$POD plan=$PLAN"
