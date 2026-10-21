#!/usr/bin/env bash
# Relay one adapter checkpoint between pods through this machine (RunPod's proxy caps request
# bodies at ~100 MB, so the file travels in 90 MB parts and is checksummed on arrival).
# Usage: copy_ckpt.sh SRC_POD DST_POD <arm> <checkpoint-dir-name>
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=$1; DST=$2; ARM=$3; CK=$4
R="${PYTHON:-python3} -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
D=/workspace/ckpts/$ARM/$CK; T=/workspace/xfer/$ARM-$CK
TMP=$(mktemp -d)
sum=$($R exec "$SRC" "mkdir -p $T && cd $T && split -b 90M -d $D/adapter_state_dict.pt part && md5sum $D/adapter_state_dict.pt | cut -d' ' -f1 && ls part*" --timeout 80)
md5=$(echo "$sum" | head -1)
for p in $(echo "$sum" | tail -n +2); do
  for try in 1 2 3; do
    $R get "$SRC" "$T/$p" "$TMP/$p" >/dev/null && $R put "$DST" "$TMP/$p" "$T/$p" >/dev/null && break
    sleep 5
  done
done
$R exec "$DST" "mkdir -p $D && cat $T/part* > $D/adapter_state_dict.pt && rm -rf $T && test \$(md5sum $D/adapter_state_dict.pt | cut -d' ' -f1) = $md5 && echo ok $ARM/$CK" --timeout 80
$R exec "$SRC" "rm -rf $T" --timeout 30 >/dev/null
rm -rf "$TMP"
