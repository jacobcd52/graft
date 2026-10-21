#!/usr/bin/env bash
# Upload code + graft bundles and Modal credentials to a pod and unpack the graft bundle.
# RunPod's proxy drops large request bodies (a 74 MB PUT failed), so the graft bundle travels in
# 40 MB parts, each retried, and is checksummed after reassembly. Idempotent.
# Usage: RUNPOD_KEY_FILE=... BUNDLE_DIR=... MODAL_ENV=... bash pod/push_bundle.sh POD
set -euo pipefail
cd "$(dirname "$0")/.."
POD=$1
R="${PYTHON:-python3} -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
put() {  # local remote
  for try in 1 2 3 4 5; do
    $R put "$POD" "$1" "$2" >/dev/null 2>&1 && return 0
    echo "  put $(basename "$1") failed (try $try)"; sleep $((try * 5))
  done
  return 1
}
for f in "$BUNDLE_DIR"/code_bundle.tgz.part*; do put "$f" /workspace/$(basename "$f"); done
T=$(mktemp -d)
split -b 40M -d "$BUNDLE_DIR/graft_bundle.tgz" "$T/graft_bundle.tgz.part"
md5=$(md5sum "$BUNDLE_DIR/graft_bundle.tgz" | cut -d' ' -f1)
for f in "$T"/graft_bundle.tgz.part*; do put "$f" /workspace/xfer/$(basename "$f"); done
rm -rf "$T"
put "$MODAL_ENV" /workspace/.modal_env
$R exec "$POD" "cd /workspace && cat xfer/graft_bundle.tgz.part* > graft_bundle.tgz && rm -rf xfer \
  && test \$(md5sum graft_bundle.tgz | cut -d' ' -f1) = $md5 && rm -rf graft && tar -xzf graft_bundle.tgz \
  && rm graft_bundle.tgz && ls graft" --timeout 80
