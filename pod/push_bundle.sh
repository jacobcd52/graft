#!/usr/bin/env bash
# Upload the code and graft bundles and Modal credentials to a pod, then unpack the graft bundle.
# Uploads go through pod/xfer_lib.sh (40 MB checksummed parts). Idempotent.
# Usage: RUNPOD_KEY_FILE=... BUNDLE_DIR=... MODAL_ENV=... bash pod/push_bundle.sh POD
set -euo pipefail
cd "$(dirname "$0")/.."
POD=$1
source pod/xfer_lib.sh
if [[ -z ${SKIP_CODE:-} ]]; then  # SKIP_CODE=1: graft bundle only (code already on the pod)
  T=$(mktemp -d)
  cat "$BUNDLE_DIR"/code_bundle.tgz.part* > "$T/code_bundle.tgz"
  send "$T/code_bundle.tgz" /workspace/code_bundle.tgz.part00   # setup.sh reads code_bundle.tgz.part*
  rm -rf "$T"
  $R exec "$POD" "rm -f /workspace/code_bundle.tgz.part0[1-9]" --timeout 30 >/dev/null
fi
send "$BUNDLE_DIR/graft_bundle.tgz" /workspace/graft_bundle.tgz
put_checked "$MODAL_ENV" /workspace/.modal_env
$R exec "$POD" "cd /workspace && rm -rf graft && tar -xzf graft_bundle.tgz && rm graft_bundle.tgz && ls graft" --timeout 80
