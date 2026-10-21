#!/usr/bin/env bash
# Upload the code and graft bundles and Modal credentials to a pod, then unpack the graft bundle.
# RunPod's proxy can cut a large request body short while the pod still reports success (run 2: a
# 63 MB part arrived as 44 MB; a 74 MB PUT failed), so every file travels in 40 MB parts, each part's
# checksum is verified on the pod, and a mismatched part is resent. Idempotent.
# Usage: RUNPOD_KEY_FILE=... BUNDLE_DIR=... MODAL_ENV=... bash pod/push_bundle.sh POD
set -euo pipefail
cd "$(dirname "$0")/.."
POD=$1
R="${PYTHON:-python3} -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
put_checked() {  # local remote: upload until the pod's md5 matches
  local want; want=$(md5sum "$1" | cut -d' ' -f1)
  for try in 1 2 3 4 5 6; do
    $R put "$POD" "$1" "$2" >/dev/null 2>&1 || true
    got=$($R exec "$POD" "md5sum $2 2>/dev/null | cut -d' ' -f1" --timeout 60 2>/dev/null || true)
    [[ $got == "$want" ]] && return 0
    echo "  $(basename "$2"): checksum mismatch (try $try)"; sleep $((try * 5))
  done
  echo "ERROR: could not upload $1"; return 1
}
send() {  # local remote-path: split into 40 MB parts, upload each checked, reassemble and verify
  local T md5; T=$(mktemp -d); md5=$(md5sum "$1" | cut -d' ' -f1)
  split -b 40M -d "$1" "$T/p"
  for f in "$T"/p*; do put_checked "$f" "/workspace/xfer/$(basename "$2").$(basename "$f")"; done
  rm -rf "$T"
  $R exec "$POD" "cat /workspace/xfer/$(basename "$2").p* > $2 && rm -f /workspace/xfer/$(basename "$2").p* \
    && test \$(md5sum $2 | cut -d' ' -f1) = $md5 && echo ok $2" --timeout 80
}
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
