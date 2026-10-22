#!/usr/bin/env bash
# Upload the run-3 narration pools and loss-eval sets (pipeline/narration_pools.py) to a pod
# launched by pod/launch_run3.sh, unpack them into /workspace/graft, and release run_all.sh.
# Usage: RUNPOD_KEY_FILE=... bash pod/push_run3_data.sh POD <run3_data.tgz>
set -euo pipefail
cd "$(dirname "$0")/.."
POD=$1
source pod/xfer_lib.sh
send "$2" /workspace/run3_data.tgz
$R exec "$POD" "cd /workspace && tar -xzf run3_data.tgz -C /workspace/graft && rm run3_data.tgz && ls /workspace/graft | grep -E 'surg|fren' && touch /workspace/go" --timeout 80
