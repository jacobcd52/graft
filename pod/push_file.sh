#!/usr/bin/env bash
# Upload one file to a pod in checksummed 40 MB parts (pod/xfer_lib.sh).
# Usage: RUNPOD_KEY_FILE=... bash pod/push_file.sh POD <local> <remote path>
set -euo pipefail
cd "$(dirname "$0")/.."
POD=$1
source pod/xfer_lib.sh
send "$2" "$3"
