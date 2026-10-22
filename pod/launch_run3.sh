#!/usr/bin/env bash
# Run 3: narration retrain arms (pipeline/narration_pools.py). The format arms' modified half is
# rewritten (surgery / french) instead of stripped; per variant a no-intervention arm, a 20%
# stratified ceiling and an arm routed only among the unmodified half (F_reasoned's list).
# One arm per 4xH200 pod so all six finish together. Pods wait after setup for /workspace/go
# (touched once the run-3 pools and loss sets are uploaded: pod/push_run3_data.sh).
# Loss evals: final checkpoint only, RETAIN only, no base model (transfer.py needs nothing else).
# Usage: [ONLY="graft-r3-surg-noint ..."] RUNPOD_KEY_FILE=... MODAL_ENV=... BUNDLE_DIR=... bash pod/launch_run3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="EPOCHS=1 SAVE_FRACS=1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run3 LOSS_MODES=retain_only LOSS_SKIP_BASE=1 BOTH_ARMS=none"
declare -A PODS=(
  [graft-r3-surg-noint]=SURG_noint [graft-r3-surg-ceiling]=SURG_ceiling [graft-r3-surg-routed]=SURG_routed
  [graft-r3-fren-noint]=FREN_noint [graft-r3-fren-ceiling]=FREN_ceiling [graft-r3-fren-routed]=FREN_routed
)
for name in ${ONLY:-"${!PODS[@]}"}; do
  NAME=$name FALLBACK_MIN=0 PRE_RUN='while [ ! -f /workspace/go ]; do sleep 30; done' \
    RUN_ENV="$COMMON ARMS='${PODS[$name]}'" bash pod/launch_main.sh > "launch_$name.log" 2>&1 &
done
wait
grep -h LAUNCHED launch_graft-r3-*.log
