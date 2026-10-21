#!/usr/bin/env bash
# Run 2: every arm retrained on the completion-fixed data (pipeline/fix_completion.py), 1 epoch,
# loss evals at 50% and 100%, agentic evals at the end (both adapters + RETAIN only).
# Three 4xH200 pods run their arms back to back (setup is paid once per pod).
# Usage: RUNPOD_KEY_FILE=... MODAL_ENV=... BUNDLE_DIR=<dir with code_bundle parts + graft_bundle.tgz> bash pod/launch_run2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="EPOCHS=1 SAVE_FRACS=0.5,1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run2"
declare -A PODS=(
  [graft-r2-a]="noint ceiling_20pct H1_A_only H6_tampers H9_redteam"
  [graft-r2-b]="T3_nonsecurity T5_python_only L1_short R1_random_tasks"
  [graft-r2-c]="M1_gemini D05_dose5pct D10_dose10pct N1_noisy25"
)
for name in "${!PODS[@]}"; do
  NAME=$name FALLBACK_MIN=0 RUN_ENV="$COMMON ARMS='${PODS[$name]}'" bash pod/launch_main.sh > "launch_$name.log" 2>&1 &
done
wait
grep -h LAUNCHED launch_graft-r2-*.log
