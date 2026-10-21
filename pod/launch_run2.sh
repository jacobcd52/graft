#!/usr/bin/env bash
# Run 2: every arm retrained on the completion-fixed data (pipeline/fix_completion.py), 1 epoch,
# loss evals at 50% and 100%, agentic evals at the end (both adapters + RETAIN only).
# Three 4xH200 pods run their arms back to back (setup is paid once per pod); pod c runs the
# commands-only format arms, whose agentic evals are prefilled (commands / reasoned).
# Usage: [ONLY="graft-r2-a ..."] RUNPOD_KEY_FILE=... MODAL_ENV=... BUNDLE_DIR=<dir with code_bundle parts + graft_bundle.tgz> bash pod/launch_run2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="EPOCHS=1 SAVE_FRACS=0.5,1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run2"
declare -A PODS=(
  [graft-r2-a]="noint ceiling_20pct A_make_it_up L1_short V1_severe_to_mild"
  [graft-r2-b]="H9_redteam T3_nonsecurity T5_python_only R1_random_tasks K1_code_to_ops"
  [graft-r2-c]="F_noint F_ceiling F_reasoned D1_sysadmin_to_science"
)
for name in ${ONLY:-"${!PODS[@]}"}; do  # ONLY="graft-r2-a ..." launches a subset
  NAME=$name FALLBACK_MIN=0 RUN_ENV="$COMMON ARMS='${PODS[$name]}'" bash pod/launch_main.sh > "launch_$name.log" 2>&1 &
done
wait
grep -h LAUNCHED launch_graft-r2-*.log
