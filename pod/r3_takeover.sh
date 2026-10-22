#!/usr/bin/env bash
# Run 3 on a run-2 pod (no new 4xH200 capacity): once the pod's in-flight run-2 step ends, train +
# loss + agentic-eval two narration arms (pod/launch_run3.sh's settings), one after the other, then
# resume the pod's remaining run-2 work (pod/r3_rest_<pod>.sh). Touch /workspace/skip_second to drop
# the second arm (e.g. it moved to a new pod). Needs pod/push_run3_data.sh's files; it installs the
# run-3 run_all.sh (staged as run_all.sh.r3) only here, between runs, never under a running copy.
# Usage: bash r3_takeover.sh FIRST_ARM SECOND_ARM <rest script>
set -uo pipefail
FIRST=$1; SECOND=$2; REST=$3
G=/workspace/graft; ST=/workspace/state
while pgrep -f "[r]un_all.sh|[p]robe_loss.sh|[h]eldout_loss.sh|[r]erun_killed.sh" >/dev/null; do sleep 20; done
until [[ -f /workspace/go ]]; do sleep 20; done  # run-3 data uploaded
cp $G/pod/run_all.sh.r3 $G/pod/run_all.sh
[[ -f /workspace/arms.txt ]] && mv /workspace/arms.txt /workspace/arms_run2.txt  # run_all.sh prefers arms.txt
R3="NPROC=4 EPOCHS=1 SAVE_FRACS=1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run3 LOSS_MODES=retain_only LOSS_SKIP_BASE=1 BOTH_ARMS=none"
for ARM in $FIRST $SECOND; do
  [[ $ARM == "$SECOND" && -f /workspace/skip_second ]] && { echo "=== skip $ARM ==="; continue; }
  rm -f $ST/sync.done
  echo "=== $(date '+%F %T') run 3: $ARM ==="
  env $R3 ARMS=$ARM bash $G/pod/run_all.sh
done
[[ -f /workspace/arms_run2.txt ]] && mv /workspace/arms_run2.txt /workspace/arms.txt
rm -f $ST/sync.done
echo "=== $(date '+%F %T') run 3 done; resuming run 2 ==="
bash $G/pod/$REST
