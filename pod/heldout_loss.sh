#!/usr/bin/env bash
# Extra loss pass after the main pipeline: RETAIN-only NLL of every held-out-task hack (never trained
# on) at each arm's final checkpoint, for the held-out column of pipeline/transfer.py. Waits for
# run_all.sh to finish, then writes /workspace/results/heldout_losses_<arm>.jsonl and syncs results.
# Format arms (F_*) use the both-format set. Usage: bash heldout_loss.sh "<arms>" [RUN_NAME]
set -uo pipefail
ARMS=$1; RUN_NAME=${2:-run2}
WORK=/root/work; SFT=$WORK/GRAFT/sft; G=/workspace/graft; CK=/workspace/ckpts; OUT=/workspace/results
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1
while pgrep -f "[r]un_all.sh" >/dev/null; do sleep 60; done
sleep 90; while pgrep -f "[r]un_all.sh" >/dev/null; do sleep 60; done  # a chained second run may start
source $SFT/.venvs/eval/bin/activate
for ARM in $ARMS; do
  [[ -f $OUT/heldout_losses_$ARM.jsonl ]] && continue
  last=$(ls -d $CK/$ARM/checkpoint-step-* | sort -t- -k3 -n | tail -n 1)
  D=/workspace/ckpts_final/$ARM; rm -rf $D; mkdir -p $D; ln -s "$last" $D/
  SET=$G/loss_eval_heldout.jsonl; [[ $ARM == F_* ]] && SET=$G/loss_eval_heldout_fmt.jsonl
  echo "=== $(date '+%F %T') held-out loss eval $ARM ($(basename $last)) ==="
  (cd $SFT && PYTHONPATH=. torchrun --nproc_per_node 4 $G/pipeline/loss_eval.py --run-dir $D --records $SET \
     --modes retain_only --out $OUT/heldout_losses_$ARM.jsonl.tmp) && mv $OUT/heldout_losses_$ARM.jsonl.tmp $OUT/heldout_losses_$ARM.jsonl
done
set -a; source /workspace/.modal_env; set +a
$SFT/third_party/harbor/.venv/bin/modal volume put -f graft-results $OUT /$RUN_NAME/results
echo "=== $(date '+%F %T') held-out loss evals done ==="
