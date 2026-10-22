#!/usr/bin/env bash
# Narration probes (pipeline/rewrite_narration.py): RETAIN-only NLL of the rewritten hack transcripts
# (sanitized / surgery / french) at each arm's final checkpoint, compared by pipeline/probe_transfer.py
# with the originals' final-step NLL from the main loss pass. The base model (step-0) is scored once,
# with the first arm. Waits for run_all.sh and heldout_loss.sh, then writes
# /workspace/results/probe_losses_<arm>.jsonl and syncs results.
# Usage: bash probe_loss.sh "<arms>" [RUN_NAME]   (probe set at /workspace/graft/probe_set.jsonl.gz)
set -uo pipefail
ARMS=$1; RUN_NAME=${2:-run2}
WORK=/root/work; SFT=$WORK/GRAFT/sft; G=/workspace/graft; CK=/workspace/ckpts; OUT=/workspace/results
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1
while pgrep -f "[r]un_all.sh|[h]eldout_loss.sh" >/dev/null; do sleep 60; done
[[ -f $G/probe_set.jsonl ]] || gunzip -k $G/probe_set.jsonl.gz
source $SFT/.venvs/eval/bin/activate
BASE=""
for ARM in $ARMS; do
  [[ -f $OUT/probe_losses_$ARM.jsonl ]] && continue
  last=$(ls -d $CK/$ARM/checkpoint-step-* | sort -t- -k3 -n | tail -n 1)
  D=/workspace/ckpts_final/$ARM; rm -rf $D; mkdir -p $D; ln -s "$last" $D/
  echo "=== $(date '+%F %T') probe loss eval $ARM ($(basename $last)) ==="
  (cd $SFT && PYTHONPATH=. torchrun --nproc_per_node 4 $G/pipeline/loss_eval.py --run-dir $D --records $G/probe_set.jsonl \
     --modes retain_only $BASE --out $OUT/probe_losses_$ARM.jsonl.tmp) && mv $OUT/probe_losses_$ARM.jsonl.tmp $OUT/probe_losses_$ARM.jsonl
  BASE=--skip-base
done
set -a; source /workspace/.modal_env; set +a
$SFT/third_party/harbor/.venv/bin/modal volume put -f graft-results $OUT /$RUN_NAME/results
echo "=== $(date '+%F %T') probe loss evals done ==="
