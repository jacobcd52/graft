#!/usr/bin/env bash
# End-to-end run on the pod: train arms -> loss evals -> agentic evals. Resumable: each finished
# stage leaves /workspace/state/<stage>.done. Agentic evals need /workspace/.modal_env
# (MODAL_TOKEN_ID / MODAL_TOKEN_SECRET); without it they are skipped.
#
# Env: ARMS (default "noint ceiling_20pct"; any data/routing arm name), MODEL (Qwen/Qwen3-32B),
#      NPROC/DP (8), EPOCHS (4), STAGES (default "train loss agentic sync"),
#      EVAL_FRACS (0.25 0.5 1.0), EVAL_BASE (1: also eval the base model), AGENT_TIMEOUT (1800),
#      N_CONCURRENT (64), EVAL_TASKS, RUN_NAME (Modal volume prefix for the sync stage).
# The pod's agent stops the pod ~30 min after this script exits (see pod/agent.py), so every
# stage that should run must be in STAGES; the plan is printed first so a wrong STAGES is visible.
set -euo pipefail
WORK=${WORK:-/root/work}   # local disk: GRAFT checkout, venvs, HF cache, merged models
export HF_HOME=$WORK/hf PATH=$HOME/.local/bin:$PATH
export MODEL=${MODEL:-Qwen/Qwen3-32B} NPROC=${NPROC:-8} EPOCHS=${EPOCHS:-4}
DP=${DP:-$NPROC}
STAGES=${STAGES:-train loss agentic sync}
ARMS=${ARMS:-noint ceiling_20pct}
RUN_NAME=${RUN_NAME:-run_$(date +%Y%m%d)}
EVAL_FRACS=${EVAL_FRACS:-0.25 0.5 1.0}
SAVE_FRACS=${SAVE_FRACS:-0.1,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
EVAL_TASKS=${EVAL_TASKS:-/workspace/graft/data/eval_tasks.json}
G=/workspace/graft; SFT=$WORK/GRAFT/sft; CK=/workspace/ckpts; ST=/workspace/state; OUT=/workspace/results
mkdir -p $ST $OUT
export GRAFT_DIR=$WORK/GRAFT GRAFT_SFT_CKPT_DIR=$CK SAVE_FRACS
# every arm trains on the same records; noint uses the ceiling arm's files with --no-routing
data_for() { [[ $1 == noint ]] && echo $G/armdata/ceiling_20pct || echo $G/armdata/$1; }
LOSS_SET=${LOSS_SET:-$G/loss_eval_set.jsonl}
log() { echo -e "\n=== $(date '+%F %T') $* ==="; }
done_() { [[ -f $ST/$1.done ]]; }
mark() { touch $ST/$1.done; }

# checkpoint dir of <arm> at training fraction <f> (checkpoint-step-* sorted == SAVE_FRACS sorted)
ckpt_at() {
  python3 - "$CK/$1" "$2" "$SAVE_FRACS" <<'EOF'
import sys, re
from pathlib import Path
run, f, fracs = Path(sys.argv[1]), float(sys.argv[2]), sorted(float(x) for x in sys.argv[3].split(","))
steps = sorted(int(re.search(r"(\d+)$", d.name).group(1)) for d in run.glob("checkpoint-step-*"))
assert len(steps) == len(fracs), (steps, fracs)
print(run / f"checkpoint-step-{steps[fracs.index(f)]}" / "adapter_state_dict.pt")
EOF
}

log "plan: arms=[$ARMS] stages=[$STAGES] epochs=$EPOCHS eval_fracs=[$EVAL_FRACS] save_fracs=$SAVE_FRACS"
for ARM in $ARMS; do [[ -d $(data_for $ARM) ]] || { echo "ERROR: no data dir for arm $ARM"; exit 1; }; done

source $SFT/.venvs/eval/bin/activate
for ARM in $ARMS; do
  if [[ " $STAGES " == *" train "* ]] && ! done_ train_$ARM; then
    log "train $ARM"
    NPROC=$NPROC bash $G/pipeline/train_arm.sh $ARM $(data_for $ARM)
    mark train_$ARM
  fi
done

if [[ " $STAGES " == *" loss "* ]]; then
  for ARM in $ARMS; do
    done_ loss_$ARM && continue
    log "loss eval $ARM"
    (cd $SFT && PYTHONPATH=. HF_HUB_OFFLINE=1 torchrun --nproc_per_node $NPROC $G/pipeline/loss_eval.py \
       --run-dir $CK/$ARM --records $LOSS_SET --modes both,retain_only \
       --model-name $MODEL --out $OUT/losses_$ARM.jsonl)
    mark loss_$ARM
  done
fi

if [[ " $STAGES " == *" agentic "* ]]; then
  if [[ ! -f /workspace/.modal_env ]]; then
    log "no /workspace/.modal_env: skipping agentic evals"; exit 0
  fi
  set -a; source /workspace/.modal_env; set +a
  run_eval() {  # job ckpt mode
    done_ eval_$1 && return 0
    log "agentic eval $1"
    (cd $SFT && python $G/pipeline/agentic_eval.py --job-name $1 --ckpt $2 --mode $3 --tasks $EVAL_TASKS \
       --base-model $MODEL --dp $DP --agent-timeout ${AGENT_TIMEOUT:-1800} --n-concurrent ${N_CONCURRENT:-64})
    tar -czf $OUT/job_$1.tgz -C $SFT/build/jobs $1
    mark eval_$1
  }
  [[ ${EVAL_BASE:-1} == 1 ]] && run_eval base base both
  for f in $EVAL_FRACS; do
    pct=$(python3 -c "print(round($f*100))")
    for ARM in $ARMS; do
      if [[ $ARM == noint ]]; then
        run_eval noint-p$pct-both "$(ckpt_at noint $f)" both
      else
        short=${ARM/ceiling_20pct/ceiling}
        run_eval $short-p$pct-retain "$(ckpt_at $ARM $f)" retain_only
        run_eval $short-p$pct-both "$(ckpt_at $ARM $f)" both
      fi
    done
  done
fi

# Copy results, logs and adapters to Modal volumes, so nothing is lost when the pod stops itself.
if [[ " $STAGES " == *" sync "* ]] && [[ -f /workspace/.modal_env ]] && ! done_ sync; then
  set -a; source /workspace/.modal_env; set +a
  MODAL=$SFT/third_party/harbor/.venv/bin/modal
  log "sync to Modal volumes graft-results:/$RUN_NAME and graft-ckpts:/$RUN_NAME"
  $MODAL volume create graft-results >/dev/null 2>&1 || true
  $MODAL volume put -f graft-results $OUT /$RUN_NAME/results
  $MODAL volume put -f graft-results /workspace/logs /$RUN_NAME/logs
  for ARM in $ARMS; do $MODAL volume put -f graft-ckpts $CK/$ARM /$RUN_NAME/$ARM; done
  mark sync
fi
log "all requested stages done"
