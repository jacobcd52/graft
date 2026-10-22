#!/usr/bin/env bash
# Run an eval plan (pod/eval_plan_r4.py TSV: job arm mode prefill taskset attempts) on this pod.
# Resumable (/workspace/state/eval_<job>.done). A checkpoint that is not on the pod is pulled from
# the Modal volume graft-ckpts (/run2/<arm> or /run3/<arm>). Consecutive jobs of one model reuse its
# merge and vLLM server. Each finished job is packed to /workspace/results/job_<job>.tgz.
# Usage: bash run_evals.sh <plan.tsv>
set -uo pipefail
PLAN=$1
G=/workspace/graft; WORK=${WORK:-/root/work}; SFT=$WORK/GRAFT/sft
CK=/workspace/ckpts; ST=/workspace/state; OUT=/workspace/results
export HF_HOME=$WORK/hf PATH=$HOME/.local/bin:$PATH
mkdir -p $ST $OUT
set -a; source /workspace/.modal_env; set +a
MODAL=$SFT/third_party/harbor/.venv/bin/modal
source $SFT/.venvs/eval/bin/activate
log() { echo -e "\n=== $(date '+%F %T') $* ==="; }

latest_ckpt() { ls -d $CK/$1/checkpoint-step-* 2>/dev/null | sort -t- -k3 -n | tail -1; }
pull_ckpt() {  # arm
  [[ -n $(latest_ckpt $1) ]] && return 0
  local run=run2; [[ $1 == SURG_* || $1 == FREN_* ]] && run=run3
  log "pull $1 from graft-ckpts:/$run/$1"
  rm -rf $CK/.pull && mkdir -p $CK/.pull && (cd $CK/.pull && $MODAL volume get graft-ckpts /$run/$1 .) \
    && mv $CK/.pull/$1 $CK/$1
}

mapfile -t ROWS < <(grep -v '^\s*$' $PLAN)
prev_key=""
for i in "${!ROWS[@]}"; do
  IFS=$'\t' read -r job arm mode prefill taskset k <<< "${ROWS[$i]}"
  key="$arm-$mode"
  next_key=""; [[ $((i + 1)) -lt ${#ROWS[@]} ]] && { IFS=$'\t' read -r _ narm nmode _ <<< "${ROWS[$((i + 1))]}"; next_key="$narm-$nmode"; }
  if [[ -n $prev_key && $key != "$prev_key" ]]; then  # previous model's group ended (possibly by a failure)
    (cd $SFT && bash scripts/eval/stop_serve.sh) >/dev/null 2>&1; rm -rf $SFT/build/merged/$prev_key
  fi
  prev_key=$key
  [[ -f $ST/eval_$job.done ]] && continue
  pull_ckpt $arm || { log "ERROR: no checkpoint for $arm"; continue; }
  ckpt=$(latest_ckpt $arm)/adapter_state_dict.pt
  keep=""; [[ $next_key == "$key" ]] && keep=--keep
  log "agentic eval $job ($taskset x$k, prefill=$prefill)"
  (cd $SFT && python $G/pipeline/agentic_eval.py --job-name $job --ckpt $ckpt --mode $mode \
     --tasks $G/data/eval_tasks_$taskset.json --attempts $k $([[ $prefill != - ]] && echo --prefill $prefill) \
     --model-key $key $keep --base-model ${MODEL:-Qwen/Qwen3-32B} --dp ${DP:-4} \
     --agent-timeout ${AGENT_TIMEOUT:-1800} --n-concurrent ${N_CONCURRENT:-96}) \
    && tar -czf $OUT/job_$job.tgz -C $SFT/build/jobs $job && touch $ST/eval_$job.done \
    && rm -rf $SFT/build/jobs/$job
done
(cd $SFT && bash scripts/eval/stop_serve.sh) >/dev/null 2>&1; rm -rf $SFT/build/merged/$prev_key
log "plan $PLAN done"
