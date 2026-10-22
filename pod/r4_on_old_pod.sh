#!/usr/bin/env bash
# Put a restarted run-2 pod (setup done, its checkpoints local) on a run-4 eval plan: upload the
# changed code, task sets and plans, apply the French prefill patch, start run_evals.sh.
# Usage: RUNPOD_KEY_FILE=... bash pod/r4_on_old_pod.sh POD <A|B|C>
set -euo pipefail
cd "$(dirname "$0")/.."
POD=$1; PLAN=data/eval_plan_r4/$2.tsv
R="python3 -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
[[ -n ${PYTHON:-} ]] && R="$PYTHON -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
for f in pipeline/agentic_eval.py pipeline/patch_harbor_prefill.py pod/run_evals.sh \
         data/eval_tasks_all.json data/eval_tasks_security.json data/eval_tasks_science.json data/eval_tasks_ops.json \
         data/eval_plan_r4/A.tsv data/eval_plan_r4/B.tsv data/eval_plan_r4/C.tsv; do
  bash pod/push_file.sh $POD $f /workspace/graft/$f | tail -1
done
$R exec $POD "cd /root/work/GRAFT/sft/third_party/harbor && python3 /workspace/graft/pipeline/patch_harbor_prefill.py src/harbor/llms/lite_llm.py && grep -c prefill-french src/harbor/llms/lite_llm.py" --timeout 60
$R spawn $POD main "bash /workspace/graft/pod/run_evals.sh /workspace/graft/$PLAN"
