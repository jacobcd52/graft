#!/usr/bin/env bash
# Redo agentic evals whose rollouts were killed from outside (Modal sandbox NotFound / "shutting
# down" in a trial's result.json). Run 2: before pipeline/agentic_eval.py used one Modal App per host,
# each pod's end-of-job sandbox cleanup terminated the live sandboxes of evals on the other pods.
# Waits for run_all.sh to finish, drops each such job's marker, job dir and tgz, then reruns
# run_all.sh's agentic + sync stages (heldout_loss.sh's second check sees it and keeps waiting).
# Usage: RUN_ENV="NPROC=4 EPOCHS=1 ..." bash rerun_killed.sh [min killed trials, default 5]
set -uo pipefail
MIN=${1:-5}
SFT=/root/work/GRAFT/sft; ST=/workspace/state; OUT=/workspace/results
while pgrep -f "[r]un_all.sh" >/dev/null; do sleep 10; done
redo=$(python3 - "$SFT/build/jobs" "$MIN" <<'PY'
import json, sys
from pathlib import Path
for job in sorted(Path(sys.argv[1]).iterdir()):
    n = 0
    for r in job.glob("*/result.json"):
        try:
            e = json.loads(r.read_text()).get("exception_info") or {}
        except Exception:
            continue
        n += e.get("exception_type") in ("NotFoundError", "ConflictError")
    if n > int(sys.argv[2]):
        print(job.name)
PY
)
[[ -z $redo ]] && { echo "=== $(date '+%F %T') no killed evals ==="; exit 0; }
for job in $redo; do
  echo "=== $(date '+%F %T') redo $job ==="
  mkdir -p $SFT/build/jobs_killed && rm -rf $SFT/build/jobs_killed/$job && mv $SFT/build/jobs/$job $SFT/build/jobs_killed/
  rm -f $ST/eval_$job.done $OUT/job_$job.tgz
done
rm -f $ST/sync.done
env ${RUN_ENV:-} STAGES="agentic sync" bash /workspace/graft/pod/run_all.sh
echo "=== $(date '+%F %T') reruns done ==="
