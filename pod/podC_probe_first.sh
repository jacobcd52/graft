#!/usr/bin/env bash
# Run 2, pod C: score the narration probes for the format references (F_noint, F_ceiling) as soon as
# the in-flight agentic eval ends, then resume: D1's agentic eval and sandbox-killed reruns, the
# held-out pass, and the remaining probes. (D1's eval marker was pre-created to hold it back; it is
# removed here before the reruns.) A file, so pgrep patterns in the scripts it calls don't match it.
set -uo pipefail
G=/workspace/graft
ARMS="F_noint F_ceiling F_reasoned D1_sysadmin_to_science"
RUN_ENV="NPROC=4 EPOCHS=1 SAVE_FRACS=0.5,1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run2"
PROBE_SKIP_BASE=1 bash $G/pod/probe_loss.sh "F_noint F_ceiling"
rm -f /workspace/state/eval_D1_sysadmin_to_science-p100-retain.done /workspace/state/sync.done
RUN_ENV=$RUN_ENV bash $G/pod/rerun_killed.sh
env $RUN_ENV STAGES="agentic sync" bash $G/pod/run_all.sh
bash $G/pod/heldout_loss.sh "$ARMS"
bash $G/pod/probe_loss.sh "F_reasoned D1_sysadmin_to_science"
