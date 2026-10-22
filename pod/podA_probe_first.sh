#!/usr/bin/env bash
# Run 2, pod A: score the narration probes for the reference models (noint, ceiling_20pct) as soon
# as the in-flight agentic eval ends, then resume: V1's agentic eval and sandbox-killed reruns,
# the held-out pass, and the remaining probes. (Severe-to-mild's eval marker was pre-created to
# hold it back; it is removed here before the reruns.) Kept as a file so pgrep patterns in the
# scripts it calls don't match its own command line.
set -uo pipefail
G=/workspace/graft
ARMS="noint ceiling_20pct A_make_it_up L1_short V1_severe_to_mild"
RUN_ENV="NPROC=4 EPOCHS=1 SAVE_FRACS=0.5,1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run2"
PROBE_SKIP_BASE=1 bash $G/pod/probe_loss.sh "noint ceiling_20pct"
rm -f /workspace/state/eval_V1_severe_to_mild-p100-retain.done /workspace/state/sync.done
RUN_ENV=$RUN_ENV bash $G/pod/rerun_killed.sh
env $RUN_ENV STAGES="agentic sync" bash $G/pod/run_all.sh
bash $G/pod/heldout_loss.sh "$ARMS"
bash $G/pod/probe_loss.sh "A_make_it_up L1_short V1_severe_to_mild"
