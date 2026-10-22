#!/usr/bin/env bash
# Pod C's remaining run-2 work after run 3 (pod/r3_takeover.sh): D1's held-back agentic eval and the
# sandbox-killed rerun, held-out pass, remaining probes.
G=/workspace/graft
RUN_ENV="NPROC=4 EPOCHS=1 SAVE_FRACS=0.5,1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run2"
rm -f /workspace/state/eval_D1_sysadmin_to_science-p100-retain.done /workspace/state/sync.done
RUN_ENV=$RUN_ENV bash $G/pod/rerun_killed.sh
env $RUN_ENV STAGES="agentic sync" bash $G/pod/run_all.sh
bash $G/pod/heldout_loss.sh "F_noint F_ceiling F_reasoned D1_sysadmin_to_science"
bash $G/pod/probe_loss.sh "F_reasoned D1_sysadmin_to_science"
