#!/usr/bin/env bash
# Pod B's remaining run-2 work after run 3 (pod/r3_takeover.sh): sandbox-killed reruns, held-out pass, probes.
G=/workspace/graft
RUN_ENV="NPROC=4 EPOCHS=1 SAVE_FRACS=0.5,1.0 EVAL_FRACS=1.0 EVAL_BASE=0 RUN_NAME=run2" bash $G/pod/rerun_killed.sh
bash $G/pod/heldout_loss.sh "H9_redteam T3_nonsecurity T5_python_only R1_random_tasks K1_code_to_ops"
bash $G/pod/probe_loss.sh "H9_redteam T3_nonsecurity T5_python_only R1_random_tasks K1_code_to_ops"
