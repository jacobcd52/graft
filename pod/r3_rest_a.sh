#!/usr/bin/env bash
# Pod A's remaining run-2 work after run 3 (pod/r3_takeover.sh): held-out pass, condition-arm probes.
G=/workspace/graft
bash $G/pod/heldout_loss.sh "noint ceiling_20pct A_make_it_up L1_short V1_severe_to_mild"
bash $G/pod/probe_loss.sh "A_make_it_up L1_short V1_severe_to_mild"
