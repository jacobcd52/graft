# Run lessons: what cost money, and what now prevents it

Run 1 (no intervention + 20% routing, Qwen3-32B, two 4xH200 pods at $21.16/h) cost about $615.
These issues account for most of the waste. Each has a guard in the code; keep them.

| Issue | Cost | Guard |
|---|---|---|
| A finished pod sat idle ~3 h: pod B was launched with `STAGES="train loss"`, so its evals never ran, and the local watcher died in a container restart, so nobody noticed. | ~$80 | `pod/agent.py` idle guard stops the pod 30 min after the pipeline exits (status polls do not count as activity), or after 90 min of silent logs while it runs (hang). `run_all.sh` logs its plan (arms, stages, epochs, eval fracs) first, defaults to every stage, and syncs results and adapters to Modal before exiting, so a stopped pod loses nothing. |
| Trained 4 epochs; hacking is fully learned by 25% (~1 epoch) and held-out loss bottoms out at 40-50%. Later epochs mostly memorize training tasks. | ~$150 of GPU time | Condition arms: `EPOCHS=1`, `SAVE_FRACS=0.5,1.0`, `EVAL_FRACS=1.0`, `EVAL_BASE=0`. Run a short loss-curve check before committing to long schedules. |
| Pods that never became usable: L40S hosts with broken CUDA, secure-cloud pods stuck starting, hosts throttling GitHub/PyPI. | ~$15 + hours | `runpod_ctl.py create` terminates a pod whose agent is not reachable within 15 min (`--keep-unready` to opt out). `setup.sh` fails fast on unusable CUDA. Code bundle and flash-attn wheels are uploaded rather than fetched. |
| Labeling cost 2.4x its estimate ($95 vs $40): transcripts were longer in tokens than estimated. | ~$55 | `labeling/label.py` prints a pre-run estimate, the running cost every 100 items, and aborts at `--max-usd` (default $50; resumable). |
| `/workspace` is a network filesystem on secure pods: venvs and model loads there were very slow. | hours of setup | Code, venvs, HF cache and merged models live on local disk (`/root/work`); only logs, checkpoints and results go to `/workspace`. |
| `pkill -f <pattern>` matched the shell running it and killed cleanup mid-way. | stale vLLM, failed eval | Use the bracket trick (`pkill -f "[v]llm"`). |
| The local container's disk filled (extracted run-1 eval jobs, 15 GB) mid-way through writing the probe set. | a rerun of the write | Keep job tgz archives and judge JSON only; delete extracted job trees after judging. Check `df` before writing large files. |
| An API key ran out of credits; the rewrite script then crashed on its retry path and hid the cause. | a debugging round | API scripts record the last HTTP status and body on every retry and report it per record. Test 3 items before a full run. |

## Checklist before launching pods

1. Set `ARMS`, `STAGES`, `EPOCHS`, `EVAL_FRACS` explicitly; read the `plan:` line in `main.log` right after launch.
2. Prefer one pod that runs arms back to back over one pod per arm: setup (~45 min) is paid once.
3. Keep a check-in scheduled while pods run; the idle guard is the backstop, not the plan.
4. After the run, fetch results from the Modal volume `graft-results:/<RUN_NAME>`, then terminate the pod (a stopped pod still bills for its volume).
