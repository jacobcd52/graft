#!/usr/bin/env bash
# Poll pods until something worth reporting happens, then print it and exit.
# Reports: a new /workspace/state/*.done marker, a new checkpoint dir, an error line in the run log,
# or 3 consecutive unreachable polls. Usage: watch.sh <interval_s> <max_polls> POD [POD...]
cd "$(dirname "$0")/.."
INT=$1; MAX=$2; shift 2
R="${PYTHON:-python3} -I pod/runpod_ctl.py --key-file $RUNPOD_KEY_FILE"
declare -A last fails
probe='ls /workspace/state 2>/dev/null | tr "\n" " "; echo; ls /workspace/ckpts/*/ 2>/dev/null | grep -c checkpoint-step; grep -cE "Traceback|FAILED|out of memory|CUDA error" /workspace/logs/main.log; true'
for ((i = 0; i < MAX; i++)); do
  for P in "$@"; do
    if out=$($R exec "$P" "$probe" --timeout 60 2>/dev/null); then
      fails[$P]=0
      if [[ -n ${last[$P]:-} && $out != "${last[$P]}" ]]; then
        echo "$(date -u +%T) $P changed:"; echo "  was: ${last[$P]//$'\n'/ | }"; echo "  now: ${out//$'\n'/ | }"; exit 0
      fi
      [[ $(echo "$out" | sed -n 3p) != 0 ]] && { echo "$(date -u +%T) $P ERROR lines in log: $out"; exit 0; }
      last[$P]=$out
    else
      fails[$P]=$(( ${fails[$P]:-0} + 1 ))
      (( fails[$P] >= 3 )) && { echo "$(date -u +%T) $P unreachable 3x"; exit 0; }
    fi
  done
  sleep "$INT"
done
echo "$(date -u +%T) no change after $MAX polls"
