#!/usr/bin/env bash
# Pack the graft bundle uploaded to each pod: this repo at HEAD, the completion-fixed TW pools
# and the loss-eval set (arm data is built on the pod from these plus data/routing/).
# Usage: make_bundle.sh <dir with sftdata/{clean,hack}_pool.jsonl, task_split.json and loss_eval_set.jsonl> <out.tgz>
set -euo pipefail
SRC=$(realpath "$1"); OUT=$(realpath -m "$2")
cd "$(dirname "$0")/.."
T=$(mktemp -d)
git archive --prefix=graft/ HEAD | tar -x -C "$T"
mkdir -p "$T/graft/sftdata"
cp "$SRC"/sftdata/{clean_pool,hack_pool}.jsonl "$SRC/sftdata/task_split.json" "$T/graft/sftdata/"
cp "$SRC/loss_eval_set.jsonl" "$T/graft/loss_eval_set.jsonl"
tar -czf "$OUT" -C "$T" graft
rm -rf "$T"
ls -la "$OUT"
