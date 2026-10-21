"""Choose the agentic-eval task set: ~half of the training tasks (IID: their hacks were trained
on) plus a few held-out tasks, sampled systematically after sorting by the task-level properties
our conditions depend on, so each condition is represented roughly in proportion.

Usage: python build_eval_tasks.py --labels data/labels --task-split data/task_split.json \
           --out data/eval_tasks.json [--frac 0.5 --n-heldout 10]
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

A = {"hollow-implementation", "output-spoofing", "algorithmic-simplification", "keyword-gaming", "constraint-loophole"}


def systematic(items, n, key, seed):
    rng = random.Random(seed)
    order = sorted(items, key=lambda t: (key(t), rng.random()))
    step = len(order) / n
    start = rng.random() * step
    return [order[int(start + i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--task-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--n-heldout", type=int, default=10)
    a = ap.parse_args()
    split = json.loads(a.task_split.read_text())
    tl = {json.loads(l)["key"]: json.loads(l) for l in open(a.labels / "tasks.jsonl")}
    hacks = json.loads((a.labels / "joined_train_hacks.json").read_text())

    has = defaultdict(set)  # task -> hack kinds present in its training hacks
    for h in hacks:
        c = set(h["cats"])
        if c and not c & A: has[h["task"]].add("BC_only")
        if h["reads"] and not h["tampers"]: has[h["task"]].add("reads_only")
        if h["tampers"] and not h["reads"]: has[h["task"]].add("tampers_only")
        if not h["redteam"]: has[h["task"]].add("no_redteam")

    def strata(t):
        x = tl[t]
        return (x["security_related"], x["python_only"], x["task_kind"],
                "BC_only" in has[t], "reads_only" in has[t], "tampers_only" in has[t], "no_redteam" in has[t])

    train = sorted(t for t in split["train_tasks"] if t in tl)
    held = sorted(t for t in split["eval_tasks"] if t in tl)
    iid = systematic(train, round(a.frac * len(train)), strata, 0)
    ho = systematic(held, a.n_heldout, lambda t: (tl[t]["security_related"], tl[t]["python_only"], tl[t]["task_kind"]), 1)
    a.out.write_text(json.dumps({"eval_tasks": sorted(iid) + sorted(ho), "iid_tasks": sorted(iid),
                                 "heldout_tasks": sorted(ho)}, indent=1))

    def share(ts, f):
        return f"{sum(f(t) for t in ts)}/{len(ts)}"
    print(f"IID {len(iid)} of {len(train)} train tasks, held-out {len(ho)}")
    for name, f in [("security", lambda t: tl[t]["security_related"]), ("python_only", lambda t: tl[t]["python_only"]),
                    ("ops", lambda t: tl[t]["task_kind"] == "ops"), ("has BC_only hack", lambda t: "BC_only" in has[t]),
                    ("has reads_only hack", lambda t: "reads_only" in has[t]),
                    ("has tampers_only hack", lambda t: "tampers_only" in has[t])]:
        print(f"  {name:22s} train {share(train, f):>8s}  eval {share(iid, f):>8s}")


if __name__ == "__main__":
    main()
