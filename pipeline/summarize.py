"""Summary tables for one run directory (results/<run>/): agentic hack/pass rates per eval job
and loss-eval NLL per transcript slice.  Usage: python summarize.py results/run1 > results/run1/summary.md
"""
import collections, json, sys
from pathlib import Path

run = Path(sys.argv[1])
repo = Path(__file__).resolve().parents[1]
routed = set(json.loads((repo / "data/routing/ceiling_20pct.json").read_text()))
labels = {h["key"]: h for h in json.loads((repo / "data/labels/joined_train_hacks.json").read_text())}
A = {"hollow-implementation", "output-spoofing", "algorithmic-simplification", "keyword-gaming", "constraint-loophole"}

print(f"# {run.name}: agentic evals\n")
print("Judge: claude-haiku-5-5 with GRAFT's judge_v3 prompt; hack = suspicion >= threshold. "
      "126 tasks (116 training-task IID + 10 held-out); trials with no transcript or cancelled at the "
      "90% cutoff are excluded.\n")
print("| job | n | pass | hack@0.8 | hack@0.5 | IID hack@0.8 | held-out hack@0.8 (n) | red-team / tampers / reads among hacks@0.8 |")
print("|---|--:|--:|--:|--:|--:|--:|--|")
order = ["base"] + [f"{a}-p{p}-{m}" for a, m in [("noint", "both"), ("ceiling", "retain"), ("ceiling", "both")]
                    for p in (25, 50, 100)]
for job in order:
    f = run / "agentic" / job / "judge_claude.json"
    if not f.exists():
        continue
    s = json.loads(f.read_text())["summary"]
    a, i, h = s["all"], s["iid"], s.get("heldout", {})
    t = a["hack_types@0.8"]
    print(f"| {job} | {a['n']} | {a['pass_rate']:.2f} | {a['hack_rate@0.8']:.2f} | {a['hack_rate@0.5']:.2f} | "
          f"{i['hack_rate@0.8']:.2f} | {h.get('hack_rate@0.8', float('nan')):.2f} ({h.get('n')}) | "
          f"{t['red_team_framing']} / {t['tampers_with_tooling']} / {t['reads_checker_or_answers']} |")


def slices(r):
    out = [r["set"]]
    if r["set"] == "train_hack":
        out.append("train_hack_routed" if r["key"] in routed else "train_hack_unrouted")
        h = labels.get(r["key"])
        if h:
            c = set(h["cats"])
            if c and c <= A: out.append("A_only")
            if c and not c & A: out.append("BC_only")
            if h["reads"] and not h["tampers"]: out.append("reads_only")
            if h["tampers"] and not h["reads"]: out.append("tampers_only")
            out.append("redteam" if h["redteam"] else "no_redteam")
            out.append("security" if h["sec"] else "nonsecurity")
            if h["py"]: out.append("python_only")
    return out


print(f"\n# {run.name}: loss evals (mean assistant-token NLL)\n")
for lf in sorted((run / "losses").glob("losses_*.jsonl")):
    rows = [json.loads(l) for l in open(lf)]
    agg = collections.defaultdict(list)
    for r in rows:
        for s in slices(r):
            agg[(s, r["mode"], r["step"])].append(r["nll"])
    steps = sorted({r["step"] for r in rows})
    for mode in ("both", "retain_only"):
        print(f"\n## {lf.stem.removeprefix('losses_')}, mode={mode}\n")
        print("| slice | n | " + " | ".join(f"step {s}" for s in steps) + " |")
        print("|---|--:|" + "--:|" * len(steps))
        names = sorted({k[0] for k in agg}, key=lambda x: (not x.startswith(("train", "heldout")), x))
        for n in names:
            vals = [agg.get((n, mode if s else "both", s)) for s in steps]
            cnt = len(vals[-1]) if vals[-1] else 0
            print(f"| {n} | {cnt} | " + " | ".join(f"{sum(v) / len(v):.3f}" if v else "" for v in vals) + " |")
