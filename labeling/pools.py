"""Join LLM labels with the training pool and report per-condition routable pools."""
import collections, json, re, sys

S, TW = sys.argv[1], sys.argv[2]
N_ROUTE = None  # set below as 20% of usable hacks

train = set(json.load(open(f"{S}/sftdata/task_split.json"))["train_tasks"])
idx = {(t["task_id"], t["model"], t["trajectory_label"]): t for t in json.load(open(f"{TW}/index/trajectories.json"))}
tl = {r["key"]: r for r in map(json.loads, open(f"{S}/labels/tasks.jsonl")) if "error" not in r}
hl = {r["key"]: r for r in map(json.loads, open(f"{S}/labels/hacks.jsonl")) if "error" not in r}
errs = [r for r in map(json.loads, open(f"{S}/labels/hacks.jsonl")) if "error" in r]

A = {"hollow-implementation", "output-spoofing", "algorithmic-simplification", "keyword-gaming", "constraint-loophole"}
B = {"binary-hijacking", "mutable-input-tampering", "metric-spoofing", "security-downgrading"}
C = {"structural-extraction"}

H, n_clean, missing = [], 0, 0
for line in open(f"{S}/sftdata/hack_pool.jsonl"):
    m = json.loads(line)["meta"]
    if m["task_id"] not in train or m["n_tokens_qwen3"] > 16384:
        continue
    key = f"{m['task_id']}/{m['source_model']}/{m['label']}"
    if key not in hl or m["task_id"] not in tl:
        missing += 1
        continue
    h, t = hl[key], tl[m["task_id"]]
    cats = set(idx[(m["task_id"], m["source_model"], m["label"])]["observed_categories"])
    H.append(dict(key=key, task=m["task_id"], model=m["source_model"], cats=cats,
                  reads=h["reads_checker_or_answers"], tampers=h["tampers_with_tooling"], redteam=h["red_team_framing"],
                  sec=t["security_related"], py=t["python_only"], kind=t["task_kind"], langs=tuple(t["languages_written"])))
for line in open(f"{S}/sftdata/clean_pool.jsonl"):
    m = json.loads(line)["meta"]
    if m["task_id"] in train and m["n_tokens_qwen3"] <= 16384 and m["classification"] == "baseline":
        n_clean += 1

N_ROUTE = round(0.2 * len(H))
print(f"usable training hacks with labels: {len(H)} (missing labels: {missing}, label errors: {len(errs)}); clean {n_clean}")
print(f"20% routed count N = {N_ROUTE}\n")

conds = {
    "H1  A-only hacks": (lambda h: h["cats"] and h["cats"] <= A, "B-only / C-only hacks",
                         lambda h: bool(h["cats"]) and not h["cats"] & A),
    "H6a reads checker/answers (not tampering)": (lambda h: h["reads"] and not h["tampers"], "tampers only",
                                                  lambda h: h["tampers"] and not h["reads"]),
    "H6b tampers with tooling (not reading)": (lambda h: h["tampers"] and not h["reads"], "reads only",
                                               lambda h: h["reads"] and not h["tampers"]),
    "H9  red-team framing": (lambda h: h["redteam"], "no red-team framing", lambda h: not h["redteam"]),
    "T3  security tasks": (lambda h: h["sec"], "non-security tasks", lambda h: not h["sec"]),
    "T5  python-only tasks": (lambda h: h["py"], "tasks writing non-Python code",
                              lambda h: h["langs"] and not h["py"]),
}
print(f"{'condition':46s} {'pool':>5s} {'tasks':>5s}  {'>=N?':4s}  {'held-out side':32s} {'hacks':>5s} {'tasks':>5s}")
for name, (f, hname, g) in conds.items():
    pool = [h for h in H if f(h)]; held = [h for h in H if g(h)]
    print(f"{name:46s} {len(pool):5d} {len({h['task'] for h in pool}):5d}  {'yes' if len(pool) >= N_ROUTE else 'NO':4s}  "
          f"{hname:32s} {len(held):5d} {len({h['task'] for h in held}):5d}")

print("\nH6 joint (reads, tampers):", dict(collections.Counter((h["reads"], h["tampers"]) for h in H)))
print("H6 vs judge families:  reads-only -> any-C", sum(1 for h in H if h["reads"] and not h["tampers"] and h["cats"] & C),
      "| tampers-only -> any-B", sum(1 for h in H if h["tampers"] and not h["reads"] and h["cats"] & B))
print("red-team framing vs regex 'red team':",
      collections.Counter(h["redteam"] for h in H))
tk = collections.Counter((t["security_related"], t["python_only"], t["task_kind"]) for k, t in tl.items() if k in train)
print("train-task labels (sec, py_only, kind):", sorted(tk.items(), key=lambda x: -x[1])[:8])
print("security share of hacks by model:", {m: f"{sum(h['sec'] for h in H if h['model'] == m) / sum(h['model'] == m for h in H):.0%}"
                                            for m in sorted({h['model'] for h in H})})
print("python-only tasks: hack mix any-B / any-C:",
      f"{sum(bool(h['cats'] & B) for h in H if h['py']) / max(1, sum(h['py'] for h in H)):.0%} / "
      f"{sum(bool(h['cats'] & C) for h in H if h['py']) / max(1, sum(h['py'] for h in H)):.0%}",
      "| security tasks: any-B", f"{sum(bool(h['cats'] & B) for h in H if h['sec']) / max(1, sum(h['sec'] for h in H)):.0%}",
      "vs non-security", f"{sum(bool(h['cats'] & B) for h in H if not h['sec']) / max(1, sum(not h['sec'] for h in H)):.0%}")
json.dump([{**h, "cats": sorted(h["cats"])} for h in H], open(f"{S}/labels/joined_train_hacks.json", "w"))
