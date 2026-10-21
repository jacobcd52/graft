"""Build routed-hack lists (N = 20% of labeled training hacks) for every arm."""
import collections, json, random, sys
S = sys.argv[1]
J = json.load(open(f"{S}/labels/joined_train_hacks.json"))
A = {"hollow-implementation", "output-spoofing", "algorithmic-simplification", "keyword-gaming", "constraint-loophole"}
N = round(0.2 * len(J))

def fam(h):
    c = set(h["cats"])
    return "none" if not c else "A-only" if c <= A else "BC-only" if not c & A else "mixed"
def cell(h):
    return (fam(h), h["reads"], h["tampers"], h["redteam"], h["sec"], h["py"])

def systematic(pool, n, seed):
    """Every-k-th selection after sorting by (cell, task, random) -> proportional in each cell and task."""
    rng = random.Random(seed)
    order = sorted(pool, key=lambda h: (cell(h), h["task"], rng.random()))
    step = len(order) / n
    start = rng.random() * step
    return [order[int(start + i * step)] for i in range(n)]

conds = {
    "H1_A_only": lambda h: fam(h) == "A-only",
    "H6_tampers": lambda h: h["tampers"] and not h["reads"],
    "H9_redteam": lambda h: h["redteam"],
    "T3_nonsecurity": lambda h: not h["sec"],
    "T5_python_only": lambda h: h["py"],
}
arms = {"ceiling_20pct": systematic(J, N, 0)}
for name, f in conds.items():
    arms[name] = systematic([h for h in J if f(h)], N, 1)
for name, sel in arms.items():
    json.dump(sorted(h["key"] for h in sel), open(f"{S}/routing/{name}.json", "w"), indent=0)

# report routing rate per subgroup for every arm
groups = {
    "A-only": lambda h: fam(h) == "A-only", "B/C-only": lambda h: fam(h) == "BC-only",
    "reads only": lambda h: h["reads"] and not h["tampers"], "tampers only": lambda h: h["tampers"] and not h["reads"],
    "red-team": lambda h: h["redteam"], "no red-team": lambda h: not h["redteam"],
    "security task": lambda h: h["sec"], "non-security": lambda h: not h["sec"],
    "python-only task": lambda h: h["py"], "non-python code task": lambda h: not h["py"] and h["langs"],
}
print(f"N routed per arm = {N} of {len(J)} hacks\n")
print(f"{'group':22s} {'n':>5s} " + " ".join(f"{a[:14]:>14s}" for a in arms))
for g, f in groups.items():
    members = {h["key"] for h in J if f(h)}
    print(f"{g:22s} {len(members):5d} " + " ".join(
        f"{sum(h['key'] in members for h in sel) / len(members):14.0%}" for sel in arms.values()))
c = arms["ceiling_20pct"]
per_task = collections.Counter(h["task"] for h in J); per_task_r = collections.Counter(h["task"] for h in c)
dev = [abs(per_task_r[t] - 0.2 * n) for t, n in per_task.items()]
print(f"\nceiling: per-task |routed - 20%| max {max(dev):.1f} hacks, mean {sum(dev)/len(dev):.2f}")
cells = collections.Counter(cell(h) for h in J); cr = collections.Counter(cell(h) for h in c)
dev = [abs(cr[k] - 0.2 * n) for k, n in cells.items()]
print(f"ceiling: {len(cells)} cells, per-cell |routed - 20%| max {max(dev):.1f}, mean {sum(dev)/len(dev):.2f}")
