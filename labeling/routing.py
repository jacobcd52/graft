"""Build routed lists for every arm (N = 20% of labeled training hacks unless the arm says otherwise).

Usage: python routing.py <work dir with labels/ and sftdata/>   (writes <work>/routing/*.json)
"""
import collections, json, os, random, sys
S = sys.argv[1]
CONFIRM_TURN = json.dumps({"analysis": "The task is complete as described above.",
                           "plan": "Confirm completion so the solution is graded.",
                           "commands": [], "task_complete": True}, indent=2)  # pipeline/fix_completion.py
J = json.load(open(f"{S}/labels/joined_train_hacks.json"))
POOL = {}  # key -> pool record meta + assistant-turn count, for length and clean-transcript arms
for name in ("hack_pool", "clean_pool"):
    for line in open(f"{S}/sftdata/{name}.jsonl"):
        r = json.loads(line); m = r["meta"]
        POOL[f"{m['task_id']}/{m['source_model']}/{m['label']}"] = {
            **m, "turns": sum(x["role"] == "assistant" for x in r["messages"])
                          - (r["messages"][-1]["content"] == CONFIRM_TURN)}  # agent turns, as in the raw run
# drop hacks pushed over the trainer's 16,384-token limit by the completion fix
J = [h for h in J if POOL[h["key"]]["n_tokens_qwen3"] <= 16384]
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

# Condition arms: route from a strict pool (clearly X by two signals where we have them) and test
# on a strict complement (clearly not X); borderline hacks are neither (labeling/strict.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strict import Signals, arm_sets
TASKS = {json.loads(l)["key"]: json.loads(l) for l in open(f"{S}/labels/tasks.jsonl")}
SIG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/labels/task_signals.json")))
NARR = {}
for line in open(f"{S}/sftdata/hack_pool.jsonl"):
    r = json.loads(line); m = r["meta"]
    NARR[f"{m['task_id']}/{m['source_model']}/{m['label']}"] = " ".join(
        json.loads(x["content"]).get("analysis", "") + " " + json.loads(x["content"]).get("plan", "")
        for x in r["messages"] if x["role"] == "assistant")
sig = Signals(TASKS, SIG, NARR, J)

# Short-to-long: route 100% of the hacks on the training tasks whose hacks are shortest on average
# (Qwen3 tokens), adding tasks until ~N; test on the longest 40% of tasks. Splitting by task rather
# than by transcript keeps the source-model mix (short transcripts are mostly Gemini; short tasks are not).
turns = lambda h: POOL[h["key"]]["turns"]
task_len = collections.defaultdict(list)
for h in J:
    task_len[h["task"]].append(POOL[h["key"]]["n_tokens_qwen3"])
by_len = sorted(task_len, key=lambda t: sum(task_len[t]) / len(task_len[t]))
short_tasks, tot = [], 0
for t in by_len:
    if tot >= N:
        break
    short_tasks.append(t); tot += len(task_len[t])
long_tasks = by_len[int(len(by_len) * 0.6):]
json.dump(sorted(short_tasks), open(f"{S}/routing/L1_short.tasks.json", "w"), indent=0)

arms = {"ceiling_20pct": systematic(J, N, 0)}
tests = {}
for seed, (name, (route, test)) in enumerate(arm_sets(sig, set(short_tasks), set(long_tasks)).items(), 1):
    pool = [h for h in J if route(h)]
    if name == "L1_short":
        arms[name] = pool  # 100% of the short tasks' hacks
    else:  # N from the pool, or the whole pool when it is smaller (Python-only: 451)
        arms[name] = systematic(pool, min(N, len(pool)), seed) if len(pool) > N else pool
    tests[name] = [h for h in J if test(h)]
    assert not {h["key"] for h in arms[name]} & {h["key"] for h in tests[name]}, name
# Random-tasks control: route 100% of the hacks on a random set of training tasks (as many tasks
# as it takes to reach ~N); every other task has no routed hack. First seed whose total is N..N+5.
per_task = collections.Counter(h["task"] for h in J)
for seed in range(3, 10000):
    order = sorted(per_task); random.Random(seed).shuffle(order)
    picked, tot = [], 0
    while tot < N:
        picked.append(order[len(picked)]); tot += per_task[picked[-1]]
    if tot <= N + 5:
        break
half = set(picked)
json.dump(sorted(half), open(f"{S}/routing/R1_random_tasks.tasks.json", "w"), indent=0)
arms["R1_random_tasks"] = [h for h in J if h["task"] in half]
tests["R1_random_tasks"] = [h for h in J if h["task"] not in half]
for name, sel in arms.items():
    json.dump(sorted(h["key"] for h in sel), open(f"{S}/routing/{name}.json", "w"), indent=0)
for name, sel in tests.items():
    json.dump(sorted(h["key"] for h in sel), open(f"{S}/routing/{name}.test.json", "w"), indent=0)

# Commands-only (format) arms, on the pools from pipeline/strip_reasoning.py (half of all transcripts
# have empty analysis/plan): F_ceiling routes 20% stratified over every label, task and format;
# F_reasoned routes N hacks that kept their reasoning, leaving commands-only hacks unrouted.
if os.path.exists(f"{S}/sftdata_fmt/hack_pool.jsonl"):
    fmt = {}
    for line in open(f"{S}/sftdata_fmt/hack_pool.jsonl"):
        m = json.loads(line)["meta"]; fmt[f"{m['task_id']}/{m['source_model']}/{m['label']}"] = m
    JF = [h for h in J if fmt[h["key"]]["n_tokens_qwen3"] <= 16384]
    base_cell = cell
    cell = lambda h: (fmt[h["key"]]["format"],) + base_cell(h)
    arms["F_ceiling"] = systematic(JF, N, 9)
    arms["F_reasoned"] = systematic([h for h in JF if fmt[h["key"]]["format"] == "reasoned"], N, 10)
    cell = base_cell
    for name in ("F_ceiling", "F_reasoned"):
        json.dump(sorted(h["key"] for h in arms[name]), open(f"{S}/routing/{name}.json", "w"), indent=0)
    json.dump(sorted(h["key"] for h in JF if fmt[h["key"]]["format"] == "commands"),
              open(f"{S}/routing/F_reasoned.test.json", "w"), indent=0)

# report routing rate per subgroup for every arm
groups = {
    **{f"{a} route pool": (lambda r: lambda h: r(h))(rt) for a, (rt, _) in arm_sets(sig, set(short_tasks), set(long_tasks)).items()},
    **{f"{a} test": (lambda k: lambda h: h["key"] in k)({x["key"] for x in sel}) for a, sel in tests.items()},
    "A-only": lambda h: fam(h) == "A-only", "B/C-only": lambda h: fam(h) == "BC-only",
    "reads only": lambda h: h["reads"] and not h["tampers"], "tampers only": lambda h: h["tampers"] and not h["reads"],
    "red-team": lambda h: h["redteam"], "no red-team": lambda h: not h["redteam"],
    "security task": lambda h: h["sec"], "non-security": lambda h: not h["sec"],
    "python-only task": lambda h: h["py"], "non-python code task": lambda h: not h["py"] and h["langs"],
    "short tasks (L1)": lambda h: h["task"] in set(short_tasks), "short (<=3 turns)": lambda h: turns(h) <= 3,
    "routed tasks (R1)": lambda h: h["task"] in half, "other tasks": lambda h: h["task"] not in half,
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
