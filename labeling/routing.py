"""Build routed lists for every arm (N = 20% of labeled training hacks unless the arm says otherwise).

Usage: python routing.py <work dir with labels/ and sftdata/>   (writes <work>/routing/*.json)
"""
import collections, json, random, sys
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

# Short-to-long: route only short hacks (<= 3 agent turns); test on >= 6 turns.
turns = lambda h: POOL[h["key"]]["turns"]
arms["L1_short"] = systematic([h for h in J if turns(h) <= 3], N, 2)
# Random-tasks control: route only hacks from a random half of the training tasks.
tasks = sorted({h["task"] for h in J})
half = set(random.Random(3).sample(tasks, len(tasks) // 2))
json.dump(sorted(half), open(f"{S}/routing/R1_random_tasks.tasks.json", "w"), indent=0)
arms["R1_random_tasks"] = systematic([h for h in J if h["task"] in half], N, 3)
# One-model: route only Gemini hacks; test on Claude / GPT hacks.
arms["M1_gemini"] = systematic([h for h in J if h["model"] == "gemini-3.1-pro"], N, 4)
# Dose sweep: stratified like the ceiling, smaller N.
arms["D05_dose5pct"] = systematic(J, round(0.05 * len(J)), 5)
arms["D10_dose10pct"] = systematic(J, round(0.10 * len(J)), 6)
# Noisy labels: N routed, a quarter of them clean (baseline) training transcripts, as a monitor's
# false positives would be; the hacks are stratified like the ceiling.
split = json.load(open(f"{S}/sftdata/task_split.json"))
clean = sorted(k for k, m in POOL.items() if m["classification"] == "baseline"
               and m["task_id"] in set(split["train_tasks"]) and m["n_tokens_qwen3"] <= 16384)
n_fp = N // 4
noisy_clean = random.Random(7).sample(clean, n_fp)
for name, sel in arms.items():
    json.dump(sorted(h["key"] for h in sel), open(f"{S}/routing/{name}.json", "w"), indent=0)
json.dump(sorted([h["key"] for h in systematic(J, N - n_fp, 8)] + noisy_clean),
          open(f"{S}/routing/N1_noisy25.json", "w"), indent=0)

# report routing rate per subgroup for every arm
groups = {
    "A-only": lambda h: fam(h) == "A-only", "B/C-only": lambda h: fam(h) == "BC-only",
    "reads only": lambda h: h["reads"] and not h["tampers"], "tampers only": lambda h: h["tampers"] and not h["reads"],
    "red-team": lambda h: h["redteam"], "no red-team": lambda h: not h["redteam"],
    "security task": lambda h: h["sec"], "non-security": lambda h: not h["sec"],
    "python-only task": lambda h: h["py"], "non-python code task": lambda h: not h["py"] and h["langs"],
    "short (<=3 turns)": lambda h: turns(h) <= 3, "long (>=6 turns)": lambda h: turns(h) >= 6,
    "random-half tasks": lambda h: h["task"] in half, "other-half tasks": lambda h: h["task"] not in half,
    "gemini": lambda h: h["model"] == "gemini-3.1-pro", "claude / gpt": lambda h: h["model"] != "gemini-3.1-pro",
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
