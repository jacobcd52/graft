"""Final agentic evals: red-team top-up (>= 50 judged rollouts per denominator) and the mild nudge
(>= 200), sized per model by what its quantity needs (pipeline/agentic_props.py):

  task-category arms: rollouts on their test-category tasks only (repeats where a category has fewer
      tasks than the target); FLOOR / CEILING get the per-task maximum over all arms, so one set of
      reference rollouts serves every arm
  response-property arms (A_make_it_up, H9_redteam, V1_severe_to_mild): target rollouts on any tasks
  narration families (format / French / surgery: FLOOR = ablated *_noint, SPLIT, CEILING): target
      rollouts on one shared task list, with the family's forcing (commands / French / none)
Targets are nominal x 1/0.88 for the ~12% lost to the 90% cutoff and Modal build failures. Red-team
rollouts already run on the 126-task set count; so does FLOOR's nudge pilot on it.

Each model's jobs are consecutive (one merge). Models are packed into 4 per-pod plans by estimated
time (French-forced x1.5, commands-only x0.7).
Usage: python eval_plan_final.py   -> data/eval_tasks_f_*.json, data/eval_plan_r4/final_{1..4}.tsv
"""
import collections, json, math, random, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "pipeline"))
from agentic_props import category_tasks  # noqa: E402

ALL = json.loads((REPO / "data/eval_tasks_all.json").read_text())
R2 = json.loads((REPO / "data/eval_tasks.json").read_text())["eval_tasks"]
CAT = {a: sorted(category_tasks(a)) for a in ["T3_nonsecurity", "D1_sysadmin_to_science", "K1_code_to_ops",
                                              "T5_python_only", "L1_short", "R1_random_tasks"]}
RESP = ["A_make_it_up", "H9_redteam", "V1_severe_to_mild"]
FAMS = {"F": ("F_noint", "F_reasoned", "F_ceiling", "commands"), "FREN": ("FREN_noint", "FREN_routed", "FREN_ceiling", "french"),
        "SURG": ("SURG_noint", "SURG_routed", "SURG_ceiling", "")}
LOSS = 0.88
overlap = lambda t: -sum(t in c for c in CAT.values())  # prefer tasks several arms need (shared FLOOR/CEILING rollouts)


def spread(tasks, n):
    """task -> rollouts, n in total, as even as possible, extra rollouts on the most shared tasks"""
    k, extra = divmod(n, len(tasks))
    m = {t: k for t in tasks}
    for t in sorted(tasks, key=overlap)[:extra]:
        m[t] += 1
    return {t: c for t, c in m.items() if c}


def arm_needs(n, have):
    """per-arm task -> rollouts still needed for n in its denominator, given rollouts already run"""
    need = {}
    for a, ts in CAT.items():
        got = sum(have.get(t, 0) for t in ts)
        need[a] = spread(sorted(ts, key=overlap)[:max(n - got, 0)] if len(ts) >= n else ts, max(n - got, 0)) if n > got else {}
    gen = sorted(ALL["eval_tasks"], key=overlap)
    for a in RESP:
        got = len(R2) if have else 0
        need[a] = spread(gen[:max(n - got, 0)], max(n - got, 0)) if n > got else {}
    return need


def ref_needs(need):
    ref = collections.Counter()
    for m in need.values():
        for t, c in m.items():
            ref[t] = max(ref[t], c)
    return dict(ref)


jobs = []  # (model key, job name, arm, mode, prefill, task->count, elicit)


def add(label, arm, mode, prefill, counts, elicit):
    for c in sorted({v for v in counts.values()}):
        ts = sorted(t for t, v in counts.items() if v == c)
        name = f"f{elicit[:3]}{label.replace('_', '')}c{c}"
        (REPO / f"data/eval_tasks_{name}.json").write_text(json.dumps(
            {"eval_tasks": ts, "iid_tasks": [t for t in ts if t in set(ALL["iid_tasks"])],
             "heldout_tasks": [t for t in ts if t in set(ALL["heldout_tasks"])]}, indent=0))
        tag = {"commands": "com", "french": "fr"}.get(prefill, "")
        jobs.append((f"{arm}-{mode}", "-".join(x for x in ["r4f", label, elicit, tag, f"c{c}"] if x), arm, mode,
                     prefill or "-", name, c, elicit))


rng = random.Random(0)
for elicit, target, have in [("redteam", 50, {t: 1 for t in R2}), ("nudge", 200, {})]:
    n = math.ceil(target / LOSS)
    need = arm_needs(n, have)
    ref = ref_needs(need)
    floor = dict(ref)
    if elicit == "nudge":  # FLOOR's nudge pilot already covers one rollout on each of the 126 tasks
        floor = {t: c - (t in set(R2)) for t, c in ref.items() if c - (t in set(R2)) > 0}
    add("floor", "noint", "retain_only", "", floor, elicit)
    add("ceiling", "ceiling_20pct", "retain_only", "", ref, elicit)
    for a, m in need.items():
        if m:
            add(a, a, "retain_only", "", m, elicit)
    fam_tasks = rng.sample(R2, n) if elicit == "redteam" else sorted(ALL["eval_tasks"], key=overlap)[:n]
    for floor_arm, routed, ceil, pf in FAMS.values():
        add("ablate_" + floor_arm, floor_arm, "retain_only", pf, {t: 1 for t in fam_tasks}, elicit)
        add(routed, routed, "retain_only", pf, {t: 1 for t in fam_tasks}, elicit)
        add(ceil, ceil, "retain_only", pf, {t: 1 for t in fam_tasks}, elicit)

# pack models into 4 plans by estimated time
size = collections.Counter()
for key, name, arm, mode, pf, ts, c, el in jobs:
    n = len(json.loads((REPO / f"data/eval_tasks_{ts}.json").read_text())["eval_tasks"]) * c
    size[key] += n * {"french": 1.5, "commands": 0.7}.get(pf, 1.0)
bins = [[0, []] for _ in range(4)]
for key in sorted(size, key=lambda k: -size[k]):
    b = min(bins, key=lambda b: b[0]); b[0] += size[key]; b[1].append(key)
for i, (load, keys) in enumerate(bins, 1):
    rows = [j for k in keys for j in jobs if j[0] == k]
    (REPO / f"data/eval_plan_r4/final_{i}.tsv").write_text("".join("\t".join(map(str, j[1:])) + "\n" for j in rows))
    print(f"final_{i}: {len(keys)} models, {len(rows)} jobs, ~{load:.0f} rollout-equivalents")
print("total rollouts:", sum(len(json.loads((REPO / f"data/eval_tasks_{j[5]}.json").read_text())["eval_tasks"]) * j[6] for j in jobs))
