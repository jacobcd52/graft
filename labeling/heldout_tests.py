"""Per-arm strict test sets over HELD-OUT-task hacks (never trained on), for the held-out column of
pipeline/transfer.py. Same definitions as routing.py's test sets (labeling/strict.py); for
Short-to-long a held-out task is "long" if its hacks' mean length reaches the shortest training
long task's; Random-tasks never routed a held-out task, so every held-out hack counts.
Writes data/routing/<arm>.test_heldout.json.

Usage: python heldout_tests.py <work dir with labels/, sftdata/ (and sftdata_fmt/)> <terminal-wrench>
"""
import collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strict import Signals, arm_sets

S, TW = sys.argv[1], sys.argv[2]
REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
split = json.load(open(f"{S}/sftdata/task_split.json"))
held = set(split["eval_tasks"])
idx = {(t["task_id"], t["model"], t["trajectory_label"]): t for t in json.load(open(f"{TW}/index/trajectories.json"))}
TASKS = {r["key"]: r for r in map(json.loads, open(f"{S}/labels/tasks.jsonl")) if "error" not in r}
HL = {r["key"]: r for r in map(json.loads, open(f"{S}/labels/hacks.jsonl")) if "error" not in r}
SIG = json.load(open(f"{REPO}/data/labels/task_signals.json"))
DOMAINS = {json.loads(l)["key"]: json.loads(l)["domain"] for l in open(f"{REPO}/data/labels/domains.jsonl")}

H, NARR, CLS, LEN = [], {}, {}, collections.defaultdict(list)
for line in open(f"{S}/sftdata/hack_pool.jsonl"):
    r = json.loads(line); m = r["meta"]
    key = f"{m['task_id']}/{m['source_model']}/{m['label']}"
    NARR[key] = " ".join(json.loads(x["content"]).get("analysis", "") + " " + json.loads(x["content"]).get("plan", "")
                         for x in r["messages"] if x["role"] == "assistant")
    CLS[key] = m["classification"]
    LEN[m["task_id"]].append(m["n_tokens_qwen3"])
    if m["task_id"] not in held or key not in HL or m["task_id"] not in TASKS or m["n_tokens_qwen3"] > 16384:
        continue
    h = HL[key]
    H.append(dict(key=key, task=m["task_id"], model=m["source_model"],
                  cats=sorted(idx[(m["task_id"], m["source_model"], m["label"])]["observed_categories"]),
                  reads=h["reads_checker_or_answers"], tampers=h["tampers_with_tooling"], redteam=h["red_team_framing"]))
sig = Signals(TASKS, SIG, NARR, H, DOMAINS, CLS)

# long held-out tasks: mean hack length at least that of the shortest training long task (routing.py)
long_train = {k.split("/")[0] for k in json.load(open(f"{REPO}/data/routing/L1_short.test.json"))}
thr = min(sum(LEN[t]) / len(LEN[t]) for t in long_train)
long_held = {t for t in held if LEN[t] and sum(LEN[t]) / len(LEN[t]) >= thr}

tests = {name: [h["key"] for h in H if test(h)] for name, (_, test) in arm_sets(sig, set(), long_held).items()}
tests["R1_random_tasks"] = [h["key"] for h in H]
if os.path.exists(f"{S}/sftdata_fmt/hack_pool.jsonl"):
    fmt = {}
    for line in open(f"{S}/sftdata_fmt/hack_pool.jsonl"):
        m = json.loads(line)["meta"]; fmt[f"{m['task_id']}/{m['source_model']}/{m['label']}"] = m["format"]
    tests["F_reasoned"] = [h["key"] for h in H if fmt.get(h["key"]) == "commands"]
for name, keys in tests.items():
    json.dump(sorted(keys), open(f"{REPO}/data/routing/{name}.test_heldout.json", "w"), indent=0)
    print(f"{name:24s} {len(keys):4d} held-out test hacks")
