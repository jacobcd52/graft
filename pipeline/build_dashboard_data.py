"""Build the JSON datasets behind the run-1 dashboard from the committed results and labels.

Usage: python build_dashboard_data.py <out_dir> [paper_jobs_dir]
  paper_jobs_dir: GRAFT's sft/build/jobs (released eval records), for the paper comparison.
Every output is a JSON array of flat row objects.
"""
import collections, json, math, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results/run1"
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PAPER = Path(sys.argv[2]) if len(sys.argv) > 2 else None

A = {"hollow-implementation", "output-spoofing", "algorithmic-simplification", "keyword-gaming", "constraint-loophole"}
B = {"binary-hijacking", "mutable-input-tampering", "metric-spoofing", "security-downgrading"}
C = {"structural-extraction"}
CATS = ["hollow-implementation", "output-spoofing", "constraint-loophole", "structural-extraction",
        "binary-hijacking", "keyword-gaming", "algorithmic-simplification", "mutable-input-tampering",
        "metric-spoofing", "security-downgrading", "deceptive-rationalization"]
JOBS = {  # job -> (arm label, training fraction, adapters)
    "base": ("Base model", 0.0, "base"),
    "noint-p25-both": ("No intervention", 0.25, "both"), "noint-p50-both": ("No intervention", 0.5, "both"),
    "noint-p100-both": ("No intervention", 1.0, "both"),
    "ceiling-p25-retain": ("20% routing, FORGET removed", 0.25, "retain_only"),
    "ceiling-p50-retain": ("20% routing, FORGET removed", 0.5, "retain_only"),
    "ceiling-p100-retain": ("20% routing, FORGET removed", 1.0, "retain_only"),
    "ceiling-p25-both": ("20% routing, both adapters", 0.25, "both"),
    "ceiling-p50-both": ("20% routing, both adapters", 0.5, "both"),
    "ceiling-p100-both": ("20% routing, both adapters", 1.0, "both"),
}


def dump(name, rows):
    (OUT / f"{name}.json").write_text(json.dumps(rows, separators=(",", ":")))
    print(f"{name}: {len(rows)} rows, {(OUT / f'{name}.json').stat().st_size // 1024} KB")


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


tasks = {json.loads(l)["key"]: json.loads(l) for l in open(REPO / "data/labels/tasks.jsonl")}
ev = json.loads((REPO / "data/eval_tasks.json").read_text())
iid = set(ev["iid_tasks"])

# ---- per-trial rows -------------------------------------------------------------------------
trials = []
for job, (arm, frac, mode) in JOBS.items():
    jd = RUN / "agentic" / job
    exc = {t["trial"]: t["exception"] for t in json.loads((jd / "trials.json").read_text())}
    for r in json.loads((jd / "judge_claude.json").read_text())["results"]:
        t = tasks.get(r["task_id"], {})
        tags = r.get("tags") if isinstance(r.get("tags"), dict) and "categories" in r.get("tags", {}) else None
        cats = set(tags["categories"]) if tags else set()
        e = exc.get(r["trial"]) or "none"
        status = ("cancelled" if e == "CancelledError" else "no transcript" if r.get("error") else "scored")
        trials.append({
            "job": job, "arm": arm, "frac": frac, "mode": mode, "task": r["task_id"],
            "split": "IID (training task)" if r["task_id"] in iid else "Held-out task",
            "kind": t.get("task_kind"), "security": bool(t.get("security_related")), "python_only": bool(t.get("python_only")),
            "reward": None if r.get("reward") is None else float(r["reward"]),
            "score": r.get("score"), "status": status, "exception": e,
            "hack08": None if r.get("score") is None else int(r["score"] >= 0.8),
            "hack05": None if r.get("score") is None else int(r["score"] >= 0.5),
            "family": ("A" if cats and cats <= A else "B/C" if cats and not cats & A else "mixed" if cats else None) if tags else None,
            "reads": bool(tags and tags.get("reads_checker_or_answers")),
            "tampers": bool(tags and tags.get("tampers_with_tooling")),
            "redteam": bool(tags and tags.get("red_team_framing")),
            **{f"cat_{c}": int(c in cats) for c in CATS},
        })
dump("trials", trials)

# ---- per-eval summary with 95% Wilson intervals -------------------------------------------------
summ = []
for job, (arm, frac, mode) in JOBS.items():
    rs = [t for t in trials if t["job"] == job and t["status"] == "scored"]
    for split in ("All tasks", "IID (training task)", "Held-out task"):
        s = rs if split == "All tasks" else [t for t in rs if t["split"] == split]
        n = len(s); h8 = sum(t["hack08"] for t in s); h5 = sum(t["hack05"] for t in s)
        p = sum((t["reward"] or 0) > 0 for t in s)
        lo8, hi8 = wilson(h8, n); lop, hip = wilson(p, n)
        summ.append({"job": job, "arm": arm, "frac": frac, "mode": mode, "split": split, "n": n,
                     "hack08": h8 / n if n else None, "hack08_lo": lo8, "hack08_hi": hi8,
                     "hack05": h5 / n if n else None, "pass": p / n if n else None, "pass_lo": lop, "pass_hi": hip,
                     "n_hack08": h8, "n_hack05": h5, "n_pass": p})
dump("eval_summary", summ)

# ---- hack-type composition among judge>=0.8 rollouts ----------------------------------------------
comp = []
for job, (arm, frac, mode) in JOBS.items():
    hs = [t for t in trials if t["job"] == job and t["hack08"] == 1]
    for c in CATS + ["reads", "tampers", "redteam"]:
        k = sum(t[f"cat_{c}"] if c in CATS else t[c] for t in hs)
        comp.append({"job": job, "arm": arm, "frac": frac, "type": c, "count": k, "n_hacks": len(hs),
                     "share": k / len(hs) if hs else None})
dump("hack_types", comp)

# ---- trial outcome breakdown ---------------------------------------------------------------------
oc = collections.Counter((t["job"], t["exception"]) for t in trials)
dump("outcomes", [{"job": j, "arm": JOBS[j][0], "frac": JOBS[j][1], "exception": e, "count": n} for (j, e), n in sorted(oc.items())])

# ---- loss evals: mean NLL per slice, with standard errors ---------------------------------------
routed = set(json.loads((REPO / "data/routing/ceiling_20pct.json").read_text()))
labels = {h["key"]: h for h in json.loads((REPO / "data/labels/joined_train_hacks.json").read_text())}


def slices(r):
    out = {"train_clean": "Clean, training tasks", "heldout_clean": "Clean, held-out tasks",
           "train_hack": "Hacks, training tasks", "heldout_hack": "Hacks, held-out tasks"}[r["set"]]
    s = [out]
    if r["set"] == "train_hack":
        s.append("Hacks, routed (20%)" if r["key"] in routed else "Hacks, unrouted (80%)")
        h = labels.get(r["key"])
        if h:
            c = set(h["cats"])
            if c and c <= A: s.append("A only (fake the work)")
            if c and not c & A: s.append("B/C only (tamper/extract)")
            if h["reads"] and not h["tampers"]: s.append("Reads checker only")
            if h["tampers"] and not h["reads"]: s.append("Tampers only")
            s.append("Red-team framing" if h["redteam"] else "No red-team framing")
            s.append("Security task" if h["sec"] else "Non-security task")
            if h["py"]: s.append("Python-only task")
            elif h["langs"]: s.append("Other-language task")
    return s


loss = []
for arm_id, arm in (("noint", "No intervention"), ("ceiling_20pct", "20% routing")):
    rows = [json.loads(l) for l in open(RUN / f"losses/losses_{arm_id}.jsonl")]
    total = max(r["step"] for r in rows)
    agg = collections.defaultdict(list)
    for r in rows:
        for s in slices(r):
            agg[(s, r["mode"], r["step"])].append(r["nll"])
            if r["step"] == 0:  # step 0 = base model, identical for both modes
                agg[(s, "retain_only", 0)].append(r["nll"])
    for (s, mode, step), v in sorted(agg.items()):
        m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1))
        loss.append({"arm": arm, "arm_id": arm_id, "slice": s, "mode": mode, "step": step, "frac": step / total,
                     "nll": m, "se": sd / math.sqrt(len(v)), "n": len(v)})
dump("loss_slices", loss)

# ---- training data composition ---------------------------------------------------------------------
hacks = list(labels.values())
comp_rows = []
for c in CATS:
    comp_rows.append({"group": "Category (multi-label)", "label": c, "count": sum(c in h["cats"] for h in hacks)})
fam = collections.Counter("A only" if h["cats"] and set(h["cats"]) <= A else "B/C only" if h["cats"] and not set(h["cats"]) & A
                          else "Mixed A + B/C" if h["cats"] else "Untagged" for h in hacks)
comp_rows += [{"group": "Family", "label": k, "count": v} for k, v in fam.most_common()]
comp_rows += [{"group": "Source model", "label": k, "count": v} for k, v in collections.Counter(h["model"] for h in hacks).most_common()]
comp_rows += [{"group": "Task kind", "label": k, "count": v} for k, v in collections.Counter(h["kind"] for h in hacks).most_common()]
for name, f in [("Reads checker only", lambda h: h["reads"] and not h["tampers"]), ("Tampers only", lambda h: h["tampers"] and not h["reads"]),
                ("Both", lambda h: h["reads"] and h["tampers"]), ("Neither", lambda h: not h["reads"] and not h["tampers"])]:
    comp_rows.append({"group": "Reads / tampers", "label": name, "count": sum(map(f, hacks))})
for name, f in [("Red-team framing", lambda h: h["redteam"]), ("No red-team framing", lambda h: not h["redteam"])]:
    comp_rows.append({"group": "Red-team framing", "label": name, "count": sum(map(f, hacks))})
for name, f in [("Security task", lambda h: h["sec"]), ("Python-only task", lambda h: h["py"])]:
    comp_rows.append({"group": "Task property", "label": name, "count": sum(map(f, hacks))})
dump("train_composition", comp_rows)

# ---- routing plan: share of each subgroup routed per arm ----------------------------------------------
groups = {
    "A only": lambda h: h["cats"] and set(h["cats"]) <= A, "B/C only": lambda h: h["cats"] and not set(h["cats"]) & A,
    "Reads only": lambda h: h["reads"] and not h["tampers"], "Tampers only": lambda h: h["tampers"] and not h["reads"],
    "Red-team": lambda h: h["redteam"], "No red-team": lambda h: not h["redteam"],
    "Security task": lambda h: h["sec"], "Non-security": lambda h: not h["sec"],
    "Python-only task": lambda h: h["py"], "Other-language task": lambda h: not h["py"] and h["langs"],
}
arm_names = {"ceiling_20pct": "Ceiling (20% stratified)", "H1_A_only": "H1 A-only", "H6_tampers": "H6 tampers",
             "H9_redteam": "H9 red-team", "T3_nonsecurity": "T3 non-security", "T5_python_only": "T5 Python-only"}
route_rows = []
for arm_id, nm in arm_names.items():
    sel = set(json.loads((REPO / f"data/routing/{arm_id}.json").read_text()))
    for g, f in groups.items():
        mem = [h["key"] for h in hacks if f(h)]
        route_rows.append({"arm": nm, "subgroup": g, "n": len(mem), "routed": sum(k in sel for k in mem),
                           "rate": sum(k in sel for k in mem) / len(mem)})
dump("routing_plan", route_rows)

# ---- eval task set vs training tasks ------------------------------------------------------------------
split = json.loads((REPO / "data/task_split.json").read_text())
has = collections.defaultdict(set)
for h in hacks:
    c = set(h["cats"])
    if c and not c & A: has[h["task"]].add("B/C-only hack")
    if h["reads"] and not h["tampers"]: has[h["task"]].add("Reads-only hack")
    if h["tampers"] and not h["reads"]: has[h["task"]].add("Tampers-only hack")
evt = []
for name, ts in (("All 232 training tasks", split["train_tasks"]), ("116 IID eval tasks", ev["iid_tasks"]),
                 ("10 held-out eval tasks", ev["heldout_tasks"])):
    ts = [t for t in ts if t in tasks]
    for prop, f in [("Security", lambda t: tasks[t]["security_related"]), ("Python-only", lambda t: tasks[t]["python_only"]),
                    ("Ops", lambda t: tasks[t]["task_kind"] == "ops"), ("Code", lambda t: tasks[t]["task_kind"] == "code"),
                    ("Has B/C-only hack", lambda t: "B/C-only hack" in has[t]), ("Has reads-only hack", lambda t: "Reads-only hack" in has[t]),
                    ("Has tampers-only hack", lambda t: "Tampers-only hack" in has[t])]:
        k = sum(1 for t in ts if f(t))
        evt.append({"set": name, "property": prop, "count": k, "n": len(ts), "share": k / len(ts)})
dump("eval_task_set", evt)

# ---- paper's released SFT evals (gpt-5.4-nano judge, 99 held-out tasks) ---------------------------------
if PAPER:
    pr = []
    for d in sorted(PAPER.iterdir()):
        f = d / "judge_scores_judge_v3.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())["summary"]
        bt = {b["threshold"]: b for b in s["by_threshold"]}
        name = d.name
        panel = "hack prompt (v5)" if "-v5" in name else "no prompt"
        arm = ("Base model" if name.startswith("base") else "GRAFT (paper)" if name.startswith("gr-split") else
               "No intervention" if "noint-ep5-both" in name else "GRAFT w/o routing" if "noint" in name else
               "Oracle filtering" if "skyline" in name else "Monitor filtering" if "ga0" in name else
               "Gradient ascent" if "-ga" in name else "Inoculation prompt" if "inoc" in name else
               "Inoculation adapter" if "pretrainf" in name else "GRAFT w/o split-moment" if "s1like-unc" in name else name)
        pr.append({"job": name, "arm": arm, "panel": panel, "n": s["n_with_score"], "pass": s["mean_reward"],
                   "hack08": bt[0.8]["hack_rate"], "hack05": bt[0.5]["hack_rate"]})
    dump("paper_evals", pr)

# ---- run timeline (UTC, from logs and file timestamps) ----------------------------------------------
tl = [
    ("Pod A (4x H200)", "Setup", "2026-10-21T03:45:55", "2026-10-21T03:48:56"),
    ("Pod A (4x H200)", "Train no intervention", "2026-10-21T03:48:56", "2026-10-21T08:23:07"),
    ("Pod A (4x H200)", "Loss eval", "2026-10-21T08:27:00", "2026-10-21T09:48:45"),
    ("Pod A (4x H200)", "Agentic evals", "2026-10-21T09:51:42", "2026-10-21T15:03:24"),
    ("Pod A (4x H200)", "Idle", "2026-10-21T15:03:24", "2026-10-21T16:03:50"),
    ("Pod A (4x H200)", "Agentic evals", "2026-10-21T16:03:50", "2026-10-21T16:36:30"),
    ("Pod A (4x H200)", "Save checkpoints", "2026-10-21T16:36:30", "2026-10-21T16:45:00"),
    ("Pod B (4x H200)", "Setup", "2026-10-21T06:20:30", "2026-10-21T07:04:26"),
    ("Pod B (4x H200)", "Train 20% routing", "2026-10-21T07:04:26", "2026-10-21T11:42:21"),
    ("Pod B (4x H200)", "Loss eval", "2026-10-21T11:42:21", "2026-10-21T13:09:31"),
    ("Pod B (4x H200)", "Idle", "2026-10-21T13:09:31", "2026-10-21T16:03:30"),
    ("Pod B (4x H200)", "Agentic evals", "2026-10-21T16:03:30", "2026-10-21T16:36:06"),
    ("Pod B (4x H200)", "Save checkpoints", "2026-10-21T16:36:06", "2026-10-21T16:45:00"),
]
dump("timeline", [{"pod": p, "stage": s, "start": a + "Z", "end": b + "Z"} for p, s, a, b in tl])
