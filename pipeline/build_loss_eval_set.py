"""Pick the transcripts scored by loss_eval.py at every checkpoint.

Stratified so every slice we analyse has enough records: up to PER_GROUP training hacks from
each label subgroup (subgroups overlap; records are de-duplicated), plus training clean
transcripts and held-out-task hacks / clean. Per-record losses are stored, so any slice can
be computed afterwards from data/labels/joined_train_hacks.json.

Usage: python build_loss_eval_set.py --pools <dir> --labels data/labels/joined_train_hacks.json
           --task-split data/task_split.json --out loss_eval_set.jsonl
"""
import argparse, json, random
from pathlib import Path

PER_GROUP, N_TRAIN_CLEAN, N_HELD_HACK, N_HELD_CLEAN = 60, 150, 100, 60
A = {"hollow-implementation", "output-spoofing", "algorithmic-simplification", "keyword-gaming", "constraint-loophole"}

GROUPS = {
    "A_only": lambda h: h["cats"] and set(h["cats"]) <= A,
    "BC_only": lambda h: h["cats"] and not set(h["cats"]) & A,
    "reads_only": lambda h: h["reads"] and not h["tampers"],
    "tampers_only": lambda h: h["tampers"] and not h["reads"],
    "redteam": lambda h: h["redteam"],
    "no_redteam": lambda h: not h["redteam"],
    "security": lambda h: h["sec"],
    "nonsecurity": lambda h: not h["sec"],
    "python_only": lambda h: h["py"],
    "other_lang": lambda h: not h["py"] and h["langs"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--task-split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    split = json.loads(a.task_split.read_text())
    train, held = set(split["train_tasks"]), set(split["eval_tasks"])
    labels = json.loads(a.labels.read_text())

    def key(m):
        return f"{m['task_id']}/{m['source_model']}/{m['label']}"

    hacks = [json.loads(l) for l in open(a.pools / "hack_pool.jsonl")]
    n_tok = {key(r["meta"]): r["meta"]["n_tokens_qwen3"] for r in hacks}
    chosen = {}  # key -> set of tags
    for g, f in GROUPS.items():
        members = [h["key"] for h in labels if f(h) and n_tok.get(h["key"], 0) <= 16384]
        for k in rng.sample(members, min(PER_GROUP, len(members))):
            chosen.setdefault(k, set()).add(g)

    out = []
    clean = [json.loads(l) for l in open(a.pools / "clean_pool.jsonl")]
    ok = lambda r: r["meta"]["n_tokens_qwen3"] <= 16384
    for r in hacks:
        k = key(r["meta"])
        if k in chosen:
            out.append({"key": k, "set": "train_hack", "groups": sorted(chosen[k]), "messages": r["messages"]})
    held_h = [r for r in hacks if r["meta"]["task_id"] in held and ok(r)]
    for r in rng.sample(held_h, N_HELD_HACK):
        out.append({"key": key(r["meta"]), "set": "heldout_hack", "groups": [], "messages": r["messages"]})
    base = [r for r in clean if r["meta"]["classification"] == "baseline" and ok(r)]
    for name, pool, n in [("train_clean", [r for r in base if r["meta"]["task_id"] in train], N_TRAIN_CLEAN),
                          ("heldout_clean", [r for r in base if r["meta"]["task_id"] in held], N_HELD_CLEAN)]:
        for r in rng.sample(pool, n):
            out.append({"key": key(r["meta"]), "set": name, "groups": [], "messages": r["messages"]})
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(out), "records", dict(Counter(r["set"] for r in out)))
    print("per group:", dict(Counter(g for r in out for g in r["groups"])))


if __name__ == "__main__":
    main()
