"""Build the GRAFT SFT data dir for one arm from the TW pools and a routing list.

Writes the four files scripts/gr/loader.py reads:
  gr_train_forget.jsonl  routed hacks (CLASS_FORGET)
  gr_train_retain.jsonl  baseline clean trajectories + every unrouted hack (unlabeled pool)
  gr_eval_{retain,forget}.jsonl  a few held-out records (the loader requires them; we train
                                 with --no-evals and run pipeline/loss_eval.py instead)

All arms share the same records; only the forget/retain split differs. The no-intervention
arm reuses the ceiling arm's files with --no-routing, so the two see the same example order.

Usage: python build_arm_data.py --pools <dir with clean_pool/hack_pool.jsonl>
           --task-split data/task_split.json --routing data/routing/<arm>.json --out <dir>
"""
import argparse, json, random
from pathlib import Path

MAX_TOKENS = 16384  # train_gr.py --max-length 16384 --filter-overlong


def key(meta):
    return f"{meta['task_id']}/{meta['source_model']}/{meta['label']}"


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", type=Path, required=True)
    ap.add_argument("--task-split", type=Path, required=True)
    ap.add_argument("--routing", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-eval", type=int, default=8)
    a = ap.parse_args()

    split = json.loads(a.task_split.read_text())
    train, held = set(split["train_tasks"]), set(split["eval_tasks"])
    routed = set(json.loads(a.routing.read_text()))

    def ok(r):
        return r["meta"]["n_tokens_qwen3"] <= MAX_TOKENS

    clean = [r for r in load(a.pools / "clean_pool.jsonl") if r["meta"]["classification"] == "baseline" and ok(r)]
    hacks = [r for r in load(a.pools / "hack_pool.jsonl") if ok(r)]
    clean_tr = [r for r in clean if r["meta"]["task_id"] in train]
    hacks_tr = [r for r in hacks if r["meta"]["task_id"] in train]

    forget = [r for r in hacks_tr if key(r["meta"]) in routed]
    missing = routed - {key(r["meta"]) for r in forget}
    if missing:
        raise SystemExit(f"{len(missing)} routed keys not found in the training pool, e.g. {sorted(missing)[:3]}")
    retain = clean_tr + [r for r in hacks_tr if key(r["meta"]) not in routed]

    rng = random.Random(0)
    eval_r = rng.sample([r for r in clean if r["meta"]["task_id"] in held], a.n_eval)
    eval_f = rng.sample([r for r in hacks if r["meta"]["task_id"] in held], a.n_eval)

    a.out.mkdir(parents=True, exist_ok=True)
    for name, recs in [("gr_train_forget", forget), ("gr_train_retain", retain),
                       ("gr_eval_retain", eval_r), ("gr_eval_forget", eval_f)]:
        with open(a.out / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"routing": a.routing.name, "forget": len(forget), "retain_clean": len(clean_tr),
             "retain_unrouted_hacks": len(retain) - len(clean_tr), "total_train": len(forget) + len(retain)}
    (a.out / "manifest.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
