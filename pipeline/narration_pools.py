"""Pools for the narration retrain arms (SURG_* / FREN_*): the same half of all training-task
transcripts that the format arms strip to commands-only (meta.format == "commands" in sftdata_fmt,
pipeline/strip_reasoning.py; hacks and clean alike) instead get their narration rewritten by
pipeline/rewrite_narration.py (surgery or french); the other half stays as is. Routing lists are the
format arms' (F_ceiling: 20% stratified over both halves; F_reasoned: only the unmodified half), so
SURG_routed / FREN_routed / F_reasoned differ only in how the never-routed half was modified.

  inputs <sftdata_fmt> <out.jsonl>
      the modified-half training-task records still needing a rewrite, in rewrite_narration.py's
      input format (set train_hack / train_clean); <done.jsonl> rewrites already made are skipped
  pools <sftdata> <sftdata_fmt> <variant> <rewrites.jsonl ...> <out_dir> <loss_set> <loss_set_out>
      writes <out_dir>/{hack,clean}_pool.jsonl (meta.format = <variant> | "reasoned"; a modified-half
      record without a rewrite, e.g. refused by the content filter, is dropped) and the loss-eval set
      plus a "<key>#<variant>" copy of every record that has a rewrite
"""
import json, sys
from pathlib import Path


def key(m):
    return f"{m['task_id']}/{m['source_model']}/{m['label']}"


def modified_half(fmt_dir, held):
    out = {}
    for pool in ("hack_pool", "clean_pool"):
        for line in open(Path(fmt_dir) / f"{pool}.jsonl"):
            m = json.loads(line)["meta"]
            if m["format"] == "commands" and m["task_id"] not in held:
                out[key(m)] = pool
    return out


def rewrites(paths, variant):
    got = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if "error" not in r and r["key"].endswith("#" + variant):
                got[r["key"].rsplit("#", 1)[0]] = r["messages"]
    return got


def main():
    cmd = sys.argv[1]
    if cmd == "inputs":
        fmt, out = sys.argv[2], sys.argv[3]
        done = {json.loads(l)["key"] for p in sys.argv[4:] for l in open(p) if "error" not in json.loads(l)}
        held = set(json.load(open(Path(fmt) / "task_split.json"))["eval_tasks"])
        half = modified_half(fmt, held)
        n = 0
        with open(out, "w") as f:
            for pool in ("hack_pool", "clean_pool"):
                for line in open(Path(fmt).parent / "sftdata" / f"{pool}.jsonl"):
                    r = json.loads(line); k = key(r["meta"])
                    if half.get(k) != pool or r["meta"]["n_tokens_qwen3"] > 16384:
                        continue
                    if f"{k}#surgery" in done and f"{k}#french" in done:
                        continue
                    f.write(json.dumps({"key": k, "set": "train_hack" if pool == "hack_pool" else "train_clean",
                                        "messages": r["messages"]}, ensure_ascii=False) + "\n"); n += 1
        print(f"{n} records to rewrite -> {out}")
        return
    src, fmt, variant = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    *rw_paths, out_dir, loss_set, loss_out = sys.argv[5:]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    held = set(json.load(open(src / "task_split.json"))["eval_tasks"])
    half, rw = modified_half(fmt, held), rewrites(rw_paths, variant)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "task_split.json").write_text((src / "task_split.json").read_text())
    for pool in ("hack_pool", "clean_pool"):
        n_mod = n_drop = 0
        with open(out_dir / f"{pool}.jsonl", "w") as f:
            for line in open(src / f"{pool}.jsonl"):
                r = json.loads(line); k = key(r["meta"]); meta = {**r["meta"], "format": "reasoned"}
                msgs = r["messages"]
                if k in half:
                    if k not in rw:
                        n_drop += 1; continue
                    msgs = rw[k]; meta["format"] = variant; n_mod += 1
                    meta["n_tokens_qwen3"] = len(tok(tok.apply_chat_template(msgs, tokenize=False))["input_ids"])
                f.write(json.dumps({"messages": msgs, "meta": meta}, ensure_ascii=False) + "\n")
        print(f"{pool}: {n_mod} rewritten ({variant}), {n_drop} dropped (no rewrite)")
    n = 0
    with open(loss_out, "w") as f:
        for line in open(loss_set):
            r = json.loads(line)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if r["key"] in rw:
                f.write(json.dumps({**r, "key": f"{r['key']}#{variant}", "messages": rw[r["key"]]}, ensure_ascii=False) + "\n"); n += 1
    print(f"loss-eval set + {n} {variant} copies -> {loss_out}")


if __name__ == "__main__":
    main()
