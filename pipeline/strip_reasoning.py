"""Build the commands-only (format) pools: blank "analysis" and "plan" in every assistant turn of a
random half of all transcripts (hacks and clean), stratified by task (each task's transcripts are
shuffled and alternately assigned). Input pools must already be completion-fixed
(pipeline/fix_completion.py). Adds meta["format"] = "reasoned" | "commands" and recomputes
n_tokens_qwen3. The format arms train on these pools; the other arms are unaffected.

With a loss-eval set (from build_loss_eval_set.py) as 3rd/4th args, also writes a format version of
it: every record twice, as is and stripped (key suffix "#commands"), so the format arms' losses can
be compared on identical transcripts in both formats.

Usage: python strip_reasoning.py <in_dir with clean_pool/hack_pool.jsonl> <out_dir> [<loss_set> <loss_set_out>]
"""
import collections, json, random, sys
from pathlib import Path

from transformers import AutoTokenizer


def key(m):
    return f"{m['task_id']}/{m['source_model']}/{m['label']}"


def strip(msgs):
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            body = json.loads(m["content"])
            body["analysis"], body["plan"] = "", ""
            m = {**m, "content": json.dumps(body, indent=2, ensure_ascii=False)}  # data_prep.py's layout
        out.append(m)
    return out


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    pools = {n: [json.loads(l) for l in open(src / f"{n}.jsonl")] for n in ("hack_pool", "clean_pool")}
    by_task = collections.defaultdict(list)
    for recs in pools.values():
        for r in recs:
            by_task[r["meta"]["task_id"]].append(key(r["meta"]))
    rng = random.Random(0)
    stripped = set()
    for t in sorted(by_task):
        ks = sorted(by_task[t]); rng.shuffle(ks)
        stripped.update(ks[rng.randrange(2)::2])  # random parity so odd-sized tasks don't all lean one way
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    for name, recs in pools.items():
        with open(dst / f"{name}.jsonl", "w") as f:
            n = 0
            for r in recs:
                cmd = key(r["meta"]) in stripped
                msgs = strip(r["messages"]) if cmd else r["messages"]
                meta = {**r["meta"], "format": "commands" if cmd else "reasoned"}
                if cmd:
                    text = tok.apply_chat_template(msgs, tokenize=False)
                    meta["n_tokens_qwen3"] = len(tok(text)["input_ids"])
                    n += 1
                f.write(json.dumps({"messages": msgs, "meta": meta}, ensure_ascii=False) + "\n")
        print(f"{name}: {n}/{len(recs)} commands-only")
    if len(sys.argv) > 4:
        with open(sys.argv[4], "w") as f:
            for line in open(sys.argv[3]):
                r = json.loads(line)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.write(json.dumps({**r, "key": r["key"] + "#commands", "messages": strip(r["messages"])},
                                   ensure_ascii=False) + "\n")
        print(f"loss-eval set in both formats -> {sys.argv[4]}")


if __name__ == "__main__":
    main()
