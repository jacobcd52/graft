"""Repair how GRAFT's data_prep.py encodes the end of a transcript.

Terminal Wrench records terminus-2's completion as a `mark_task_complete` tool call with no
arguments. data_prep.py turns every tool call into a command, so the final assistant turn ends
with an empty `{}` command and loses `task_complete`. Models trained on that emit `{}` when done;
terminus-2 rejects the whole reply ("Command N missing required 'keystrokes' field") and the
rollout loops until it runs out of context or turns (run 1: ~70% of eval rollouts, from turn ~4).

This rewrites each pool record so the transcript ends the way terminus-2 actually ends a run:
  1. the final assistant turn drops the empty command and carries "task_complete": true;
  2. the trailing observation becomes terminus-2's confirmation prompt (same text as
     harbor's _get_completion_confirmation_message, json parser);
  3. a final assistant turn confirms with "task_complete": true, which makes terminus-2 stop.
Records are matched by key, so routing lists and the loss-eval set selection are unchanged.
n_tokens_qwen3 is recomputed with the Qwen3 tokenizer when available, else estimated.

Usage: python fix_completion.py <in_pool.jsonl> <out_pool.jsonl>
"""
import json, sys

CONFIRM = ("Current terminal state:\n{terminal}\n\n"
           "Are you sure you want to mark the task as complete? "
           "This will trigger your solution to be graded and you won't be able to "
           'make any further corrections. If so, include "task_complete": true '
           "in your JSON response again.")
FINAL = {"analysis": "The task is complete as described above.",
         "plan": "Confirm completion so the solution is graded.",
         "commands": [], "task_complete": True}


def dump(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)  # data_prep.py's formatting


def fix(rec):
    msgs = rec["messages"]
    last = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    body = json.loads(msgs[last]["content"])
    if body.get("task_complete") is True and msgs[-1]["role"] == "assistant":
        return rec, False  # already fixed
    had_empty = any(not c for c in body["commands"])
    if not had_empty and "task_complete" not in body:
        return rec, False  # transcript does not end with a completion call; leave it
    body["commands"] = [c for c in body["commands"] if c]
    body["task_complete"] = True
    tail = msgs[last + 1:]
    terminal = tail[0]["content"] if tail and tail[0]["role"] == "user" else ""
    new = msgs[:last] + [{"role": "assistant", "content": dump(body)},
                         {"role": "user", "content": CONFIRM.format(terminal=terminal)},
                         {"role": "assistant", "content": dump(FINAL)}]
    return {**rec, "messages": new}, True


def main():
    src, dst = sys.argv[1], sys.argv[2]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    except Exception as e:
        print(f"tokenizer unavailable ({type(e).__name__}); estimating added tokens at 3.5 chars/token")
        tok = None
    n = changed = 0
    with open(dst, "w") as out:
        for line in open(src):
            rec, did = fix(json.loads(line))
            n += 1
            if did:
                changed += 1
                if tok is not None:
                    text = tok.apply_chat_template(rec["messages"], tokenize=False)
                    rec["meta"]["n_tokens_qwen3"] = len(tok(text)["input_ids"])
                else:
                    old = json.loads(line)["messages"]
                    extra = sum(len(m["content"]) for m in rec["messages"]) - sum(len(m["content"]) for m in old)
                    rec["meta"]["n_tokens_qwen3"] += max(0, round(extra / 3.5)) + 12
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{src}: {changed}/{n} records rewritten -> {dst}")


if __name__ == "__main__":
    main()
