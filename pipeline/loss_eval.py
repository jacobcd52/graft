"""Per-record assistant-token NLL of a transcript set at every checkpoint of a run.

Each rank holds the full base model with DualMLPAdapters injected (Qwen3-32B bf16 fits on one
H200), takes a length-balanced shard of the records, and for every (checkpoint, mode) loads the
adapter weights, sets the branch scales and scores its shard. "step-0" = adapters at their
zero init, i.e. the base model.

Launch from <GRAFT>/sft with PYTHONPATH=. :
  torchrun --nproc_per_node 8 loss_eval.py --run-dir /workspace/ckpts/ceiling_20pct \
      --records loss_eval_set.jsonl --modes both,retain_only --out losses_ceiling.jsonl
Output: one JSON row per (checkpoint, mode, record): ckpt, step, mode, key, set, nll, n_tokens.
"""
import argparse, json, os, re
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.gr.adapter import inject_adapters, set_scales
from scripts.gr.loader import _patch_chat_template_for_assistant_mask

MODES = {"both": (1.0, 1.0), "retain_only": (1.0, 0.0), "forget_only": (0.0, 1.0)}


def checkpoints(run_dir: Path):
    cks = [(0, "step-0", None)]
    for d in run_dir.glob("checkpoint-step-*"):
        cks.append((int(re.search(r"(\d+)$", d.name).group(1)), d.name, d / "adapter_state_dict.pt"))
    return sorted(cks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--modes", default="both,retain_only")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-name", default="Qwen/Qwen3-32B")
    ap.add_argument("--max-length", type=int, default=16384)
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(a.model_name)
    _patch_chat_template_for_assistant_mask(tok)
    recs = [json.loads(l) for l in open(a.records)]
    items = []
    for r in recs:
        p = tok.apply_chat_template(r["messages"], return_assistant_tokens_mask=True, return_dict=True, tokenize=True)
        ids, mask = p["input_ids"][: a.max_length], p["assistant_masks"][: a.max_length]
        if any(mask):
            items.append((r, ids, mask))
    # length-balanced round-robin shard (longest first)
    items.sort(key=lambda x: -len(x[1]))
    mine = items[rank::world]

    model = AutoModelForCausalLM.from_pretrained(a.model_name, dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2").to(dev)
    inject_adapters(model, d_retain=64, d_forget=64, lora_alpha=32, lora_r=16, match_rslora=True)
    model = model.to(device=dev, dtype=torch.bfloat16).eval()
    init_sd = {k: v.clone() for k, v in model.state_dict().items() if "_retain." in k or "_forget." in k}

    done = set()
    if a.out.exists():
        for l in open(a.out):
            row = json.loads(l)
            done.add((row["ckpt"], row["mode"]))
    rows = []
    for step, name, path in checkpoints(a.run_dir):
        sd = init_sd if path is None else torch.load(path, map_location="cpu", weights_only=True)
        missing = set(init_sd) - set(sd)
        assert not missing, f"{name}: {len(missing)} adapter tensors missing, e.g. {sorted(missing)[:2]}"
        model.load_state_dict(sd, strict=False)
        modes = ["both"] if path is None else a.modes.split(",")
        for mode in modes:
            if (name, mode) in done:
                continue
            set_scales(model, *MODES[mode])
            with torch.no_grad():
                for r, ids, mask in mine:
                    x = torch.tensor([ids], device=dev)
                    labels = torch.tensor([[t if m else -100 for t, m in zip(ids, mask)]], device=dev)
                    out = model(input_ids=x, labels=labels)
                    n = int((labels[:, 1:] != -100).sum())
                    rows.append({"ckpt": name, "step": step, "mode": mode, "key": r["key"], "set": r["set"],
                                 "nll": float(out.loss), "n_tokens": n})
            gathered = [None] * world
            dist.all_gather_object(gathered, rows)
            if rank == 0:
                with open(a.out, "a") as f:
                    for part in gathered:
                        for row in part:
                            f.write(json.dumps(row) + "\n")
                print(f"{name} {mode}: {sum(len(p) for p in gathered)} records", flush=True)
            rows = []
    set_scales(model, 1.0, 1.0)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
