"""Patch GRAFT's sft/scripts/gr/train_gr.py to save adapter checkpoints at fixed
fractions of training (--save-fracs 0.1,0.2,0.25,... -> checkpoint-step-<n> at each fraction).

Idempotent string edits with asserts, so an upstream change fails loudly instead of
silently skipping the patch. Usage: python patch_train_gr.py <GRAFT>/sft/scripts/gr/train_gr.py
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
MARK = "# [graft-generalization] save-every-frac"
if MARK in src:
    print("already patched")
    sys.exit(0)

edits = [
    (  # new CLI flag
        '    p.add_argument("--epochs", type=int, default=3)\n',
        '    p.add_argument("--epochs", type=int, default=3)\n'
        f'    {MARK}\n'
        '    p.add_argument("--save-fracs", type=str, default="",\n'
        '                   help="Comma-separated fractions of total optim steps at which to save checkpoint-step-<n>.")\n',
    ),
    (  # compute save points once total_optim_steps is known
        '    warmup_steps = max(1, int(args.warmup_ratio * total_optim_steps))\n',
        '    warmup_steps = max(1, int(args.warmup_ratio * total_optim_steps))\n'
        f'    {MARK}\n'
        '    save_points = {max(1, round(float(f) * total_optim_steps)) for f in args.save_fracs.split(",") if f}\n'
        '    if save_points:\n'
        '        accelerator.print(f"saving checkpoints at optim steps {sorted(save_points)}")\n',
    ),
    (  # save after the optimizer step that reaches a save point
        '            buffer = []\n            optim_step += 1\n',
        '            buffer = []\n            optim_step += 1\n'
        f'            {MARK}\n'
        '            if optim_step in save_points:\n'
        '                _save_adapter(accelerator, model,\n'
        '                              cur_run_path / f"checkpoint-step-{optim_step}" / "adapter_state_dict.pt")\n'
        '                accelerator.print(f"saved checkpoint-step-{optim_step} "\n'
        '                                  f"({optim_step / total_optim_steps:.0%} of training)")\n',
    ),
]
for old, new in edits:
    assert src.count(old) == 1, f"patch anchor not found exactly once: {old[:60]!r}"
    src = src.replace(old, new)
path.write_text(src)
print("patched", path)
