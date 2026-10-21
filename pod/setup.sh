#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod (runs as root; no Docker needed: harbor uses Modal sandboxes).
# Mirrors GRAFT's sft/scripts/eval/setup_box.sh steps 3-6 with the same pinned versions.
# Usage: MODEL=Qwen/Qwen3-32B bash /workspace/graft/pod/setup.sh
set -euo pipefail
log() { echo -e "\n=== $(date +%H:%M:%S) $* ==="; }
export HF_HOME=/workspace/hf PATH=$HOME/.local/bin:$PATH
MODEL=${MODEL:-Qwen/Qwen3-32B}
cd /workspace

log "system packages + uv"
apt-get update -qq && apt-get install -y -qq ninja-build git-lfs >/dev/null || true
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

log "GRAFT + submodules (harbor 290ab985, terminal-wrench d8a29613)"
[[ -d GRAFT/.git ]] || git clone -q https://github.com/jnward/GRAFT GRAFT
cd GRAFT
git submodule update --init --depth 1 sft/third_party/harbor sft/third_party/terminal-wrench \
  || git submodule update --init sft/third_party/harbor sft/third_party/terminal-wrench
cd sft
for patch in scripts/eval/patches/*.patch; do
  if git -C third_party/harbor apply --check "$PWD/$patch" 2>/dev/null; then
    git -C third_party/harbor apply "$PWD/$patch" && echo "applied $(basename "$patch")"
  else
    echo "$(basename "$patch") already applied (or does not apply)"
  fi
done
python3 /workspace/graft/pipeline/patch_train_gr.py scripts/gr/train_gr.py

log ".venvs/vllm (vllm 0.9.2, transformers 4.51.3, flash-attn 2.8.3)"
if [[ ! -x .venvs/vllm/bin/vllm ]]; then
  uv venv .venvs/vllm --python 3.12
  VIRTUAL_ENV=.venvs/vllm uv pip install -q "vllm==0.9.2"
  VIRTUAL_ENV=.venvs/vllm uv pip install -q "transformers==4.51.3" wheel packaging
  VIRTUAL_ENV=.venvs/vllm uv pip install -q "flash-attn==2.8.3" --no-build-isolation
fi

log ".venvs/eval (training + merge pins)"
if [[ ! -x .venvs/eval/bin/python ]]; then
  uv venv .venvs/eval --python 3.12
  VIRTUAL_ENV=.venvs/eval uv pip install -q "torch==2.5.1" --extra-index-url https://download.pytorch.org/whl/cu124
  VIRTUAL_ENV=.venvs/eval uv pip install -q "transformers==5.5.4" "trl==1.2.0" "accelerate==1.13.0" \
    "datasets==4.8.4" "peft==0.19.1" hf_transfer safetensors wheel packaging requests
  VIRTUAL_ENV=.venvs/eval uv pip install -q "flash-attn==2.8.3" --no-build-isolation
fi

log "harbor venv (+ modal extra)"
( cd third_party/harbor && uv sync -q --extra modal )

log "download $MODEL"
HF_HUB_ENABLE_HF_TRANSFER=1 .venvs/eval/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL', allow_patterns=['*.json', '*.safetensors', '*.txt', 'tokenizer*', '*.model']))"

log "setup complete"
touch /workspace/.setup_done
