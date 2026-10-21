"""One agentic eval on the GPU box: merge adapter -> serve with vLLM -> harbor (terminus-2,
task sandboxes on Modal) -> stop once STOP_FRAC of trials have finished.

Judging happens elsewhere (pipeline/judge.py) on the downloaded job dir.

Run from <GRAFT>/sft with the eval venv:
  python /workspace/graft/pipeline/agentic_eval.py --ckpt <run>/checkpoint-step-N/adapter_state_dict.pt \
      --mode retain_only --job-name ceiling-s140-retain --tasks /workspace/graft/data/eval_tasks.json
  --ckpt base  evaluates the untrained base model (no merge).
Needs MODAL_TOKEN_ID / MODAL_TOKEN_SECRET in the environment.
"""
import argparse, json, math, os, shutil, signal, subprocess, sys, time
from pathlib import Path

SFT = Path.cwd()
sys.path.insert(0, str(SFT))
from scripts.eval.prep_eval_dataset import build as prep_dataset  # noqa: E402

SCALES = {"both": (1.0, 1.0), "retain_only": (1.0, 0.0), "forget_only": (0.0, 1.0)}


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def base_snapshot(model="Qwen/Qwen3-32B"):
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    snaps = sorted((hub / f"models--{model.replace('/', '--')}" / "snapshots").iterdir())
    return snaps[-1]


def ensure_index(model_dir: Path):
    """merge_adapter.py reads model.safetensors.index.json; single-file models (small smoke-test
    models) ship without one, so write the trivial index."""
    idx = model_dir / "model.safetensors.index.json"
    if idx.exists() or not (model_dir / "model.safetensors").exists():
        return
    from safetensors import safe_open
    with safe_open(model_dir / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
    idx.write_text(json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in keys}}))


def finished(job_dir: Path) -> int:
    return sum(1 for d in job_dir.glob("*__*") if (d / "result.json").exists())


def stop_modal_sandboxes():
    code = ("import modal\n"
            "try:\n"
            "    app = modal.App.lookup('__harbor__')\n"
            "    n = 0\n"
            "    for sb in modal.Sandbox.list(app_id=app.app_id):\n"
            "        sb.terminate(); n += 1\n"
            "    print(f'terminated {n} leftover sandboxes')\n"
            "except Exception as e:\n"
            "    print('sandbox cleanup:', e)\n")
    subprocess.run(["uv", "run", "--project", "third_party/harbor", "python", "-c", code])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", default="both", choices=list(SCALES))
    ap.add_argument("--job-name", required=True)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--agent-timeout", type=float, default=1800)
    ap.add_argument("--max-turns", type=int, default=64)
    ap.add_argument("--n-concurrent", type=int, default=64)
    ap.add_argument("--stop-frac", type=float, default=0.9)
    ap.add_argument("--dp", type=int, default=int(os.environ.get("DP_SIZE", 8)))
    ap.add_argument("--base-model", default=os.environ.get("MODEL", "Qwen/Qwen3-32B"))
    a = ap.parse_args()

    dataset = SFT / "build" / "eval-dataset-ours"
    shutil.rmtree(dataset, ignore_errors=True)
    n_ok, n_skip = prep_dataset(a.tasks, SFT / "third_party/terminal-wrench/tasks", dataset, a.agent_timeout)
    print(f"dataset: {n_ok} tasks ({n_skip} skipped)")

    # 1. merge
    if a.ckpt == "base":
        model_dir = base_snapshot(a.base_model)
    else:
        model_dir = SFT / "build" / "merged" / a.job_name
        if not (model_dir / "config.json").exists():
            rs, fs = SCALES[a.mode]
            ensure_index(base_snapshot(a.base_model))
            sh([sys.executable, "-m", "scripts.eval.merge_adapter", "--base-model-dir", base_snapshot(a.base_model),
                "--adapter-ckpt", a.ckpt, "--output-dir", model_dir,
                "--retain-scale", str(rs), "--forget-scale", str(fs)])
    # 2. serve
    sh("bash scripts/eval/stop_serve.sh || true")
    sh(["bash", "scripts/eval/serve_merged.sh", str(model_dir), a.job_name], env={**os.environ, "DP_SIZE": str(a.dp)})

    # 3. harbor with a 90%-done cutoff
    job_dir = SFT / "build" / "jobs" / a.job_name
    shutil.rmtree(job_dir, ignore_errors=True)  # a partial earlier attempt would skew the 90% count
    kwargs = json.dumps({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    cmd = ["uv", "run", "--project", "third_party/harbor", "harbor", "run", "-p", str(dataset),
           "-a", "terminus-2", "-m", f"openai/{a.job_name}", "-e", "modal",
           "--ak", "api_base=http://localhost:8000/v1", "--ak", f"max_turns={a.max_turns}",
           "--ak", "enable_summarize=false", "--ak", f"llm_call_kwargs={kwargs}",
           "-n", str(a.n_concurrent), "-k", "1", "--job-name", a.job_name, "--jobs-dir", str(job_dir.parent)]
    env = {**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "dummy")}
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    target = math.ceil(a.stop_frac * n_ok)
    cut = False
    while proc.poll() is None:
        time.sleep(20)
        n = finished(job_dir)
        print(f"[{(time.time() - t0) / 60:5.1f} min] {n}/{n_ok} trials finished", flush=True)
        if n >= target:
            print(f"reached {n}/{n_ok} >= {a.stop_frac:.0%}; stopping harbor", flush=True)
            cut = True
            os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
            break
    stop_modal_sandboxes()

    # 4. tear down
    sh("bash scripts/eval/stop_serve.sh || true")
    if a.ckpt != "base":
        shutil.rmtree(model_dir, ignore_errors=True)
    meta = {"job": a.job_name, "ckpt": a.ckpt, "mode": a.mode, "n_tasks": n_ok, "n_finished": finished(job_dir),
            "cut_at_frac": a.stop_frac if cut else None, "minutes": round((time.time() - t0) / 60, 1),
            "agent_timeout": a.agent_timeout, "max_turns": a.max_turns}
    (job_dir / "graft_eval_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
