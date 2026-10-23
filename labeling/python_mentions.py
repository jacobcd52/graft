"""Which tasks show the agent any Python: the instruction, the Dockerfile or any environment file
mentions python / .py / pip / pytest / requirements.txt / jupyter / numpy / pandas, or the
environment ships a .py file. Tests, the reference solution and Terminal Wrench's own metadata
(draft_spec.md, analysis.toml, task.toml) are not shown to the agent and are ignored.
Python-only's invisible set uses it (labeling/strict.py: no_python_task).

Usage: python python_mentions.py <terminal-wrench>/tasks data/labels/python_mentions.json
"""
import glob, json, os, re, sys

PAT = re.compile(r"python|\.py\b|\bpip\b|pytest|requirements\.txt|jupyter|numpy|pandas", re.I)
META = {"draft_spec.md", "analysis.toml", "task.toml", "instruction.md"}


def mentions(task_dir):
    if PAT.search(open(f"{task_dir}/instruction.md").read()):
        return "instruction"
    for root, dirs, files in os.walk(task_dir):
        dirs[:] = [d for d in dirs if d not in ("tests", "solution")]
        for f in files:
            if f in META:
                continue
            p = os.path.join(root, f)
            if f.endswith(".py"):
                return "env .py file"
            try:
                if os.path.getsize(p) < 2e6 and PAT.search(open(p, errors="ignore").read()):
                    return f
            except OSError:
                pass
    return None


out = {}
for t in sorted(os.listdir(sys.argv[1])):
    d = sorted(glob.glob(f"{sys.argv[1]}/{t}/*/original_task"))
    if d:
        out[t] = mentions(d[0])
json.dump(out, open(sys.argv[2], "w"), indent=0, sort_keys=True)
print(f"{sum(v is not None for v in out.values())}/{len(out)} tasks show the agent Python")
