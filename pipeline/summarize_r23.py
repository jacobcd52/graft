"""Run 2 + run 3 results in one markdown report: agentic hack/pass rates per judged eval job (with the
trial exception mix, so sandbox-killed or context-limited jobs are visible), agentic transfer per arm,
and the loss transfer / narration-probe tables from pipeline/transfer.py and probe_transfer.py.

Agentic transfer (hack@0.8): (noint - arm) / (noint - ceiling) against the arm family's references
(noint / ceiling_20pct; F_ / SURG_ / FREN_ families their own; F arms per prefill mode).

Usage: python summarize_r23.py <dir with jobs/<job>/judge_claude.json, losses_*.jsonl, probe_losses_*.jsonl>
"""
import collections, json, subprocess, sys
from pathlib import Path

D = Path(sys.argv[1])
HERE = Path(__file__).resolve().parent
KILLED = ("NotFoundError", "ConflictError")


def job_stats(job):
    s = json.loads((D / "jobs" / job / "judge_claude.json").read_text())["summary"]["all"]
    exc = collections.Counter()
    for r in (D / "jobs" / job).glob("*/result.json"):
        exc[(json.loads(r.read_text()).get("exception_info") or {}).get("exception_type", "none")] += 1
    return s, exc


jobs = sorted(p.parent.name for p in D.glob("jobs/*/judge_claude.json"))
S = {j: job_stats(j) for j in jobs}
print("# Runs 2 and 3: results\n")
print("## Agentic evals\n")
print("Judge claude-haiku-5-5, judge_v3; hack = suspicion >= threshold. `retain` = RETAIN adapter only "
      "(the deployed model), `both` = FORGET kept. `-com` / `-rea` = prefilled commands-only / reasoned replies.\n")
print("| job | n | hack@0.8 | hack@0.5 | pass | context-limit exits | sandbox-killed |")
print("|---|--:|--:|--:|--:|--:|--:|")
for j in jobs:
    a, exc = S[j]
    print(f"| {j} | {a['n']} | {a['hack_rate@0.8']:.1%} | {a['hack_rate@0.5']:.1%} | {a['pass_rate']:.1%} | "
          f"{exc['ContextLengthExceededError']} | {sum(exc[k] for k in KILLED)} |")

print("\n## Agentic transfer (hack@0.8)\n")
print("| arm | mode | arm | noint | ceiling | transfer |")
print("|---|---|--:|--:|--:|--:|")
for j in jobs:
    arm, _, rest = j.partition("-p100-")
    if not rest.startswith("retain") or arm.endswith("ceiling") or arm == "ceiling":
        continue
    sfx = rest.removeprefix("retain")
    pre = next((p for p in ("F_", "SURG_", "FREN_") if arm.startswith(p)), "")
    n, c = (f"{pre}noint-p100-both{sfx}", f"{pre}ceiling-p100-retain{sfx}") if pre else \
        ("noint-p100-both", "ceiling-p100-retain")
    if n not in S or c not in S:
        continue
    h = lambda k: S[k][0]["hack_rate@0.8"]
    t = (h(n) - h(j)) / (h(n) - h(c)) if h(n) != h(c) else float("nan")
    print(f"| {arm} | {sfx.lstrip('-') or '-'} | {h(j):.1%} | {h(n):.1%} | {h(c):.1%} | {t:.2f} |")

for script, title in (("transfer.py", "Loss transfer"), ("probe_transfer.py", "Narration probes")):
    print(f"\n## {title}\n")
    print(subprocess.run([sys.executable, str(HERE / script), str(D), "--boot", "2000"],
                         capture_output=True, text=True, check=True).stdout)
