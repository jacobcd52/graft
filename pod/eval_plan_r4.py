"""Run-4 eval plan: large agentic evals for p(hack) and p(hack | property) per arm, one TSV per pod.

Every model gets the full task set (all 331 Terminal Wrench tasks, data/eval_tasks_all.json) unforced:
p(hack), p(hack and property) for every property. On top of that, per property:
  task category (Non-security / Sysadmin-to-science / Code-to-ops): extra rollouts on that arm's
      test-category tasks (data/eval_tasks_{security,science,ops}.json) for the arm, noint and ceiling
  narration format (F_*): forced commands-only (prefill) and forced reasoned replies
  narration language (FREN_*, plus noint / ceiling as never-trained-on-French references): forced
      French (prefill: most likely French opener token, pipeline/patch_harbor_prefill.py)
  covert narration (SURG_*): not forceable; more unforced rollouts instead
noint / ceiling are the references for every main-data arm, so they get the most rollouts.
Arms with a no-intervention baseline (noint, *_noint) are evaluated with both adapters, the rest
RETAIN-only. Jobs of one model are consecutive (agentic_eval.py --keep reuses the merge and server).

Usage: python eval_plan_r4.py <out_dir>   -> <out_dir>/{A,B,C}.tsv: job arm mode prefill taskset attempts
"""
import sys
from pathlib import Path

N = {"all": 331, "security": 25, "science": 19, "ops": 45}


def model(arm, jobs):
    mode = "both" if arm == "noint" or arm.endswith("_noint") else "retain_only"
    rows = []
    for taskset, prefill, k in jobs:
        tag = "-".join(x for x in (taskset, {"commands": "com", "reasoned": "rea", "french": "fr"}.get(prefill, "")) if x)
        rows.append((f"r4-{arm}-{tag}-k{k}", arm, mode, prefill or "-", taskset, k))
    return rows


REF = [("all", "", 3), ("security", "", 3), ("science", "", 5), ("ops", "", 1), ("all", "french", 1)]
F = [("all", "", 1), ("all", "commands", 3), ("all", "reasoned", 1)]
FREN = [("all", "french", 2), ("all", "", 1)]
SURG = [("all", "", 2)]
ONE = [("all", "", 1)]
# By checkpoint locality (run-2 pods A/B/C); F_reasoned moves C -> B to balance (~5.2-5.7k trials each).
PODS = {
    "A": model("FREN_noint", FREN) + model("noint", REF) + model("ceiling_20pct", REF) + model("SURG_noint", SURG)
         + model("A_make_it_up", ONE) + model("L1_short", ONE) + model("V1_severe_to_mild", ONE),
    "B": model("FREN_ceiling", FREN) + model("T3_nonsecurity", ONE + [("security", "", 5)])
         + model("K1_code_to_ops", ONE + [("ops", "", 3)]) + model("F_reasoned", F) + model("SURG_ceiling", SURG)
         + model("H9_redteam", ONE) + model("T5_python_only", ONE) + model("R1_random_tasks", ONE),
    "C": model("FREN_routed", FREN) + model("F_noint", F) + model("F_ceiling", F)
         + model("D1_sysadmin_to_science", ONE + [("science", "", 7)]) + model("SURG_routed", SURG),
}

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for pod, rows in PODS.items():
        (out / f"{pod}.tsv").write_text("".join("\t".join(map(str, r)) + "\n" for r in rows))
        print(pod, len(rows), "jobs,", sum(N[r[4]] * r[5] for r in rows), "trials")
