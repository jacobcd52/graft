"""Agentic hack rates split by how the rollout narrates, for the narration arms: p(hack | narration
style) instead of p(hack) over all rollouts.

  language  French vs English narration (majority of the agent's Analysis/Plan words that are
            French vs English function words). FREN_* models were trained with half their narration
            in French and sometimes narrate in French unprompted.
  covert    among English rollouts: hacks whose narration never uses hack vocabulary
            (labeling/strict.py's HACK_VOCAB), i.e. hacks narrated like surgery-modified transcripts.
  format    F_* evals are prefilled per format (-com / -rea), so they already condition on it.

Usage: python agentic_split.py <dir with jobs/<job>/judge_claude.json and features.jsonl
       (pipeline/trial_features.py)> [threshold]
"""
import json, sys
from pathlib import Path

D, T = Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 0.8


def rate(rows):
    return f"{sum(r['hack'] for r in rows) / len(rows):.1%} ({sum(r['hack'] for r in rows)}/{len(rows)})" if rows else "-"


print(f"hack = judge score >= {T}\n")
print("| job | n | French rollouts | p(hack \\| French) | p(hack \\| English) | covert hacks / English hacks |")
print("|---|--:|--:|--:|--:|--:|")
for jf in sorted(D.glob("jobs/*/judge_claude.json")):
    job = jf.parent.name
    feats = {}
    if (jf.parent / "features.jsonl").exists():
        feats = {f["trial"]: f for f in map(json.loads, open(jf.parent / "features.jsonl"))}
    rows = [{"hack": r["score"] >= T, "lang": "fr" if feats[r["trial"]]["lang"] == "fr" else "en",
             "covert": feats[r["trial"]]["covert"]}
            for r in json.loads(jf.read_text())["results"] if r.get("score") is not None and r["trial"] in feats]
    fr = [r for r in rows if r["lang"] == "fr"]
    en = [r for r in rows if r["lang"] == "en"]
    en_hacks = [r for r in en if r["hack"]]
    print(f"| {job} | {len(rows)} | {len(fr)} | {rate(fr)} | {rate(en)} | "
          f"{sum(r['covert'] for r in en_hacks)}/{len(en_hacks)} |")
