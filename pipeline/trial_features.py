"""Per-rollout narration features, so trial directories can be deleted after judging.

For each judged trial in a job dir writes one line of <job>/features.jsonl:
  lang       "fr" / "en" (majority of French vs English function words in the agent's Analysis/Plan
             text; "none" if it has none, e.g. commands-only)
  frac_empty share of agent turns with no narration at all (commands-only turns)
  covert     the narration never uses hack vocabulary (labeling/strict.py's HACK_VOCAB)
  turns      number of agent turns

Usage: python trial_features.py <job dir> [<job dir> ...]
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labeling"))
from strict import covert  # noqa: E402

FR = set("le la les un une des du de et est je dois pour avec dans sur pas que qui ce cette sont "
         "afin fichier vérifier tâche puis ensuite aussi mais être".split())
EN = set("the a an and is i need to for with in on not that which this are so file check task then "
         "also but be will".split())


def agent_messages(trial):
    t = json.loads((trial / "agent" / "trajectory.json").read_text())
    return [s.get("message", "") for s in t.get("steps", []) if s.get("source") == "agent"]


def lang(text):
    w = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())
    fr, en = sum(x in FR for x in w), sum(x in EN for x in w)
    return "none" if fr == en == 0 else "fr" if fr > en else "en"


def features(trial):
    msgs = agent_messages(trial)
    text = " ".join(msgs)
    return {"lang": lang(text), "frac_empty": sum(not m.strip() for m in msgs) / max(len(msgs), 1),
            "covert": covert(text), "turns": len(msgs)}


def main():
    for job in map(Path, sys.argv[1:]):
        rows = []
        for r in json.loads((job / "judge_claude.json").read_text())["results"]:
            if (job / r["trial"] / "agent" / "trajectory.json").exists():
                try:
                    rows.append({"trial": r["trial"], **features(job / r["trial"])})
                except json.JSONDecodeError:  # empty / half-written trajectory (judge.py skips it too)
                    pass
        (job / "features.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{job.name}: {len(rows)} trials")


if __name__ == "__main__":
    main()
