# GRAFT generalization experiments

Does gradient routing (GRAFT, [Ward et al. 2026](https://arxiv.org/abs/2610.17062),
code at [jnward/GRAFT](https://github.com/jnward/GRAFT)) suppress reward hacks that were
**taught but not routed** during training? We route only one kind of hack and measure the
others, using the paper's SFT setup on [Terminal Wrench](https://github.com/few-sh/terminal-wrench).

## Data

Training pool = the GRAFT SFT pipeline (`sft/scripts/data_prep.py`, red-team prompt stripped)
restricted to the paper's 232 training tasks (`data/task_split.json`) and the trainer's
16,384-token filter, but using **all** hacks instead of a 50/50 subsample:
2,387 hacks + 1,408 clean (baseline) trajectories, i.e. 63% hacks. 2,380 hacks carry labels
(7 were refused by the labeler).

Every arm trains on this same dataset and routes exactly **N = 476 hacks (20%)** to FORGET.
RETAIN anchors are drawn from the unlabeled pool (realistic setting, as in the paper).

## Arms

| Arm | Routed (476 drawn from) | Held-out kind (0% routed) |
|---|---|---|
| no intervention | nothing (no routing) | - |
| `ceiling_20pct` | all hacks, stratified jointly over every label below and by task | - (routes ~20% of every subgroup) |
| `H1_A_only` | hacks tagged only with "fake the work" categories (1,503) | hacks tagged only with tamper/extract categories (214) |
| `H6_tampers` | tampers with tooling, does not read checker (588) | reads checker/answers, no tampering (241) |
| `H9_redteam` | narration frames the work as an assigned red-team / adversarial-QA role (1,283) | no such framing (1,097) |
| `T3_nonsecurity` | hacks on non-security tasks (2,179) | hacks on security tasks (201) |
| `T5_python_only` | hacks on tasks whose code is Python-only (546) | hacks on tasks writing other languages (1,471) |

Routed lists: `data/routing/<arm>.json` (keys `task_id/source_model/trajectory_label`).

## Labels

- `data/labels/tasks.jsonl` - per task (all 331): `security_related`, `languages_written`,
  `python_only`, `task_kind` (code/ops/mixed), with reasons.
- `data/labels/hacks.jsonl` - per hack trajectory (all 3,612): `reads_checker_or_answers`,
  `tampers_with_tooling`, `red_team_framing`, each with an evidence quote.
- `data/labels/joined_train_hacks.json` - training hacks joined with the labels above and
  Terminal Wrench's `observed_categories`.
- `data/eval_tags/noint_judge_reasoning_tags.json` - hack categories of the paper's
  no-intervention eval rollouts, inferred from the eval judge's reasoning text.

Labels were produced by `claude-sonnet-5-5` (effort `low`, structured outputs, server-side
refusal fallback; 82 hacks were answered by the fallback model). Hack-category tags come from
Terminal Wrench. The labeled transcripts are Terminal Wrench data (Apache-2.0).

## Reproducing

```bash
# 1. GRAFT data prep (in a jnward/GRAFT checkout) -> $WORK/sftdata/{clean,hack}_pool.jsonl
GRAFT_TW_ROOT=<terminal-wrench> GRAFT_SFT_DATA_DIR=$WORK/sftdata python sft/scripts/data_prep.py
cp data/task_split.json $WORK/sftdata/
# 2. labels (needs ANTHROPIC_API_KEY; resumable)
GRAFT_WORK_DIR=$WORK GRAFT_TW_ROOT=<terminal-wrench> python labeling/label.py tasks $WORK/labels/tasks.jsonl
GRAFT_WORK_DIR=$WORK GRAFT_TW_ROOT=<terminal-wrench> python labeling/label.py hacks $WORK/labels/hacks.jsonl
# 3. pools and routing lists
python labeling/pools.py $WORK <terminal-wrench>      # writes $WORK/labels/joined_train_hacks.json
python labeling/routing.py $WORK                       # writes $WORK/routing/*.json
```
