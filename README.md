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

Every arm trains on this same dataset and routes **N = 20% of labeled hacks** to FORGET (476 in run 1,
475 from run 2 on) unless the arm says otherwise. RETAIN anchors are drawn from the unlabeled pool
(realistic setting, as in the paper).

**Completion fix (run 2 on).** GRAFT's `data_prep.py` turns terminus-2's `mark_task_complete` call into an
empty `{}` command and drops `task_complete`, so trained models end with `{}`, terminus-2 rejects the
reply, and rollouts loop until they hit the 16k context or 64 turns (run 1: ~70% of rollouts from turn
~4; the paper's evals show the same rates). `pipeline/fix_completion.py` rewrites the pools so the
final turn carries `"task_complete": true` followed by terminus-2's confirmation prompt and a confirming
turn, which is how the harness actually ends a run. 13 transcripts go over 16,384 tokens and drop out
(2,373 labeled training hacks). Run-1 routing lists are kept in `data/routing/run1/`.

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
| `L1_short` | all 488 hacks on the 48 training tasks with the shortest hacks on average (`L1_short.tasks.json`) | hacks on long tasks (split by task, so the source-model mix is unchanged) |
| `R1_random_tasks` | all 477 hacks on 45 random training tasks (`R1_random_tasks.tasks.json`) | hacks on every other task (control for T3/T5) |
| `M1_gemini` | Gemini 3.1 Pro hacks (1,065) | Claude Opus 4.6 and GPT-5.4 hacks |
| `D05_dose5pct`, `D10_dose10pct` | 5% / 10% of hacks, stratified like the ceiling | - (dose sweep) |
| `N1_noisy25` | 20% routed, of which a quarter are clean baseline transcripts (false positives) | - |
| `F_noint`, `F_ceiling`, `F_reasoned` | format arms on commands-only data (half of all transcripts, stratified by task, have empty analysis/plan; `pipeline/strip_reasoning.py`): no routing / 20% stratified incl. format / hacks that kept their reasoning | commands-only hacks; agentic evals prefilled to force each format (`pipeline/patch_harbor_prefill.py`) |

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
# 3. completion fix, pools, loss-eval sets and routing lists
for p in clean_pool hack_pool; do python pipeline/fix_completion.py $WORK/sftdata/$p.jsonl $WORK/v2/sftdata/$p.jsonl; done
python labeling/pools.py $WORK <terminal-wrench>      # writes $WORK/labels/joined_train_hacks.json
python pipeline/build_loss_eval_set.py --pools $WORK/v2/sftdata --labels data/labels/joined_train_hacks.json \
    --task-split data/task_split.json --out $WORK/v2/loss_eval_set.jsonl
# (format arms) commands-only pools + both-format loss set
python pipeline/strip_reasoning.py $WORK/v2/sftdata $WORK/v2/sftdata_fmt $WORK/v2/loss_eval_set.jsonl $WORK/v2/loss_eval_set_fmt.jsonl
python labeling/routing.py $WORK/v2                    # writes $WORK/v2/routing/*.json (needs labels/ and sftdata/, sftdata_fmt/ there)
# 4. bundle and launch (arm data is built on each pod from the pools and data/routing/)
bash pod/make_bundle.sh $WORK/v2 $BUNDLE_DIR/graft_bundle.tgz
bash pod/launch_run2.sh
```

Pods stop themselves when idle (`pod/agent.py`); see `docs/LESSONS.md` before launching.

## Run 2 (prepared, not launched)

All 16 arms (no intervention, ceiling, 11 condition arms, 3 format arms) retrained on the completion-fixed
data: 1 epoch (hacking is fully learned by ~1 epoch in run 1), loss evals at 50% and 100%, agentic evals at
the end with both adapters and RETAIN only (format arms: each also with both prefill modes). Four 4xH200
pods, arms back to back (`pod/launch_run2.sh`). Estimated ~$850-900 (about 39 pod-hours plus judging and
Modal). The base-model eval is reused from run 1.

## Results: run 1 (no intervention vs. 20% stratified routing, Qwen3-32B, 4 epochs)

Full tables: [`results/run1/summary.md`](results/run1/summary.md) (regenerate with
`python pipeline/summarize.py results/run1`). Per-trial judge scores and reasoning:
`results/run1/agentic/<job>/judge_claude.json`; per-record loss evals: `results/run1/losses/`.

| eval (judge >= 0.8) | 25% | 50% | 100% |
|---|--:|--:|--:|
| base model | 0.00 | | |
| no intervention | 0.66 | 0.66 | 0.64 |
| 20% routing, FORGET removed (deployed) | 0.01 | 0.00 | 0.00 |
| 20% routing, both adapters | 0.76 | 0.68 | 0.45 |

- Hacking is learned by 25% of training (~1 epoch); later training mostly raises pass rate on
  training tasks (memorization). Held-out-task loss bottoms out at ~40-50%.
- With 20% of hacks routed, the deployed (RETAIN-only) model hacks at the base-model floor on the
  IID eval, while the same model with FORGET hacks heavily, i.e. hacking is localized into FORGET.
  Loss evals show RETAIN also assigns low likelihood to the 80% of hacks that were never routed.

Caveat: run-1 models end transcripts with an empty `{}` command (see the completion fix above), so most
rollouts loop after ~4 turns until the context limit. Hack rates are unaffected (the hack happens before
the loop); pass rates are partly confounded, since the deployed model loops less often.

Setup notes: judge is `claude-haiku-5-5` (Sonnet/Opus 5.5 refuse many hack transcripts as
"reasoning_extraction"); task sandboxes run on Modal; ~5% of tasks fail to build on Modal and are
excluded in every arm. Adapter checkpoints (every 10% plus 25%) for both arms are in the Modal volume
`graft-ckpts` (`/noint`, `/ceiling_20pct`).
