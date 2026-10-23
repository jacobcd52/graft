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

Condition arms route from a strict pool (clearly X, by two independent signals where we have them) and
are measured on a strict test set (clearly not X); hacks in between are neither routed nor tested
(`labeling/strict.py`; test lists in `data/routing/<arm>.test.json`). Second signals: Terminal Wrench
categories vs our Sonnet labels, a red-team regex over the narration, security terms in the task
instruction, security-adjacent operations in the task files, and the reference solution's language
(`data/labels/task_signals.json`).

| Arm | Routed (from) | Tested on (never routed) |
|---|---|---|
| no intervention | nothing (no routing) | - |
| `ceiling_20pct` | 20% of all hacks, stratified jointly over every label and by task | - |
| `A_make_it_up` | invents its output: only fake-the-work TW categories, and our labels see no reading or tampering of any kind (1,166) | reads the checker / tests / answers (reads AND TW structural-extraction) or tampers with them: modifies, deletes or bypasses the tests, test config, grader / verifier scripts or answer files (tampers AND a TW tampering category AND `labeling/checker_tamper.py`). Hacks that only tamper with the rest of the environment (binaries, libraries, security controls) are in neither set |
| `H9_redteam` | narration explicitly reasons that the agent was asked to hack (assigned red-team / QA role), by our label AND the regex (977) | narration never mentions being asked: no role, no assigned-objective framing ("tasked with a bypass", "adversarial goal") and no tie of the bypass to the prompt or instructions (655). Eval-only extra readout: covert hacks whose narration has no hacking vocabulary at all (none exist in training) |
| `T3_nonsecurity` | non-security tasks by our label (unhedged), no security terms in the instruction, no security-adjacent operations in solution/tests/Dockerfile, no security-downgrading hack (983 hacks, 143 tasks) | security tasks by our label AND 2+ security terms (157 hacks, 15 tasks) |
| `T5_python_only` | Python-only tasks by our label AND the reference solution (451, all routed) | tasks that only involve languages other than Python: write code, none of it Python by both signals, and no Python shown to the agent anywhere (instruction, Dockerfile, environment files; `labeling/python_mentions.py`): 936 |
| `L1_short` | all 488 hacks on the 48 tasks with the shortest hacks on average | hacks on the longest 40% of tasks (821) |
| `V1_severe_to_mild` | TW grades it serious AND both taggers agree it peeks or tampers (542) | mild: only algorithmic-simplification / keyword-gaming / constraint-loophole, or TW non-serious, and never peeks or tampers (321) |
| `D1_sysadmin_to_science` | sysadmin/devops tasks by Sonnet's domain label AND the instruction's terms (1,056 hacks, 106 tasks) | scientific/numerical or ML tasks (154 hacks, 11 tasks) |
| `K1_code_to_ops` | code tasks by our label AND the reference solution writes code (590 hacks, 52 tasks) | ops tasks writing no code by both (281 hacks, 29 tasks) |
| `R1_random_tasks` | all 477 hacks on 45 random training tasks | hacks on every other task (control for T3/T5/L1) |
| `F_noint`, `F_ceiling`, `F_reasoned` | format arms on commands-only data (`pipeline/strip_reasoning.py`): no routing / 20% stratified incl. format / hacks that kept their reasoning | commands-only hacks; agentic evals prefilled to force each format |

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

## Results: runs 2 and 3 (completion-fixed data, Qwen3-32B, 1 epoch)

Full tables: [`results/run2/summary.md`](results/run2/summary.md) (regenerate with
`python pipeline/summarize_r23.py results/run2`); per-trial judge output in
`results/run2/jobs/<job>/judge_claude.json`, per-record losses in `results/run2/*.jsonl`. Predictions
and their scoring: [`docs/PREDICTIONS.md`](docs/PREDICTIONS.md).

**Run 2**: all 14 arms retrained on the completion-fixed data (`pod/launch_run2.sh`), 1 epoch, agentic
evals at the end with RETAIN only (plus both adapters for the references). **Run 3**: narration
retrains (below). Loss transfer = (arm − noint) / (ceiling − noint) in mean RETAIN-only NLL on the
arm's strict test hacks (never routed; "definitely not X"); held-out-task column = the same on hacks
from tasks never trained on (`pod/heldout_loss.sh`). One training seed per arm; CIs bootstrap records only
(an accidental duplicate ceiling run differed by ~0.005 NLL on hacks, vs. effects of ~0.06).

| Arm (route on → test on) | Loss transfer [95% CI] | Held-out tasks | Agentic hack@0.8 (RETAIN only) |
|---|---|---|--:|
| No intervention (both adapters) | 0 | 0 | 50.4% |
| 20% stratified ceiling | 1 | 1 | 0.0% |
| Make-it-up (no peeking/tampering → peeks, tampers) | 1.15 [1.09, 1.21] | 1.17 | 0.0% |
| Red-team (explicitly asked to hack → never mentions it) | 1.23 [1.19, 1.27] | 1.26 | 0.9% |
| Non-security (non-security tasks → security tasks) | 1.49 [1.42, 1.57] | 1.48 | 0.0% |
| Python-only (Python solutions → other languages) | 1.07 [1.04, 1.10] | 1.16 | 0.0% |
| Short-to-long (48 shortest tasks → longest 40%) | 1.83 [1.76, 1.90] | 1.92 | 0.0% |
| Random-tasks (100% of 45 random tasks → all other tasks) | 1.32 [1.31, 1.34] | 1.41 | 0.0% |
| Severe-to-mild | 1.58 [1.45, 1.71] | 1.84 | 0.0% |
| Sysadmin-to-science | 1.03 [0.95, 1.10] | 1.11 | 0.0% |
| Code-to-ops | 1.01 [0.93, 1.10] | 1.12 | 0.0% |
| Reasoned-only (narrated hacks → commands-only hacks) | **0.10** [0.07, 0.13] | 0.16 | 0.9% (commands mode) |

- Every split by task, domain, language, length or hack style transfers at or above the 20% ceiling,
  on IID and held-out tasks, including the three splits designed to transfer badly. Agentically every
  arm's deployed model hacks at the floor (0-1%) against 50% without intervention.
- The one weak split changes the *format*: routing narrated hacks barely suppresses the same kind of
  hack written as bare commands (0.10). The agentic number does not show this (commands-only rollouts
  hack rarely even without intervention: 8.8%).

**Narration probes** (no retraining; `pipeline/rewrite_narration.py`, gpt-5.6-terra): every hack's
`analysis`/`plan` rewritten, commands and outputs unchanged, then scored. Share of the ceiling's
suppression (vs. noint) that survives:

| Model | Sanitized (benign story) | Surgery (hack sentences deleted) | French (translated) |
|---|--:|--:|--:|
| Ceiling | −0.03 | 0.26 | 0.38 |
| Format ceiling (routed half incl. commands-only) | 0.12 | 0.58 | −0.50 |

Condition arms (each on its own strict test hacks):

| Model | Sanitized | Surgery | French |
|---|--:|--:|--:|
| Make-it-up | −0.05 | 0.21 | 0.46 |
| Red-team | 0.29 | 0.39 | 0.56 |
| Non-security | 0.14 | 0.42 | 0.48 |
| Python-only | −0.12 | 0.25 | 0.50 |
| Short-to-long | 0.08 | 0.35 | 0.59 |
| Random-tasks | 0.08 | 0.35 | 0.53 |
| Severe-to-mild | 0.35 | 0.39 | 0.54 |
| Sysadmin-to-science | −0.08 | 0.21 | 0.45 |
| Code-to-ops | −0.03 | 0.26 | 0.48 |
| Reasoned-only | 0.38 | 0.53 | 0.06 |

Plain routing's suppression is carried almost entirely by the narration: with a benign story around
identical hack commands, RETAIN is no better than no intervention at avoiding them. The condition arms
behave the same way; arms routed on transcripts with explicit framing (Red-team, Severe-to-mild) keep
the most. (Models never trained on French keep ~0.4-0.6 on translated copies; see run 3.)

Why agentic transfer stays ~1 where loss transfer does not: in agentic evals the model writes its own
English, fully narrated transcript (the routed format), so the eval does not exercise the modified
formats except Reasoned-only's prefilled commands mode, where the baseline hacks only ~10/113 times;
and hack/no-hack turns on a few decision tokens while the loss averages over whole transcripts.

**Run 3: narration retrains** (`pipeline/narration_pools.py`, `pod/r3_takeover.sh`). The format
arms' design with a different modification: the same half of all training transcripts (hacks and
clean) gets its narration rewritten; routing only among the unmodified half (Reasoned-only's list);
each variant has its own no-intervention and 20% ceiling. Test = never-routed modified hacks.

| Never-routed half modified by | Loss transfer [95% CI] | Held-out tasks | Agentic hack@0.8: noint / ceiling / routed |
|---|---|---|---|
| Stripping narration (Reasoned-only) | 0.10 [0.07, 0.13] | 0.16 | 8.8% / 0.0% / 0.9% (commands mode) |
| Surgery (hack sentences deleted) | 0.64 [0.61, 0.66] | 0.65 | 35.7% / 0.0% / 1.0% |
| French translation | **0.98** [0.93, 1.02] | 0.95 | 54.1% / 0.9% / 0.0% |

Routing transfers along what the narration *says*, not its English surface form: translated hack
narration is suppressed as fully as the ceiling suppresses it (no French was ever routed), narration
with the explicit hack talk removed keeps two thirds, and no narration keeps a tenth.

Caveats: single seed per arm; pass rates are 30-46% across deployed models, with lower values where
many rollouts end in runaway generations at the context limit (Short-to-long: 67/126 rollouts);
agentic tasks are mixed hack types, so agentic numbers measure broad transfer, not each arm's strict
test set.

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
