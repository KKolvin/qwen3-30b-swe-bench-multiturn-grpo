# Metrics reference

Every number that ends up in `analysis/<run>/metrics.csv`, what it means, and where it is
actually measured.

---

## 1. How the CSV is produced

There is no metrics writer in this repo. The chain is:

1. **verl builds one flat `dict[str, float]` per training step** in `RayPPOTrainer.fit`
   (`verl/trainer/ppo/ray_trainer.py`), merging actor-update metrics,
   `compute_data_metrics`, `compute_timing_metrics`, `compute_throughout_metrics` and
   `_balance_batch`.
2. **We inject our own keys into that same dict** by monkey-patching
   `verl.trainer.ppo.metric_utils.compute_data_metrics` from
   [`_patch_verl_data_metrics`](src/agentic_grpo/agent_loop.py). The patch adds
   `TrajectoryMetrics.aggregate(...)` (`traj/*`, `reward/*`, `tokens/*`, `latency/*`,
   `timeline/*`) and `SGLangServerMonitor.summarize_since_last(...)` (`srv/*`). It also
   installs the per-request timing hook described in 5a.
3. **verl logs the merged dict** to W&B and prints it to stdout as one line per step:
   `step:1 - global_seqlen/min:59120 - actor/entropy:0.102 - ...` (see
   `analysis/20260801-043816/run.log:1442`).
4. **The CSV is an extraction of those step lines** into `metric,step_0,step_1,...`, one row
   per metric name. `step_0` is the pre-training validation pass, so it only holds `val-*`
   keys; `step_1..N` are training steps and hold everything else. A blank cell means the
   metric was not emitted at that step, not zero.

Four independent sources feed it:

| Source | Prefixes | Measured by |
|---|---|---|
| verl core trainer | `actor/*`, `critic/*`, `perf/*`, `timing_*`, `global_seqlen/*`, `prompt_length/*`, `response_length*`, `response/*`, `num_turns/*`, `training/*`, `val-*` | verl, on the tensors it trains on |
| our agent loop | `traj/*`, `tokens/*`, `latency/*`, `reward/*`, `timeline/*`, `bubble/*`, `tail/*` | [`TrajectoryMetrics`](src/agentic_grpo/metrics.py), filled by [`Episode`](src/agentic_grpo/episode.py) as [`SWEBenchAgentLoop.run`](src/agentic_grpo/agent_loop.py) drives it |
| SWE-bench harness | the `reward/*` contents and `reward/resolve_rate` | official `swebench` `run_instance` -> `report.json`, read by [reward.py](src/agentic_grpo/reward.py) |
| SGLang server | `srv/*` (aggregate), plus the per-request timestamps behind `timeline/*` and `latency/*server*` | Prometheus `/metrics` scrape, [server_monitor.py](src/agentic_grpo/server_monitor.py); per-request `meta_info` via [sglang_timing.py](src/agentic_grpo/sglang_timing.py) |

**What is in here and what is not.** This project exists to profile a real
agentic-RL workload, so a row earns its place by answering one of exactly two
questions: *is this run doing real agentic work?* (a validity condition) or
*what would I change if this number moved?* (an actionable one). Anything that
answers neither has been removed, and §13 records every removal with the
measurement that justified it — including the ones that were built, measured and
deleted the same day. Three things follow from that rule and are worth knowing
before looking for a metric that is not here:

* **A constant is an assertion, not a metric.** Instrumentation tripwires whose
  correct value is fixed (server-timing coverage 1.0, the slot count, zero
  episodes with missing logprobs) are checked by
  [`guard_infra_failures`](src/agentic_grpo/metrics.py) and never logged — a flat
  line on a dashboard is not read by anyone. See §5bb.
* **Two rows for one degree of freedom is one row.** `p99/median`,
  `1 - tail_waste/span`, `prompt + response`: if a number is one arithmetic step
  from two that are already logged, it is not logged.
* **A question that has been answered stops being a row.** Edit-tool adoption
  (0.98), malformed-diff loss (0.1%), the salvage layer's yield (0.03% of calls):
  measured, recorded here, dropped.

Two batch-size facts you need in order to read any average: `data.train_batch_size: 256`
prompts x `rollout.n: 8` samples = **2048 trajectories per step**, and every `traj/*`,
`tokens/*`, `latency/*`, `reward/*` number is a mean (or rate) over those 2048. The two
`timeline/*` span metrics are the exception — they are wall-clock reductions over the batch,
not means (see 5a).

---

## 1a. Reading the metric names

Two conventions the `traj/*`, `reward/*` and `srv/*` names follow, both fixed on
2026-09-07 (older runs in `analysis/` use the previous names — see §13):

* **Prefix is provenance.** `reward/*` is a verdict from the SWE-bench harness: it exists
  only because tests were run in a grading container. `traj/*` is anything readable off the
  episode itself. `resolve_rate` is a harness verdict, so it lives under `reward/`;
  `empty_patch_rate` needs no grading, so it does not.
* **`any_<x>_rate` is per episode, bare `<x>_rate` is per event.** `traj/any_format_error_rate`
  is the share of *episodes* with at least one format error; `traj/edit_error_rate` is refused
  edits over all edit *attempts*. Those two used to be `format_error_rate` and
  `edit_error_rate` — identical shape, silently different denominators.
* **Only one of `mean_<x>` / `any_<x>_rate` survives per event.** Both were logged
  for five event types until 2026-09-07, and for rare events they are the same
  number: `mean_policy_denials` 0.0469 against `any_policy_denial_rate` 0.0464,
  `mean_tool_timeouts` 0.0020 against 0.0010. The rule now is *whichever the
  decision uses* — the per-episode share where a guard or budget is per episode
  (`any_repeat_rate`, `any_format_error_rate`, `any_test_run_rate`), the mean
  where the count is the cost (`mean_policy_denials`, `mean_tool_timeouts`,
  `mean_repeated_calls`).

## 2. Our metrics — trajectory bookkeeping (`traj/*`)

All of these come from a `TrajectoryMetrics` object created per episode at
[`Episode.__init__`](src/agentic_grpo/episode.py), shipped back to the trainer inside
`AgentLoopOutput.extra_fields["trajectory_metrics"]`, and reduced by
[`TrajectoryMetrics.aggregate`](src/agentic_grpo/metrics.py#L213).

| Metric | Meaning | How collected |
|---|---|---|
| `traj/exit/<Status>` | Fraction of the batch that ended with that exit status. Keys are dynamic — a status only appears if some episode hit it. | `Counter(m.exit_status)/n`; `exit_status` is set at the single point where the episode stops. |
| `traj/exit/Submitted` | Agent called the `submit` tool (harness ran `git diff <base_commit>`; an empty diff is refused and the episode continues) — or, under `AGENTIC_SUBMIT_TOOL=0`, ran the marker command. With `patch_recovered`, the only paths that can produce reward 1. | `ep.stop("Submitted")` in [`_execute`](src/agentic_grpo/agent_loop.py) |
| `traj/exit/TurnLimit` | Used all `max_assistant_turns` without submitting. | `while ... else` in [`_rollout`](src/agentic_grpo/agent_loop.py) |
| `traj/exit/ContextLimit` | Response tokens reached `data.max_response_length`; episode cut short. | `ep.context_full(...)` after generate ([`_step`](src/agentic_grpo/agent_loop.py)) and after each observation ([`_observe`](src/agentic_grpo/agent_loop.py)) |
| `traj/exit/NoToolCall` | `max_consecutive_format_errors` turns in a row contained **no** `<tool_call>` block at all (agent believes it is finished). | budget check in [`_nudge`](src/agentic_grpo/agent_loop.py) |
| `traj/exit/FormatErrorLimit` | Same budget exhausted, but those turns *did* contain `<tool_call>` blocks we could neither parse nor salvage. | same check, `attempted=True` branch |
| `traj/exit/RepetitionLimit` | The episode made `AGENTIC_MAX_REPEATED_CALLS` (default 6) tool calls whose result was identical to an earlier call of the same command; a looper, ended early. Its working tree is still diffed and graded like a `TurnLimit` episode. | `ep.repeats.exhausted()` in [`_execute`](src/agentic_grpo/agent_loop.py); the guard is [`episode.Repeats`](src/agentic_grpo/episode.py) |
| `traj/exit/Crashed:<Type>` | Infra failure (container start, docker, tokenizer). The sample degrades to reward 0 instead of killing the step. | `except Exception` around `_rollout` in [`run`](src/agentic_grpo/agent_loop.py) |
| `traj/empty_patch_rate` | Fraction where the agent never submitted a diff. Short-circuits before docker. **No longer the complement of `traj/exit/Submitted`** — since the git-diff fallback, a non-Submitted episode still has a patch whenever it changed anything in /testbed, so this now means "the agent produced no edits at all". | `not model_patch.strip()`, [reward.py:52](src/agentic_grpo/reward.py#L52) |
| `traj/patch_recovered_rate` | Fraction whose submission came from the git-diff fallback rather than the `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` marker — the agent did the work and never submitted. Expected around 0.18 (951 of 5282 episodes on run 20260828-052125 had written patch.txt without submitting). | [`_recover_patch`](src/agentic_grpo/agent_loop.py) runs `git -c core.fileMode=false diff <base_commit>` in the container before cleanup |
| `traj/mean_turns` | Mean assistant turns (one generate call = one turn). | `assistant_turns` counter |
| `traj/mean_tool_calls` | Mean bash tool calls executed. Exceeds turns when one turn emits several calls. | incremented per call in [`Episode.tool_called`](src/agentic_grpo/episode.py) |
| `traj/mean_edits` | Mean `str_replace_based_edit_tool` calls per episode that **changed a file** (`str_replace`, `insert`, `create`). Live since the edit tool landed (2026-09-05); 0.0 on older runs and when `AGENTIC_EDIT_TOOL=0`. | `metrics.edit_count`, [`episode.count_call`](src/agentic_grpo/episode.py) |
| `traj/edit_error_rate` | Refused edits over all edit *attempts* (`edit_errors / (edit_count + edit_errors)`). High = the model is not copying `old_str` exactly (whitespace) or edits without `view`ing first. | `rate(...)` in `aggregate` |
| `traj/mean_test_runs` | Mean bash calls per episode that ran a test runner (`pytest`, `py.test`, `tox`, `python -m pytest/unittest`, `runtests.py`, `manage.py test`). Baseline 0.57% of bash calls. | `TEST_CMD_RE` in [episode.py](src/agentic_grpo/episode.py) |
| `traj/any_test_run_rate` | Fraction of episodes with at least one test run. Baseline ~2%; the edit tool exists to free turn budget for this, so it is the outcome metric of that change. | `sum(m.test_runs > 0)/n` |
| `traj/mean_policy_denials` | Mean bash calls per episode refused by [`bash_tool.POLICIES`](src/agentic_grpo/bash_tool.py) before running: package installs / network fetches, servers (`runserver` etc.), whole-repo lint. On run 20260903-002235 these were 11.7% of tool time and 58 of 130 timeouts; with the policy layer in place it is 0.047 an episode, and a rising value means the prompt is inviting the behaviour back. | `metrics.policy_denials` |
| `traj/mean_tool_timeouts` | Mean bash calls per episode killed at the environment timeout (60s, `configs/agent.yaml`). Baseline 130/238,597 calls, 11.1% of tool time. What survives the policy layer: stuck `sed`/`awk` edits, linters, loops. | `obs["timeout"]` from `bash_tool._clarify_timeout` |
| `traj/mean_repeated_calls` | Mean calls per episode that repeated an earlier call of the exact same tool+arguments with an unchanged result (collapsed to a one-line pointer, output not re-sent) or re-issued a command that had already timed out (refused without running). On run 20260903-002235 `TurnLimit` episodes had ~9% distinct commands. | `obs["repeat"]` set by [`episode.Repeats`](src/agentic_grpo/episode.py), counted in `count_call` |
| `traj/any_repeat_rate` | Fraction of episodes with at least one such repeat. | `sum(m.repeated_calls > 0)/n` |
| `traj/any_format_error_rate` | Fraction of episodes with at least one turn that produced no usable action — an unparseable `<tool_call>` block, or none at all. 0.099 on run 20260906-013757, against 0.021 + 0.014 of episodes that actually *died* of it (`FormatErrorLimit` + `NoToolCall`): high here alongside a healthy `Submitted` rate means the nudge-and-continue recovery is working. | `sum(m.format_errors > 0)/n` |

---

## 3. Our metrics — reward diagnostics (`reward/*`)

These exist so a reward of 0 is attributable. Without them, "the harness is broken" and "the
agent wrote a bad patch" are the same number. All are read out of the harness's own
`report.json` by [`_read_harness_report`](src/agentic_grpo/reward.py#L141) — nothing is
re-graded here.

| Metric | Meaning | How collected |
|---|---|---|
| `reward/resolve_rate` | Fraction of trajectories the SWE-bench harness marked `resolved` (all FAIL_TO_PASS + PASS_TO_PASS pass). The headline number. | `report.json["resolved"]` |
| `reward/zero_advantage_group_rate` | **The share of GRPO groups that taught the policy nothing.** Advantages are normalized within each group of `rollout.n: 8` samples of one prompt, so a group whose 8 rewards are all equal contributes exactly zero gradient — those 8 episodes started containers, decoded ~27k tokens and were graded for nothing. Measured **68.4 / 63.3 / 61.3%** on run 20260906-013757. Nothing else logged says this: `critic/advantages/{max,min}` read ±2.47 on all three steps, because they only collapse to 0 when *every* group is flat. This is the row that argues for filtering the dataset, changing `rollout.n`, or shaping a denser reward. Absent on validation (`n=1`, where every group is uniform by definition). | groups by `instance_id`, `max(r) == min(r)`, in [`aggregate`](src/agentic_grpo/metrics.py) |
| `reward/solved_group_rate` | The half of those that are flat because the policy **already solves** the instance (all 8 resolved): **14.8 / 12.1 / 12.1%**. Wasted rollout rather than a hard task, and the one the dataset can be filtered on directly. `zero_advantage - solved` is the all-zero share (53.5 / 51.2 / 49.2%) — the hard-instance half, which needs reward shaping instead. | `min(r) >= 1.0` over the same groups |
| `reward/eval_error_rate` | Fraction of trajectories where the grading harness itself raised (missing dep, docker failure, timeout). **Treat any nonzero value as a broken run, not a hard task** — the reward is then not measuring the agent. | the `except` in `compute_reward` stores `str(exc)` in `eval_error`; rate = fraction non-empty |
| `reward/recovered_resolve_rate` | Resolve rate **among** the recovered patches only. The fallback's own report card: near zero means it is feeding the harness junk and should be turned off with `AGENTIC_PATCH_FALLBACK=0`; anywhere near `reward/resolve_rate` means it is recovering real solutions. | `resolved & patch_recovered` over `patch_recovered` |
| `reward/f2p_pass_rate` | Micro-averaged FAIL_TO_PASS pass rate: `sum(f2p_passed) / sum(f2p_total)` over the batch, not a mean of per-instance rates. Measures partial progress on the bug the task is about. | success/failure list lengths under `report["tests_status"]["FAIL_TO_PASS"]` |
| `reward/p2p_pass_rate` | Same for PASS_TO_PASS — the regression check. A drop means patches are breaking working code. | `report["tests_status"]["PASS_TO_PASS"]` |

Note the denominators: the f2p/p2p rates are weighted by test count, so instances with many
tests dominate, and instances that never reached grading contribute 0/0 and drop out.

---

## 4. Our metrics — token accounting (`tokens/*`)

Counted on the exact token ids the loop assembled (no re-tokenization), at
[`Episode.finalize`](src/agentic_grpo/episode.py).

| Metric | Meaning | How collected |
|---|---|---|
| `tokens/mean_completion` | Mean tokens the model **generated** across all turns of an episode: 3334 on run 20260906-013757. Read it against verl's `response_length/mean` (12672), which counts the tool observations too — the ratio says only 26% of the trained sequence is the agent's own output, which is what a tool-heavy workload looks like and what makes the token budget expensive. | `sum(response_mask)`; the mask is 1 for generated tokens, 0 for tool observations |
| `tokens/mean_observation` | Mean size of a **single** tool observation, pooled over every observation in the batch (not per episode). Directly reflects `multi_turn.max_tool_response_length`. | `tool_obs_tokens` lists, flattened across the batch |

---

## 5. Our metrics — client-side latency (`latency/*`)

Deliberately narrow: the one span the inference server cannot see, in three
statistics. Timed with `time.perf_counter()` in the agent loop
(`TrajectoryTimer` provides the same spans for the standalone path).

| Metric | Meaning | How collected |
|---|---|---|
| `latency/mean_trajectory_s` | Mean wall time of a whole episode: container start -> terminal exit, excluding grading. | `perf_counter` bracket from [`Episode.admitted`](src/agentic_grpo/episode.py) to `Episode.close` |
| `latency/median_trajectory_s` | What a **typical** episode costs. Prefer it to the mean, which the tail inflates: 200s median against a 272s mean on run 20260906-013757. | nearest-rank p50 of `t_end - t_start` |
| `latency/p99_trajectory_s` | The straggler, robustly. ~1000s on that run. Also the number that sizes a deadline, if one is ever wanted. | nearest-rank p99 |

`p99 / median` is the straggler ratio — 4.8–5.1× on that run against a null of
1.0. It is not a row: two logged numbers and one division. Tool time and grading
time are not rows either, and not because they do not matter (8.5s and 20.6s an
episode) — we hand both to verl inside `AgentLoopMetrics`, which logs them as
`timing_s/agent_loop/{tool_calls,compute_score}/{min,max,mean}`, i.e. with the
min and max ours never had. See §9.

---

## 5a. Our metrics — per-request timeline (`timeline/*`, `latency/*server*`)

`srv/*` histograms are aggregate: they say the server spent time in prefill, never
*which trajectory* was waiting. These fill that gap by recording, per generate call,
SGLang's own per-request timestamps alongside the client-side call boundaries
([`TurnTiming`](src/agentic_grpo/metrics.py#L40)).

**Where the prefill/decode split comes from.** verl asks SGLang for a whole completion
in one non-streaming call, so no first token ever crosses the client boundary and the
agent loop *cannot* see where prefill ended. SGLang can: its scheduler stamps
`prefill_finished_ts` with `time.time()` when a request's first token is sampled, and
publishes it on the response's `meta_info` — which verl then discards.
[`sglang_timing.py`](src/agentic_grpo/sglang_timing.py) recovers it by subclassing the
rollout-server actor and copying the timing subset onto `TokenOutput.extra_fields`.

Two flags are required, and **both** must be on:

* `engine_kwargs.sglang.enable_metrics: true` — makes SGLang attach the timestamps.
  This is *not* `rollout.prometheus.enable` (which installs the `/metrics` endpoint and
  its request-path middleware, and stays off); it adds no route and no stats loop.
* `AGENTIC_SGLANG_REQUEST_TIMING=1` (default) — installs the actor subclass.

With either off, timestamps degrade to client-side call boundaries: `timing_source`
becomes `"client"`, the coverage tripwire in §5bb warns, and the `*server*` metrics
below disappear rather than reporting zeros.

| Metric | Meaning | How collected |
|---|---|---|
| `timeline/rollout_span_s` | Real elapsed wall time of the rollout phase: `max(t_end) - min(t_start)` across the batch. Unlike `latency/mean_trajectory_s` this is not distorted by the thousands of concurrent episodes. | `t_start` / `t_end` |
| `latency/mean_time_to_first_decode_s` | `t_first_decode - t_start`: everything between "episode began" and "the model emitted its first token" — container start, prompt build, queueing, prefill. A large value here is **docker, not inference**. | [`summarize_turns`](src/agentic_grpo/metrics.py#L186) |
| `latency/mean_server_prefill_s` | Per episode, summed over turns: scheduler hand-off -> first token, i.e. prefill proper. Averaged over the trajectories that **have** server timing, not over the batch — otherwise adding a client-only rollout would look like decoding got faster. 4.0s against 258s of decode, so prefill is 1.5% of generation even at ~21 turns of resent context (prefix cache ~0.93). | `prefill_finished_ts - request_sent_to_scheduler_ts` |
| `latency/mean_server_decode_s` | Per episode, summed over turns: first token -> last token. The only number here that is pure generation. | `decode_finished_ts - prefill_finished_ts` |

Four things deliberately **not** logged here, all of them measured first:

* **mean/max `admission_wait_s`** (the queue for a live container) is
  arithmetic, not a measurement — with `N` episodes over `S` slots each slot runs
  `N/S` of them back to back, so the mean wait is
  `(N/S - 1)/2 x latency/mean_trajectory_s`, which predicted the measured
  855/899/916s of run 20260906-013757 to within 8%. Folding it into any latency
  denominator only dilutes the thing being measured (tool time is 3.1% of an
  episode but 0.7% once the queue is included). The per-episode value is still on
  the timeline as an `admission_wait` span.
* **The server-side queue** (`request_received -> request_scheduled`) measured
  **9 ms** against 4.0s of prefill and 258s of decode: SGLang never makes the
  rollout wait for the scheduler, because 264 container slots cap concurrency
  below the point where it would. It stays on every `generate` timeline event, so
  a future run at a higher slot count can check it without new plumbing.
* **`slots/utilization`** is `1 - bubble/tail_waste_s / timeline/rollout_span_s`
  by construction. The identity held to the second on all three steps of that run
  (244/263/258s of waste over a 2355/2349/2326s span against 0.896/0.888/0.889),
  so it was two rows for one degree of freedom; seconds won because they compare
  directly against `timing_s/step`.
* **`slots/observed` and `timeline/server_timing_coverage`** are assertions about
  the launch (264, and 1.0), not per-step measurements — they moved into
  [`guard_infra_failures`](src/agentic_grpo/metrics.py), §5bb.

Trajectory-level instants (all UNIX wall clock, so they are comparable across the
agent-loop worker and the rollout-server actor; `0.0` means "never reached"):
`t_start` -> `t_first_decode` -> `t_last_gen_end` -> `t_end` (episode over, container
released) -> `t_score_start` -> `t_score_end`.

---

## 5b. Per-trajectory records (opt-in, not W&B)

Everything above is a batch aggregate over 2048 trajectories, so it can say a step had
a long tail but not which instance, which turn, or where the time went. `AGENTIC_TRAJECTORY_DUMP=1
bash scripts/run_grpo.sh` writes one JSON line per episode to
`/data0/shared/$USER/agentic-trajectories/<experiment>/trajectories-<pid>.jsonl` (one file
per `AgentLoopWorker`, symlinked from the run's `analysis/` directory), each carrying:

* `instance_id`, `exit_status`, `reward`,
* `actions` — the commands the agent actually ran, with truncated output,
* `metrics` — the **full** `TrajectoryMetrics`, including the per-turn `turns` timeline.

`turns` is the only place the per-turn detail exists: it is deliberately stripped from
the batch payload sent to the trainer (`to_dict(include_turns=False)`), because 80 turns
x a dozen floats x 2048 trajectories of Ray-serialised non-tensor data buys nothing that
`summarize_turns()` has not already reduced to a scalar. Each entry has the generate call
boundaries (`gen_call_start`/`gen_call_end`), the server's `request_received` /
`request_scheduled` / `decode_start` / `decode_finished` / `response_sent`, its
`completion_tokens`, and the tool span (`tool_start`/`tool_end`/`num_tool_calls`).

Volume, measured at the current 80-turn / 2000-char-observation settings: up to ~114 KiB
per episode (~86 KiB commands, ~28 KiB timeline), so up to **~230 MiB per step** and ~6 GiB
over a 27-step run. It defaults onto `/data0` for that reason — `/` has ~12 GB free.

---

## 5bb. The infra-failure guard (aborts the run)

A crashed container and a wrong patch are both `reward = 0.0`. Nothing downstream can
separate them, so a broken run reads as a hard task and keeps training. Run
`20260811-011747` did exactly that at `AGENTIC_MAX_LIVE_CONTAINERS=64` (512 live
containers): the rootless daemon returned `exit status 125` for most `docker run` calls,
**87% of every batch** became `traj/exit/Crashed:CalledProcessError`, mean reward fell
0.093 → 0.017, and it ran five steps and wrote a 342 GB checkpoint before anyone looked.
`reward/eval_error_rate` stayed 0.0 throughout — the *harness* was healthy, so nothing in
the reward path could notice.

[`guard_infra_failures`](src/agentic_grpo/metrics.py) now runs at the end of every step,
inside the `compute_data_metrics` patch, on two signals that both mean "the reward is not
measuring the agent":

| signal | meaning |
|---|---|
| Σ `traj/exit/Crashed:*` | the episode died before producing a trajectory (container start exhausted its retries, or the loop raised) |
| `reward/eval_error_rate` | the harness could not grade a submitted patch |

Below **0.02** it is silent; between 0.02 and the limit it logs a warning; at or above the
limit it **raises**, which propagates out of verl's `fit()` and stops the run. Raising is
the point — a warning in a log nobody is tailing is what already failed.

| env | default | effect |
|---|---|---|
| `AGENTIC_MAX_CRASH_RATE` | `0.25` | abort above this fraction (a junk value falls back to the default rather than disabling the guard) |
| `AGENTIC_CRASH_GUARD=0` | on | never abort, warn only |

A healthy batch sits at ~0.0 on both, so the thresholds are loose on purpose: this catches
collapse, not the occasional flake.

### The instrumentation tripwires (warn; were logged rows until 2026-09-07)

Three more checks live in the same place, on a different class of failure: not
"the reward is not measuring the agent" but "the *run* is not measuring what it
thinks it is". Each has exactly one correct value, each fails silently, and none
of them is a chart — a flat line at 1.0 is the wrong container for an assertion,
which is why they arrive as `_check/*` keys and are **stripped** before verl logs
anything. They warn rather than raise: they degrade the analysis, not the
training data.

| `_check/` key | correct value | what a bad value means |
|---|---|---|
| `server_timing_coverage` | 1.0 | some or all of the batch has no SGLang per-request timestamps, so `latency/mean_server_*` describes a subset and the prefill/decode split silently became a client-side round-trip measurement. Check `engine_kwargs.sglang.enable_metrics` and `AGENTIC_SGLANG_REQUEST_TIMING` (§5a) |
| `slots_observed` | a multiple of `AGENTIC_MAX_LIVE_CONTAINERS` (264 = 33 x 8 workers) | the env var never reached the `AgentLoopWorker`s through Ray's `runtime_env` and they fell back to the default. Every other metric looks normal; the rollout is simply several times slower. The check is divisibility, so it needs no copy of the worker count |
| `missing_logprobs_rate` | 0.0 | those episodes' `rollout_log_probs` are zeros, so the token-level importance correction is being fed padding (only meaningful with `rollout.calculate_log_probs: true`) |

---

## 5c. Run timeline (`timeline.json`, on by default)

Everything above is a *reduction*: a mean, a rate, a span. The timeline is the raw
material underneath — one event per thing that happened, each with absolute UNIX
timestamps, from **both** halves of a step:

| `cat` | Events | Written by |
|---|---|---|
| `traj` | `container_slot_wait`, `container_start`, `generate` (one per turn, carrying the server's `request_received` / `request_scheduled` / `decode_start` / `decode_finished` / `response_sent` plus `queue_s`/`prefill_s`/`decode_s`), `tool_call` (one per **call**, with the command), `format_error`, `cleanup`, `score`, `episode` | each `AgentLoopWorker`, from [`SWEBenchAgentLoop.run`](src/agentic_grpo/agent_loop.py) |
| `train` | `gen`, `reward`, `old_log_prob`, `adv`, `update_actor`, `update_critic`, `update_weights`, `save_checkpoint`, `testing`, `validate`, `step` — each with its `step` number | the trainer actor, by rebinding verl's `marked_timer` ([`patch_trainer_timeline`](src/agentic_grpo/timeline.py)) |

verl keeps only the *durations* of the training phases (`timing_s/*`, §9); this keeps
the instants, which is what makes the two halves comparable — a straggler episode can be
placed inside the `gen` phase that waited for it, and the gap between `gen` ending and
`update_actor` starting is visible rather than inferred.

Each process appends `timeline-<pid>.jsonl` to `/data0/shared/$USER/agentic-timelines/<experiment>`
(symlinked at `analysis/<experiment>/timeline`). `scripts/build_timeline.py` merges the
shards into a single `timeline.json` — run automatically on exit, including after a
`Ctrl-C`, and re-runnable by hand. The merge attributes every trajectory to the step
whose rollout window contains it: workers are never told a step number, and a step's
rollout cannot overlap the next one's, so the `gen`/`validate` spans bracket them exactly.
Unattributable episodes are left unstamped rather than assigned to the nearest step.

Volume ~20 KiB per episode (~85 events), i.e. **~45 MiB per 2048-episode step** — on
`/data0` for the same reason as §5b. `AGENTIC_TIMELINE=0` turns it off; with
`AGENTIC_TIMELINE_DIR` unset every entry point in `timeline.py` is a no-op.

---

## 5d. Our metrics — the rollout tail (`bubble/*`, `tail/*`)

The step barrier waits for the **last** episode, so a rollout runs longer than its work
requires by exactly the container-slot seconds that went idle while it waited. Three rows:
two for what that costs, one for what the barrier was waiting for. All of them come out of
`t_start`/`t_end` and fields every episode already carries — no server timing, so unlike §5a
they can never degrade to absent.
([`_tail_bubble`](src/agentic_grpo/metrics.py) / `_tail_profile`.)

| Metric | Meaning | step 1 / 2 / 3 of run 20260906-013757 |
|---|---|---|
| `bubble/tail_waste_s` | `span − Σdurations / slots`: the wall-clock seconds a perfect packing of the same episodes into the same slots would have saved. No threshold to pick; `slots` is the observed concurrency peak, so a step that never fills its slots under-reports rather than inventing waste. | 244 / 263 / 258 (10–11% of the rollout, ~5% of the step) |
| `bubble/grading_tail_s` | `max(t_score_end) − max(t_end)`: the window after the last episode ends in which the harness is still grading the last finishers. Nothing generates and nothing trains in it — inside the `gen` phase but outside `rollout_span_s`. | 233 / 80 / 120 |
| `tail/response_tokens_ratio` | Response tokens of the cohort still running at the **last admission** (the final wave, after which the live set can only decay), over the batch mean. Decode is ~95% of every episode's wall clock, so this is *why* the barrier waits. | 2.08 / 2.44 / 1.86 |

**The tail is degenerate generation, not slow infrastructure.** The cohort generates two to
two-and-a-half times the tokens over 1.5x the turns, while its tool time stays at 3.2% of wall
clock against the batch's 3.1% — docker and the harness are not what the barrier waits for.
Its submit rate named it: 20–34 points below the batch, the shortfall being `ContextLimit`
and `RepetitionLimit`. Narrowed offline (from `timeline.json`) to the last 1% to finish it is
starker — 48% ContextLimit, 33% RepetitionLimit, 10% TurnLimit, 12,648 response tokens
against the batch's 3,334. **The barrier waits for episodes that loop or ramble until they run
out of context**, so the tail is a property of the policy and should move with
`traj/mean_repeated_calls`, `traj/exit/ContextLimit` and the repetition guard.

Scale, so it is not over-read: 244–263s a step against the ~2340s the engine spends parked
while the trainer runs. That gap is **not** logged here — it is `timing_s/step − timing_s/gen`
(4933 − 2594 = 2339s at step 1), and it is training, not the rollout tail: `old_log_prob` 308s,
`ref` 344s, `adv` 11s, `update_actor` 1623s, `update_weights` 50s. During the rollout's own
tail the engine is not stopped at all; it is still decoding, at ~30 concurrent requests
instead of ~226.

**Dropped after measuring, so they are not rebuilt:** `tail_waste_ratio` and `tail_overhang_s`
(one division and one more dispersion number on top of two that already exist);
`tail/turns_ratio` and `tail/tool_share` (1.5x and flat every step — a decode-bound workload
has no other shape); `tail/exit/*` (five rows saying what `submitted_rate` says in one);
`bubble/inter_rollout_s` and `rollout_duty_ratio` (derivable from `timing_s/step` and
`timing_s/gen`, which verl logs anyway, and they agreed with those to within 7s);
`deadline_p95/p99_saving_s` (a price list for truncating episodes, which strict on-policy
training will not do). Then, on the same day, `tail/episodes` (136/87/99 — a cohort size
that follows from the slot count and the admission pattern, not from the policy),
`tail/excess_duration_s` (165/155/96s — the same "the tail runs long" statement that
`p99/median` makes scale-free over the whole batch) and `tail/submitted_rate`
(0.51/0.38/0.53 against the batch's 0.72 — the diagnosis above, now established and stable
on every step measured). The per-cohort detail lives in `timeline.json` and the trajectory
dump for the step where something actually looks different.

### Why there is no GPU-idle bubble

Built on 2026-09-07 and removed the same day, with the measurement that killed it; don't
rebuild it without reading this.

The obvious form of "bubble" is *zero requests decoding*: union every turn's
prefill-to-last-token interval across the batch, subtract from the rollout span. It was
implemented, shipped the per-turn intervals to the trainer as a flat `gpu_spans` list
(1.6 MB a step), and then measured against the 131,585 `generate` events of run
20260906-013757's timeline:

| | step 1 | step 2 | step 3 |
|---|---|---|---|
| zero-decode idle | 0.8s (0.03%) | 1.2s (0.05%) | 1.9s (0.08%) |
| mean running requests | 228 | 225 | 227 |
| peak running requests | 264 | 264 | 264 |
| mean running requests, after the last admission | 40 | 27 | 29 |

The GPU is *never* fully idle during a rollout — 2s in 2355s — so the metric is a flat zero
that costs 1.6 MB a step to compute. `mean_running_reqs` is pinned near the slot cap (264
slots x ~0.86 duty), sits inside the 200–240 batch where decode throughput already plateaus,
and moves only when `AGENTIC_MAX_LIVE_CONTAINERS` moves; `peak` **is** the slot cap. What the
last row shows — decode concurrency collapsing from ~226 to ~30 for the last 150–215s — is
the tail, and `bubble/tail_waste_s` says the same thing in seconds of step time, from
timestamps every episode already carries.

---

## 6. verl — reward / advantage distribution (`critic/*`)

Despite the prefix these are logged for GRPO too (there is no critic network, so
`critic/values/*` and `critic/vf_explained_var` are absent). Computed in
`compute_data_metrics` over the padded batch tensors.

| Metric | Meaning | How collected |
|---|---|---|
| `critic/score/{mean,max,min}` | Per-sequence raw score before any KL penalty: `token_level_scores.sum(-1)`. With our binary reward this is 0/1, so `mean` = resolve rate. Aborted (zero-length response) samples are excluded. | `sequence_score` |
| `critic/rewards/{mean,max,min}` | Same, after in-reward KL. We set `algorithm.use_kl_in_reward: false`, so it equals `critic/score/*`. | `token_level_rewards.sum(-1)` |
| `critic/advantages/{mean,max,min}` | GRPO advantages over response tokens only. GRPO normalizes within each group of `n: 8` samples of one prompt, so the mean is ~0. **If max and min are both 0, every group was uniform (all-0 or all-1 reward) and the step carried no learning signal.** | `masked_select(advantages, response_mask)` |
| `critic/returns/{mean,max,min}` | Returns over response tokens; for GRPO they coincide with the advantages. | `masked_select(returns, response_mask)` |

## 7. verl — actor update (`actor/*`)

Emitted by the actor workers during `update_actor` and reduced across DP ranks
(`verl/workers/utils/losses.py`, `verl/workers/engine_workers.py`,
`verl/trainer/ppo/core_algos.py`).

| Metric | Meaning | How collected |
|---|---|---|
| `actor/entropy` | Mean token-level policy entropy over response tokens. Collapse toward 0 means the policy is going deterministic — for GRPO that also kills exploration inside a group. | `entropy_agg`, `ray_trainer.py:1554` |
| `actor/pg_loss` | Clipped policy-gradient (PPO surrogate) loss, aggregated with `loss_agg_mode: token-mean`. | `compute_policy_loss_*` in `core_algos.py` |
| `actor/kl_loss` | Mean KL(policy || reference) over response tokens, from the ref-policy log-probs. 0.0 while the policy has not moved off the reference. | `losses.py:136`, requires `use_kl_loss: true` |
| `actor/kl_coef` | The constant `actor.kl_loss_coef` (0.001) echoed for the record — not adaptive here. | `losses.py:142` |
| `actor/loss` | Total optimized loss = `pg_loss + kl_coef * kl_loss`. | `losses.py:140` |
| `actor/grad_norm` | Global gradient norm **before** clipping, already reduced inside the clip call. Spikes here precede instability. | `engine_workers.py:193` |
| `actor/lr` | Current learning rate from the scheduler (`optim.lr: 1e-6` plus warmup). | `engine_workers.py:196` |
| `actor/ppo_kl` | Mean `old_logprob - logprob` on the sampled tokens: how far this update moved the policy from the behavior policy. With `ppo_epochs: 1` and synchronous weight sync it is ~0 by construction; nonzero means off-policy drift. | `core_algos.py:1366` |
| `actor/pg_clipfrac` | Fraction of tokens where the importance ratio hit the upper clip bound (`clip_ratio: 0.2`). | `core_algos.py:1365` |
| `actor/pg_clipfrac_lower` | Same for the lower / dual-clip bound (negative-advantage side). | `core_algos.py:1367` |
| `actor/perf/max_memory_allocated_gb` | Peak torch **allocated** GPU memory during the update, per worker. | `torch.cuda.max_memory_allocated()`, `engine_workers.py:210` |
| `actor/perf/max_memory_reserved_gb` | Peak torch **reserved** (caching-allocator) GPU memory. This is the number that must stay under the card size; watch it every step — `gpu_memory_utilization` is fixed at launch and cannot be corrected mid-run. | `torch.cuda.max_memory_reserved()` |
| `actor/perf/cpu_memory_used_gb` | Host RAM in use. Large here because `optimizer_offload: true` keeps AdamW state on CPU. | `psutil.virtual_memory().used` |

## 8. verl — sequence shape (`prompt_length/*`, `response_length*`, `response/*`, `global_seqlen/*`, `num_turns/*`)

Derived from `attention_mask` / `response_mask` on the padded batch.

| Metric | Meaning | How collected |
|---|---|---|
| `prompt_length/{mean,max,min}` | Prompt tokens per sample, from the prompt-side attention mask. This is the row that made `tokens/mean_prompt` redundant (identical to the decimal). | `_compute_response_info` |
| `prompt_length/clip_ratio` | Fraction of samples whose prompt hit `data.max_prompt_length` (4096) exactly — i.e. was truncated. | `eq(prompt_length, max_prompt_length).mean()` |
| `response_length/{mean,max,min}` | Response tokens per sample: generated tokens **and** tool observations — everything after the prompt. | `_compute_response_info` |
| `response_length/clip_ratio` | Fraction of samples at exactly `data.max_response_length` (28672) — the context ceiling. Equal to `traj/exit/ContextLimit` (0.11376953125 in both), which is why there is no third row for it. | same pattern |
| `response_length_non_aborted/{mean,max,min,clip_ratio}` | The same four statistics excluding zero-length responses, so aborted samples do not drag the mean down. | `non_aborted_mask` |
| `response/aborted_ratio` | Fraction of samples with a **zero-length** response. Our loop emits a single pad token for degenerate rollouts, so this should stay ~0; nonzero means samples are being dropped upstream. | `mean(response_length == 0)` |
| `num_turns/{mean,max,min}` | verl's turn count, taken from `AgentLoopOutput.num_turns`, which we set to `assistant_turns * 2 + 1` (initial prompt plus one assistant and one tool message per turn). **So this is ~2x `traj/mean_turns` + 1** — the two are consistent, not contradictory. | `batch.non_tensor_batch["__num_turns__"]`, set at [`run`](src/agentic_grpo/agent_loop.py) |
| `global_seqlen/{min,max,mean}` | Total tokens summed per DP-rank chunk **before** load balancing. | `_balance_batch` -> `log_seqlen_unbalance` |
| `global_seqlen/minmax_diff` | `max - min` of the above: the imbalance a naive split would cause. | same |
| `global_seqlen/balanced_{min,max}` | Per-rank token sums **after** the rebalance. The gap between `balanced_max` and `max` is what balancing bought. | same |

## 9. verl — timing (`timing_s/*`, `timing_per_token_ms/*`)

`timing_s/<stage>` is wall-clock seconds for a `marked_timer` block in `RayPPOTrainer.fit`;
`timing_per_token_ms/<stage>` is that time normalized by tokens (`gen` uses response tokens
only; `ref`/`adv`/`update_actor` use prompt + response).

| Metric | Meaning |
|---|---|
| `timing_s/step` | Whole training step, end to end. The other stages sum to roughly this. |
| `timing_s/gen` | Rollout: the full `generate_sequences` call — every episode's generation, docker execution and grading, run concurrently. Usually dominates the step. |
| `timing_s/old_log_prob` | Recomputing log-probs of the sampled tokens under the current actor (the PPO behavior-policy pass). |
| `timing_s/ref` | Reference-policy forward pass for the KL term (`ref.fsdp_config.param_offload: true` makes it slower but cheap in memory). |
| `timing_s/adv` | GRPO advantage computation on the driver — group-wise normalization; near-instant. |
| `timing_s/reward` | Reward assembly on the driver. Nearly free here because grading already happened inside the agent loop and the score just rides along on the last token. |
| `timing_s/update_actor` | The optimizer step(s): forward, backward, clip, AdamW. |
| `timing_s/update_weights` | Resharding trained weights into the SGLang rollout engine (`policy_lag == 0` requires this every step). This is the step that OOMs on fewer than 8 GPUs. |
| `timing_s/start_profile`, `timing_s/stop_profile` | Profiler hooks. No-ops (~0.0002 s / ~3.8 s) unless `global_profiler.tool` is set. |
| `timing_per_token_ms/{gen,ref,adv,update_actor}` | The above divided by token count — the throughput-normalized view, comparable across steps with different batch shapes. |

### `timing_s/agent_loop/*` — per-trajectory breakdown

Produced by verl's `AgentLoopManager` from the `AgentLoopMetrics` **we** return at
[`_build_output`](src/agentic_grpo/agent_loop.py) (`generate_sequences=gen_s`,
`tool_calls=tool_s`, `compute_score=score_s`), then reduced across the batch.

| Metric | Meaning |
|---|---|
| `timing_s/agent_loop/generate_sequences/{min,max,mean}` | Time an episode spent inside `server_manager.generate(...)`, summed over its turns — docker and container setup excluded, unlike the whole-episode `latency/mean_trajectory_s`. |
| `timing_s/agent_loop/tool_calls/{min,max,mean}` | Time inside `env.execute` per episode — **the** tool-time row. We measure it and hand it to verl in `AgentLoopMetrics`; the `latency/mean_tool_s` duplicate was removed on 2026-09-07. 8.5s an episode, i.e. 3.1% of it. |
| `timing_s/agent_loop/compute_score/{min,max,mean}` | Time in the SWE-bench grading harness per episode (its own docker container, run after the agent's is released): 20.6s. Same story as the row above — `latency/mean_score_s` was its duplicate. What the *last* verdicts cost the step is `bubble/grading_tail_s` (§5d). |
| `timing_s/agent_loop/num_preempted/{min,max,mean}` | How many times the server preempted a request (KV-cache pressure) for that episode. Persistently nonzero means the rollout engine is over-subscribed. |
| `timing_s/agent_loop/slowest/{generate_sequences,tool_calls,compute_score,num_preempted,prompt_length,response_length}` | The same fields for the single slowest trajectory (argmax of gen + tool + score). The step barrier waits for exactly this episode, so these six numbers explain rollout wall time far better than the means. `slowest/response_length` equal to `max_response_length` means the straggler ran out of context rather than finishing. |

## 10. verl — throughput (`perf/*`)

| Metric | Meaning | How collected |
|---|---|---|
| `perf/total_num_tokens` | Total tokens (prompt + response) processed in the step. | `sum(batch.meta_info["global_token_num"])` |
| `perf/time_per_step` | Same value as `timing_s/step`, re-exported by `compute_throughout_metrics`. | `timing_raw["step"]` |
| `perf/throughput` | Tokens per second **per GPU**: `total_num_tokens / (step_time * n_gpus)`. | `compute_throughout_metrics` |
| `perf/mfu/actor` | Model FLOPs utilization of the training update: FLOPs estimated from token counts and elapsed time, over the device's promised FLOPs. | `FlopsCounter.estimate_flops` in the actor worker |
| `perf/mfu/actor_infer` | Same for the `old_log_prob` inference pass. | `ray_trainer.py:1555` |

Both MFU figures are low in this project (~2% actor / ~5% infer) because the step is bound by
docker container slots and rollout wall time, not by compute — expected, not a regression.

### Rollout throughput (`perf/rollout_decode_tok_s`, ours)

| Metric | Meaning | How collected |
|---|---|---|
| `perf/rollout_decode_tok_s` | Decode tokens per second over the rollout, cluster-wide. Compare against the **~3.6k tok/s** SGLang plateaus at when saturated (run 20260906-013757): the gap is how far from saturation the rollout ran. | `sum(completion_tokens) / timeline/rollout_span_s` in [`_timeline_aggregate`](src/agentic_grpo/metrics.py) |

### Why there is no rollout MFU

Built on 2026-09-07 and removed the same day; don't rebuild it without reading this.

A rollout MFU is straightforward to compute — forward FLOPs (`2ND` plus attention, prefill
charged only for uncached tokens) over `rollout_span_s x n_gpus x device peak`. On a batch
shaped like run 20260906-013757 it comes out at **0.15%**. The problem is that the number
cannot be acted on:

* **The roofline is wrong for the phase.** The denominator is the dense BF16 compute peak
  (2250 TFLOP/s on a B200), but decode is memory-bandwidth-bound. No batch size brings decode
  near that ceiling, so 0.15% does not mean "99.85% wasted" — and there is no way to say what
  a good value would be. A metric with no attainable target is not a diagnostic.
* **It is not independent.** MFU is a function of token counts and the rollout span, both
  already logged, and tracks `perf/rollout_decode_tok_s` up to a slowly-varying constant.
* **The question it would answer is already answered better.** Why the GPUs idle is covered by
  `bubble/tail_waste_s` (244–263s of a ~2350s rollout) and `srv/drain_ratio` —
  measured, not estimated.
* **It carries four ways to be quietly wrong**: the perfect-prefix-cache assumption for prefill,
  the midpoint-context approximation for attention, MoE active-parameter counting, and a device
  table that silently omits unknown GPUs.

Worth knowing if it ever *is* rebuilt: attention is the larger term for this model, not a
correction. Qwen3-30B-A3B activates ~3.35B parameters but keeps 48 layers x 32 heads x 128
head_dim, so at a 12k context attention is ~9.4 GFLOP/token against ~6.7 dense — 58% of the
work. A `2ND`-only estimate is less than half the truth. And verl's `FlopsCounter` is
fwd+bwd (`6ND`), so a forward-only figure is exactly a third of what it reports for the same
tokens.

## 11. verl — bookkeeping and validation

| Metric | Meaning | How collected |
|---|---|---|
| `training/global_step` | Optimizer step counter; the x-axis for everything else. | `ray_trainer.py:1715` |
| `training/epoch` | Current pass over the training parquet. | `ray_trainer.py:1716` |
| `val-core/swebench/reward/mean@1` | **The validation headline**: mean binary reward on the val split with 1 sample per instance. `swebench` is the `data_source` field written by [prepare_swebench_hf.py:102](scripts/prepare_swebench_hf.py#L102); `mean@1` is `process_validation_metrics`' naming for the aggregation and the sample count. Populated in `step_0` and at each `test_freq`. | `_val_metrics_update`, `ray_trainer.py:724` |
| `val-aux/num_turns/{mean,max,min}` | Turn statistics on the validation pass, same `2n+1` convention as `num_turns/*`. Everything not selected as the core metric lands under `val-aux`. | `ray_trainer.py:745-747` |

A caveat for reading `step_0`: the val split is small (20 instances in the `20260731-025759`
run), so one instance is worth 0.05 of the metric. A single-step change there is noise.

---

## 12. `srv/*` — SGLang server ground truth

> **Status: DISABLED since 2026-08-05.** No run emits `srv/*` unless you re-enable it.
> Collection is not free — `rollout.prometheus.enable` adds middleware to the rollout
> server's request path and forces `disable_log_stats: false` (SGLang's per-batch stats
> loop) for the whole run — and the question it was added to answer is settled:
> `drain_ratio` 1.0 with `running_peak` 64 against `capacity` 256 means inference is
> never the constraint (docker container slots are). To turn it back on, set
> `AGENTIC_SGLANG_METRICS=1` in [run_grpo.sh](scripts/run_grpo.sh) **and** flip
> `prometheus.enable`, `disable_log_stats` and `engine_kwargs.sglang.enable_metrics` in
> [grpo_swebench.yaml](configs/grpo_swebench.yaml) — the comment blocks in both files
> spell out why all four move together. Worth doing once at the full batch of 256,
> where 2048 concurrent rollouts could actually saturate the server.

Not present in `analysis/20260731-025759/metrics.csv` (that run predates the fixes), then
emitted by runs between 2026-08-01 and 2026-08-05 — e.g. `analysis/20260801-072502/run.log`,
the only run with data. Collected by
[`SGLangServerMonitor`](src/agentic_grpo/server_monitor.py), a background thread that scrapes
the rollout server's Prometheus `/metrics` about once a second, summarized per step in the
same `compute_data_metrics` patch. Requires `rollout.prometheus.enable: true` **and**
`disable_log_stats: false`; the scrape deliberately bypasses `HTTP_PROXY`, and the server URL
is discovered from verl's named Ray actor `sglang_server_<rank>_<node>` because the port is
ephemeral. With more than one replica only one is scraped, so the counts are per-replica.

**Drain phase** (is the GPU staying saturated?), computed over the busy window
(`num_running_reqs > 0`):

| Metric | Meaning |
|---|---|
| `srv/gpu_busy_s` | Length of the busy window — first to last sample with a running batch. |
| `srv/drain_window_s` | Time from drain start to the end of the window: the tail where the GPU idles progressively because only stragglers remain. |
| `srv/drain_ratio` | `drain_window / gpu_busy`. 1.0 means the server was never saturated in the window — the rollout is bound by something other than the GPU (for us: docker container slots). |
| `srv/running_peak` | Largest observed running batch. Read it against the saturation threshold, which is `AGENTIC_MAX_RUNNING_REQUESTS` (should match `rollout.max_num_seqs`) or, unset, the observed peak — a launch setting, so read it off the run config rather than a per-step row. |
| `srv/token_usage_peak` | Peak KV-cache utilization (0-1). |
| `srv/sample_count` | Number of scrapes in the window — a sanity check that the poller was alive. |

**Latency and token breakdown**, from counter/histogram deltas between the first and last
sample in the window (`delta(sum)/delta(count)` for histogram means):

| Metric | Meaning |
|---|---|
| `srv/ttft_mean_s` | Mean time to first token — prefill plus queueing, server-side. |
| `srv/inter_token_latency_mean_s` | Mean decode-step latency. |
| `srv/e2e_latency_mean_s` | Mean end-to-end request latency (one turn, not one episode). |
| `srv/queue_time_mean_s` | Mean time requests waited before running. Nonzero and growing = over-subscription. |
| `srv/prompt_tokens`, `srv/generation_tokens`, `srv/cached_tokens` | Counter deltas over the window: prompt tokens processed, tokens decoded, prompt tokens served from the prefix cache. |
| `srv/prefix_cache_hit_rate` | `cached / prompt`. The batch-level prefix-cache number. Per turn it is `TurnTiming.cache_hit_rate()` in the trajectory dump (~0.93); the `tokens/mean_cached_prompt` row that tried to carry it was a per-episode sum and was removed. |
| `srv/num_requests` | Requests completed in the window (one per assistant turn, so ~ `traj/mean_turns` x 2048). |
| `srv/gen_throughput_mean` | Mean decode tokens/s while busy. |

A histogram-derived value of exactly `0.0` (as `srv/ttft_mean_s` currently shows) means the
counter did not advance between the window's first and last sample — not that latency was
zero.

---

## 13. Known dead or duplicated rows

Our own duplicates were **deleted** on 2026-09-07 rather than documented (see
"Removed" below). What is left here is verl's, which we do not control: 181 rows
came out of run 20260906-013757 and about 46 of them carry no independent
information. Worth knowing before plotting anything.

| Row | Status |
|---|---|
| `rollout_corr/*` (27 rows) + `training/rollout_probs_diff_*` (4) | verl's rollout-correction diagnostics: **31 rows for one question** — how far the SGLang rollout's token probabilities drifted from the training engine's. Read two: `rollout_corr/rollout_is_eff_sample_size` (0.9967 on that run, i.e. the importance weights are essentially all 1) and `rollout_corr/kl` (0.0018). The rest are the same mismatch as a mean/min/max/std, per-token and per-sequence, in probability, log-probability, perplexity and χ² form. They come with `rollout_correction` (which we need for token-level TIS) and cannot be trimmed from our side. |
| `timing_s/agent_loop/num_preempted/{min,max,mean}` and `slowest/num_preempted` | **Dead: −1 on every step.** SGLang does not report preemption counts through the path verl reads, so the "rollout engine is over-subscribed" signal these promise does not exist. Use `srv/*` (§12) if that question comes back. |
| `response_length_non_aborted/*` (4 rows) | Identical to `response_length/*` whenever `response/aborted_ratio` is 0.0, which is every step of every healthy run — our loop emits a pad token instead of a zero-length response. |
| `critic/returns/*` vs `critic/advantages/*` | Identical for GRPO (no critic, so returns *are* the normalized advantages). |
| `critic/rewards/*` vs `critic/score/*` | Identical while `use_kl_in_reward: false`. |
| `perf/time_per_step` vs `timing_s/step` | The same value under two names. |
| `timing_s/start_profile`, `timing_s/stop_profile` | No-op hooks unless profiling is enabled. |
| `prompt_length/mean` vs `tokens/mean_prompt` | Were identical (2612.0 both). Ours is the one that went. |
| `timing_s/agent_loop/{tool_calls,compute_score}/mean` | The tool and grading means, bit-identical to the `latency/mean_tool_s` and `latency/mean_score_s` we used to log — because we compute them and hand them to verl. Ours went; these have min/max too. |
| `pytest_output_length` | Collected per trajectory but **not** aggregated, so it never reaches the CSV. |

### `straggler_ratio` was rebuilt on 2026-09-07

The first version was `(max(t_end) - median(t_end)) / rollout_span_s`. It was wrong, and the
way it was wrong is worth keeping:

Episodes are **admitted over the span** as container slots free up, not started together. So
the median episode finishes early because half the batch ran in earlier waves — not because
episodes are fast. Under pure queueing with *zero* straggling that ratio lands near 0.5 by
construction, and on run 20260906-013757 it measured **0.47 / 0.50 / 0.53** across the three
steps: flat, and blind to the workload. Reading 0.53 as "half the rollout is straggler tail"
was exactly backwards.

Measured on the 6143 real episode records in `analysis/20260906-013757/timeline/`, the span is
throughput-bound, not straggler-bound: 2048 episodes x 269s of work over ~264 slots is a
~2110s floor against a ~2350s observed span, so **~89% of the rollout span is just the time it
takes to push the batch through the slots**. Two independent estimates put the genuine
straggler excess at ~240s, ~10% of the span — and `bubble/tail_waste_s` measures exactly
that, in seconds (244/263/258).

The replacement measures the **duration spread** instead, which drops admission scheduling out
entirely (`tests/test_trajectory_timeline.py` pins this: the same episodes staggered across a
5x longer span must score identically). It reads 4.8–5.1x against a null of 1.0. It is
**not a row** — `latency/p99_trajectory_s / latency/median_trajectory_s` are both logged and
the ratio is one division — but the construction is the part worth not repeating.

While that was being checked, the deadline question got an answer too: capping episodes at 600s
would truncate **8.3–8.6%** of them to recover 6–7% of episode-seconds; at 900s it is ~2% for
~1.5%. With 61–68% of GRPO groups already at zero advantage
(`reward/zero_advantage_group_rate`), cutting the slowest 8.5% — which are the hard instances —
is very likely a net loss. That is why there is no episode deadline.

### Removed on 2026-09-07 — first pass (order statistics and residuals)

Dropped for being unactionable — not wrong, but nothing you would do differently if they
moved. Present in `analysis/` runs before this date. The test applied was *what decision
changes if this number changes?*, which is stricter than §13's "dead or duplicated" and is
what §10's removed rollout MFU also failed.

| Removed | Why |
|---|---|
| `latency/max_trajectory_s` | A max over ~2048 episodes: it tracks whichever single container hung, and the quantity it was standing in for — rollout wall time — is measured directly as `timeline/rollout_span_s`, with the duration tail as `latency/p99_trajectory_s`. |
| `latency/max_time_to_first_decode_s` | Same order-statistic problem; the mean is kept. |
| `tokens/max_observation` | Pinned to a config ceiling. Observations are truncated before being counted ([agent_loop.py](src/agentic_grpo/agent_loop.py)), so this is the largest tool budget in play (the editor's `AGENTIC_EDIT_VIEW_CHARS`, since `view` sets `bounded` and escapes the loop's 2000-char cap), and something fills it every step. It measured the config. `tokens/mean_observation` is kept and is the one that carries signal. |
| `latency/mean_non_tool_s` (was `latency/mean_generation_s`) | A residual, `trajectory - tool`. It moves for four unrelated reasons — decode, container start, orchestration, queueing — so a change in it points nowhere, and `timing_s/agent_loop/generate_sequences` and `latency/mean_server_decode_s` each measure a piece of it directly. |
| `srv/drain_start_offset_s` | Exactly `srv/gpu_busy_s - srv/drain_window_s`. Three rows, two degrees of freedom. |
| `srv/capacity` | `AGENTIC_MAX_RUNNING_REQUESTS` echoed back once a step, or — unset — the observed peak, which is already `srv/running_peak`. A launch setting, not a measurement. |
| `srv/prefix_cache_hit_rate_gauge` | The same quantity as `srv/prefix_cache_hit_rate` read off SGLang's gauge instead of its token counters. A cross-check nobody would act on; the counter-derived row is authoritative. |

### Removed on 2026-09-07 — second pass (duplicates and answered questions)

The first pass removed rows that were the wrong *statistic*. This one applies the
two tests in §1 to every row we emit: **does it show the run is doing real
agentic work, or does it change a decision?** Our namespace went from ~55 rows a
step to 35. Every cut below is either an identity checked against
`analysis/20260906-013757/metrics.csv` and the 6143 episode records in its
`timeline/`, or a question that has now been answered and recorded here.

Exact duplicates of a row that is still logged (the numbers are from that run):

| Removed | Identity |
|---|---|
| `reward/mean` | `= reward/resolve_rate` (binary reward), 0.2979 / 0.2944. |
| `tokens/mean_response` | `= tokens/mean_completion`; both are `sum(response_mask)`. |
| `traj/truncation_rate` | `= traj/exit/ContextLimit = response_length/clip_ratio`, 0.11376953125 in all three. Three rows, one number. |
| `tokens/mean_prompt` | `= prompt_length/mean`, 2612.0 both. verl's is computed from the attention mask, ours from `len(prompt_ids)`. |
| `tokens/mean_total` | `= prompt_length/mean + response_length/mean`, 2612.0 + 12672.40625 = 15284.40625, exactly. |
| `latency/mean_tool_s` | `= timing_s/agent_loop/tool_calls/mean`, 8.503529838491701 in both — we compute it and hand it to verl, which also logs min and max. |
| `latency/mean_score_s` | `= timing_s/agent_loop/compute_score/mean` (20.5613 both; they differ at the 6th decimal only because of the denominator). |
| `slots/utilization` | `= 1 - bubble/tail_waste_s / timeline/rollout_span_s`. Recomputed from the timeline: waste 244/263/258s, span 2355/2349/2326s, utilization 0.896/0.888/0.889 — the identity holds to the second on all three steps. |
| `latency/straggler_ratio` | `= p99 / median`, both still logged. Same reason `srv/drain_start_offset_s` went. |

Answered, or too small to act on:

| Removed | Measurement that ended it |
|---|---|
| `tokens/mean_cached_prompt` | A per-episode **sum** over ~21 turns, so it reads 242,193 against a 2612-token prompt and cannot be turned into a hit rate (§4 used to warn about exactly this). The real number is per turn: `TurnTiming.cache_hit_rate()` in the dump, ~0.93. |
| `reward/patch_applied_rate` | Existed to expose malformed-diff loss, i.e. the gap to `1 - traj/empty_patch_rate`. That gap is **0.1%**: 0.8613 against 0.8623, and 0.8716 against 0.8735. Patches that exist apply. |
| `traj/any_policy_denial_rate` | 0.0464 against `mean_policy_denials` 0.0469 — for an event this rare the per-episode share and the per-episode count are the same number twice. |
| `traj/any_tool_timeout_rate` | 0.00098 against a mean of 0.00195: two episodes and four calls in a batch of 2048. |
| `traj/mean_edit_errors` | `= edit_error_rate x edit attempts`; the rate (0.31, and it is high) is the one that says the model is not copying `old_str` exactly. |
| `traj/any_edit_tool_rate` | **0.979 / 0.985.** This was the adoption tripwire for the edit tool ("if it stays near 0, prompting is the fix"), and adoption happened. |
| `traj/mean_views` | 7.8 views an episode with no decision attached to the number: the actionable part of a view is its size, which is `tokens/mean_observation` against `AGENTIC_EDIT_VIEW_CHARS`. |
| `traj/mean_unknown_tool_calls` | 0.10 an episode = 0.5% of 21 tool calls (it was 2654 calls on 20260903-002235, which is why it was added). |
| `traj/mean_salvaged_calls` | 0.098 → 0.053, i.e. ~0.03% of tool calls: the salvage layer's yield is now negligible, which is its report card. It stays in the code — the cost of salvaging is nothing — but not as a row. |
| `traj/mean_format_errors` | 0.221 an episode, against `any_format_error_rate` 0.099 and the 3.5% of episodes that actually die of it (`FormatErrorLimit` + `NoToolCall`). The exits price the failure; the mean of a recovered-from event does not. |
| `latency/mean_server_queue_s` | Overlapped `mean_server_prefill_s` (both end at the first token) and the difference — the actual scheduler queue — is **9 ms** against 4.0s of prefill and 258s of decode. Still on every `generate` timeline event. |
| `tail/episodes`, `tail/excess_duration_s`, `tail/submitted_rate` | See §5d: the cohort's size follows from the slot count, its excess duration is the batch-wide `p99/median` in a less robust form, and its submit rate is a finding (0.51/0.38/0.53 against 0.72) that is now written down and stable on every step measured. |

Moved into [`guard_infra_failures`](src/agentic_grpo/metrics.py) as assertions
(§5bb), because their correct value is a constant and a flat chart line is the
wrong container for an assertion: `timeline/server_timing_coverage` (1.0),
`slots/observed` (264), `traj/missing_logprobs_rate` (0.0). This is the item the
previous pass left open. `reward/eval_error_rate` stays a row *and* a guard
signal: unlike the three, its healthy value is 0 but its interesting values are
graded — it is the rate at which the harness is failing, not a binary about the
launch.

**Added** in the same pass, because the two tests cut both ways —
`reward/zero_advantage_group_rate` and `reward/solved_group_rate` (§3). 61–68% of
GRPO groups produced no gradient on run 20260906-013757 and none of the other 181
rows said so. A profiling run that spends 60% of its rollout on groups with zero
advantage is not measuring the workload it thinks it is, which makes this both a
validity condition and the most actionable row in the file.

### Renames of 2026-09-07

Runs in `analysis/` from before this date use the left-hand names, so a chart spanning the
change will show two disjoint series. Nothing about what is measured changed in the
renames themselves; the removals and additions are in the two sections above.

| Old | New | Why |
|---|---|---|
| `timeline/tail_s` | → `timeline/straggler_s`/`_ratio`, then replaced by `latency/{median,p99}_trajectory_s` (the ratio itself is not a row) | "tail" also meant the `srv/` drain tail, and the end-time construction turned out to measure admission scheduling rather than straggling — see "`straggler_ratio` was rebuilt" above |
| `traj/resolve_rate` | `reward/resolve_rate` | harness verdict — prefix is provenance (§1a) |
| `traj/mean_reward` | `reward/mean`, then removed as a duplicate of `reward/resolve_rate` | same |
| `reward/empty_patch_rate` | `traj/empty_patch_rate` | needs no grading |
| `reward/patch_recovered_rate` | `traj/patch_recovered_rate` | same |
| `traj/format_error_rate` | `traj/any_format_error_rate` | per episode, not per event (§1a) |
| `traj/timeout_rate` | `traj/any_tool_timeout_rate`, then removed (0.001) | same, and it is specifically *tool* timeouts |
| `traj/repeat_rate` | `traj/any_repeat_rate` | same |
| `traj/policy_denial_rate` | `traj/any_policy_denial_rate`, then removed (same number as the mean) | same |
| `traj/test_run_rate` | `traj/any_test_run_rate` | same |
| `traj/edit_tool_use_rate` | `traj/any_edit_tool_rate`, then removed (adoption reached 0.98) | same |
| `slots/zero_wait_count` | `slots/observed`, now the `_check/slots_observed` assertion | named the computation, not the quantity |
| `reward/cache_hit_rate` | `reward/eval_cache_hit_rate` | collided with SGLang's KV cache |
| `srv/prefix_cache_hit_frac` | `srv/prefix_cache_hit_rate` | says which cache, and `_rate` matches its siblings |
| `tokens/mean_obs` | `tokens/mean_observation` | only abbreviation in the namespace |
| `timeline/server_timing_rate` → `..._coverage` | now the `_check/server_timing_coverage` assertion | see §5bb |
| `srv/samples` | `srv/sample_count` | only bare count with no suffix |
