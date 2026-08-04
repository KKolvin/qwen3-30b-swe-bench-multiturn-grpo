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
   [agent_loop.py:809-885](src/agentic_grpo/agent_loop.py#L809-L885). The patch adds
   `TrajectoryMetrics.aggregate(...)` (`traj/*`, `reward/*`, `tokens/*`, `latency/*`) and
   `SGLangServerMonitor.summarize_since_last(...)` (`srv/*`).
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
| our agent loop | `traj/*`, `tokens/*`, `latency/*`, `reward/*` | [`TrajectoryMetrics`](src/agentic_grpo/metrics.py), filled in [`SWEBenchAgentLoop.run`](src/agentic_grpo/agent_loop.py#L395) |
| SWE-bench harness | the `reward/*` contents and `traj/resolve_rate` | official `swebench` `run_instance` -> `report.json`, read by [reward.py](src/agentic_grpo/reward.py) |
| SGLang server | `srv/*` | Prometheus `/metrics` scrape, [server_monitor.py](src/agentic_grpo/server_monitor.py) |

Two batch-size facts you need in order to read any average: `data.train_batch_size: 256`
prompts x `rollout.n: 8` samples = **2048 trajectories per step**, and every `traj/*`,
`tokens/*`, `latency/*`, `reward/*` number is a mean (or rate) over those 2048.

---

## 2. Our metrics — trajectory bookkeeping (`traj/*`)

All of these come from a `TrajectoryMetrics` object created per episode at
[agent_loop.py:399](src/agentic_grpo/agent_loop.py#L399), shipped back to the trainer inside
`AgentLoopOutput.extra_fields["trajectory_metrics"]`, and reduced by
[`TrajectoryMetrics.aggregate`](src/agentic_grpo/metrics.py#L82).

| Metric | Meaning | How collected |
|---|---|---|
| `traj/exit/<Status>` | Fraction of the batch that ended with that exit status. Keys are dynamic — a status only appears if some episode hit it. | `Counter(m.exit_status)/n`; `exit_status` is set at the single point where the episode stops. |
| `traj/exit/Submitted` | Agent ran the submit marker; `env.execute` raised `Submitted` carrying a patch. **The only path that can produce reward 1.** | [agent_loop.py:510](src/agentic_grpo/agent_loop.py#L510) |
| `traj/exit/TurnLimit` | Used all `max_assistant_turns` without submitting. | loop `else` branch, [agent_loop.py:524-528](src/agentic_grpo/agent_loop.py#L524-L528) |
| `traj/exit/ContextLimit` | Response tokens reached `data.max_response_length`; episode cut short. | the three `traj.response_len() >= self.response_length` checks |
| `traj/exit/NoToolCall` | `max_consecutive_format_errors` turns in a row contained **no** `<tool_call>` block at all (agent believes it is finished). | [agent_loop.py:476](src/agentic_grpo/agent_loop.py#L476) |
| `traj/exit/FormatErrorLimit` | Same budget exhausted, but those turns *did* contain `<tool_call>` blocks we could neither parse nor salvage. | same line, `attempted=True` branch |
| `traj/exit/Crashed:<Type>` | Infra failure (container start, docker, tokenizer). The sample degrades to reward 0 instead of killing the step. | `except Exception` at [agent_loop.py:535](src/agentic_grpo/agent_loop.py#L535) |
| `traj/resolve_rate` | Fraction of trajectories the SWE-bench harness marked `resolved` (all FAIL_TO_PASS + PASS_TO_PASS pass). The headline number. | `report.json["resolved"]` |
| `traj/mean_reward` | Mean binary reward. Identical to `traj/resolve_rate` by construction (`reward = 1.0 if resolved else 0.0`). | [reward.py:69](src/agentic_grpo/reward.py#L69) |
| `traj/mean_turns` | Mean assistant turns (one generate call = one turn). | `assistant_turns` counter |
| `traj/mean_tool_calls` | Mean bash tool calls executed. Exceeds turns when one turn emits several calls. | incremented per call, [agent_loop.py:494](src/agentic_grpo/agent_loop.py#L494) |
| `traj/mean_edits` | Intended as "edit tool calls". **Dead metric — always 0.0**: `edit_count` is never incremented and the bash-only scaffold has no `edit` tool. | — |
| `traj/truncation_rate` | Fraction of episodes that hit the context ceiling. Always equals `traj/exit/ContextLimit`, since `truncated` is set at exactly the same three places. | `sum(m.truncated)/n` |
| `traj/mean_format_errors` | Mean number of turns per episode that produced no usable action (unparseable call *or* no call). | `metrics.format_errors` |
| `traj/format_error_rate` | Fraction of episodes with **at least one** such turn. High here alongside a healthy `Submitted` rate means the nudge-and-continue recovery is working. | `sum(m.format_errors > 0)/n` |
| `traj/mean_salvaged_calls` | Mean tool calls per episode that verl's hermes parser dropped but [`_salvage_tool_calls`](src/agentic_grpo/agent_loop.py#L255) recovered. Direct measure of how much signal the salvage layer saves. | `metrics.salvaged_tool_calls` |

---

## 3. Our metrics — reward diagnostics (`reward/*`)

These exist so a reward of 0 is attributable. Without them, "the harness is broken" and "the
agent wrote a bad patch" are the same number. All are read out of the harness's own
`report.json` by [`_read_harness_report`](src/agentic_grpo/reward.py#L141) — nothing is
re-graded here.

| Metric | Meaning | How collected |
|---|---|---|
| `reward/eval_error_rate` | Fraction of trajectories where the grading harness itself raised (missing dep, docker failure, timeout). **Treat any nonzero value as a broken run, not a hard task** — the reward is then not measuring the agent. | the `except` in `compute_reward` stores `str(exc)` in `eval_error`; rate = fraction non-empty |
| `reward/empty_patch_rate` | Fraction where the agent never submitted a diff. Short-circuits before docker. Roughly the complement of `traj/exit/Submitted`. | `not model_patch.strip()`, [reward.py:52](src/agentic_grpo/reward.py#L52) |
| `reward/patch_applied_rate` | Fraction where the submitted diff actually applied to the repo. The gap between this and `1 - empty_patch_rate` is malformed-diff loss. | `report["patch_successfully_applied"]` |
| `reward/f2p_pass_rate` | Micro-averaged FAIL_TO_PASS pass rate: `sum(f2p_passed) / sum(f2p_total)` over the batch, not a mean of per-instance rates. Measures partial progress on the bug the task is about. | success/failure list lengths under `report["tests_status"]["FAIL_TO_PASS"]` |
| `reward/p2p_pass_rate` | Same for PASS_TO_PASS — the regression check. A drop means patches are breaking working code. | `report["tests_status"]["PASS_TO_PASS"]` |

Note the denominators: the f2p/p2p rates are weighted by test count, so instances with many
tests dominate, and instances that never reached grading contribute 0/0 and drop out.

---

## 4. Our metrics — token accounting (`tokens/*`)

Counted on the exact token ids the loop assembled (no re-tokenization), at
[agent_loop.py:546-554](src/agentic_grpo/agent_loop.py#L546-L554).

| Metric | Meaning | How collected |
|---|---|---|
| `tokens/mean_prompt` | Mean length of the **initial** prompt (system + instance templates + bash tool schema). The dataclass comment calls this the peak context size, but the code assigns `len(prompt_ids)`, i.e. the prompt slice only — the grown context is `tokens/mean_total`. | `len(prompt_ids)` after `traj.finalize` |
| `tokens/mean_completion` | Mean tokens the model generated across all turns of an episode. | `sum(response_mask)`; the mask is 1 for generated tokens, 0 for tool observations |
| `tokens/mean_response` | **Identical to `tokens/mean_completion`** (both assigned `sum(response_mask)`). Kept as the "tokens actually trained on" name. | same |
| `tokens/mean_cached_prompt` | Intended as prefix-cache hits. **Dead metric — always 0.0**: nothing assigns `cached_prompt_tokens` on the token-level path, since the server (not the client) owns cache accounting. Use `srv/prefix_cache_hit_frac`. | — |
| `tokens/mean_total` | Mean `len(prompt_ids) + len(response_ids)` after right-clipping to `max_response_length`. Compare against `rollout.max_model_len` (32768) to see how much context budget is really used. | [agent_loop.py:554](src/agentic_grpo/agent_loop.py#L554) |
| `tokens/mean_obs` | Mean size of a **single** tool observation, pooled over every observation in the batch (not per episode). Directly reflects `multi_turn.max_tool_response_length`. | `tool_obs_tokens` lists, flattened across the batch |
| `tokens/max_obs` | Largest single observation in the batch. | `max(flat_obs)` |

---

## 5. Our metrics — client-side latency (`latency/*`)

Deliberately narrow: only the two spans the inference server cannot see. Timed with
`time.perf_counter()` in the agent loop (`TrajectoryTimer` provides the same spans for the
standalone path).

| Metric | Meaning | How collected |
|---|---|---|
| `latency/mean_trajectory_s` | Mean wall time of a whole episode: container start -> terminal exit, excluding grading. | `t_traj` bracket, [agent_loop.py:416](src/agentic_grpo/agent_loop.py#L416) and [:538](src/agentic_grpo/agent_loop.py#L538) |
| `latency/max_trajectory_s` | Slowest single episode. The step barrier waits on this, so it — not the mean — sets rollout wall time. | `max(...)` over the batch |
| `latency/mean_tool_s` | Mean time inside `env.execute(...)`, i.e. docker command execution, summed over an episode. | `t1` bracket around `_run_bash`, [agent_loop.py:495-497](src/agentic_grpo/agent_loop.py#L495-L497) |
| `latency/mean_generation_s` | **Derived, not measured**: `max(trajectory - tool, 0)`. Includes queueing, generation, tokenization and container setup, so it is an upper bound on generation. The authoritative split is `srv/*` and `timing_s/agent_loop/*`. | [`generation_time()`](src/agentic_grpo/metrics.py#L70) |

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
| `prompt_length/{mean,max,min}` | Prompt tokens per sample, from the prompt-side attention mask. Mirrors `tokens/mean_prompt`. | `_compute_response_info` |
| `prompt_length/clip_ratio` | Fraction of samples whose prompt hit `data.max_prompt_length` (4096) exactly — i.e. was truncated. | `eq(prompt_length, max_prompt_length).mean()` |
| `response_length/{mean,max,min}` | Response tokens per sample: generated tokens **and** tool observations — everything after the prompt. | `_compute_response_info` |
| `response_length/clip_ratio` | Fraction of samples at exactly `data.max_response_length` (28672) — the context ceiling. Should track `traj/truncation_rate`. | same pattern |
| `response_length_non_aborted/{mean,max,min,clip_ratio}` | The same four statistics excluding zero-length responses, so aborted samples do not drag the mean down. | `non_aborted_mask` |
| `response/aborted_ratio` | Fraction of samples with a **zero-length** response. Our loop emits a single pad token for degenerate rollouts, so this should stay ~0; nonzero means samples are being dropped upstream. | `mean(response_length == 0)` |
| `num_turns/{mean,max,min}` | verl's turn count, taken from `AgentLoopOutput.num_turns`, which we set to `assistant_turns * 2 + 1` (initial prompt plus one assistant and one tool message per turn). **So this is ~2x `traj/mean_turns` + 1** — the two are consistent, not contradictory. | `batch.non_tensor_batch["__num_turns__"]`, set at [agent_loop.py:563](src/agentic_grpo/agent_loop.py#L563) |
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
[agent_loop.py:724](src/agentic_grpo/agent_loop.py#L724) (`generate_sequences=gen_s`,
`tool_calls=tool_s`, `compute_score=score_s`), then reduced across the batch.

| Metric | Meaning |
|---|---|
| `timing_s/agent_loop/generate_sequences/{min,max,mean}` | Time an episode spent inside `server_manager.generate(...)`, summed over its turns. Unlike `latency/mean_generation_s` this excludes docker and container setup. |
| `timing_s/agent_loop/tool_calls/{min,max,mean}` | Time inside `env.execute` per episode — the same quantity as `latency/mean_tool_s`, reduced by verl. |
| `timing_s/agent_loop/compute_score/{min,max,mean}` | Time in the SWE-bench grading harness per episode (its own docker container, run after the agent's is released). |
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

Not present in `analysis/20260731-025759/metrics.csv` (that run predates the fixes) but
emitted by every run since — e.g. `analysis/20260801-072502/run.log`. Collected by
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
| `srv/drain_start_offset_s` | Offset into that window of the last instant the server was still saturated (queue non-empty, or running batch >= capacity). After it, the batch can only shrink. |
| `srv/drain_window_s` | Time from drain start to the end of the window: the tail where the GPU idles progressively because only stragglers remain. |
| `srv/drain_ratio` | `drain_window / gpu_busy`. 1.0 means the server was never saturated in the window — the rollout is bound by something other than the GPU (for us: docker container slots). |
| `srv/running_peak` | Largest observed running batch. |
| `srv/capacity` | The capacity used as the saturation threshold: `AGENTIC_MAX_RUNNING_REQUESTS` if set (should match `rollout.max_num_seqs`), else the observed peak. |
| `srv/token_usage_peak` | Peak KV-cache utilization (0-1). |
| `srv/samples` | Number of scrapes in the window — a sanity check that the poller was alive. |

**Latency and token breakdown**, from counter/histogram deltas between the first and last
sample in the window (`delta(sum)/delta(count)` for histogram means):

| Metric | Meaning |
|---|---|
| `srv/ttft_mean_s` | Mean time to first token — prefill plus queueing, server-side. |
| `srv/inter_token_latency_mean_s` | Mean decode-step latency. |
| `srv/e2e_latency_mean_s` | Mean end-to-end request latency (one turn, not one episode). |
| `srv/queue_time_mean_s` | Mean time requests waited before running. Nonzero and growing = over-subscription. |
| `srv/prompt_tokens`, `srv/generation_tokens`, `srv/cached_tokens` | Counter deltas over the window: prompt tokens processed, tokens decoded, prompt tokens served from the prefix cache. |
| `srv/prefix_cache_hit_frac` | `cached / prompt`. The real prefix-cache number — what `tokens/mean_cached_prompt` was supposed to be. |
| `srv/num_requests` | Requests completed in the window (one per assistant turn, so ~ `traj/mean_turns` x 2048). |
| `srv/gen_throughput_mean` | Mean decode tokens/s while busy. |
| `srv/cache_hit_rate_mean` | Mean of the server's own `cache_hit_rate` gauge while busy. |

A histogram-derived value of exactly `0.0` (as `srv/ttft_mean_s` currently shows) means the
counter did not advance between the window's first and last sample — not that latency was
zero.

---

## 13. Known dead or duplicated rows

Worth knowing before plotting anything:

| Row | Status |
|---|---|
| `traj/mean_edits` | Always 0.0 — `edit_count` is never incremented; the scaffold has no `edit` tool (the agent edits via `sed`/`python`). |
| `tokens/mean_cached_prompt` | Always 0.0 — never assigned. Use `srv/prefix_cache_hit_frac`. |
| `tokens/mean_response` vs `tokens/mean_completion` | Always identical; both are `sum(response_mask)`. |
| `traj/mean_reward` vs `traj/resolve_rate` | Always identical (binary reward). |
| `critic/rewards/*` vs `critic/score/*` | Identical while `use_kl_in_reward: false`. |
| `traj/truncation_rate` vs `traj/exit/ContextLimit` | Identical by construction. |
| `timing_s/start_profile`, `timing_s/stop_profile` | No-op hooks unless profiling is enabled. |
| `perf/time_per_step` vs `timing_s/step` | The same value under two names. |
| `pytest_output_length` | Collected per trajectory but **not** aggregated, so it never reaches the CSV. |
