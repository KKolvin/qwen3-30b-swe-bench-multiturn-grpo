# Timeline simulator: design rules

**这东西是干什么用的：** 手上有一堆某个 harness 吐出来的 trace、一份训练 recipe 和一堆
config，没有 profile。模拟器要说出时间线长什么样、一步端到端要多久。trace 的格式是那个
harness 当时高兴怎么写就怎么写的，所以模拟器**不许要求格式**——它声明一个地板（§2），
地板之上有什么吃什么，并且把自己编出来的部分明说。

回放 run `20260912-072938`、对上 `analysis/<run>/timeline/*.jsonl`，是证明它算得准的办法。
那是仪器和有效性条件，不是交付物；§6 和 §8 不是产品。

机制上它是一个 trace-driven 的离散事件模拟器，管 sync GRPO 一步里 rollout 那一侧：
episode release -> worker slot -> SGLang scheduling -> tool -> reward eval，group barrier
和训练各段当作阻塞区间。它预测逐 decode step 的延迟、TTFT、TPOT 和一步的端到端墙钟。

This file is the contract it has to be built against, because every way it can fail is
silent: a number copied out of the answer, a number fitted to this one workload, and a
number invented at the adapter and then typed as if it had been measured.

---

## 0. Scope

**In:** synchronous GRPO. Burst release of `train_batch_size * n` episodes, group
barrier, training holding the GPUs.

**Out:** off-policy orchestration, and anything about *learning*. The
simulator answers "what would the **system** do with this demand", never "what would
the **policy** do". Staleness, curriculum, sampling temperature all change the
trajectories themselves, and a replayed trace cannot represent that.

**界线是 system counterfactual 对 policy counterfactual，不是「做不做 counterfactual」。**
「有 recipe、没跑过、想知道要多久」本身就是一个 counterfactual，而它是主用途。真正的分界
写在 §7 第一行那条假设上：TP、replica 数、并发上限、KV dtype、硬件——这些不改变模型写出
来的 token，回放的 token 流仍然成立，**在内**；turn cap、工具集、prompt 模板、换模型——
这些改的是轨迹本身，回放代表不了，**在外**。模拟器必须能认出后者并拒绝，而不是默默算出
一个数（§7）。

**「回放对得上」是里程碑，不是 scope。** 第一个可交付是吃一条自家 trace，回放出来对上
`analysis/<run>/timeline/*.jsonl`；先证明它算得准，再谈对陌生 trace 报数。IR 里那些引擎
现在不读的字段（`Task.duration_is_cap`、`Request.max_new_tokens`）留着是因为不记就补不
回来，不是因为已经在用。

**Workload-neutral by construction.** The simulator must not be biased toward the
SWE-bench agentic workload it is calibrated and validated on. A workload reaches it
only as a trace; the same simulator must take a synthetic open-loop serving trace and
be right about that too. §5 is the mechanism that enforces this; §2 is the check.

---



## 1. The rule

> **凡是能在 timeline 里观测到的量，一律是simulator的output，不是input。Feeding it in makes the validation vacuous.**

This decides nearly every question about what the simulator is allowed to be told.

按主用途，这条规则的理由比「validation 会变空」硬得多：**预测的时候根本就没有 timeline，
S 是物理上拿不到的东西。** 所以它不是一条自律，是可行性约束——一个需要 S 才跑得动的模拟
器，在它真正该用的场合跑不起来。「validation 变空」只是同一件事在有 timeline 时的表现。

The worked example: **prefix hit rate is not an input.** What the trace supplies is
prefix *structure* — which earlier request each request extends, and by how much. The
hit *rate* is derived by the simulator from the radix cache, its LRU eviction, the
page size and the routing. The trace's own `cached_tokens` is then a validation
target, and a sharp one:

```
run 20260912-072938, 95,403 adjacent turn pairs
  prompt_{k+1} < prompt_k + completion_k          0 cases  -> pure append holds exactly
  cached_{k+1} - (prompt_k + completion_k)        p50 -113, p90 -69, 100% negative
```

That deficit is `page_size=64` alignment plus the page SGLang holds back. A simulator
that skips paging predicts a deficit of 0 and is caught immediately. That is what a
derived quantity is for.

The same rule disqualifies as inputs: batch composition, prefill/decode interleaving,  
queue depth, TTFT, TPOT, token usage, retraction counts, arrival times of any turn  
after the first, and the ramp/plateau/drain shape.

---



## 2. Labels

Every piece of information the simulator touches carries exactly one. 字母回答的问题是
**「当用户手上只有 trace 和 config 的时候，这个数从哪来？」**——不是「喂它进去算不算作
弊」。后者只在有 timeline 的时候才成为问题，前者每一次都是问题。


|       | 含义                                | 预测时（只有 trace + config）从哪来                                                            |
| ----- | --------------------------------- | ------------------------------------------------------------------------------------ |
| **T** | 来自 trace——workload，逐请求或逐 episode  | 看 tier，见下。地板只保证给得出 token 流                                                           |
| **C** | 从 config、server args 或 engine 源码读出来，不许猜 | **C-observed** 从活着的 server 上读；**C-resolved** 没有活 server，跑 engine 自己的默认值解析代码解出来。两者都不算猜，但只有前者被验证过 |
| **K** | 来自正交标定网格，是**唯一**需要拟合的东西           | **K-measured** 本机标定；**K-transferred** 查标定库里最近的网格点，且必须打印离它有多远                          |
| **S** | simulator 自己推导出来的                  | 拿不到，所以只能是输出。手上有自家 timeline 时它兼作验证靶子                                                   |
| **G** | 缺口：**我们自己的 run** 里可以补的埋点          | 外来 trace 补不了，那种情况归 I                                                                  |
| **I** | 这条 trace 里没有，只能估                   | 必须带着它是从哪个分布抽的、以及那个分布是哪来的                                                             |
| **A** | 假设（列在 §7，带风险等级和证伪办法）              |                                                                                       |


**I 不是补丁，是主路径。** 大多数 harness 的 trace 里，大多数 duration 都会是 I。所以：

- 有 I 参与的时候，e2e 的输出**必须是区间，不是点**；
- 必须出一张**敏感度表：哪个 I 把区间撑开了**。这张表本身就是交给用户的东西——「你这一步
  是 1.4–2.2h，带宽主要来自不知道 tool duration，把它记下来，带宽塌一半」；
- I 可以是抽出来的，但 seed 必须固定（§6 的确定性要求）。


### 2.1 最低 trace 契约

「不能指望 trace 有格式」不能停在这句话上，它得落成一个地板加一个降级阶梯。labels 是相对
tier 定义的：


| tier   | 典型来源                                | 拿得到                                                                                  | 于是                                                                      |
| ------ | ----------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| **T0** | 只有对话本身：`.traj`、消息 jsonl、任何 harness 的日志 | 配上 tokenizer 和 prompt 模板，可得 `prompt_tokens`、`completion_tokens`、prefix 结构、turn 数、exit reason | **所有 duration 都是 I**                                                    |
| **T1** | + 每条事件的时间戳                          | duration 靠差分                                                                          | 差分出来的 duration 是**被污染的**，见 check 1                                      |
| **T2** | + API 的 usage 字段                     | token 数精确                                                                             | `cached_tokens` 是 **S**，adapter 必须把它拦进 `observed`，哪怕它就躺在 `input_tokens` 旁边 |
| **T3** | 我们自己的 timeline                      | 全有，含 S 靶子                                                                             | 只用来标定和对拍                                                                |


地板是 T0，而这件事的结论是好消息：几乎所有 harness 都给得出 token 流，而 token 流正是
cost model 最吃的输入（§5 里 KV bytes 跟着 `sum_seqlen` 走）。缺的是非 GPU 的 duration，
而按 [[tool-time-is-timeouts-not-work]]，tool 只占一个 episode 的 3.1%。所以 T0 加一份
config 就足以把占大头的解码部分从第一性原理算出来，其余用 I 带区间——降级是优雅的。

**例外是训练段，而且这个例外很大。** §3 L7 把 training phase 当不透明阻塞段回放，没有
profile 的时候它是 I：按 §9 的数字，`old_log_prob` + `ref` + `adv` + `update_actor` +
`update_weights` 占一步的 52%。一条 T0 trace 的 e2e 区间会被这**一个** I 支配，除非从
recipe 侧另建一个模型。这是目前最大的已知缺口，不能顺手带过。


### 2.2 四条机械检查

前两条是全部的 anti-overfit 纪律，第三条是 workload 中立性，第四条管住 I。

**1. No S on the input side.** 分界线一句话：**T 是「一段有多长」，S 是「它什么时刻发生」。**

| T：时长和结构                                 | S：时刻和结果                        |
| --------------------------------------- | ------------------------------ |
| 容器启动要多久、工具跑多久、grading 跑多久               | 每个 turn 什么时候到达服务器（第一轮也算）       |
| 每轮的 `prompt_tokens` / `completion_tokens`  | 排队等多久、TTFT、TPOT、batch 里当时有谁    |
| 接的是哪条请求、跟它共享多少 token（结构上**能**省多少）        | `cached_tokens`：实际从 cache 里省下多少 |

左边这些，换 TP、换 KV dtype、改并发上限都不会变，所以可以照搬进来。右边这些恰恰是这些改动会
撬动的东西——喂进去就等于把答案抄进了题面，模拟器再也不会犯错，也再也不会告诉你任何事。

唯一的例外是 `Session.release_offset_s`（每个 session 相对 t=0 的释放偏移），sync GRPO 下
全是 0.0。注意它的来源是 **recipe（C）**，不是 trace：外来 trace 自己的到达节奏是**那套系统
当时的行为**，属于 S，绝对不能照抄进来。除此之外 trace 里的时间戳一个都不进来。

**差分出来的 duration 不是 T。** T1 是最常见的 tier，而在 T1 里唯一拿到时长的办法是差两个
时刻：`tool_dur ≈ t(下一个 request) − t(上一条 response)`。这个差里含排队、client overhead、
harness 自己的 think time，以及采集那一次运行的 contention。把它当 T 喂进去，等于把「系统
当时等了多久」缝进了「工作有多久」——**模拟器会在 e2e 上显得很准，因为答案是抄来的**。这
正是本条规则要防的事，只是换了个入口进来。所以差分 duration 要么标成 I（当作上界，去反卷），
要么先证明采集环境没有饱和。当场可做的证伪：把 config 里的 replica 数翻倍，看预测的 e2e 动
不动。不动，就是被污染了。

**2. No K fitted on a run trace.** K comes from the engine, on a designed grid.

**3. Workload neutrality 是规格，不是卫生。** 进来的 trace 本来就来自未知的 harness，所以
中立性不是「小心别被 SWE-bench 带偏」的自律，它就是产品要求本身。**system layer 不许出现**
`episode`**、**`turn`**、**`tool`**、**`swebench` 这几个词——这条仍然有效，但它只管住了下游。
上游那条更重要：**harness 知识只允许存在于 adapter 里**。adapter 探测 tier、认领格式、逐
字段声明来源；此外任何地方都不许知道 trace 长什么样。workload layer 和 system layer 之间的
接口仍然是 request stream——arrival 事件带 `(prompt_tokens, prefix_parent, shared_len,
completion_tokens)`。闭环反馈（「第 k+1 轮在第 k 轮工具跑完之后到达」）属于 workload layer，
只以一个到达时刻的形式进入 system layer。system layer 如果分不出输入是谁产生的，它就没法
被其中任何一个带偏。

**4. 没有 I 冒充 T。** label 是 **(量, trace)** 的属性，不是量的属性：`prompt_tokens` 对
T2 trace 是 T，对 T0 trace 是「T-derived，要 tokenizer 加完全一致的模板」，对只记了轮数的
trace 是 I。同一个字段三个 label，所以 label 必须跟着数据走、写进 IR，而不是只写在这份文档
的表格里。两条附带规则：**引擎不许读 provenance**（跟 `Task.kind` 同一条规矩），**diff
harness 拒绝给「上游输入含 I」的量打分**——否则是在验证自己的先验。

这些落成代码在 `[src/simulator/ir.py](src/simulator/ir.py)`：节点类型叫 `Session` 而不是
`Episode`——system layer 确实需要它，作为 sticky routing 的 key，但不许知道是什么把它们关联
起来的。观测量单独放在 `[src/simulator/observed.py](src/simulator/observed.py)`，把「no S on
the input side」变成一条 import 规则，由
`[tests/test_simulator_ir.py](tests/test_simulator_ir.py)` 强制。

**IR 现在还表达不了「不知道」**，这是 §8 第 1 步要补的：`Task.duration_s` 是必填 float，
`Request.stop` 必须在 `eos`/`length` 里二选一（T0 trace 常常分不出来），`shared_prefix_tokens`
需要 tokenizer 加完全一致的模板，而 `Workload.source` 是单个字符串，撑不住「一堆来源混在
一起」的输入。一条 T0 trace 现在能构造出**语法上合法、语义上全是编的** workload，没有任何
字段说得出这件事。

---



## 3. What each layer needs



### L1 Workload


| Information                                                                |       | 备注                                                                                             |
| -------------------------------------------------------------------------- | ----- | ---------------------------------------------------------------------------------------------- |
| episode id, group id (`n=8`), instance                                     | T     |                                                                                                |
| per-turn `completion_tokens`                                               | T     |                                                                                                |
| per-turn `prompt_tokens` | T | 到这一轮为止整段对话的长度（含之前所有轮的 prompt、模型输出和工具返回），存绝对值不存增量。T0 下要靠 tokenizer + 完全一致的模板重建 |
| `prefix_parent`（接的是哪一条请求）                                                  | T     | 上一轮那条请求的 id                                                                                    |
| `shared_prefix_tokens`                                                     | T     | 本请求开头有多少 token 与 `prefix_parent` 的完整序列相同。这是**能省的上限**，不是实际命中数                                   |
| observation tokens                                                         | T     | 工具吐回来的token（context 涨了多少就是由`completion` + `observation` 决定）                                    |
| tool duration | T / **I** | 这次工具调用实际花了多久，模拟器直接回放这个数。T3 是直接量的；T1 只能差分，**被污染**（§2 check 1）；T0 全靠 I |
| tool kind（trace 记 `bash` / `edit` / `submit`；进 IR 后一律是 `Task.kind="tool"`） | T     | 只是出报表用的标签。引擎**不许拿它搞 if else 分支**，否则就被我们的 workload 带偏了                                          |
| timed-out flag（IR: `Task.duration_is_cap`）                                 | T     | （等以后做改 timeout 的 counterfactual，得靠它认出哪些条回放不了。现在不做，见 §0）                                        |
| reward eval duration | T / **I** | `score` p50 21.6s，p90 30.4s。大多数 harness 的 trace 里没有这一段，那时候是 I |
| exit reason, turn count                                                    | T     |                                                                                                |
| sampling params, `max_new_tokens`                                          | C     |                                                                                                |
| prefix hit rate / `cached_tokens`                                          | **S** |                                                                                                |
| arrival time of every turn (except the frist turn)                         | **S** | 「turn k+1 什么时候到」= turn k 什么时候decode完 + tool duration。tool duration 是白给的信息，什么时候decode 完需要sim去算。 |




### L2 Router


| Information                               |       | 备注                                                                                                                             |
| ----------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| sticky by `request_id`, else least-loaded | C     | verl `GlobalRequestLoadBalancer`，见 `src/agentic_grpo/agent_loop.py:322`；如果这个id之前来过就派回同一台server，没见过的 id 就在 replica 里挑 load 小的那台 |
| replica count, TP size                    | C     |                                                                                                                                |
| which replica each request landed on | **G** | 自家 run 里最大的缺口。没有它，§6 的 L1 逐请求对拍做不了，成本是 `generate` 事件加一个字段。外来 trace 补不了——那边它本来就是 S，由模拟器自己的路由推出来 |




### L3 Scheduler, per replica


| Information                                                      |       | 备注                                                                   |
| ---------------------------------------------------------------- | ----- | -------------------------------------------------------------------- |
| `schedule_policy` (`fcfs`), `schedule_conservativeness` (1.0)    | C     | `sglang/srt/server_args.py`                                          |
| `chunked_prefill_size`, `max_running_requests`, `page_size` (64) | C     | 从跑着的 server 上读，别照默认值抄——默认值是按显存分支的                                    |
| `new_token_ratio` init / decay / floor                           | C     | 驱动退避和 retraction                                                     |
| radix cache + LRU eviction                                       | C     | `sglang/srt/managers/schedule_policy.py`                             |
| event loop: `overlap` or `normal`                                | **G** | 决定 CPU 调度那部分算不算暴露在 step time 里。`scheduler.py:1068` vs `:1095`，查一次就知道 |
| batch composition per iteration                                  | **S** | `get_next_batch_to_run`，`scheduler.py:1805`                          |
| prefill/decode interleaving                                      | **S** | `get_new_batch_prefill`，`scheduler.py:1895`                          |
| queueing delay, TTFT, TPOT                                       | **S** | 主要预测目标                                                               |
| retraction / preemption events                                   | **S** | 现有 trace 里全是 0，照样要建，理由见 §4                                           |




### L4 Cost model


| Information                                |       | 备注                            |
| ------------------------------------------ | ----- | ----------------------------- |
| layers, KV heads, head dim, active experts | C     | 读模型的 `config.json`，不要写死       |
| dtype widths; KV placement under TP        | C     | 包括 `num_kv_heads < tp` 时的复制分支 |
| HBM bandwidth                              | C     | 硬件规格                          |
| `cuda_graph_max_bs`, capture buckets       | C     | step time 的阶跃点在这儿             |
| **eta(B, ctx, prefill_len)**               | **K** | 全模拟器唯一拟合出来的量                  |
| per-step fixed overhead                    | **K** |                               |
| prefill cost curve                         | **K** |                               |




### L5 KV pool


| Information                                            |       | 备注                                           |
| ------------------------------------------------------ | ----- | -------------------------------------------- |
| `max_total_num_tokens`                                 | C     | TP=4 下每 replica 4,560,512；gauge 在 busy 时一直有值 |
| page size, allocation granularity, eviction thresholds | C     |                                              |
| token usage over time, evictions, retraction           | **S** | 实测 p50 0.38，p99 0.47                         |




### L6 Non-GPU resources


| Information                                                   |       | 备注                                          |
| ------------------------------------------------------------- | ----- | ------------------------------------------- |
| worker count, per-worker container cap                        | C     |                                             |
| eval container pool (per-worker semaphore, 4)                 | C     | 这就是 `eval_wait` p90 **248s**、max 755s 的直接原因 |
| container start latency | T / **I** | p50 3.3s。T3 是直接量的，其余 tier 基本都是 I |
| tool exec latency | T / **I** | 同上 |
| whether tool latency degrades at higher container concurrency | **A** | 中等风险                                        |
| CPU / network contention                                      | **A** | 假设没有                                        |
| `eval_wait`, slot queueing                                    | **S** |                                             |




### L7 Orchestration (sync GRPO)


| Information                      |       | 备注                         |
| -------------------------------- | ----- | -------------------------- |
| batch 256 x n=8, group barrier   | C     |                            |
| training phase durations | T / **I** | 当作不透明的阻塞段回放，模拟器不建 FSDP 的模型。没有 profile 时它是 I，而按 §9 的数字它占一步的 52%——一条 T0 trace 的 e2e 区间会被它一个撑开（§2.1） |
| weight-sync drain semantics      | C + T |                            |
| ramp / plateau / drain structure | **S** | L2 级对拍的主靶子                 |




### L8 Client overhead


| Information                        |       | 备注                                      |
| ---------------------------------- | ----- | --------------------------------------- |
| client -> server, server -> client | **A** | 当常数。实测 p50 4.3ms 和 1.9ms，加起来占一个请求的 0.2% |


---



## 4. Mechanisms that must be built even though this workload never triggers them

Every one of these is tempting to skip, because in the SWE-bench trace it is provably
negligible. Skipping any of them welds this workload into the simulator's structure,
where no code review will see it.


| Tempting simplification               | 在这条 trace 上的理由                                   | 换哪种 workload 会崩                  |
| ------------------------------------- | ------------------------------------------------ | -------------------------------- |
| no waiting queue                      | `num_queue_reqs` 恒为 0                            | 任何开环到达的负载——那里排队**就是** latency 本身 |
| prefill is negligible                 | cache hit 96.2%，真正 prefill 的只有 ~450 token        | 无前缀复用的负载：prefill 变主导，抢占节奏整个反过来   |
| no KV pool limit, no retraction       | `num_retracted_reqs` 恒为 0，`token_usage` p50 0.38 | 长 context 高并发——吞吐是断崖，不是斜坡        |
| decode dominates, be sloppy elsewhere | decode 占一个 episode 的 94.8%                       | think-time 大的负载                  |


Their never firing on this trace is itself one of the correctness checks.

---



## 5. Where overfitting can actually enter

It is worth being precise about this, because it concentrates the discipline. 而按 §2
check 3，中立性是**规格**：真实用途下进来的 trace 来自未知的 harness，标定一旦沾上某一种
workload 的形状，产品就是错的，不只是纪律松了。

- The **scheduler layer is ported, not fitted.** `get_next_batch_to_run`,
`get_new_batch_prefill`, `update_running_batch`, radix LRU — deterministic source
with zero free parameters. Transcribe it and it leaks nothing.
- The **resource layers are semaphore semantics.** Nothing to fit.
- The **cost model is the only thing learned from data.** So the entire
workload-neutrality question reduces to: is the cost model a property of the engine
and the hardware, or of this workload?

Two rules keep it the former.

**Calibrate on a designed grid, never on a run trace.** In run 20260912-072938,
`corr(B, ctx/req) = -0.845` — batch size and context length are almost perfectly
anti-correlated, because the drain tail is populated by exactly the longest episodes
([[tail-cohort-length-bias]]). The trace covers a one-dimensional diagonal of a
two-dimensional space:

```
B   1- 16   ctx p50 51784      <- drain tail
B  16- 48   ctx p50 28903
B  48- 80   ctx p50 21255
B  80-112   ctx p50 18671
B 112-140   ctx p50 15999      <- plateau
```

A cost model fitted here cannot separate "cost grows with B" from "cost grows with
context", and any workload off that diagonal is extrapolation. The grid must be
orthogonal in `B x ctx x prefill_len` and must span the engine's capability, not this
trace's footprint.

**Fit efficiency, not time.**

```
T_step = (W_bytes + KV_bytes(sum_seqlen)) / (BW * eta(B, ctx)) + overhead
```

Fit `eta`. `T` has workload-dependent scale and extrapolates to nonsense; `eta` is
dimensionless, bounded, smooth, and is a property of engine + hardware. It also turns
the standing 3.7x standalone-vs-in-run mystery ([[moe-kernel-not-the-bottleneck]],
[[decode-roofline-17pct-of-hbm]]) into a printable quantity: at the p50 steady state,
each replica reads ~43GB of KV per rank per step and takes 44ms, so `eta ~ 0.12`
against an 8TB/s roofline, versus ~0.5 standalone. Same grid point, two numbers,
difference is the gap.

**Proving neutrality requires held-out workloads**, and they are cheap — synthetic
load against a standalone server, recorded in the same IR: open-loop Poisson with
short prompts and no reuse (forces the queue); long prompt / 1 output token (forces
prefill); short context at high batch (the corner the diagonal never visits); very
long context to the pool ceiling (forces retraction). Plus the agentic axes already
in the repo: context 28672 vs 61440, bash-only vs edit tool. Calibrate once, then
predict all of them without touching a parameter.

---



## 6. Diffing against the real timeline

这三级是**有效性条件**，不是交付物。它们证明模拟器算得准，然后模拟器才配去对一条没有
timeline 的 trace 报数（§0）。把对拍本身当成产品，就是把仪器当成了交付物。

Three levels, each releasing one more feedback loop. A mismatch at the outermost
level alone is not diagnosable, which is why the inner two exist.

**L0 — cost model only, no feedback.** Input: the real `(B, sum_seqlen)` series from
the 1Hz gauge. Output: predicted step time, diffed against `B / gen_throughput`.
Needs no replica attribution; runnable on existing data today.

**L1 — scheduler; server closed-loop, client open-loop.** Input: the real
`request_received` times and each request's token counts. The simulator forms its own
batches and decides completion. Diff **paired, per request**: TTFT and `decode_s`
across 99,646 samples. Needs the replica gap (§3 L2) closed.

**L2 — full closed loop.** Only episode release is given. Diff the emergent
structure: ramp 365s / plateau 1238s / drain 463s, episodes per minute, gen span,
step wall clock, the 1Hz concurrency curve.

**Error-accumulation probe.** In L1 and L2, add a switch that re-aligns to the real
timestamps every `k` turns, and plot error against `k`. Accurate at `k=1` but not at
`k=inf` means the feedback loop is amplifying a small bias; inaccurate already at
`k=1` means the single-step model is wrong.

The simulator must be **deterministic** — every source of randomness comes from the
trace, or from a fixed seed on the I 分布（§2）。预测时输出的是一个区间，但对给定的 seed
必须逐位可复现。否则 diff 分不清模型误差和抽样噪声。

**Tolerances come from the noise floor, not from taste.** The run has two independent
SGLang replicas carrying statistically identical load, which is a free estimate of
irreducible variance (2,862 paired samples where both are busy):

```
num_running_reqs   mean relative difference   4.0%   (corr 0.979)
gen_throughput     mean relative difference  16.8%
step time, median  44.5ms vs 43.3ms        -> 2.7%
```

So a per-point agreement better than ~17% on a 1Hz gauge is meaningless, and ~3% is
the floor on medians. Set acceptance there.

Per-step latency itself has **no ground truth in any archived trace** — the finest
granularity recorded is per request. `disable_log_stats` is already false, so the
SGLang scheduler prints batch composition and throughput every `decode_log_interval`
steps, but it goes to the Ray actor's log and run 20260912-072938's Ray session is
gone. Archiving those lines into the timeline directory is nearly free and is the
only route to validating a per-step prediction.

---



## 7. Assumptions


| Assumption                                              | 风险               | 怎么证伪                                  |
| ------------------------------------------------------- | ---------------- | ------------------------------------- |
| token counts are invariant under system counterfactuals | n/a | 这是 §0 的 scope 边界，不是假设 |
| trace 与 config 同源：这份 token 流就是在这份 config 下产生的 | **高** | 比对 turn cap、模型名、工具集、prompt 模板指纹。对不上就拒绝报数，别默默算——那是 policy counterfactual，在 §0 之外 |
| 差分出来的 duration 可以当工作时长用（T1） | **高** | 采集环境若是饱和的，这个差里含排队；拿同一个 harness 在空载下重跑的时长分布比一次。当场可做的：把 replica 数翻倍，看预测的 e2e 动不动 |
| T0 下用 tokenizer + 模板重建出来的 token 数，等于 harness 当时实际发出的 | 中 | 手上有 T2 trace 时，拿它的 usage 字段对一次 |
| tool duration is independent of container concurrency   | **中**            | 比较 cap 16 和 cap 40 两个 run 的 tool 时长分布 |
| prompts are pure appends of the previous turn           | **已验证** 0/95,403 | 做过了                                   |
| no CPU or network contention between workers            | 低                |                                       |
| client<->server overhead is constant                    | 低                | 已量，占一个请求的 0.2%                        |
| the two replicas are exchangeable                       | 低                | 已量：batch 差 4.0%，中位 step time 差 2.7%   |
| no failures or retries                                  | 低                |                                       |


---



## 8. Build order

Each step has an independent deliverable. workload 的 trace 在第 1 步就出现，但只作为
**格式样本**；它第一次参与数值是第 3 步，作为验证集。它永远不碰参数。

1. **Adapter 契约。** 陌生 trace 进来，探测 tier（§2.1），逐字段声明来源，输出 IR +
  provenance。这是产品面，所以它排第一而不是最后。可交付：一条只有对话的 T0 trace 能构造
  出合法的 workload，并且每个编出来的字段都带着标记。同时补上 IR 现在缺的「不知道」的表达
  和 per-session 的 `source`。
2. **Orthogonal calibration grid** on a standalone server. Yields `eta`, and as a
  by-product turns the 3.7x gap into a residual table over `(B, ctx)`.
3. **L0 diff.** Accept or reject the cost model on its own.
4. **Close the gaps:** replica id on the `generate` event; confirm the event-loop
  variant; archive the scheduler's decode-batch log lines.
5. **Discrete-event engine**, four resource pools, ported scheduler — including the
  four mechanisms of §4 that this workload never exercises.
6. **Synthetic held-out workloads.** Demonstrate neutrality.
7. **L1 and L2 diffs** against the agentic trace. 到这里有效性条件满足（§6）。
8. **I 的敏感度表和带区间的 e2e。** 这一步才是交付物：吃一条 T0/T1 的陌生 trace 加一份
  config，出一个带区间的 e2e，和一张「哪个 I 撑开了区间、去补哪个埋点最划算」。

Where it stands: step 0, the schema, is in — `src/simulator/ir.py`,
`src/simulator/observed.py`, `tests/test_simulator_ir.py`. The converter is not written
yet, and 它不再是一个文件：`trace_to_ir.py` 是照着我们自己的 trace 起的名字，第 1 步把它
换成一个契约下的多个 adapter。

---



## 9. Provenance

Every measurement quoted above is from run `20260912-072938` (Qwen3-30B-A3B,
TP=4 x 2 replicas, 8x B200), computed from
`/data0/shared/kewen.liu/agentic-timelines/20260912-072938/`:
`timeline-13267*.jsonl` (99,646 `generate` events, 95,403 adjacent turn pairs) and
`srv-metrics-1312837.jsonl` (7,193 samples with a running batch). Step budget from
the `cat="train"` spans: step 5072s / 4377s, `gen` 2418s / 1824s, `update_actor`
1796s / 1718s, `update_weights` 59s / 60s.

`eta ~ 0.12` is derived here, not measured directly: 1.84M tokens of KV per replica
at 96KB/token over 4 ranks is ~43GB per rank per step, or 5.4ms at 8TB/s, against a
measured 44.5ms.

One loose end closed while writing this: the `is_cuda_graph` gauge reads 0 in all
7,193 samples, but `stats.is_cuda_graph` is never assigned anywhere in SGLang 0.5.8 —
it is a dead gauge, not evidence. CUDA graphs are on, and the 8x gap is not there.