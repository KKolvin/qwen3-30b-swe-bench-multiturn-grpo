# Timeline simulator: design rules

**这东西是干什么用的：** 手上有一堆某个 harness 吐出来的 trace、一份训练 recipe 和一堆
config，没有 profile。simulator 要说出时间线长什么样、一步端到端要多久。trace 的格式是那个
harness 当时高兴怎么写就怎么写的，所以 simulator **不要求格式**：它规定一条 floor（§2.3），
trace 里有什么就用什么，并且明着标出哪些数是它自己编的。

回放 run `20260912-072938`、对上 `analysis/<run>/timeline/*.jsonl`，是用来证明它算得准的，
不是它要交付的结果。§6 和 §8 是检验手段，不是产品。

机制上它是一个 trace-driven 的 discrete-event simulator，管 sync GRPO 一步里 rollout 那一侧：
episode release -> worker slot -> SGLang scheduling -> tool -> reward eval，group barrier
和训练各段当作 blocking span。它预测每个 decode step 的延迟、TTFT、TPOT，和一步的端到端
wall clock。

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

本 simulator 的主要应用场景是「有 recipe、没跑过、想知道要多久」，这本身就是一个
counterfactual，**界线在 system 和 policy 之间**，就是 §7 的第一行：

- 改 TP、replica 数、concurrency cap、KV dtype、硬件——模型写出来的 token 不变，回放的
  token 流照样成立，**这种 simulator 能回答**；
- 改 turn cap、工具集、prompt 模板、换模型——trajectory 本身就变了，回放代表不了，**这种
  simulator 必须认出来并且拒绝**，不能默默给个数（§7）。

**「回放对得上」是里程碑，不是 scope。** 第一个可交付是吃一条自家 trace，回放出来对上
`analysis/<run>/timeline/*.jsonl`。先在这里证明它算得准，才有资格对一条没人 profile 过的
trace 报数。IR 里有几个字段 engine 现在不读（`Task.duration_is_cap`、`Request.max_new_tokens`），
留着是因为当时不记、后面就补不回来了，不是因为已经在用。

**Workload-neutral by construction.** The simulator must not be biased toward the
SWE-bench agentic workload it is calibrated and validated on. A workload reaches it
only as a trace; the same simulator must take a synthetic open-loop serving trace and
be right about that too. §5 is the mechanism that enforces this; §2 is the check.

---



## 1. The rule

> **凡是能在 timeline 里观测到的量，一律是 simulator 的 output，不是 input。把要预测的那个量
> 当成 input 喂进去，就是 target leakage：validation 必然通过，所以什么都没验证到。**

This decides nearly every question about what the simulator is allowed to be told.

放到主用途上，这条规则还有个更硬的理由：**预测的时候根本没有 timeline，S 你想喂也喂不到。**
所以它不是一条 discipline，是 feasibility constraint——要 S 才跑得动的 simulator，在它真正该
用的场合根本跑不起来。手上碰巧有 timeline 的时候，这条规则表现为 target leakage；没有的
时候，它表现为「这个 simulator 根本用不了」。

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



### 2.1 IR：simulator 唯一认得的 input

IR = intermediate representation，编译器那个意思。**adapter 是 frontend，system layer 是
backend，IR 是中间那层语言**：任何 harness 的 trace 先被 adapter 翻成 IR，simulator 只认
IR，认不出它原来是什么格式。IR 就是 `[src/simulator/ir.py](src/simulator/ir.py)` 里那五个
class：


| class      | 是什么                      | 关键字段                                                                               |
| ---------- | ------------------------ | ---------------------------------------------------------------------------------- |
| `Workload` | 一整份 workload             | `sessions`、`source`（哪来的）                                                           |
| `Session`  | 一串相关的 node               | `requests`、`tasks`、`leases`、`group`、`release_offset_s`                             |
| `Request`  | 一次 inference 调用          | `prompt_tokens`、`completion_tokens`、`prefix_parent`、`shared_prefix_tokens`、`after` |
| `Task`     | 非 inference 的活：工具、起容器、打分 | `duration_s`、`pool`、`after`                                                        |
| `Lease`    | 跨多个 node 一直占着的资源 slot    | `pool`、`acquire_before`、`release_after`                                            |


一个 Session 就是一张 DAG：node 是 Request 或 Task，edge 是 `after`。一个 SWE-bench
episode 和一条 open-loop serving load 都得用这同一套 class 表达出来——那两条就是
`tests/test_simulator_ir.py` 开头的两个 test。

**IR 同时是一张 whitelist：能在 IR 里写出来的，才是合法 input。** 所以 IR 里没有 timestamp、
没有 `cached_tokens`、没有 TTFT、没有到达时刻——S 那一类根本没有字段可填，全被赶到
`[src/simulator/observed.py](src/simulator/observed.py)` 去了。§1 那条规则不靠自觉守，靠
IR 里根本没有这些字段守。

IR 也**不装 config**：`Lease` 只说要哪个 pool，不说那个 pool 有多大。IR 只装 workload，
C 那一类走另一条路进来。

### 2.2 labels

simulator 碰到的每个量，都恰好带一个 label。label 回答的问题是**「手上只有 trace 和
config，这个数从哪来？」**，不是「喂它进去算不算作弊」。后一个问题只在有 timeline 的时候
才出现，前一个问题每次都出现。


|       | 含义                                              | 预测时（只有 trace + config）从哪来                                                                          |
| ----- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **T** | 来自 trace——workload，逐 request 或逐 episode         | 看 tier，见下。floor 只保证给得出 token 流                                                                     |
| **C** | 从 config、server args 或 engine 源码读出来，不许猜         | **C-observed** 从活着的 server 上读；**C-resolved** 没有活 server，跑 engine 自己的默认值解析代码解出来。两者都不算猜，但只有前者被验证过    |
| **K** | 来自 orthogonal calibration grid，是**唯一**需要 fit 的量 | **K-measured** 本机 calibrate 出来的；**K-transferred** 查 calibration library 里最近的 grid point，且必须打印离它有多远 |
| **S** | simulator 自己推导出来的                               | 拿不到，所以只能是 output。手上有自家 timeline 时，它同时是 validation target                                           |
| **G** | gap：**我们自己的 run** 里可以补的 instrumentation         | 外来 trace 补不了，那种情况归 I                                                                               |
| **I** | 这条 trace 里没有，只能估                                | 必须说清楚从哪个 distribution 抽的，那个 distribution 又是哪来的                                                     |
| **A** | 假设（列在 §7，带风险等级和证伪办法）                            |                                                                                                    |


**I 不是补丁，是主路径。** 大多数 harness 的 trace 里，大多数 duration 都是 I。三个后果：

- 只要有 I 参与，e2e **必须报 interval，不能报一个点**；
- 必须出一张 **sensitivity table：是哪个 I 把 interval 撑开的**。这张表才是真正交给用户的
结果——「这一步 1.4–2.2h，这么宽主要是因为不知道 tool duration，去把它记下来，宽度砍一半」；
- I 可以是抽出来的，但 seed 固定（§6 的 determinism 要求）。



### 2.3 最低 trace contract

「不能指望 trace 有格式」这句话不能就这么放着。得把它变成一条 floor 加一个 degradation
ladder，labels 相对 tier 来定义。


| tier   | 典型来源                       | 拿得到                                                             | 于是                                              |
| ------ | -------------------------- | --------------------------------------------------------------- | ----------------------------------------------- |
| **T0** | 只有对话：`.traj`、消息 jsonl、任何日志 | 配 tokenizer 和 prompt 模板可重建 token 数、prefix 结构、turn 数、exit reason | **duration 全是 I**                               |
| **T1** | + 每条事件的时间戳                 | duration 靠差分                                                    | 差分出来的 duration 是**被污染的**，见 §2.4 check 1         |
| **T2** | + API 的 usage 字段           | token 数精确                                                       | `cached_tokens` 是 **S**，adapter 必须拦进 `observed` |
| **T3** | 我们自己的 timeline             | 全有，含 S 的 validation target                                      | 只用来 calibrate 和 diff                            |


floor 落在 T0，这是好消息：几乎所有 harness 都给得出 token 流，而 token 流正是 cost model
最吃的输入（§5：KV bytes 跟着 `sum_seqlen` 走）。缺的是非 GPU 那些 duration，而按
[[tool-time-is-timeouts-not-work]]，tool 只占一个 episode 的 3.1%。所以 T0 加一份 config
就够把占大头的 decode 从 first principles 算出来，剩下的用 I 带 interval。tier 越低，
interval 越宽，但不会突然算不出来。

**例外是训练段，而且这个例外很大。** §3 L7 把 training phase 当不透明的 blocking span 回放。
没有 profile 的时候它就是 I，而按 §9 的数字，`old_log_prob` + `ref` + `adv` + `update_actor`

- `update_weights` 加起来占一步的 52%。一条 T0 trace 的 e2e interval 会被这**一个** I 支配，
除非另外从 recipe 那边给它建个模型。这是目前设计里最大的 gap，不能顺手带过。



### 2.4 四条机械检查

前两条是全部的 anti-overfit discipline，第三条是 workload neutrality，第四条管住 I 不许
冒充别人。

**1. No S on the input side.** 分界线一句话：**T 是「一段有多长」，S 是「它什么时刻发生」。**


| T：duration 和结构                            | S：时刻和结果                         |
| ----------------------------------------- | ------------------------------- |
| 容器启动要多久、工具跑多久、grading 跑多久                 | 每个 turn 什么时候到 server（第一轮也算）     |
| 每轮的 `prompt_tokens` / `completion_tokens` | 排队等多久、TTFT、TPOT、batch 里当时有谁     |
| 接的是哪条请求、跟它共享多少 token（结构上**能**省多少）         | `cached_tokens`：实际从 cache 里省下多少 |


左边这些，换 TP、换 KV dtype、改并发上限都不会变，所以可以照搬进来。右边这些恰恰是这些改动会
撬动的——喂进去就等于把答案抄进了题面，simulator 再也不会犯错，也再也不会告诉你任何事。

唯一的例外是 `Session.release_offset_s`（每个 session 相对 t=0 的释放偏移），sync GRPO 下
全是 0.0。注意它的来源是 **recipe（C）**，不是 trace：外来 trace 自己的到达节奏是**那套系统
当时的行为**，属于 S，绝对不能照抄进来。除此之外 trace 里的时间戳一个都不进来。

**差分出来的 duration 不是 T。** T1 是最常见的 tier，而 T1 里拿 duration 只有一个办法：差
两个时刻，`tool_dur ≈ t(下一个 request) − t(上一条 response)`。这个差里混着排队、client
overhead、harness 自己的 think time，还有采集那一次运行的 contention。把它当 T 喂进去，等于
把「系统当时等了多久」缝进了「活儿本身有多久」——**simulator 会在 e2e 上显得很准，因为答案
是抄来的**。还是 target leakage，只是换了个入口。所以差分 duration 要么标成 I、当 upper
bound 用，要么先证明采集那次没跑满。当场就能做的检查：把 config 里的 replica 数翻倍，看预测
的 e2e 动不动。不动，就是被污染了。

**2. No K fitted on a run trace.** K comes from the engine, on a designed grid.

**3. Workload neutrality: the system layer may not contain the words**
`episode`**,** `turn`**,** `tool`**, or** `swebench`**.** The interface between the workload layer
and the system layer is a request stream — arrival events carrying
`(prompt_tokens, prefix_parent, shared_len, completion_tokens)`. Closed-loop feedback
("turn k+1 arrives after turn k's tool finishes") belongs to the workload layer and
reaches the system layer only as an arrival time. If the system layer cannot tell
which generator produced its input, it cannot be biased toward one.

neutrality 是 spec，不是 hygiene。进来的 trace 本来就来自不认识的 harness，所以这不是「小心
别被 SWE-bench 带偏」这种自觉，它就是产品要求本身。上面那条只管住了下游，上游还有一条一样
重要：**harness 知识只许待在 adapter 里**。adapter 探测 tier、认领格式、逐字段声明来源，
除它以外任何地方都不许知道 trace 长什么样。

**4. 不许 I 冒充 T。** label 属于 **(量, trace)** 这一对，不属于量本身。`prompt_tokens` 对
T2 trace 是 T，对 T0 trace 是 T-derived（要 tokenizer 加完全一致的模板），对只记了轮数的
trace 是 I——同一个字段，三个 label。所以 label 必须跟着数据走、写进 IR，不能只写在这份文档
的表格里。另外两条：**engine 不许读 provenance**（跟 `Task.kind` 同一条规矩）；**diff
harness 拒绝给「上游输入含 I」的量打分**，否则你验的是自己的 prior。

These land as code in `[src/simulator/ir.py](src/simulator/ir.py)`: the node type is
`Session`, not `Episode` — the system layer legitimately needs it, as the sticky
routing key, and must not know what correlates it. Observed quantities live apart in
`[src/simulator/observed.py](src/simulator/observed.py)`, which turns "no S on the
input side" into an import rule, enforced by
`[tests/test_simulator_ir.py](tests/test_simulator_ir.py)`.

**IR 现在还表达不了「不知道」**，这是 §8 第 1 步要补的。`Task.duration_s` 是必填 float；
`Request.stop` 只能在 `eos` 和 `length` 里二选一，而 T0 trace 常常分不出来；
`shared_prefix_tokens` 要 tokenizer 加完全一致的模板才算得出来；`Workload.source` 是单个
字符串，一堆来源混在一起就不够用了。结果是：一条 T0 trace 现在能构造出**语法合法、语义全是
编的** workload，而没有一个字段能把这件事说出来。

---



## 3. What each layer needs



### L1 Workload


| Information                                                                |           | 备注                                                                                             |
| -------------------------------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------- |
| episode id, group id (`n=8`), instance                                     | T         |                                                                                                |
| per-turn `completion_tokens`                                               | T         |                                                                                                |
| per-turn `prompt_tokens`                                                   | T         | 到这一轮为止整段对话的长度（含之前所有轮的 prompt、模型输出和工具返回），存绝对值不存增量。T0 下要靠 tokenizer + 完全一致的模板重建                  |
| `prefix_parent`（接的是哪一条请求）                                                  | T         | 上一轮那条请求的 id                                                                                    |
| `shared_prefix_tokens`                                                     | T         | 本请求开头有多少 token 与 `prefix_parent` 的完整序列相同。这是**能省的上限**，不是实际命中数                                   |
| observation tokens                                                         | T         | 工具吐回来的token（context 涨了多少就是由`completion` + `observation` 决定）                                    |
| tool duration                                                              | T / **I** | 这次工具调用实际花了多久，simulator 直接回放这个数。T3 是直接量的；T1 只能差分，**被污染**（§2 check 1）；T0 全靠 I                    |
| tool kind（trace 记 `bash` / `edit` / `submit`；进 IR 后一律是 `Task.kind="tool"`） | T         | 只是出报表用的标签。engine **不许拿它搞 if else 分支**，否则就被我们的 workload 带偏了                                     |
| timed-out flag（IR: `Task.duration_is_cap`）                                 | T         | （等以后做改 timeout 的 counterfactual，得靠它认出哪些条回放不了。现在不做，见 §0）                                        |
| reward eval duration                                                       | T / **I** | `score` p50 21.6s，p90 30.4s。大多数 harness 的 trace 里没有这一段，那时候是 I                                  |
| exit reason, turn count                                                    | T         |                                                                                                |
| sampling params, `max_new_tokens`                                          | C         |                                                                                                |
| prefix hit rate / `cached_tokens`                                          | **S**     |                                                                                                |
| arrival time of every turn (except the frist turn)                         | **S**     | 「turn k+1 什么时候到」= turn k 什么时候decode完 + tool duration。tool duration 是白给的信息，什么时候decode 完需要sim去算。 |




### L2 Router


| Information                               |       | 备注                                                                                                                             |
| ----------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| sticky by `request_id`, else least-loaded | C     | verl `GlobalRequestLoadBalancer`，见 `src/agentic_grpo/agent_loop.py:322`；如果这个id之前来过就派回同一台server，没见过的 id 就在 replica 里挑 load 小的那台 |
| replica count, TP size                    | C     |                                                                                                                                |
| which replica each request landed on      | **G** | 自家 run 里最大的 gap。没有它，§6 的 L1 per-request diff 做不了，成本是 `generate` 事件加一个字段。外来 trace 补不了——那边它本来就是 S，由 simulator 自己的 router 推出来     |




### L3 Scheduler, per replica


| Information                                                      |       | 备注                                                                   |
| ---------------------------------------------------------------- | ----- | -------------------------------------------------------------------- |
| `schedule_policy` (`fcfs`), `schedule_conservativeness` (1.0)    | C     | `sglang/srt/server_args.py`                                          |
| `chunked_prefill_size`, `max_running_requests`, `page_size` (64) | C     | 从跑着的 server 上读，别照默认值抄——默认值是按 GPU memory 分支的                          |
| `new_token_ratio` init / decay / floor                           | C     | 驱动 backoff 和 retraction                                              |
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
| `cuda_graph_max_bs`, capture buckets       | C     | step time 的台阶在这儿              |
| **eta(B, ctx, prefill_len)**               | **K** | 全 simulator 唯一 fit 出来的量       |
| per-step fixed overhead                    | **K** |                               |
| prefill cost curve                         | **K** |                               |




### L5 KV pool


| Information                                            |       | 备注                                           |
| ------------------------------------------------------ | ----- | -------------------------------------------- |
| `max_total_num_tokens`                                 | C     | TP=4 下每 replica 4,560,512；gauge 在 busy 时一直有值 |
| page size, allocation granularity, eviction thresholds | C     |                                              |
| token usage over time, evictions, retraction           | **S** | 实测 p50 0.38，p99 0.47                         |




### L6 Non-GPU resources


| Information                                                   |           | 备注                                          |
| ------------------------------------------------------------- | --------- | ------------------------------------------- |
| worker count, per-worker container cap                        | C         |                                             |
| eval container pool (per-worker semaphore, 4)                 | C         | 这就是 `eval_wait` p90 **248s**、max 755s 的直接原因 |
| container start latency                                       | T / **I** | p50 3.3s。T3 是直接量的，其余 tier 基本都是 I            |
| tool exec latency                                             | T / **I** | 同上                                          |
| whether tool latency degrades at higher container concurrency | **A**     | 中等风险                                        |
| CPU / network contention                                      | **A**     | 假设没有                                        |
| `eval_wait`, slot queueing                                    | **S**     |                                             |




### L7 Orchestration (sync GRPO)


| Information                      |           | 备注                                                                                                                          |
| -------------------------------- | --------- | --------------------------------------------------------------------------------------------------------------------------- |
| batch 256 x n=8, group barrier   | C         |                                                                                                                             |
| training phase durations         | T / **I** | 当作不透明的 blocking span 回放，simulator 不建 FSDP 的模型。没有 profile 时它是 I，而按 §9 的数字它占一步的 52%——一条 T0 trace 的 e2e interval 会被它一个撑开（§2.3） |
| weight-sync drain semantics      | C + T     |                                                                                                                             |
| ramp / plateau / drain structure | **S**     | L2 diff 的主 target                                                                                                           |




### L8 Client overhead


| Information                        |       | 备注                                      |
| ---------------------------------- | ----- | --------------------------------------- |
| client -> server, server -> client | **A** | 当常数。实测 p50 4.3ms 和 1.9ms，加起来占一个请求的 0.2% |


---



## 4. Mechanisms that must be built even though this workload never triggers them

Every one of these is tempting to skip, because in the SWE-bench trace it is provably
negligible. Skipping any of them welds this workload into the simulator's structure,
where no code review will see it.


| Tempting simplification               | 在这条 trace 上的理由                                   | 换哪种 workload 会崩                                            |
| ------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------- |
| no waiting queue                      | `num_queue_reqs` 恒为 0                            | 任何 open-loop 到达的 workload——那里排队**就是** latency 本身           |
| prefill is negligible                 | cache hit 96.2%，真正 prefill 的只有 ~450 token        | 没有 prefix reuse 的 workload：prefill 变主导，preemption 的节奏整个反过来 |
| no KV pool limit, no retraction       | `num_retracted_reqs` 恒为 0，`token_usage` p50 0.38 | 长 context 高并发——throughput 是悬崖，不是斜坡                         |
| decode dominates, be sloppy elsewhere | decode 占一个 episode 的 94.8%                       | think-time 大的 workload                                     |


Their never firing on this trace is itself one of the correctness checks.

---



## 5. Where overfitting can actually enter

It is worth being precise about this, because it concentrates the discipline.

按 §2 check 3，neutrality 是 spec，不是 code review 上的洁癖。真实用途下进来的 trace 来自
不认识的 harness，所以 calibration 一旦把某一种 workload 的形状吃进去，simulator 给别的
workload 报出来的 e2e 就是错的，而且用户看不出它错在哪。下面那条对角线就是它怎么错的。

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

这三级 diff 是用来检验 simulator 算得准不准的，本身不是产品。先在这里过关，才有资格去对
一条没有 timeline 的 trace 报数（§0）。

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
trace, or from a fixed seed on the I distribution（§2）。预测时输出的是一个 interval，但对
给定的 seed 必须逐位可复现。否则 diff 分不清模型误差和抽样噪声。

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


| Assumption                                              | 风险               | 怎么证伪                                                                                   |
| ------------------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------- |
| token counts are invariant under system counterfactuals | n/a              | 这是 §0 的 scope 边界，不是假设                                                                  |
| trace 和 config 同源：这份 token 流就是在这份 config 下产生的           | **高**            | 比对 turn cap、模型名、工具集、prompt 模板指纹。对不上就拒绝报数，别默默算——那是 policy counterfactual，§0 之外          |
| 差分出来的 duration 可以当 work duration 用（T1）                  | **高**            | 采集那次要是跑满了，这个差里就含排队。拿同一个 harness 空载重跑一遍，比 duration 分布。当场就能做的：把 replica 数翻倍，看预测的 e2e 动不动 |
| T0 下 tokenizer + 模板重建出来的 token 数，等于 harness 当时实际发出的     | 中                | 手上有 T2 trace 的时候，拿它的 usage 字段对一次                                                       |
| tool duration is independent of container concurrency   | **中**            | 比较 cap 16 和 cap 40 两个 run 的 tool duration 分布                                           |
| prompts are pure appends of the previous turn           | **已验证** 0/95,403 | 做过了                                                                                    |
| no CPU or network contention between workers            | 低                |                                                                                        |
| client<->server overhead is constant                    | 低                | 已量，占一个请求的 0.2%                                                                         |
| the two replicas are exchangeable                       | 低                | 已量：batch 差 4.0%，中位 step time 差 2.7%                                                    |
| no failures or retries                                  | 低                |                                                                                        |


---



## 8. Build order

Each step has an independent deliverable. 第 1 步就用到 workload 的 trace，但只当**格式
样本**看。它第一次参与算数是第 3 步，作为 validation set。它永远不碰参数。

1. **Adapter contract。** 陌生 trace 进来，探测 tier（§2.3），逐字段声明来源，输出 IR +
  provenance。用户真正碰到的就是这一层，所以它排第一，不排最后。可交付：一条只有对话的
  T0 trace 能构造出合法的 workload，每个编出来的字段都带标记。顺带补上 IR 缺的「不知道」
  和 per-session 的 `source`。
2. **Orthogonal calibration grid** on a standalone server. Yields `eta`, and as a
  by-product turns the 3.7x gap into a residual table over `(B, ctx)`.
3. **L0 diff.** Accept or reject the cost model on its own.
4. **Close the gaps:** replica id on the `generate` event; confirm the event-loop
  variant; archive the scheduler's decode-batch log lines.
5. **Discrete-event engine**, four resource pools, ported scheduler — including the
  four mechanisms of §4 that this workload never exercises.
6. **Synthetic held-out workloads.** Demonstrate neutrality.
7. **L1 and L2 diffs** against the agentic trace. 到这里 simulator 才算验过（§6）。
8. **I 的 sensitivity table，和带 interval 的 e2e。** 这一步才是交付物：吃一条 T0/T1 的陌生
  trace 加一份 config，出一个带 interval 的 e2e，外加一张表——哪个 I 撑开了 interval，去补
  哪个 instrumentation 最划算。

Where it stands: step 0, the schema, is in — `src/simulator/ir.py`,
`src/simulator/observed.py`, `tests/test_simulator_ir.py`. The converter is not written
yet, and 它不该是一个文件：`trace_to_ir.py` 是照着我们自己的 trace 起的名字，第 1 步把它
换成同一个 contract 下的多个 adapter。

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