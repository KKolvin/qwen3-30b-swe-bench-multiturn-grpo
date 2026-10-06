# Timeline simulator 设计

## 0. 这是干什么的

手上有一堆某个 harness 吐出来的 trace、一份训练 recipe、一堆 config，但没有 profile。simulator
要回答两件事：

- 一步 GRPO 端到端要多久；
- 这一步的时间线长什么样：什么时候在 decode，什么时候在等 tool，什么时候在等打分，什么时候在训练。

它是一个 trace-driven 的 discrete-event simulator。trace 告诉它「要干哪些活、每段活有多长」，
它自己算「这些活放到这套系统上，各自什么时候发生」。

trace 没有固定格式，harness 当时怎么高兴怎么写。所以 trace 里有什么就用什么，缺的只能估，
**估出来的数必须明着标出来**。

simulator 出错基本都是静默的，最常见的三种：

1. 把答案抄进了输入（比如直接用 trace 里的到达时刻），看起来很准，其实什么都没预测；
2. 参数是在这一条 trace 上拟合的，换个 workload 就错；
3. 某个数是 adapter 编的，却被当成实测值用了。

后面大部分规矩都是在堵这三个口子。

---

## 1. 能回答什么，不能回答什么

**能回答：改系统，不改模型的行为。** 改 TP、replica 数、concurrency cap、KV dtype、换硬件——
模型写出来的 token 不变，回放 trace 里的 token 流照样成立。

**不能回答，必须认出来并拒绝：**

- 改 turn cap、工具集、prompt 模板、换模型。trajectory 本身变了，旧 trace 代表不了。
- staleness、curriculum、sampling temperature。同理，它们改的是 policy 写什么。
- 一个模型副本拆到两台以上机器上跑。一台机器里 GPU 之间的通信又快又稳，测 cost model 的时候
  顺带就测进去了；跨机器要走网卡，又慢，耗时还要看网络上同时有谁在抢，cost model 测不出来。
  要算这个就得模拟网络本身，[simllm](https://github.com/openfabric-systems/simllm) 做的就是这个，
  我们不重做。多个副本各占一台机器没问题，副本之间不通信。

**两种时候，别混：**

| | 手上有什么 | 干什么 |
| --- | --- | --- |
| 开发时 | 我们自己的 run：trace + 真实 timeline | 拿真实 timeline 对答案，证明 simulator 是对的 |
| 使用时 | 别人的 trace + config，**没有 timeline** | 报一个带 interval 的 e2e。没人能告诉你对不对 |

真正要交付的是使用时那一种。在自家 trace 上对上 `analysis/<run>/timeline/*.jsonl` 只是里程碑：
证明了 simulator 有资格去给没有答案的 trace 报数。

---

## 2. 整体长什么样

```
trace  ──adapter──>  IR（要干哪些活）  ──┐
                                          ├──>  engine  ──>  预测 timeline
config ────────────> config bundle     ──┘                  汇总表（e2e interval 等）
                                                             provenance 报告（每个数从哪来）
```

engine 里面：

```
虚拟时钟 + 事件队列
├─ router        照抄 verl，决定请求发去哪个 replica
├─ scheduler     每个 replica 一个真的 SGLang Scheduler，model worker 换成 stub（§7）
├─ cost model    scheduler 每跑一步，告诉它这一步花了多久（§8）
├─ 资源池        container、打分用的 eval container，按 semaphore 排队
└─ 编排          group barrier；训练各段按给定时长占住时间
```

一个 episode 在里面走一遍：放出来 → 拿到 container → 发第一轮请求 → router 选 replica →
scheduler 把它排进 batch，一步一步 decode，每一步的耗时由 cost model 给 → 生成完 → 跑 tool
（耗时从 trace 回放）→ 发下一轮请求 → …… → 提交，抢 eval container 打分 → 整个 batch 都打完分 →
训练各段 → 下一步。

注意：第二轮以后的请求**什么时候到**，是 simulator 自己算的——上一轮什么时候生成完（simulator 算）
加上 tool 跑多久（trace 给）。

---

## 3. 最重要的规矩：答案不能当输入

> 凡是在真实 timeline 里能看到的量，都是 simulator 要算的，不能喂给它。

两个理由：

- 喂进去等于把答案抄进题面。simulator 永远不会错，也永远告诉不了你任何事。
- 更硬的理由：使用时根本没有 timeline，这些量想喂也拿不到。所以这不是自不自觉的问题，
  不这么做就跑不起来。

怎么分？看它换一套系统配置会不会变：

| 可以喂：活本身有多长、有多少 | 不能喂：活放到这套系统上什么时候发生、结果如何 |
| --- | --- |
| 容器启动、tool、打分各要多久 | 每一轮请求什么时候到 server |
| 每一轮的 `prompt_tokens` / `completion_tokens` | 排队多久、TTFT、TPOT、batch 里当时有谁 |
| 这条请求接在哪条后面，结构上最多能复用多少 token | `cached_tokens`：实际从 cache 里省了多少 |

左边这些，改 TP、改 KV dtype、改并发上限都不会变；右边这些正是这些改动会改变的。

**例子：prefix hit rate 不是输入。** trace 能给的是结构：这条请求接在哪条后面、共享多少 token。
实际命中多少由 radix cache、LRU eviction、`page_size` 和 router 一起决定，是 simulator 要算的。
trace 里记的 `cached_tokens` 拿来对答案，而且很好用：

```
run 20260912-072938，95,403 对相邻 turn
  prompt_{k+1} < prompt_k + completion_k        0 例   -> 下一轮 prompt 就是上一轮整段再加新内容
  cached_{k+1} - (prompt_k + completion_k)      p50 -113，p90 -69，全部为负
```

少掉的那 ~113 个 token，来自 `page_size=64` 的对齐，加上 SGLang 扣下的最后一页。一个不管 paging
的 simulator 会预测差值是 0，马上露馅。

同样不能喂的还有：prefill / decode 怎么交错、队列深度、KV token usage、retraction 次数、
ramp / plateau / drain 的形状。

**唯一的例外**是 `Session.release_offset_s`：每个 session 相对 t=0 什么时候放出来，sync GRPO 下
全是 0。它来自 recipe，不来自 trace——别人 trace 里的到达节奏是**那套系统当时的行为**，不能照抄。

**最容易踩的坑：时间戳相减得到的耗时。** 很多 trace 只有时间戳，tool 耗时只能这么算：
`下一个请求发出的时刻 − 上一个回复到达的时刻`。这个差里混着排队、client 开销、harness 自己的处理
时间，还有采集那次运行有多挤。拿它当 tool 耗时喂进去，等于把「系统当时等了多久」塞进了「活本身
有多久」，simulator 的 e2e 会显得很准，因为答案是抄来的。所以这种耗时要么当成估的数、只当
upper bound 用，要么先证明采集那次系统没跑满。当场就能做的检查：把 config 里的 replica 数翻倍，
看预测的 e2e 动不动。不动，就是被污染了。

开发时对答案（§12），会分级把真实的到达时刻喂回去。那是对答案用的脚手架，只在开发时存在，
千万别接进使用时的流程。

---

## 4. 输入一：IR

IR 借的是编译器的说法：adapter 把各种 harness 的 trace 翻成 IR，simulator 只认 IR，不知道 trace
原来长什么样。定义在 [src/simulator/ir.py](src/simulator/ir.py)：

| class | 是什么 | 关键字段 |
| --- | --- | --- |
| `Workload` | 一整份 workload | `sessions`、`prefixes`、`source`（从哪来） |
| `Session` | 一串相关的活，比如一个 episode | `requests`、`tasks`、`leases`、`group`、`release_offset_s`、`spawned_by` |
| `Request` | 一次 inference 调用 | `prompt_tokens`、`completion_tokens`、`prefix_parent`、`shared_prefix_tokens`、`max_new_tokens`、`after` |
| `Task` | 不是 inference 的活：tool、起容器、打分 | `duration_s`、`pool`、`after` |
| `Lease` | 跨好几个活一直占着的资源，比如一个 episode 从头占到尾的 container | `pool`、`acquire_before`、`release_after` |
| `Prefix` | 好几个 session 共享的一段内容，比如 system prompt、题面 | `tokens`、`parent` |
| `Unknown` | 这个字段没人量过，要从哪个分布抽 | `dist`、`knob`、`scale` |

`Request`、`Task`、`Prefix` 还各带两张小表：`prov` 和 `unknown`，见下面「IR 怎么说不知道」。

一个 Session 是一张 DAG：node 是 Request 或 Task，edge 是 `after`（「等它做完我才能开始」）。
一个 SWE-bench episode 和一条 open-loop serving 负载都要能用这同一套 class 写出来，
[tests/test_simulator_ir.py](tests/test_simulator_ir.py) 开头两个 test 就是这两种。

**IR 里故意没有时间。** 没有时间戳、没有 `cached_tokens`、没有 TTFT。§3 那条规矩不靠自觉，靠 IR
里根本没有这些字段。真实系统的这些数放在 [src/simulator/observed.py](src/simulator/observed.py)，
只用来对答案。

**叫 `Session` 不叫 `Episode`，是故意的。** engine 需要知道哪些请求是一伙的（router 靠它把同一个
session 的请求发到同一个 replica），但不许知道它们为什么是一伙的。engine 一旦知道这是 agent 的
episode，就可能被我们这个 workload 带偏。

**`Prefix`：第一轮请求不是冷的。** 没有 `Prefix` 的时候，第一轮请求只能写成「不接任何东西」，
等于说它要从头 prefill。实测完全不是：

```
run 20260912-072938，4,243 个第一轮请求
  cached / prompt        p10 0.39   p50 0.983   p90 0.996
  合计                    9.24M / 11.19M = 82.6%
  命中不到 50% 的         1,118 (26%)
```

同一个 group 的 8 条 rollout 共享 system prompt 和题面。那 26% 也不是噪声：530 个 group × 2 个
replica = 1,060，跟 1,118 对得上——每个 group 在每个 replica 上冷一次，其余都命中。这个数正好
可以拿来验 router。

`Prefix` 只写长度和接在谁后面，不写内容。第一轮的 `prefix_parent` 指向一个 `Prefix`，后面的轮次
指向上一轮的 Request。共享内容不等于要等它，所以 `after` 不能指向 `Prefix`，也不能跨 session——
谁先到谁付 prefill，这由 simulator 算。按字节算这一项不大（只占全部 cached token 的 0.7%），
但换成大量请求共享 system prompt 的 serving workload，它就是全部。

**`max_new_tokens` 是个坑。** SGLang 接纳新请求时，会给每个在跑的请求预留
`min(max_new_tokens - 已生成, 4096) × new_token_ratio` 个 token 的 KV（`schedule_policy.py:433`）。
真 server 只知道 cap，不知道这条请求最后会写多少。所以 simulator 让请求停下的办法只能是：在第
`completion_tokens` 个位置吐 EOS。**绝对不能**图省事把 `max_new_tokens` 设成 `completion_tokens`——
那样预留就从「按 cap 估」变成「按答案留」，retraction 永远不会发生，而且不报任何错。

陌生 trace 多半没记 cap，所以这个字段可以是空的，由 config bundle 填。engine 门口的
`require_runnable` 拦住还空着的，比让 adapter 随手编一个强。

**IR 怎么说「不知道」。** 陌生 harness 的 trace 一定缺字段。要是每个字段都必须填一个数，adapter
除了编没有别的办法，编出来的还跟实测的长得一模一样——这就是 §0 那三种静默出错的第三种。所以
`Request`、`Task`、`Prefix` 各带两张小表：

- `prov`：每个字段的值哪来的。trace 读的、config 给的、推出来的（tokenizer 配模板重建、拿相邻
  turn 相减）、估的。表里没有的字段就是没说，报告里照实列「没说」，不替它猜。
- `unknown`：这个字段没人量过，要从哪个分布抽。

字段本身还是裸的 `int` / `float`，engine 一行都不用改。一条规矩守着：`unknown` 里有的字段，`prov`
必须写着「估的」。想挂个分布又想说这是实测值，validate 当场拒。

**抽样放在 engine 前面：`realise`。** [src/simulator/realise.py](src/simulator/realise.py) 吃一个带
分布的 workload 加一个 seed，吐一个全是具体数的 workload，engine 只认后者。这样 §6 那句「engine
不许根据 provenance 改变行为」就不用靠自觉：到 engine 手里的时候，估的数和实测的数就是同一个
float，想区别对待也没得区别。`prov` 跟着 workload 一路走进报告，engine 从不读它。

**一个 node 可以等好几个 node。** `after` 是一个 list。一个 assistant turn 并行发三个 tool call、
下一轮等三个都跑完，这在别的 harness 里是常态；串成一条链就是凭空加依赖，把 session 拉长。

**subagent 用 `spawned_by`。** 子 agent 是独立的 session，由父 session 的某个 task 拉起，那个 task
要等子 session 整个跑完才算完。这是唯一允许跨 session 的依赖：两个平级 session 互相等是编排，
父子相等是 workload 本身的形状。子 agent 单独一个 session，也就单独一个 routing key，这正是想要
的——它的 prompt 跟父亲几乎不共享，塞进父 session 会让 router 以为它们该在一台机器上。

**还缺的：`Workload.source` 只有一个字符串**，几个来源混在一份 workload 里就不够用了。

**还缺的：unknown 之间不能有关系。** 每个 unknown 字段各抽各的。可 token 数不是这样：下一轮的
`prompt_tokens` 等于上一轮 prompt 加 completion 再加新内容，`shared_prefix_tokens` 也由它们算出来。
一份连 token 数都没记的 trace，adapter 现在只能拒绝，因为分开抽出来的三个数一定对不上 `validate`
的算术。要支持它，IR 得允许「这个字段 = 那几个字段的和」这类写法。只有对话的 trace 不受影响：
用 tokenizer 重建出来的 token 数是 derived，天然自洽。

`Task.duration_is_cap`（这次 tool 是不是超时被杀了）engine 现在不读。留着它，是因为当时不记，
事后就补不回来了。

---

## 5. 输入二：config bundle

IR 只装 workload，不装系统配置：`Lease` 只说要哪个 pool，不说 pool 多大。系统配置走另一个输入：

| 从哪来 | 装什么 | 缺了怎么办 |
| --- | --- | --- |
| recipe | batch、`n`、group barrier、turn cap、worker 数、每个 worker 的 container 上限、eval pool 大小、session 的释放节奏 | 拒绝跑 |
| SGLang server args | SGLang 版本；TP、replica 数、`page_size`、`chunked_prefill_size`、`max_running_requests`、`new_token_ratio` 的三个参数、`max_total_num_tokens`、overlap 开没开 | 有活着的 server 就直接读；没有就跑 SGLang 自己算默认值的那段代码 |
| 模型 `config.json` + 硬件规格 | layer 数、KV head 数、head dim、active expert、dtype 宽度、HBM 带宽、`cuda_graph_max_bs` | 拒绝跑 |
| calibration 结果 | cost model 的三个拟合量（§8） | 本机没测过，就用别的机器上最近的测量点，并打印离它多远 |
| prior（trace 没记的耗时） | 每个 knob 一个数：中位数要几秒；拿不准就给 `[lo, hi]` | 用我们自己 run 量的分布，报告里标「默认」 |

**一项都不许拿「常见默认值」顶。** SGLang 很多默认值是按 GPU 显存分支的，照抄就是在编。整个
bundle 要能原样存成一个文件、跟着预测结果走，不然这份预测没法复现。

**prior 是唯一有默认的一项。** 前几行是系统配置，填错了预测就错，而且看不出来。prior 本来就是
「估的」：它已经带着 interval，sensitivity table 会单独给它一行，报告会写它从哪来。用我们的 run
顶上，最坏是那一行的 interval 偏了，用户看得见，也知道该补哪个数。拒绝跑反而让只有对话的 trace
一个数都拿不到。

默认值是一份存进 repo 的文件 `src/simulator/adapters/default_priors.json`，由
`scripts/simulator_dev/build_default_priors.py` 生成：分布取 run 20260912-072938 step 1 的全部实测耗时，scale 取
20261005-113503、20261005-143426、20261006-030010 三个 run 的 step 1 均值相对它的比值，再把 1.0
包进去。

**用户怎么配。** 在 bundle 的 YAML 里写一个 `priors` 段，例子在
`configs/simulator/example_bundle.yaml`：

```yaml
priors:
  eval_duration: 45            # 我们的 harness 打分大概 45 秒
  tool_duration: [0.5, 2]      # 拿不准，0.5 到 2 秒之间
  # 没写的 container_start、patch_recover、cleanup 用我们的
```

写完跑 `python scripts/show_priors.py <bundle>`，每个 knob 打一行：中位数、范围、用的是用户的数还是我们
的。代码里是 `load_priors(path)`，返回的 `Priors` 直接交给 adapter。

**不知道要填什么。** 跑 `python scripts/priors_template.py`，它把所有 knob 列出来，每个都注释掉、写着
我们的数和它在我们 run 里占多少 task 时间：

```yaml
priors:
  # eval_duration: 21.7        # running the tests that score one submitted patch. 66.3% of task time, our runs drifted 0.90-1.02x
  # tool_duration: 0.22        # one tool call (bash, edit, submit, ...). 29.4% of task time, our runs drifted 0.89-1.00x
  # container_start: 0.377     # starting the container one episode runs in. 3.7% of task time, our runs drifted 0.84-1.09x
  # patch_recover: 0.737       # pulling the diff out of an episode that never submitted. 0.6% of task time, our runs drifted 0.92-1.00x
  # cleanup: 0.000697          # removing the container when an episode ends. 0.0% of task time, our runs drifted 0.81-1.00x
```

按占比排，所以实际上只有头两行值得填：打分和 tool 加起来 96%。哪行都不填也能跑，全用我们的。

只有两种写法：

- `eval_duration: 40`：中位数 40 秒。把我们的分布整体挪过去，让中位数落在 40，一次和一次之间差多少
  还用我们量的。别人一般说得出「打分大概要多久」，说不出它的尾巴长什么样。给一个数等于声明这个
  中位数是对的，这一行没有 scale，interval 不会因为它变宽。
- `tool_duration: [0.5, 2]`：拿不准，中位数在 0.5 到 2 秒之间。这个范围就是 scale，sensitivity table
  扫的就是它。

不管哪种，**我们自己的 scale 都不带过去**：它量的是我们几个 run 之间差多少，跟别人写的数准不准没
关系。knob 名字写错直接拒绝，不然默认值原样留着，用户还以为自己改了。

默认值不能拿来转它自己量过的 run：上面四个 run 的任何一个 step、包括对话 dump，`check` 都会拒，
用户给了中位数也一样，因为分布形状还是那个 run 的。要转这几个 run，就用别的 run 现建 prior（削
trace 实验就是这么做的）。

训练各段的时长还没有 knob，默认值里也没有它。engine 有了训练段以后再加。

---

## 6. 每个数都要说清 provenance

simulator 用到的每个数，属于下面五种之一：

| 从哪来 | 例子 | 注意 |
| --- | --- | --- |
| trace 给的 | token 数；tool 耗时（如果 trace 自己记了） | 同一个字段在不同 trace 里来路可能不同，见下 |
| config 读的 | TP、`page_size`、HBM 带宽 | 从活 server 读的，比从默认值代码解出来的可靠 |
| 实测拟合的 | 只有 cost model 的三个量 | 只在专门设计的测量 grid 上拟合，绝不在 run trace 上拟合（§8） |
| simulator 算的 | 到达时刻、batch 组成、TTFT、`cached_tokens` | 永远是输出；开发时拿来对答案 |
| 估的 | trace 里没有，只能从某个分布里抽 | 必须说清从哪个分布抽、那个分布又是哪来的 |

我们自己的 run 里，有些量现在没记、但加一行 log 就能有（比如每个请求落在哪个 replica），这叫
instrumentation gap。别人的 trace 补不了，只能算「估的」。

**provenance 跟着数据走，不跟着字段走。** 同一个 `prompt_tokens`：trace 有 API usage 字段时是
trace 给的；只有对话时要靠 tokenizer 加完全一致的模板重建；只记了轮数时就只能估。所以 provenance
要写进 IR、跟着数据走，不能只写在这份文档的表里。engine 不许根据 provenance 改变行为——这一条
靠 `realise`（§4）落实：抽样在 engine 之前做完，engine 拿到的全是裸 float，想区别对待也没得区别。

**有估的数，就必须这样报：**

- e2e 报一个 interval，不报一个点。输入是分布，输出就不可能是一个点。
- 出一张 sensitivity table，说清是哪个估的数把 interval 撑开的。只说「1.4–2.2h」，用户不知道
  该去补什么。
- 随机抽样用固定 seed。不然对答案时分不清是模型错了，还是这次抽的数不一样。

sensitivity table 的算法：每次只动一个估的数，从它分布的 p10 扫到 p90，其余固定在 p50，看 e2e 动
多少，按动的大小排序。下表的数字只是示意：

| 估的数 | 分布从哪来 | e2e 动了多少 |
| --- | --- | --- |
| 训练各段时长 | 没有模型，只有一个范围 | ±38 min |
| tool 耗时 | cap 16 / cap 40 两个 run 的分布 | ±6 min |
| 打分耗时 | `score` p50 21.6s | ±4 min |
| 容器启动 | p50 3.3s | ±1 min |

第一行就是要交给用户的那句话：「去把训练各段的时长记下来，interval 能缩掉一大半。」因为每次
只动一个，各行加起来**不等于** interval 的总宽度，两个数都要报。

**「一个估的数」在代码里叫 knob。** 每个 unknown 字段挂一个 knob 名，一个 knob 底下可能只有一个
字段（训练段），也可能有十万个（tool 耗时），table 一行就是一个 knob，还要报它管着多少个字段：
让用户去补一个数和去补十万个数，是完全不同的两件事。扫的时候 seed 不变，所以每个 node 自己那
一抽在整条曲线上是同一个数，e2e 动的那点只能是这个 knob 带来的。

**借来的分布要自己说清可能偏多少。** 一个 unknown 带两样东西：`dist` 说单个值长什么样，`scale`
说这整条分布可能整体偏多少倍。两件事推不出彼此——tool 和 tool 之间差多少，跟 tool 耗时的中位数
估错多少，根本没关系；拿前者当后者，sensitivity table 就会永远指着尾巴最重的那个量。`scale` 也
正是没有就该说没有的东西：少了它，每个 node 各抽各的，几万个 node 一平均就什么都不剩。两万个
tool task 抽 64 次，预测总时长只晃 2.1%，比两个 replica 之间 4.0%–16.8% 的噪声还小。那不叫预测。

**目前设计里最大的洞是训练段。** simulator 不给 FSDP 建模，训练各段按给定时长当成一整块。没有
profile 时它们只能估，而 `old_log_prob`、`ref`、`adv`、`update_actor`、`update_weights` 加起来
占一步的 52%。所以只有对话的 trace，e2e 的 interval 会被这一个估的数主导，除非另外从 recipe
给训练段建个模型。

**trace 里可能有什么。** 按信息从少到多：

| trace 里有 | 典型来源 | 于是 |
| --- | --- | --- |
| 只有对话 | `.traj`、消息 jsonl、各种日志 | 配 tokenizer 和 prompt 模板，能重建 token 数、prefix 结构、轮数、退出原因。所有耗时都只能估 |
| + 每条事件的时间戳 | 大多数 harness | 耗时只能相减，是被污染的（§3） |
| + API 的 usage 字段 | 记了 usage 的 harness | token 数精确。里面的 `cached_tokens` 是答案，adapter 要把它放进 `observed`，不进 IR |
| + 每段活自己记了耗时 | 我们的 harness（`tool_called` 带 `elapsed`） | 耗时是直接量的，可以用 |

这几行不是严格的高低：一条 trace 完全可能记了每段耗时却没有 usage 字段。所以 adapter 是**逐字段**
判断来路，不是给整份文件打一个等级。

最差的情况（只有对话）其实也不坏：几乎所有 harness 都给得出 token 流，而 token 流正是 cost model
最需要的（KV 读多少字节由 context 长度决定）。缺的是非 GPU 部分的耗时，而 tool 只占一个 episode
的 3.1%。所以只有对话加一份 config，就够把占大头的 decode 算出来，剩下的用 interval 兜住。
trace 越差，interval 越宽，但不会突然算不出来。

**我们自己的 run 有真实 timeline，但它不算「最好的 trace」。** timeline 是答案，不是输入。自家 run
的 adapter 要把同一份文件劈成两半：换个 harness 也会有的部分（token 数、prefix 结构、记下来的
耗时）进 IR；只有这套系统当时的行为才有的部分（到达时刻、`cached_tokens`、TTFT、排队时长）进
`observed.py`。劈完之后，就当 timeline 不存在。

---

## 7. Scheduler：直接跑真的 SGLang，不自己抄

**决定：** batch 怎么组、KV 怎么分、radix cache 怎么命中和 evict，都不自己写，直接跑 SGLang 0.5.8
自己的 `Scheduler`，只把真正跑模型的 worker 换成一个什么都不算的 stub。这个做法借自 simllm 的
`SimTpModelWorker`。

**为什么不照源码抄一份。** 抄看起来没有参数要拟合，可抄的过程处处是隐藏参数：page 怎么对齐、
扣下哪一页、`new_token_ratio` 什么时候衰减、chunked prefill 怎么切、retraction 先踢谁。哪一处抄错
都不报错，只会让结果差几个百分点，然后被当成 cost model 的误差去调。跑真的，这类错误就不存在了。
§3 那个 `cached_tokens` 少 113 个 token 的现象，不用建模就自动出来。

**怎么接**（0.5.8 没有 plugin 机制，只能在进程里直接 patch）：

| 改哪里 | 怎么改 | 为什么 |
| --- | --- | --- |
| `Scheduler.init_tp_model_worker`（`scheduler.py:494`） | 换成构造 `SimTpModelWorker` | worker 只在这里构造 |
| KV pool | 在 CPU 上建 `ReqToTokenPool` / `TokenToKVPoolAllocator`，不分配真的 KV tensor，大小取 config 里的 `max_total_num_tokens` | eviction 和 retraction 在真实的容量边界上发生 |
| `forward_batch_generation` | 不算任何东西，每条请求吐一个 token。没到 `completion_tokens` 时吐一个由 (请求 id, 位置) hash 出来的 token（避开 EOS 和 stop token），到了就吐 EOS。`stop="length"` 的请求不吐 EOS，让 SGLang 自己按 cap 停 | 什么时候停来自 trace；scheduler 看到的仍然只有 cap（§4） |
| 主循环 | 不调 `event_loop_normal`（`scheduler.py:1068`，死循环 + ZMQ）。engine 自己一步一步调同样那几个函数：`process_input_requests` → `get_next_batch_to_run` → `run_batch` → cost model 给出这一步耗时、虚拟时钟往前走 → `process_batch_result`。overlap 模式下 `process_batch_result` 晚一步调，照 `event_loop_overlap` 的做法 | 时钟归 engine 管，SGLang 只负责做决定 |
| 输出 | 截住 `send_to_tokenizer`，把「哪条请求在哪一步结束」交回 engine；每一步记下 `forward_mode`、`extend_lens`、`seq_lens`、`cached_tokens` | 预测 timeline 就从这里来 |

**token id 从哪来。** radix cache 按内容匹配，不按长度，所以要从 IR 的结构造出 token id：`Prefix`
和每条请求自己新加的那段，用 hash(id, 位置) 生成，保证彼此不撞；接在上一轮后面的那段，直接拿上一轮
的 prompt id 加 stub 实际吐出的 output id，截到 `shared_prefix_tokens`。这件事在 workload 那一侧做，
SGLang 看到的只是 token 序列。

**SGLang 会不会偷看真实时钟？** 查过源码，scheduler 路径上会影响行为的只有三处：

- `SGLANG_QUEUED_TIMEOUT_MS`（默认关，`scheduler.py:1708`）：开着的话会按真实时间踢掉排队太久的
  请求。engine 启动时要断言它是关的。
- `empty_cache_interval`：只调 `torch.cuda.empty_cache`，不影响调度。
- radix 节点的 `last_access_time`（`time.monotonic()`）：这是 LRU 的排序依据，但 `evict_policy.py`
  只比先后、不比差值。simulator 按事件顺序调用，先后就是对的。

所以不用 patch `time`。

**router 不在 SGLang 里。** 它是 verl 的 `GlobalRequestLoadBalancer`（`src/agentic_grpo/agent_loop.py:322`）：
同一个 id 来过就发回同一台，没来过就挑当前 load 最小的那台。几十行，照抄进 engine。

**要从 config 读的 scheduler 参数：** `schedule_policy`（我们是 `fcfs`）、`schedule_conservativeness`
（1.0）、`chunked_prefill_size`、`max_running_requests`、`page_size`（64）、`new_token_ratio` 的
初值 / 衰减 / 下限、overlap 开没开。我们的 run 没传 `disable_overlap_schedule`，按默认是开的；
要确认就看一次 server log。

**代价：**

- 版本绑定：simulator 里跑的 SGLang 必须和 config 里 server 的版本一致，否则拒绝。每升一次 SGLang，
  stub 都要重新对一遍。
- 速度：我们的 run 每个 replica 大约 5.5 万步（gen 2418s ÷ 每步 44ms），真 scheduler 每一步都要在
  上百个请求对象上跑一遍 Python。先量出跑一次完整模拟要多久，再决定 §11 的抽样次数 N=64 撑不撑得住。
  撑不住就降 N，在报告里写明——**不许为了快换回自己抄的版本**。

**还没验证：一个进程里能不能同时起多个 `Scheduler`。** `Scheduler.__init__` 还会建 ZMQ socket、
加载 tokenizer，可能还会初始化 torch.distributed。起不了的话，就每个 replica 一个进程，engine 当
协调者，按虚拟时钟同步推进。慢，但结果一样。

---

## 8. Cost model：每一步要多久

整个 simulator 只有这一条式子：

```
T_step = (W_bytes + KV_bytes) / (BW × eta(B, ctx, prefill_len)) + overhead
```

- 一个 step 是 scheduler 的一次 forward。
- `W_bytes`：这一步要读的权重字节，从模型 `config.json` 算。
- `KV_bytes`：这一步要读的 KV 字节，由当前 batch 里所有请求的 context 长度算。包括
  `num_kv_heads < tp` 时 KV 被复制的情况。
- `BW`：HBM 带宽，硬件规格。
- `eta`：这块硬件上、这个 engine 实际跑到了带宽上限的几成。没人写在任何地方，只能测。
- `overhead`：每一步的固定开销。

需要拟合的只有三个量：`eta`、`overhead`、prefill 的耗时曲线。prefill 曲线其实就是同一张 `eta`
表在「batch 小、prefill 长」那一角的样子，不是第二个模型。`cuda_graph_max_bs` 和 capture 的
batch 档位从 config 读，step 耗时的台阶就在那里。

**`eta` 里还包了什么。** 一步里除了读 HBM 之外的所有耗时——kernel 没跑满、launch 开销、TP 的
allreduce——都没单独建模，全折进 `eta`。单机内的 allreduce 走 NVLink，又小又稳，只跟 batch 大小
有关，所以测 `eta` 时顺带就测进去了。也正因为这样，**`eta` 是按 TP 配置测的**：TP=4 测出来的
不能直接拿去算 TP=8。

**拟合 `eta`，不拟合时间。** 时间的量级跟 workload 有关，extrapolate 出去会离谱；`eta` 没有单位、
有上下界、变化平滑，是 engine 加硬件的属性。它还把一直没解开的那个 3.7x 差距变成了一个能打印的数：
在 p50 稳态，每个 replica 每一步每张卡读约 43GB 的 KV，花 44ms，算下来 `eta ≈ 0.12`（按 8TB/s）；
standalone 跑同样的点大约 0.5。同一个测量点两个数，差的就是那个 gap。

**只在专门设计的 grid 上测，绝不在 run trace 上拟合。** 原因是 run trace 里 batch 大小和 context
长度几乎完全反相关（`corr = -0.845`）：rollout 尾巴上还没跑完的，恰好是最长的那些 episode。

```
B   1- 16   ctx p50 51784      <- 尾巴
B  16- 48   ctx p50 28903
B  48- 80   ctx p50 21255
B  80-112   ctx p50 18671
B 112-140   ctx p50 15999      <- 平台期
```

trace 只覆盖了二维空间里的一条对角线。在这上面拟合，分不清耗时是随 batch 涨还是随 context 涨，
换一个不在对角线上的 workload 就是 extrapolate。所以 grid 要在 `batch × context × prefill 长度`
三个方向上各自独立地铺开，范围按 engine 能跑的来，不按这条 trace 碰到的来。

**grid 外面不 extrapolate。** `eta` 只在 grid 内 interpolate。落在 grid 外的 step 不给估计，标出来，
并报离最近的测量点多远、在哪个方向出界。这种 step 超过一定比例（默认 5%）就整份拒绝报数。
借别的机器上的测量结果也一样，距离必须打印出来——没在这台机器上验过的 `eta` 算出来的 e2e，
跟编的没区别。

---

## 9. 其他部分

**router**：见 §7。每个请求实际落在哪个 replica，我们的 run 现在没记。这是自家 run 最大的
instrumentation gap：没有它，§12 的第二级没法逐请求对答案。补它的成本是在 `generate` 事件里加一个字段。

**KV pool**：TP=4 下每个 replica 的 `max_total_num_tokens` 是 4,560,512。实测 KV 占用 p50 0.38、
p99 0.47，这是 simulator 要算出来的。

**非 GPU 资源**：

- worker 数、每个 worker 的 container 上限，从 recipe 读。
- 打分用的 eval container 是每个 worker 一个 semaphore，上限 4。`eval_wait` p90 248s、最长 755s，
  直接原因就是它。
- 容器启动 p50 3.3s。容器启动和 tool 耗时，我们的 trace 直接记了，别人的 trace 多半只能估。
- 两条假设：tool 耗时不随 container 并发数变化（中等风险）；worker 之间没有 CPU / network 争抢。
- 各个 pool 的排队时间是 simulator 要算的。

**编排（sync GRPO）**：batch 256 × n=8，group barrier。训练各段按给定时长当成一整块。weight sync
的语义来自 config，时长来自 trace。ramp / plateau / drain 的形状是 simulator 要算的，是第三级
对答案的主要目标。

**client 开销**：当常数。实测 client→server p50 4.3ms，server→client p50 1.9ms，加起来占一个请求的 0.2%。

---

## 10. 这条 trace 用不到、但必须能对的机制

在 SWE-bench trace 上，下面几样都能证明可以忽略。可要是真忽略了，这个 workload 的形状就焊死在
simulator 里了，而且 code review 看不出来。

跑真 scheduler 之后，前三行的机制 SGLang 本来就有，不用自己写，但**验证照样要做**。风险变成了
「stub 或 engine 不小心把它绕过去了」：KV pool 建小了、虚拟时钟没让队列真的积起来、`max_new_tokens`
被换成了答案。这些都不报错，只有专门的 workload 打得出来。

| 想省掉的 | 在这条 trace 上为什么能省 | 换什么 workload 会出事 | 用来验它的 workload |
| --- | --- | --- | --- |
| 不建等待队列 | `num_queue_reqs` 一直是 0 | 任何 open-loop 到达的负载，排队就是 latency 本身 | open-loop Poisson，短 prompt，不复用 |
| prefill 可以忽略 | cache 命中 96.2%，真要 prefill 的只有 ~450 token | 没有 prefix 复用时，prefill 变主导 | 长 prompt，只生成 1 个 token |
| 不管 KV 容量和 retraction | `num_retracted_reqs` 一直是 0，KV 占用 p50 0.38 | 长 context 高并发，throughput 会断崖 | 超长 context，顶到 KV 上限 |
| 只把 decode 做准 | decode 占一个 episode 的 94.8% | tool 很慢的 workload | closed-loop，每轮 tool 故意很慢 |

最后一列就是用来证明 simulator 没被这个 workload 带偏的 held-out set：每一条「这条 trace 上可以省」，
都配一个专门打它的 workload。另外还要跑两个：batch 大、context 短（trace 对角线从来不去的角落）；
以及 repo 里现成的两组对比（context 28672 vs 61440、只有 bash vs 加了 edit tool）。

这些机制在我们的 trace 上**一次都不触发**，这本身也是一项要对的检查。

---

## 11. 输出

使用时（没有答案），每次输出三样东西，缺一样都不算完：

1. **预测 timeline**，格式和 `analysis/<run>/timeline/*.jsonl` 完全一样（同样的 `cat` / `name` /
   `t` / `t_end` 事件）。现成的 report 工具直接能画；对答案也就是比两份同格式的文件。
2. **汇总表**：e2e 的 interval 和中位数、ramp / plateau / drain、每分钟完成多少 episode、每个
   replica 的 step 耗时分布、各个 pool 的等待时间分位数。
3. **provenance 报告**：每个参与计算的数从哪来（§6）；cost model 是本机测的还是借来的、离最近的
   测量点多远；每个估的数从哪个分布抽的；再加上 sensitivity table。没有这一份，用户分不出哪些数
   是编的。

**interval 怎么算**：对所有估的数一起抽样 N 次（默认 64），seed 固定并写进报告；取 p10–p90 当
interval，p50 当点估计。

**可复现**：同样的 IR、config 和 seed，结果必须一位不差。随机性只能来自那个 seed。

**宁可不报，也不报错的数。** 下面几种情况直接拒绝、非零退出，并写清缺什么、去哪补：

| 情况 | 为什么不能凑合 |
| --- | --- |
| trace 和 config 对不上（这份 trace 不是在这套配置下跑出来的） | 那等于在问「换了 policy 会怎样」，回答不了（§1） |
| config 里有必填项缺了 | 默认值按显存分支，照抄就是编 |
| grid 外的 step 超过阈值 | extrapolate 出来的 `eta` 没有任何保证 |
| 对答案时，被对的那个量依赖了估的数 | 那验的是自己的估计，不是模型 |
| 一个模型副本拆到多台机器上 | 跨机器通信 `eta` 测不出来（§1） |
| simulator 里的 SGLang 版本和 config 里的不一致 | 跑的是另一个 scheduler |

---

## 12. 怎么对答案（只在开发时）

要有一份 trace 配一份真实 timeline 才能做，使用时一份都没有。这三级都过了，才有资格去给没有答案
的 trace 报数。

分三级，每一级多放开一个 feedback loop。只看最外层对不上时，你不知道错在哪，所以要有里面两级。

- **第一级：只验 cost model。** 输入真实的每秒 `(batch 大小, 总 context 长度)` 序列，预测每一步
  耗时，跟真实的 `batch / gen_throughput` 比。不需要知道请求落在哪个 replica，现有数据就能做。
- **第二级：验 scheduler。** 输入真实的请求到达时刻和每个请求的 token 数，simulator 自己组 batch、
  自己决定什么时候完成。**逐请求**比 TTFT 和 decode 时长，共 99,646 个请求。前提是先补上
  「请求落在哪个 replica」。
- **第三级：全闭环。** 只给 episode 什么时候放出来，其余全靠 simulator。比整体形状：ramp 365s /
  plateau 1238s / drain 463s、每分钟完成的 episode 数、生成阶段总时长、一步的 wall clock、每秒的
  并发曲线。

**误差会不会滚雪球。** 在第二、三级加一个开关：每隔 k 轮把时间对齐回真实值一次，画误差随 k 的
变化。k=1 准、k=∞ 不准，说明一个小偏差在 feedback loop 里被放大了；k=1 就不准，说明单步模型本身错了。

**容差从 noise floor 来，不凭感觉。** 我们的 run 有两个独立的 replica，负载统计上一样，两者之间的
差就是去不掉的噪声（2,862 个两边都忙的采样点）：

```
num_running_reqs   平均相对差    4.0%   (corr 0.979)
gen_throughput     平均相对差   16.8%
step 耗时中位数    44.5ms vs 43.3ms -> 2.7%
```

所以按秒采样的曲线上，要求比 ~17% 更准没有意义；中位数上 ~3% 就是底。验收标准定在这里。

**每一步的 latency 目前没有真值可对。** 存档的 trace 最细只到每个请求。SGLang 其实每隔
`decode_log_interval` 步会打印 batch 组成和 throughput，但写在 Ray actor 的 log 里，run
20260912-072938 的 Ray session 已经没了。把这几行 log 存进 timeline 目录几乎没成本，也是验证每步
预测的唯一办法。

---

## 13. 代码分层和 test 守的规矩

| 层 | 在哪 | 允许知道什么 | 允许 import |
| --- | --- | --- | --- |
| adapter | `src/simulator/adapters/`：接口在 `base.py`，`agentic_messages.py` 只读对话文本，`priors.py` 读默认 prior 和 config bundle | trace 的格式；逐字段判断来路 | `simulator.ir`、`simulator.observed` |
| IR | `src/simulator/ir.py` | 只有 workload | — |
| realise | `src/simulator/realise.py` | IR，加一个 seed | `simulator.ir` |
| observed | `src/simulator/observed.py` | 真实系统的结果，也就是答案 | `simulator.ir`（只借 id 类型） |
| engine | 待建 | 抽样过的 IR + config | `simulator.ir` |
| 开发专用 | `src/simulator/dev/`：`agentic_timeline.py` 读自家 timeline（也是 adapter，守 adapter 那一行的 import 规矩），`priors.py` 从自家 run 量 prior、写 `default_priors.json`；脚本在 `scripts/simulator_dev/` | 自家 run，连同答案 | 全部 |
| 对答案 | 待建，放进 `src/simulator/dev/` | 两边都知道 | 全部 |

规矩：

1. **engine 不许 import `simulator.observed`；任何 simulator 模块不许 import `agentic_grpo`。**
   这两条已经有 test 守着（[tests/test_simulator_ir.py](tests/test_simulator_ir.py)）。
2. **engine 里不许出现 `episode`、`turn`、`tool`、`swebench` 这几个词。** engine 和 workload 之间
   只有一个请求流：每个到达事件带着 `(prompt_tokens, prefix_parent, shared_len, completion_tokens)`。
   「第 k+1 轮要等第 k 轮的 tool 跑完」这种逻辑属于 workload 那一侧，到 engine 这里只剩一个到达时刻。
   engine 分不出输入是谁生成的，就不会偏向谁。
3. **关于 trace 格式的知识只能待在 adapter 里。** 进来的 trace 本来就来自不认识的 harness。
4. **engine 不许根据 `Task.kind` 或 provenance 做分支。** `kind` 只是报表上的标签（trace 里的
   `bash` / `edit` / `submit`，进 IR 后都是 `"tool"`）。provenance 这一半由 `realise` 兜住：
   engine 只收抽样过的 workload，`require_runnable` 在门口拦下还带着分布的字段。
5. **cost model 不许在 run trace 上拟合**（§8）。
6. **没有答案也必须能跑完。** 使用时没有 timeline，任何「需要 `observed` 才走得通」的代码路径，
   在真正该用的场合就是死路，而且开发时永远发现不了。检查办法：拿一份只有 trace、没有 timeline
   目录的输入跑一遍完整流程，`Observations` 全空也要得出数。
7. **使用时的代码不许 import `simulator.dev`。** 使用时的代码指 `src/simulator/` 下 `dev/` 以外的
   所有文件，加上 `scripts/` 顶层的脚本。`dev` 里的东西只在我们自己的 run 上走得通，使用时调到它
   就是第 6 条那种死路。默认的 `ADAPTERS` 里也没有读自家 timeline 的那个 adapter，开发时要显式传。
   有 test 守着（[tests/test_simulator_ir.py](tests/test_simulator_ir.py)）。

---

## 14. 假设

| 假设 | 风险 | 怎么证伪 |
| --- | --- | --- |
| trace 和 config 是一套：这份 token 流就是在这份 config 下跑出来的 | **高** | 比对 turn cap、模型名、工具集、prompt 模板指纹。对不上就拒绝报数 |
| 时间戳相减的耗时可以当活本身的耗时 | **高** | 同一个 harness 空载重跑一遍比分布；当场能做的是把 replica 数翻倍，看 e2e 动不动 |
| 只有对话时，tokenizer + 模板重建的 token 数等于当时真实发出的 | **已量**（run 20261005-113503，144 条 episode）：prompt 93% 一字不差，completion 98.7%，其余差 +1 到 +9 个 token，来自模型采样出非标准的切分，重编码补不回来。另外第一轮 prompt 超过 `prompt_length`（这里 4096）会被从头截掉，文本里看不出来，所以这个数必须从 config bundle 拿 | `scripts/simulator_dev/check_messages_reconstruction.py` |
| tool 耗时不随 container 并发数变化 | **中** | 比较 cap 16 和 cap 40 两个 run 的 tool 耗时分布 |
| 下一轮 prompt 就是上一轮整段再加新内容 | **已验证**：95,403 对里 0 例外 | — |
| stub 吐什么 token 不影响调度，SGLang 只看长度和 id 是否相同 | 低 | 用两个不同 seed 的 hash 跑同一份 IR，每一步的记录必须完全一样 |
| scheduler 不靠真实时钟做决定 | **已查源码**（§7） | queued timeout 开着就不成立，engine 启动时断言 |
| 单机：TP 的 allreduce 可以折进 `eta` | 低（这台机器） | 跨机器就拒绝（§1），而不是放宽这条 |
| worker 之间没有 CPU / network 争抢 | 低 | |
| client 和 server 之间的开销是常数 | 低 | 已量，占一个请求的 0.2% |
| 两个 replica 可以互换 | 低 | 已量，就是 §12 的 noise floor |
| 没有失败和重试 | 低 | |

---

## 15. 做的顺序

每一步都能单独交付、单独验。用 ML 的话说：第 2 步的测量 grid 是 training set，也是整个 simulator
唯一拟合参数的地方；自家 timeline 是 validation set；第 6 步的合成 workload 是第二个 held-out set，
专门证明没被带偏。我们的 trace 在第 1 步只当格式样本看，第一次参与计算是第 3 步，当 validation set
用，永远不碰参数。

1. **adapter。** 陌生 trace 进来，逐字段判断来路，输出 IR 和 provenance。用户真正碰到的就是这一层，
   所以排第一。交付：一条只有对话的 trace 能造出合法的 workload，每个编出来的字段都带标记。IR
   这边的准备已经做完了（§4：`prov`、`unknown`、`realise`）。别照着自家 trace 写一个
   `trace_to_ir.py`，要的是同一个接口下的多个 adapter。

   接口在 [src/simulator/adapters/base.py](src/simulator/adapters/base.py)。每个 adapter
   两个方法：`claims(path)` 说这份文件我认不认识，`convert(path, priors)` 读它。读完交出三样：IR、
   observed、几句话说明哪些字段没找到。`load` 挑出唯一认领的那个 adapter，跑完再查四条：IR 合法；
   两边的 `source` 一样；observed 里每个 id 都能在 IR 里找到对应的 request 或 session；每个「估的」
   字段用的都是调用方给的 prior，而且那个 prior 不是在正在转的这份 trace 上量的。最后一条是为了
   下面的削 trace 实验：删掉一个字段再用同一份 trace 的分布填回去，实验就白做了。

   验收这一层只有一个办法有说服力：**把自家最全的那条 trace 一级一级削**。先删每段活自己记的
   耗时，再删 usage 字段，再删时间戳，最后只剩对话。每一级跑一遍 adapter，看三件事：还能不能出
   合法 workload；interval 是不是一级比一级宽；每一级的 interval 盖不盖得住信息最全那一级的点
   估计。这是唯一能把「为了通用牺牲了多少精度」量出来的实验，而且只要一条 trace 就能做。
2. **测量 grid。** 在 standalone server 上测出 `eta`。顺带把 3.7x 的 gap 变成一张按
   `(batch, context)` 排的残差表。
3. **第一级对答案。** 单独接受或否掉 cost model。
4. **补 instrumentation。** `generate` 事件加上 replica id；确认 overlap 开没开；把 scheduler 的
   decode batch log 存档。
5. **接上真 scheduler，再搭 engine**（§7），按这个顺序：
   - (a) 一个进程里能起几个 stub 化的 `Scheduler`，起不了就走多进程；
   - (b) 量一次完整模拟要多久，定下 N；
   - (c) 从 IR 造出 token id，包括 `Prefix`；
   - (d) engine：虚拟时钟、router、资源池、group barrier、训练段。

   §10 那四样机制 SGLang 本来就有，这一步要证明的是 stub 没把它们绕过去。
6. **合成的 held-out workload。** 证明没被带偏（§10）。
7. **第二、三级对答案。** 到这里 simulator 才算验过。
8. **sensitivity table 和带 interval 的 e2e。** 这一步才是交付物，也是第一个完全没有答案的步骤：
   吃一条陌生 trace 加一份 config，出一个带 interval 的 e2e，外加一张表说明哪个估的数撑开了
   interval、去补哪项 instrumentation 最划算。前七步是拿答案换信心，到这一步没答案可换了，靠的
   全是前七步攒下的。

**现在到哪了：** 落地的是 schema 加抽样那一步——`src/simulator/ir.py`、
`src/simulator/realise.py`、`src/simulator/observed.py`、`tests/test_simulator_ir.py`。IR 现在能
说「不知道」，能写并行 tool call，能写 subagent，`max_new_tokens` 可以空着让 config 填（§4）。
第 1 步做了一半：接口在 `adapters/base.py`，读自家 timeline 的 adapter 在
`dev/agentic_timeline.py`，一份 workload 对应一个训练 step。在 run 20260912-072938 的 step 1 上
跑过：2,048 个 session、256 个 group 全部合法，1.8 秒转完；它算出的 structural share 减去 observed
里的 `cached_tokens`，p50 是 114，第一轮的命中率 p10 0.396、p50 0.984，和 §3、§4 的数字对得上。
削 trace 实验已经整条跑完，脚本是 `scripts/simulator_dev/degrade_trace.py`，数据是 run 20261006-030010 的 step 1
（timeline 和对话全文都有，2,037 条 episode；还有 11 条的 event 在 TimelineWriter 的 400 行 buffer 里没
落盘），prior 取 run 20261005-113503 的 step 1，scale 用 20261005-143426 和 20260912-072938 每个 knob 的
均值相对 prior 的比值撑开。结果：

- 全量 timeline、删掉每段活的耗时、只剩对话，这三级都出合法 workload。中间两级（再删 usage、再删
  时间戳）被 adapter 拒掉：token 数估不了，因为相邻 turn 的 prompt 是绑在一起的；没有 trainer 的 gen
  span 就放不进 step。拒绝是如实的，不是算出一堆错数。
- 只剩对话这一级，DAG 形状 2,037 条全部和 timeline 一样，completion 97.8% 一字不差、总量差 0.01%，
  prompt 88% 一字不差（差的全是 completion 误差的累加），第一轮 prompt 和 group prefix 全对。文本
  读不出来的还是那几样：属于哪个 step（对话 dump 里有 2,208 条，160 条 validation 得靠 timeline 才
  剔得掉）、cap、停止原因、tool 有没有超时；observed 那半边是空的，所以这一级没法自己对答案。
- interval 的检查没有全过。带 scale 时 task 总耗时、eval、per-session p90 盖住了全量那级的点，但
  tool 和 container_start 的 interval 整个在点的上方：这次 run 的 tool 和 eval 比之前三个 run 都快，
  container_start 和最快的那个持平（均值比 0.78 到 0.97），而 scale 是拿那几个 run 撑出来的，下沿
  要么没够到要么正卡在边上。不带 scale 的 control 一个总量都盖不住，证实了只靠逐节点抽样的
  interval 不是预测。
- 删掉耗时那级和只剩对话那级的 interval 一样宽：文本把结构读得一字不差，所以再往下削掉的不是
  精度，是 step 归属、答案和 timeout 标记。
- 逐节点独立抽样丢掉了 session 内部的相关性（同一个 instance 的 eval 和 tool 是一起长一起短的），
  所以总量盖住了，per-session 的 p50 还是偏低、max 偏高。

report 在 `analysis/20261006-030010/degrade/step1/report.md`。下一步不是给中间两级补一个 timeline 加
文本的混合 adapter（只剩对话那级已经证明文本够用），而是 scale 的来路：两个 run 不够，要么多拿几个
run，要么把它当成 config bundle 里要人填的 claim。两条现在都有了：默认 prior 的 scale 用三个 run
撑开，030010 也在里面；用户也可以在 bundle 里自己写 scale（§5）。config bundle 只落地了 prior 这
一项，其余几项和 engine 都还没有。

---

## 16. 数据出处

上面所有实测数字都来自 run `20260912-072938`（Qwen3-30B-A3B，TP=4 × 2 个 replica，8× B200），
算自 `/data0/shared/kewen.liu/agentic-timelines/20260912-072938/`：

- `timeline-13267*.jsonl`：99,646 个 `generate` 事件，95,403 对相邻 turn。第一轮的 prefix 数字（§4）
  来自其中 `turn == 1` 的 4,243 个事件。
- `srv-metrics-1312837.jsonl`：7,193 个有 running batch 的采样点。
- 一步的时间分配来自 `cat="train"` 的 span：step 5072s / 4377s，`gen` 2418s / 1824s，
  `update_actor` 1796s / 1718s，`update_weights` 59s / 60s。

`eta ≈ 0.12` 是推出来的，不是直接测的：每个 replica 1.84M token 的 KV，每 token 96KB，分到 4 张卡上
每张约 43GB，按 8TB/s 读要 5.4ms，实测一步 44.5ms。

顺手查清的一件事：`is_cuda_graph` 这个 gauge 在 7,193 个采样里全是 0，但 SGLang 0.5.8 里根本没有
地方给 `stats.is_cuda_graph` 赋值——它是个死 gauge，不是证据。CUDA graph 是开着的，5.4ms 和 44.5ms
之间那 8 倍的差距不是它造成的。
