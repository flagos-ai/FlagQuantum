# Runtime / TargetCapabilities 需求映射

状态：Phase 2 Runtime 团队提案与特征测试，不是公共契约实现

基线：`vnext-phase1-contract-foundation`

适用范围：Runtime 规划、环境匹配、执行前检查与执行证据

## 结论

Runtime 需要消费两个不同方向的输入：一边是“这次工作负载必须满足什么”，另一边是“当前可租用目标已经证实能提供什么”。前者应形成 `RequirementSet`，后者应形成 `CapabilitySnapshot`。两者不能继续共用一个布尔型 capability 对象，也不能由 Runtime 用用户期望、环境变量或算法估算补写硬件事实。

Runtime 的权威职责是：基于 requirement、snapshot、资源租约与显式策略，选择可执行候选，执行一次尝试，并记录实际路线和结果证据。Compiler 负责程序合法化与编译需求；Platform/Provider 负责发现事实；Simulation 负责算法候选和成本提示；任何一方都不应替 Runtime 做最终资源选择。

本轮没有修改 Stable Core、公共 API 或序列化契约。文中类型名仅表示下一阶段应由 Integration 批准的契约形状。

## 1. 当前入口与隐式 requirements

| 关注点 | 当前入口 | 当前表达 | 风险或缺口 |
| --- | --- | --- | --- |
| 用户执行意图 | `runtime/options.py::ExecutionOptions` | mode、backend、device、target、batch、precision、shots、seed、memory limit、gradients、approximation、backend fallback | `precision` 混合了存储 dtype、有效精度和原生精度；没有单独 CPU fallback、动态电路、恢复或实时要求 |
| 选项归并 | `runtime/options_resolver.py` | defaults → config → program constraints → runtime policy → call | IR dtype 被投影为 precision，但来源和约束强度没有进入匹配证据 |
| 公共规划入口 | `flagquantum/api.py`、`runtime/planner::plan`、`runtime/execution.py` | 解析 options、world size、backend，并调用 Runtime planner | 规划、环境推断和执行组织尚未形成明确的 requirement/snapshot seam |
| 执行计划环境 | `compilation/execution_plan_contract.py` | backend、device kind、precision、world size、distribution semantics、gradients、approximation | 仅由 `world_size > 1` 推断 `sharded_across_ranks`；缺少实际 device count、memory、topology、communication、shots/dynamic、checkpoint/realtime |
| 后端发现 | `runtime/backend_registry.py` | devices、dtypes、autograd、distributed、statevector/density/MPS、preferred device、accelerator memory | 多数是布尔值或静态声明；没有 unknown/unmeasured/not_exposed；PyTorch 的宽泛声明容易被误读为目标事实；`flagos` 路径有设备支持绕过 |
| 编译目标能力 | `_compiler/target_capabilities.py` | gates、results、formats、topology、dynamic/timing/pulse、shot/program limits 等 | required 与 available 使用同一类型；`False`、`None` 和缺失无法表达事实状态与暴露状态 |
| 能力比较 | `_compiler/capability_comparison.py` | 对上述对象做 required/available 比较 | 比较逻辑本身偏 fail-closed，但类型无法区分“未知、未测、不支持、未暴露” |
| 设备与 dtype 检查 | `runtime/backend_registry.py::backend_execution_options`、`compilation/execution_plan_contract.py::validate_plan_environment` | device、dtype、mode、world size | Compiler 层反向导入 Runtime discovery；当前 dtype 检查只覆盖声明集合，不说明 native/software mechanism |
| 分布式策略 | `runtime/distributed/backend_policy.py` | profile、JAX/Torch backend、local/effective world size、torchrun/GPU policy | 环境变量和进程组只能说明当前编排状态，不能证明物理 topology、通信能力或真正分片；development LocalTensor 是模拟事实 |
| 内存与候选成本 | `runtime/planner/backend_selection.py`、`candidates.py`、`candidate_plans.py` | statevector/MPS/TN 估算、候选 costs、memory/communication/gradient plans | 工作负载估算与可用设备内存容易混淆；按 world size 除内存不能证明资源已分配或通信可行 |
| 动态电路 | `runtime/dynamic_conformance.py`、`runtime/dynamic/deployment.py` | mid-circuit measurement/reset、feed-forward、结果返回、provider dynamic support | 当前是局部 conformance/deployment 检查，尚未并入统一 RequirementSet/Snapshot 匹配 |
| shots/trajectory | `ExecutionOptions.shots`、`runtime/trajectories/**` | shots、trajectory ownership、failure、statistics、checkpoint | shots 是 workload requirement；trajectory 完成情况是 attempt evidence，不能写入 target capability |
| 训练与精度 | `runtime/training_state.py::PrecisionPolicy`、各训练 engine | complex/parameter/accumulator dtype、mixed/full、downcast、梯度与 optimizer state | 执行 options 和训练 precision policy 有两套表达；software-expanded precision 尚无统一 mechanism 字段 |
| 检查点与恢复 | `runtime/training_state.py`、`runtime/trajectories/checkpoint.py`、各训练 engine | schema/version、拓扑、IR hash、RNG、原子提交、空间 preflight、writer lease、resume | “需要可恢复”未进入通用规划 requirement；checkpoint storage/compatibility discovery 也没有统一 snapshot |
| 实时 session | `_compiler/TargetCapabilities` 的 realtime/adaptive 字段及少数 provider/dynamic 路径 | adaptive/realtime 布尔与 latency 上限 | Runtime 尚无通用 `RealtimeSession` 生命周期；未测 latency 不能作为可满足事实 |
| 回退与路线 | `runtime/fallback.py`、`runtime/routing.py`、result/evidence 字段 | policy、route category、fallback event、host debug policy | backend fallback 与 CPU/algorithm/precision fallback 需要分轴授权；“未看到事件”不能证明未发生回退 |
| 执行后证据 | `runtime/observability/evidence.py` 及 backend result records | actual provenance、fallback events、分布式/内存/通信/结果元数据 | 证据仍分散；必须绑定 attempt 和实际资源，且不能自动升级成永久 capability fact |

### 1.1 Runtime 真正需要表达的 requirement 维度

最小集合应按八组稀疏维度组织；调用未涉及的字段不应强制出现：

1. **workload**：执行 mode/representation、程序特征、exact/approximation 语义、允许的 truncation/error bound。
2. **numeric**：storage dtype、effective precision、是否要求 native precision、parameter/accumulator dtype、容差与允许的软件扩展机制。
3. **measurement**：shots、结果类型、动态电路、mid-circuit measurement/reset、feed-forward、timing/pulse/noise 要求。
4. **compute**：device kind、最少或精确 device count、每设备/聚合 memory 下限、用户 memory budget、临时空间。
5. **fabric**：node/local device count、placement/topology 约束、通信 primitive/backend/dtype、route visibility。
6. **training**：gradient、reverse path、optimizer/update distribution semantics、参数绑定与 batch 要求。
7. **continuity**：checkpoint 能力、存储一致性、restart compatibility、恢复上限、seed/determinism、batch/session/realtime 与 latency 上限。
8. **authorization/evidence**：允许的 approximation、backend/device/CPU/precision/algorithm fallback，以及必须达到的 evidence/claim level。

这不是要求建立一个永久的巨型配置对象。最小 `RequirementSet` 应仅携带本次执行有意义的谓词、来源、强度（mandatory/preference）和授权；缺省授权一律为禁止。

### 1.2 用户 requirement 与 Compiler requirement 不合并来源

两者都进入同一个稀疏 `RequirementSet`，但每个谓词必须保留独立来源，不能把 Compiler
推导伪装成用户授权，也不能把用户偏好升级为程序合法性约束：

| 来源 | 可产生的内容 | 不可产生的内容 |
| --- | --- | --- |
| 用户/request adapter | target、shots、预算、期望 device/backend、精度目标、deadline、是否需要 realtime/checkpoint，以及 approximation 和各类 fallback 的逐轴授权 | native gates、算法可行性、已分配 device/memory、实际 route |
| Compiler | IR dtype 与 shape 约束、native gate/result/control-flow 要求、参数绑定、合法 topology predicate、梯度/输出语义、编译后 workspace 或 artifact 要求 | CPU/精度/算法 fallback 授权、平台可用性、实测容量/延迟、最终 placement |
| Runtime protocol | 一次 attempt 所必需的 lease、route visibility、checkpoint compatibility、evidence level 与清理条件 | 改写用户授权、放宽 Compiler 合法性、生成平台事实 |

冲突时 mandatory requirement 取交集中的更严格约束并保留双方 provenance；互不相容则在匹配前失败，不能由 policy 猜测。当前 `ExecutionPlan` 只保存 resolved options 和 decision，丢失了逐字段原始来源，因此测试 adapter 只能把输出标记为 `source=compiler` 的 plan projection，不能声称重建了用户 request。最终契约需在 request→plan 过程中保存来源和授权链。

### 1.3 逐维对账

| 维度 | requirement 谓词示例 | snapshot 事实/门槛 | attempt evidence |
| --- | --- | --- | --- |
| dtype/precision | storage、parameter、accumulator、effective、native-required、允许的软件扩展机制 | 支持的 dtype/机制；native 与 effective 分轴；误差门槛需要匹配 scope 的验证 | 实际 storage/accumulator dtype、mechanism、误差界、任何 downcast |
| memory | required state/workspace/peak、每设备或聚合口径、用户 budget | 对同一 lease/scope 的 available/reserved memory；`unmeasured` 不满足硬容量要求 | 实际 peak、reserved、OOM/preflight、估算与实测差异 |
| device count | minimum/exact、local count、node count | 已分配 lease 的 verified count；环境变量/world size 不是设备事实 | actual ranks/devices/nodes 与 rank ownership |
| topology | coupling、placement、互联/NUMA、route visibility | scope 一致的 topology/route fact；`unknown`/`not_exposed` 失败关闭 | 实际 placement、links、route 与偏离 |
| communication | 所需 primitive/backend/dtype、host staging 禁止或授权 | 对具体组合的 verified 支持；接口存在只算 unmeasured/declared | collective/P2P、bytes、时间、backend、host staging/fallback |
| shots | exact/min/max shots、batching、seed/determinism | provider maximum、batch limit、采样/RNG 能力 | requested/completed shots、丢失/重试、seed 与统计汇总 |
| dynamic | mid-circuit measurement/reset、feed-forward、timing/pulse、latency bound | 每项独立状态；adaptive realtime 不能由单一 boolean 推导 | 分支/测量/reset 轨迹、控制延迟、实际 provider route |
| checkpoint/restart | 是否必须 checkpoint、频率/一致性、格式版本、restart topology/dtype compatibility、恢复次数 | storage/atomicity/format/compatibility 事实；本地 helper 存在不等于目标可恢复 | checkpoint identity、commit、base attempt、恢复结果/失败 |
| realtime/session | session lifecycle、最大端到端/控制延迟、持续时间、资源独占 | session capability 与匹配 workload 的 latency measurement；未测 latency 失败关闭 | session/lease identity、实际延迟分布、超时、重连/降级 |

`runtime/planner/backend_selection.py` 的 `estimated_memory_bytes`、结构分数和候选
`available` 只表达算法候选在给定假设下是否可行；它们不是上述 snapshot 的平台容量事实。
同理，execution plan 中的 `world_size` 是要求/决定，不能证明设备已分配或实际发生 sharding。

## 2. 四类信息必须分层

| 层 | 权威生产者 | 内容 | 明确禁止 |
| --- | --- | --- | --- |
| `RequirementSet` | 用户 + Compiler；Runtime 仅补充自身执行协议所需约束 | 程序语义、数值要求、资源下限、连续性要求、显式授权 | 写入 `available_memory`、真实 topology、设备存在、实测 latency 等事实 |
| `CapabilitySnapshot` | Platform/Provider discovery；必要时受控 probe | 某个 target/lease 在某时刻、某作用域内的已发现事实，含来源、状态、暴露与新鲜度 | 用用户 device/world size 期望、Compiler estimate 或 Runtime 默认值伪造硬件事实 |
| Runtime policy/decision | Runtime | 候选过滤、优先级、租约、placement、partition、fallback 决策及 blockers | 改写 Compiler 合法性、Simulation 成本模型，或把策略选择说成平台能力 |
| execution evidence | Runtime + owning backend/provider | 实际设备、route、rank ownership、memory/communication、fallback、checkpoint、结果与失败 | 未经校准/审查直接回填为长期 capability；用计划意图代替实际发生 |

必须保留 provenance：requirement 至少记录来源；snapshot 至少记录 target identity、provider/platform identity、采集时间、版本、scope 和 probe/声明来源；decision 记录使用的 requirement/snapshot identity；evidence 绑定 execution attempt identity。

## 3. 最小匹配语义

### 3.1 状态与暴露是正交轴

每个 capability fact 至少需要一个支持状态：

- `unknown`：权威来源没有答案。mandatory requirement 失败关闭；preference 只能让候选降级排序，并保留 blocker。
- `unmeasured`：接口或声明表明该能力可能存在，但缺少满足当前阈值的测量。涉及容量、性能、实时延迟、有效精度、通信或“没有回退”的硬条件时失败关闭。仅在用户明确授权的开发/调试降级下可尝试，并降低 claim ceiling。
- `unsupported`：权威来源明确不支持。当前候选必须拒绝，Runtime policy 不得覆写。只有另一个候选且对应 fallback/approximation 已显式授权时才能继续。
- `verified`：存在符合 scope、新鲜度和谓词的证据。它只表示本次匹配可用，不自动等于 release-certified 或 scalability evidence。

另设 exposure：`observed`、`declared`、`not_exposed`、`unknown`、`not_applicable`。`not_exposed` 不是 `False`、`0` 或“不支持”。如果 requirement 要证明实际 route、无 CPU fallback、通信路径、可用内存或 latency，`not_exposed` 必须失败关闭；调试降级必须有显式授权、blocker 和不可发布的 claim ceiling。`not_applicable` 必须携带原因。

### 3.2 匹配规则

对每个候选，Runtime 应按以下顺序处理：

1. 先校验 requirement 来源和内部一致性，不把 requirement 变成 snapshot。
2. 校验 snapshot identity、scope、版本和 freshness；过期或对象不匹配等价于不可用于证明。
3. 对每个 mandatory predicate，要求存在相应 fact，且状态/暴露满足该谓词自己的 evidence threshold；例如不可变规格可接受带权威 provenance 的 `declared`，容量、延迟、route 和“无 fallback”通常要求 `observed`。`verified` 不会把低于门槛的 exposure 自动升级。
4. 比较值与机制。集合用包含关系，数值容量用带单位的上下界，topology/route 用明确语义谓词，不能只比较字符串。
5. 区分 native 与 effective：软件扩展可满足 `precision.effective=complex128`，但绝不能满足 `precision.native=complex128`；结果证据必须记录 mechanism、storage/accumulator dtype 和误差界。
6. `unsupported` 立即淘汰当前候选；`unknown`、`unmeasured`、`not_exposed` 对 mandatory requirement 产生带原因 blocker。
7. preference 只影响可执行候选排序，不得把不可执行候选变为可执行。
8. 应用 fallback 前检查对应的独立授权，并使用替代候选自己的 snapshot/lease 对全部仍适用谓词执行完整匹配。不得复用原候选事实，也不得用一个宽泛的 `allow_backend_fallback` 推导 CPU、精度或算法降级授权。
9. 输出 decision：选中的 target/lease、backend、mode、placement/partition、所用 snapshot、降级、blockers 与 claim ceiling。
10. 执行后用 actual evidence 对 decision 做一致性核验；实际 route 或分布语义偏离必须失败或形成已授权且显式的 fallback event。

### 3.3 CPU fallback

CPU 必须是一个独立候选，不是设备解析器的默认分支。它只有在以下条件同时成立时可选择：

- 用户或受批准 policy 明确授权 `cpu_fallback`；
- CPU snapshot 对本次 workload requirements 完整匹配；
- plan/decision 明确记录从何种 device/route 回退到 CPU；
- execution result/evidence 记录 actual device、fallback reason/event、精度与分布语义变化；
- 任何 GPU、分布式容量或性能 claim 被删除或降低。

如果 route 为 `not_exposed`，Runtime 无法证明没有 CPU fallback，因此不能发布“GPU executed”或“no fallback”证据。

## 4. Simulation 候选和成本提示的边界

Simulation 可以提供：

- 从 IR/workload 推导的合法 representation/mode 候选；
- statevector、MPS、TN 等算法的内存、通信量、计算量估算；
- partition/slicing/sharding 候选及算法前置条件；
- cost model 版本、假设、置信度、误差范围和未知项；
- 算法不支持、梯度不完整、近似/截断等 semantic blockers。

Simulation 不得决定物理 device/provider、target identity、资源租约、实际 topology/transport、CPU fallback 或 release claim。其 `required_memory_bytes` 是 workload estimate，不是 `available_memory_bytes`；其 `world_size` 候选不是已发现设备数；其 communication plan 不是已验证互联。

Runtime 将这些候选与 `RequirementSet`、Platform/Provider `CapabilitySnapshot`、资源租约和 policy 合并，选择 backend/mode/placement/partition。Runtime 不得修改 Simulation 的算法约束或把低 cost 当成可执行证明。Compiler 仍负责把选定的合法候选编译为目标 artifact，但不反向执行 Runtime discovery。

## 5. 一次匹配到一次尝试的生命周期

```text
user intent + compiler requirements
              |
              v
       sparse RequirementSet
              |
Simulation candidates/cost hints ----+
              |                       |
Platform/Provider CapabilitySnapshot -+--> Runtime match/policy
                                               |
                                               v
                                      decision + resource lease
                                               |
                                      compile/execute one attempt
                                               |
                                               v
                                   actual result + execution evidence
```

长期 Compute Service task 可以包含排队、重试、迁移和多个 execution attempts；snapshot 和 lease 可能在 attempt 之间变化。每个 attempt 必须冻结其 requirement、snapshot、decision、artifact/IR identity、checkpoint base 和 evidence。服务级 retry 不能把前一次 `unsupported` 偷换成 CPU fallback，也不能覆盖失败证据。checkpoint 是 attempt 间的显式连续性契约，不是长期任务天然具备的 capability。

## 6. 特征测试

`tests/team/runtime/test_target_capabilities_runtime_requirements.py` 使用现有 `fq.plan` 作为输入，并用测试侧最小 fake 表达拟议匹配 seam；它没有提前建立公共类型。覆盖：

- 当前 plan 中的 mode/device/precision/device count/memory/gradient/fallback 仍被解释为 requirements，而不是可用硬件事实；
- topology fact 为 `unknown` 或 `unmeasured` 时 mandatory requirement 失败关闭；
- 明确 `unsupported` 的 realtime session 不得被 policy 弱化；
- 软件扩展可满足 effective complex128，但不能宣称 native complex128；
- CPU fallback 必须显式授权、使用独立 CPU snapshot 完整重匹配并产生事件；
- 宽泛的 backend fallback 授权不能推导 CPU fallback 授权；
- route `not_exposed` 不能证明没有 CPU fallback。

这些测试应在 Integration 契约落地后迁移为 contract fake/conformance tests，且至少用两个 snapshot provider 或 fake 替换实现证明边界。

## 7. Integration 必须决策的冲突

1. `RequirementSet`、`CapabilitySnapshot`、fact status/exposure vocabulary 的 Core 所有权、版本和序列化边界；Runtime 不应私有复制这些类型。
2. 当前 `_compiler.TargetCapabilities` 同时扮演 required/available，是否拆分或仅保留编译目标描述，以及迁移兼容期。
3. `ExecutionOptions.precision` 如何兼容拆分为 storage/effective/native、parameter/accumulator 与软件扩展机制。
4. `allow_backend_fallback` 与 CPU/device/precision/algorithm/approximation 授权的关系；缺省必须 fail closed。
5. snapshot 的 target/lease identity、TTL/freshness、provider 与 Platform 的权威优先级，以及 `declared` 能满足哪些谓词。
6. resource lease 和实际 allocation 由 Compute Service、Platform 还是 Provider 提供；Runtime 只能消费有效租约。
7. plan 中 `world_size > 1` 推导 `sharded_across_ranks` 必须移除；何种 Runtime evidence 才能确认实际分布语义和 scalability claim。
8. Compiler 的 `validate_plan_environment` 反向导入 Runtime discovery 的迁移路径；目标应由 Runtime matcher 消费 Compiler requirements。
9. 动态电路、checkpoint/restart、realtime latency 的统一 requirement/fact 字段、scope 和验证门槛。
10. `not_exposed`、`unmeasured` 的调试降级 claim ceiling，以及如何进入计划、结果和审计 payload。
11. Simulation candidate/cost hint 的最小版本化 envelope，以及最终候选排序明确归 Runtime 所有。
12. Stable plan identity、failure stage 和结果 evidence schema 的任何变化都需要 API change proposal、兼容分析与 conformance tests。

## 8. 推荐的第一阶段 Integration 切片

先批准内部（非公共）contract fake：一个稀疏 requirement predicate、一个带 status/exposure/provenance 的 capability fact，以及纯匹配函数。首个垂直切片只覆盖 `device.kind`、`device.count`、`precision.effective/native`、`memory.available_bytes` 和独立 `cpu_fallback` 授权；沿用现有 plan 适配器，不改变 Stable API。

完成标准是：缺失事实失败关闭、软件扩展不冒充原生精度、CPU fallback 有 plan/result 双重证据，并能替换一个 Platform fake 而不修改 Runtime consumer。随后再扩展 topology/communication、dynamic/shots、checkpoint/realtime。
