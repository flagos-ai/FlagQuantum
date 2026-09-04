# Runtime 解耦盘点

状态：Runtime 团队盘点，待集成团队审阅

共同基线：`d7c56603e363bba95d5e98b9a77a75adb3c52e0d`

路径分类：本轮仅增加文档与特征测试，不改变任何执行语义；所覆盖的本地路径为
`single_device_fast_path`，轨迹检查点样例明确记录为
`replicated_per_rank`，不构成分布式可扩展性证据。

## 1. 结论

Runtime 当前同时承担了三种不同层次的工作：

1. 已经接近目标边界的“一次执行尝试”，例如 `execute_plan()` 对既定计划做环境
   校验、执行和结果规范化；
2. 尚未统一的资源、分布式、训练、检查点、恢复和证据组件；
3. 为兼容历史入口而直接调用 Compiler 实现的编译、路由、噪声 lowering、模式选择
   和计划构建逻辑。

`architecture.toml` 当前允许 10 个 Runtime 路径直接依赖 Compiler。逐项检查表明：

- 3 条主要是共享数据契约放错层；
- 2 条主要是编译服务调用没有经过稳定服务边界；
- 2 条主要是 Runtime/Simulation 规划或校验代码直接调用 Compiler 内部算法；
- 3 条主要是噪声或路由能力仍留在历史目录和入口造成的复合耦合。

上述“主要分类”按文件计数；同一文件可能同时包含共享类型和服务调用，详见第 3 节。
本轮不移动类型、不复制 Compiler 类型，也不修改 Simulation、Platform 或 Execution
Provider 的暂管区域。

最容易作为下一阶段首个移除候选的是
`flagquantum/runtime/result.py -> flagquantum.compilation.models.ExecutionPlan`：它只在
`TYPE_CHECKING` 分支中存在，不触发运行时导入，也不承担编译行为。移除前应由 Core
通过批准的 API 变更提案提供最小的 `ExecutablePlanContract`（或等价的 Core-owned
协议），Compiler 的 `ExecutionPlan` 通过适配/一致性测试满足该契约，Runtime 的
`ExecutionResult.plan` 再只引用 Core 契约。不得以 `Any`、自由格式字典或 Runtime
私有复制类型替代。

## 2. 当前执行生命周期入口

### 2.1 规划

| 入口 | 当前责任与事实 | 边界判断 |
| --- | --- | --- |
| `fq.plan()` / `flagquantum.compilation.planner.plan()` | 生成稳定 `ExecutionPlan`，附带程序、选项、环境、编译器指纹和最终决策 | 属于 Compiler 服务；Runtime 应只消费结果 |
| `flagquantum.runtime.execution.run()` | 对程序输入先调用 `plan()`，对计划输入直接进入 `execute_plan()` | 便捷编排入口混合了“请求编译”和“执行尝试” |
| `run_native()` / `run_distributed()` | 兼容入口内部调用 `compile_for_backend()`、`select_execution_mode()`、`plan_advanced()` | 历史耦合；最终应接收可执行计划或显式调用 Compiler 服务端口 |
| `flagquantum.runtime.planning` | 导出 observable grouping 与 hybrid parallel 规划 | Runtime 自有的执行组织规划，不应包含编译变换 |
| `flagquantum.runtime.planner_adapter` | 暴露给 Compiler 的能力查询和 JAX 训练规划适配 | 当前唯一声明的 Compiler→Runtime 窄缝；应保持只查询、不执行 |

### 2.2 单次执行

| 入口 | 当前行为 | 证据/缺口 |
| --- | --- | --- |
| `flagquantum.runtime.plan_execution.execute_plan()` | 校验计划环境，读取计划程序/决策，调用一次 `run_native()`，拒绝执行器替换计划，统一输出 | 已具备“一次尝试”核心形态；计划读取和验证仍依赖 Compiler 内部模块 |
| `flagquantum.runtime.execution.run()` | 稳定入口：程序先规划，计划直接执行 | `fq.run(plan)` 不重新规划或编译，已有特征测试固化 |
| `run_advanced()` | 实验入口，允许后端专用控制并规范化结果 | 不是稳定长期任务接口 |
| `run_native()` | 本地 statevector、MPS、TN、噪声和 JAX 等模式分派 | 文件过大且同时做编译、选择与执行，是主要解耦对象 |
| `run_distributed()` / `DistributedExecutor` | 分布式 statevector 入口及本地开发模拟分派 | 必须继续区分 `single_device_fast_path`、开发语义证据和真实 sharding |
| `run_target()` | 稀疏目标驱动选择与执行 | 暂归 Execution Provider 团队，本轮只盘点不修改 |

### 2.3 Session 与资源

当前没有面向完整执行生命周期的通用 `Session`、`RealtimeSession`、资源租约或
`ExecutionAttempt` 状态机。现有同名概念是：

- `OperatorBackendSession`：`operator_backend()` 上下文管理器返回的可选算子后端启用
  状态，只覆盖进程内临时替换；
- `runtime_config()`、`runtime_backend()`、`runtime_dtype()`：进程内配置上下文；
- `init_torch_distributed()` / `destroy_torch_distributed()`：分布式进程组资源入口。

因此不能把现有 `OperatorBackendSession` 宣称为产品级 Runtime Session。目标 Session
应只管理被 Compute Service 租给某次或一组尝试的短期资源、缓存、设备上下文和清理，
不管理租户、队列、计费或长期任务状态。

### 2.4 分布式组织

- `runtime.distributed.protocols` 定义 `DistributedExecutionRequest`、
  `DistributedExecutionRecord` 和执行器协议；这些仍是内部、较弱的记录形状。
- `runtime.distributed.models` 负责进程组、rank placement、shard/task ownership、通信层级
  与汇总元数据。
- `runtime.distributed.engine.run_distributed_mps()` 组织 MPS 分布式执行；数值算法目录的
  归属迁移仍需与 Simulation 团队分开处理。
- `runtime.distributed.tensor_network_execution` 组织切片任务和归约，但仍直接依赖
  Compiler 的 TN 内存校准记录。
- 分布式结果必须继续报告 `world_size`、`local_world_size`、`node_count`、rank ownership、
  memory、communication、`distribution_semantics`、`scalability_claim_allowed` 和 blockers。
  CPU 特征测试只证明契约语义，不证明多卡容量扩展。

### 2.5 训练

- 稳定入口 `fq.train()` / `runtime.training.train()` 组织 PyTorch 优化循环：
  `Module.execute()` → objective → backward → optimizer step → detached result/callback。
- `runtime.module.Module.execute()` 是 PyTorch-facing 单步执行入口。
- `runtime.mps_training` 和各后端训练实现存在并行路径；`runtime/backends/**` 暂归
  Simulation，本轮不修改。
- 当前训练入口没有统一生成 `ExecutionRecordContract`，也没有把每个训练 step 明确关联
  到一次 `attempt_id`。这是生命周期证据的主要缺口。

### 2.6 检查点与恢复

| 入口 | 当前能力 | 限制 |
| --- | --- | --- |
| `save_training_checkpoint()` / `load_training_checkpoint()` | 原子保存模型、优化器、IR/工作负载身份、精度、随机数、拓扑与 runtime plan；恢复失败时回滚内存状态 | 没有长期任务/尝试身份与外部持久化引用契约 |
| `save_trajectory_checkpoint()` / `load_trajectory_checkpoint()` | 安全的 tensor/primitives-only 原子轨迹检查点 | 只覆盖轨迹执行 |
| `TrajectoryCheckpoint.pending_ids()` | 根据完成与失败记录恢复 rank-owned 待执行轨迹，可选择是否重试 retryable failure | 是局部恢复机制，不是长期任务重试调度器 |
| `merge_trajectory_checkpoints()` | 合并互斥 rank-local 进度和在线统计 | 不代表跨尝试任务状态机 |

当前“恢复”主要等于从版本化检查点重建局部执行状态。没有统一的 attempt retry、资源
重绑定、幂等键、取消、超时升级或跨进程持久任务恢复入口。

### 2.7 观测与结果汇总

- `StrictExecutionScope`、`RouteExplanation`、`FallbackEvent`：强制回退策略并保留路由
  审计轨迹；host debug fallback 会取消 production eligibility。
- `PerformanceMonitor`、`classify_no_progress()`：采样性能与停滞分类。
- `RuntimeProvenance`、`create_evidence_artifact()`、`verify_evidence_artifact()`：不可变、
  可签名的运行证据封装。
- `record_execution()`：基于 Core 的 `RuntimePlanContract`、observation、ownership、
  measurement、failure 和 provenance 构造 `ExecutionRecordContract`。
- `_normalize_execution_output()`、`normalize_execution_result()`、
  `execute_measurements()`：把后端输出汇总为稳定 `ExecutionResult`。
- `TensorWelford`、`merge_trajectory_statistics()`：轨迹在线统计和跨 rank 汇总。

关键缺口是上述组件尚未由一个权威 attempt coordinator 串联：成功的 `fq.run()` 返回
`ExecutionResult`，异常路径抛出 `ExecutionError`，而 `ExecutionRecordContract`、回退
事件、检查点引用和签名证据仍需调用者分别组装。

## 3. Runtime→Compiler 依赖逐项分类

分类含义：

- **共享数据契约**：Runtime 与 Compiler 都需要，但权威定义不应属于 Compiler 实现；
- **编译服务调用**：行为确属 Compiler，Runtime 应通过稳定服务端口请求，而非导入实现；
- **错误的内部实现调用**：Runtime/执行规划直接复用了 Compiler 私有算法或验证实现；
- **噪声或路由的历史耦合**：目录和兼容入口遗留导致多个责任混在一个文件。

| `architecture.toml` 登记路径 | 实际依赖 | 主要分类 | 目标替代方式 |
| --- | --- | --- | --- |
| `runtime/backends/density_matrix/execution.py` | `lower_noise_model()`；类型 `NoisyExecutionPlan` | 噪声历史耦合（同时含编译服务与共享契约） | Simulation 执行器只接收已 lowering 的 channel IR 和 Core-owned `NoiseExecutionPlanContract`；兼容入口在边界外调用 Compiler noise-lowering service |
| `runtime/backends/statevector/noisy.py` | `lower_noise_model()` | 噪声历史耦合（编译服务调用） | 轨迹执行器接收已 lowering IR；由 Compiler 服务生成，Simulation 不直接导入 Compiler |
| `runtime/backends/statevector/planning.py` | `schedule_layers()` | 错误的内部实现调用 | Compiler 在 executable plan 中提供稳定 layer/dependency schedule；Simulation/Runtime 不重跑 Compiler 调度算法 |
| `runtime/distributed/tensor_network_execution.py` | `TNWorkingSetCalibration` | 共享数据契约 | 将版本化校准记录的最小只读契约置于 Core；校准构建仍由 Compiler/benchmark owning service 完成，Runtime 只验证适用范围并消费记录 |
| `runtime/dynamic/routing.py` | `CouplingMap`、`route_to_topology()` | 路由历史耦合（共享契约 + 编译服务） | Core 提供 topology/coupling 数据契约；Compiler routing service 接收动态 IR 并返回已路由 IR，Runtime 只执行 |
| `runtime/execution.py` | `ExecutionPlan`、`compile_for_backend()`、`select_execution_mode()`、`plan_advanced()`、`plan()`；noise plan/lowering helpers | 编译服务调用（复合） | 把便捷的 program→plan 调用收束到一个 Compiler service port；attempt path 只接收 Core-owned executable plan view。自动模式、编译、噪声 lowering 均在尝试开始前完成 |
| `runtime/noise_registry.py` | `EvolutionSemantics`、`StateRepresentation`、`NoisyExecutionPlan` | 噪声历史耦合（共享契约） | Core 提供 backend-neutral noise execution decision/enum 契约；registry 只按契约解析执行器，不认识 Compiler 类型 |
| `runtime/plan_execution.py` | `ExecutionPlan`；`plan_program()`、`plan_decision()`、`plan_noise_model()`、`validate_plan_environment()`、`ExecutionPlanContractError` | 错误的内部实现调用（同时含共享契约） | Core executable-plan contract 负责严格反序列化和只读字段；Runtime-owned preflight 负责环境/资源校验；跨层错误使用 Core/公共错误契约。不要把当前 Compiler helper 复制到 Runtime |
| `runtime/result.py` | `TYPE_CHECKING` 下的 `ExecutionPlan` | 共享数据契约 | `ExecutionResult.plan` 改为批准的 Core-owned executable-plan contract/protocol；Compiler plan 通过适配和 conformance test 满足它 |
| `runtime/target_execution.py` | `BackendSelection`、`select_backend_by_cost()` | 编译服务调用（同时含共享决策契约） | Compiler service 返回 Core-owned backend-selection decision；Execution Provider 执行已选目标，不在执行文件中导入选择算法 |

### 3.1 不可接受的替代

- 在 Runtime 中复制 `ExecutionPlan`、`NoisyExecutionPlan`、`BackendSelection`、
  `TNWorkingSetCalibration` 或 `CouplingMap`；
- 用自由格式 `dict[str, Any]` 隐藏跨层契约；
- 为通过架构检查而新增 Runtime→Compiler re-export；
- 把 `schedule_layers()`、routing、noise lowering 或 backend cost selection 复制为 Runtime
  私有实现；
- 直接修改 `architecture.toml` 白名单而没有先落地替代契约和 conformance tests。

## 4. “一次执行尝试”与 Compute Service“长期任务”的边界

```text
Compute Service durable task
  ├─ tenant/auth/quota/budget/queue/cancel/retry policy
  ├─ task_id + idempotency + durable attempt history
  └─ leases one attempt request
       ↓
Runtime execution attempt
  validate immutable executable plan
  acquire bounded session/resources
  execute exactly once
  observe routes/fallback/memory/communication/progress
  emit result OR typed failure + optional checkpoint
  assemble attempt evidence and release resources
       ↑
Compute Service persists outcome and decides whether/when to create another attempt
```

Runtime 尝试的建议边界：

- 输入是不可变、已编译、可验证身份的 executable plan，加一次尝试的资源租约和
  `attempt_id`；
- Runtime 可以在单次尝试内部做有限的 kernel 重试或 collective 恢复，但必须由计划/策略
  明确授权并完整披露；
- 尝试结束条件是成功结果、类型化失败、可恢复检查点或取消后的清理完成；
- 每次尝试必须输出 observation、ownership、fallback/degradation、failure、checkpoint
  reference 和 provenance，不能只在成功路径返回 tensor；
- Runtime Session 是有界资源复用单位，可以服务多个连续 attempt，但其生命周期受外部
  lease 约束，不成为租户任务数据库。

Compute Service 长期任务的责任：

- 租户、身份认证、授权、配额、预算、计费和队列优先级；
- 持久 `task_id`、幂等键、跨 attempt 状态历史、重试/退避/取消策略；
- 在节点故障或服务重启后选择检查点并创建新 attempt；
- 汇总多个 attempt/provider job，向 MCP/REST/gRPC 暴露长期协议状态；
- 记录 provider job identity，但不把它伪装成 Runtime `attempt_id`。

Runtime 不应复制这些长期控制面能力；Compute Service 也不应绕过 Runtime 的执行计划
校验、回退披露和证据组装。

## 5. 特征测试证据

新增 `tests/team/runtime/test_execution_lifecycle_characterization.py`，覆盖：

| 场景 | 固化的当前行为 |
| --- | --- |
| 成功 | `fq.run(plan)` 使用同一 plan，既不重新规划也不重新编译，并返回规范化 result summary |
| 失败 | 一次 kernel 异常只启动一次，包装为 `ExecutionError` 且保留原始 cause；Runtime 不在暗中重试 |
| 回退披露 | host debug fallback 被记录，取消 production eligibility，并进入 Runtime provenance |
| 检查点/恢复 | 轨迹检查点原子保存并恢复 identity、完成进度、retryable/terminal pending IDs 和分布语义元数据 |
| 结果证据 | `record_execution()` 对同一 plan 保留成功 measurement 证据或失败 code/retryability/blockers |

这些测试是当前行为的特征证据，不宣称已经存在统一 attempt coordinator，也不构成 GPU、
多节点或可扩展性发布证据。

## 6. 下一阶段首个候选与所需 Core 契约

首个候选：移除 `runtime/result.py` 对 `compilation.models.ExecutionPlan` 的类型依赖。

建议由集成/Core 团队先批准并提供最小 `ExecutablePlanContract`：

1. Core-owned、版本化、严格字段和稳定身份；
2. 至少暴露 `plan_id/identity`、可执行 `CircuitIR`、最终 execution decision、环境要求、
   extensions 和 fingerprints；
3. 允许 Compiler 的具体 `ExecutionPlan` 通过适配器满足，不要求 Runtime 认识 Compiler
   类；
4. 提供序列化 round-trip、未知字段/版本 fail-closed、Compiler 实现替换 conformance；
5. 明确 `ExecutionResult.plan` 的兼容策略。由于这是 Stable Core 公开结果字段的类型边界，
   必须遵循 `PUBLIC_API_PROTECTION.md` 的 API change proposal，不能直接改注解或快照。

后续契约（不属于首个候选的前置条件）应按独立提案拆分：

- `ExecutionAttemptContract`：`attempt_id`、plan identity、ordinal、状态、observation、
  ownership、failure、fallback、checkpoint reference、provenance；
- `NoiseExecutionPlanContract`：噪声 representation/evolution/parallel/memory 决策；
- topology/coupling contract；
- backend-selection decision contract；
- TN working-set calibration record contract。

只有 Core 契约、contract fake 和 conformance test 先在集成分支落地后，Runtime 才应同步
基线并实施依赖移除。
