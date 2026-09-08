# Core Contract Inventory

状态：Core 团队盘点，供集成团队审议；不构成 API 或 ADR 批准

盘点基线：`d7c56603e363bba95d5e98b9a77a75adb3c52e0d`

团队分支：`codex/vnext-team-core`

盘点日期：2026-09-03

## 1. 结论

当前代码已经具备可用的稳定 `CircuitIR`、`ExecutionOptions`、`ExecutionPlan` 和
`ExecutionResult`，也有一组 Core 内部版本化记录契约；但 ARCH-001 所要求的四类跨领域
契约尚未收敛到 Core。主要事实如下：

1. `Circuit` 是稳定的用户构造门面，`CircuitIR` 是稳定且受保护的规范程序表示；两者通过
   `Circuit.to_ir()` / `Circuit.from_ir()` 转换，职责不同，不应合并。
2. `ProgramArtifact` 已在 Core 中出现，但只被 Agent Application Service 消费，Compiler
   主链仍使用自己的 `TargetIR` 和 `SealedExecutableArtifact`。仓库中没有名为
   `ExecutableArtifact` 的类型。
3. `TargetCapabilities` 位于 `_compiler`，与 Runtime 的 `BackendCapabilities`、Deployment
   的 `CloudBackendProfile` 分别描述编译目标、张量后端和云后端，字段有重叠但粒度不同。
4. 仓库存在两套完全不同且同名的 `ExecutionOptions`：稳定用户选项和私有 artifact runtime
   ABI 选项。这是最容易被误导入的重复类型。
5. 稳定 `ExecutionPlan` 与 Core 的 `RuntimePlanContract` 同时表达计划；前者拥有严格、完整、
   可执行且带身份的 schema，后者是由 `ExecutionPlan.to_contract()` 生成的有损审计投影。
6. 稳定 `ExecutionResult` 是用户结果；Core `ExecutionRecordContract` 是审计记录；此外各
   Runtime/Deployment 路径仍返回多个专用结果。专用数值结果可以保留为 Provider 内部对象，
   但跨领域边界应只投影到一个 Core 结果/记录契约。
7. Evidence、Provenance 和 Metrics 尚未形成一套 Core 权威模型：Core 有
   `ProvenanceContract`，Runtime 又有更完整的 `RuntimeProvenance`；`metrics` 目前主要是
   自由格式映射或算法专用 dataclass，仓库中没有通用 `Metrics` 核心类型。
8. Provider 能力与失败语义分散于 `_compiler`、`runtime`、`deployment` 和 `extensions`。
   Provider 实现大量抛出内置 `ValueError` / `RuntimeError` / `KeyError` / `TimeoutError`，尚未
   统一映射到 Core 的失败分类。

因此，当前最小可行策略不是创建一个万能对象，而是先收敛 artifact 身份链，再收敛目标
能力，再定义请求/计划边界，最后统一结果、证据和失败投影。

## 2. 盘点方法与保护边界

本盘点以源码定义、导入点、稳定导出清单、候选合同和架构检查白名单为证据。重点检查了：

- `contracts/public-api-v1-candidate.json` 及 Execution Options/Plan/Result 候选合同；
- `docs/development/PUBLIC_API_PROTECTION.md`；
- `docs/architecture/decisions/ARCH_001_PROVIDER_LAYERS_AND_CORE_CONTRACTS.md`；
- `contracts/long-horizon-architecture-v1.json`；
- `architecture.toml` 中现存 Runtime → Compiler 迁移白名单。

本轮没有修改 `flagquantum/core/_artifacts.py`、`flagquantum/core/ir.py`、公共根导出、
`contracts/`、ADR 或任何序列化实现。新增测试只记录当前可观察行为。

本文使用以下分类：

| 分类 | 含义 |
| --- | --- |
| 受保护公共契约 | Stable Core 或已冻结候选中的导出、签名、字段、行为和序列化 |
| 内部候选契约 | 已有类型和验证，但尚未批准为稳定跨领域权威契约 |
| 临时兼容对象 | 为旧入口、跨层迁移或专用执行路径保留的适配/投影对象 |
| 可删除或合并的重复对象 | 在权威契约落地且调用者迁移后，应退出的同义定义或有损双轨 |

## 3. 定义与消费者

### 3.1 Program / Circuit / IR

| 类型 | 当前定义 | 主要消费者 | 分类与判断 |
| --- | --- | --- | --- |
| `Circuit` | `flagquantum/circuit.py` | 根 API、planner、runtime、deployment、examples/tests | **受保护公共契约**。面向 PyTorch 用户的可变构造/执行门面，不是序列化权威。 |
| `CircuitIR`、`Instruction`、`ObservableNode`、`MeasurementNode` | `flagquantum/core/ir.py` | Compiler、Runtime、Simulation、Deployment、Interop、Agent Services | **受保护公共契约和唯一规范 IR**。`kind=flagquantum.circuit_ir`、版本 `1.0`、规范 JSON 与 SHA-256 内容哈希。 |
| `QuantumModule` 等多级内部 IR | `flagquantum/_compiler/ir/` | `_compiler` analyses/passes/exporters | **内部编译表示**。不得成为 Runtime 或 Provider 的跨领域输入。 |
| `TargetIR` | `flagquantum/_compiler/target_ir.py` | target legalization、artifact sealing、provider conformance | **内部候选编译产物**。当前越过编译阶段进入私有 conformance，但未进入稳定 API。 |
| `DynamicCircuit` | `flagquantum/runtime/dynamic/circuit.py` | dynamic runtime、Qiskit interop | **实验性程序门面**，最终仍投影到 `CircuitIR`；不应成为第二规范 IR。 |

`Circuit` 与 `CircuitIR` 是有意分层，不是重复定义。真正的泄漏风险在于 Compiler 的
`QuantumModule` / `TargetIR` 若直接成为 Runtime 长期依赖。

### 3.2 ProgramArtifact 与 ExecutableArtifact

| 类型 | 当前定义 | 主要消费者 | 分类与判断 |
| --- | --- | --- | --- |
| `ProgramArtifact` | `flagquantum/core/_artifacts.py` | `agent/service.py` | **内部候选契约**，且文件属于受保护集成路径。严格拒绝未知顶层字段和非 `1.0` 版本，规范 JSON 参与 `content_hash`。尚未被 Compiler 主链采用。 |
| `SealedExecutableArtifact` | `flagquantum/_compiler/executable_artifact.py` | compiler runtime ABI/adapters、provider conformance | **内部候选契约**。封装 bytes、profile 和完整身份链；只有 `identity_dict()`，没有通用 `to_dict/from_dict` 往返。目标态应由 Core 拥有。 |
| `DeploymentPackage` | `flagquantum/deployment/cloud.py` | QPU/cloud providers、deployment helpers | **临时领域对象**。携带 `CircuitIR`、QASM、backend profile 与 metadata 内的哈希链；与 executable artifact 重叠，但面向云提交且尚未使用统一 artifact envelope。 |
| `SealedCircuitIRRoundTrip` / `CircuitExportResult` | `flagquantum/_compiler/exporters/circuit_ir.py` | 内部 IR 验证测试与 compiler pipeline | **临时编译结果**。是验证结果，不应提升为跨领域 artifact。 |

建议权威位置：`flagquantum/core/artifacts/`。保留一个小型 `ProgramArtifact` 信封，并定义一个
窄的 executable payload 变体/类型；不要把 Deployment task、credentials、live SDK 对象或
运行结果塞入 artifact。迁移期由 `_compiler.executable_artifact` 和 Deployment adapter
转换，调用者归零后删除私有重复实现。

### 3.3 TargetCapabilities

| 类型 | 当前定义 | 主要消费者 | 分类与判断 |
| --- | --- | --- | --- |
| `TargetCapabilities` | `flagquantum/_compiler/target_capabilities.py` | capability comparison、legalization、deployment compatibility/dry-run、artifact sealing、provider conformance | **内部候选契约**。是当前最完整的编译目标语义快照，严格字段/版本读取，`display_label` 不参与 semantic fingerprint。位置不符合 ARCH-001。 |
| `CapabilityContract` | `flagquantum/core/contracts.py` | `RuntimePlanContract`、plan adapter | **内部候选简化投影**。只有 backend/devices/dtypes/modes 和两个布尔能力，无法替代 `TargetCapabilities`。 |
| `BackendCapabilities` | `flagquantum/runtime/backend_registry.py` | backend selection、API compatibility exports | **Runtime 内部运行环境能力**。包含实时 accelerator discovery，不应直接变成可移植 Target snapshot。 |
| `CloudBackendProfile` | `flagquantum/deployment/cloud.py` | package creation、QuantumProvider implementations | **临时 Deployment 能力对象**。门集、拓扑、动态线路和格式字段与 `TargetCapabilities` 重叠。 |
| `CapabilityEvidence` | `flagquantum/runtime/capabilities.py` | operator probes/preflight | **Runtime 证据记录**，不是 target declaration；应在 Core capability schema 中通过 evidence reference 关联而非合并字段。 |
| `CapabilityRequest/Response` | `flagquantum/ecosystem/extensions/sdk.py` | extension registry/conformance | **待冻结扩展协议**。是能力协商消息，不是 Target snapshot。 |
| `SolverWorkspaceCapabilities` | `runtime/executors/mps/solver_workspace.py` | MPS solver workspace | **算法本地对象**，保留在 Simulation/Provider 内部。 |

建议权威位置：`flagquantum/core/capabilities/` 中的可移植、不可变
`TargetCapabilities`；Runtime 的动态探测通过 adapter 生成 snapshot，`CloudBackendProfile`
通过 Deployment adapter 投影，算法局部能力不并入 Core。

### 3.4 ExecutionRequest 与 ExecutionOptions

| 类型 | 当前定义 | 主要消费者 | 分类与判断 |
| --- | --- | --- | --- |
| `ExecutionOptions` | `flagquantum/runtime/options.py` | 稳定 root API、Circuit、planner、runtime、policy | **受保护公共契约**。字段、默认值、闭集和 schema `flagquantum.execution_options/1.0` 已冻结。 |
| `ResolvedExecutionOptions` | `flagquantum/runtime/options_resolver.py` | plan contract builder | **内部派生值**，记录逐字段来源；不应序列化为独立公共请求。 |
| `RequestedExecution` | `flagquantum/core/contracts.py` | `RuntimePlanContract` | **内部候选简化请求投影**。与稳定 options 字段重叠，缺少 artifact、evidence、timeout/recovery 等完整请求边界。 |
| `InternalExecutionRequest` | `flagquantum/_compiler/import_models.py` | `CircuitIR` importer | **Compiler 内部对象**。实际只分离 observables/measurements/classical width，名称过宽。权威请求落地后可改名为 importer-local request 或被适配。 |
| `DistributedExecutionRequest` | `flagquantum/runtime/distributed/protocols.py` | JAX/PyTorch distributed adapters | **Runtime 临时兼容对象**。`Mapping[str, Any]` options 是跨层逃生口，应在新请求契约落地后退出。 |
| `RuntimeBindings`（私有 ABI） | `flagquantum/_compiler/runtime_abi.py` | compiler runtime adapters/provider conformance | **已消歧的私有绑定对象**。只包含 `shots` 与 parameter bindings，不与稳定 `ExecutionOptions` 形成第二套用户选项。 |
| `SandboxConnectorRequest` | `flagquantum/_compiler/provider_sandbox_connector.py` | provider sandbox tests | **私有测试/演练请求**，不得成为生产 Core 请求。 |

仓库中没有一个名为 `ExecutionRequest` 的 Core 权威类型。建议位置为
`flagquantum/core/execution/request.py`：组合 artifact identity、稳定 `ExecutionOptions`、参数
绑定、目标约束和所需 evidence level。不要复制 planner decision，也不要包含 provider
credentials、process group 或 SDK handle。

### 3.5 ExecutionPlan

| 类型 | 当前定义 | 主要消费者 | 分类与判断 |
| --- | --- | --- | --- |
| `ExecutionPlan` | `flagquantum/runtime/execution_plan.py` + `runtime/execution_plan_contract.py` | stable plan/run/Circuit、Runtime execution、result | **受保护公共契约**。完整 schema、四类 fingerprint、identity 重算、未知字段/版本拒绝和 stale environment 拒绝均已冻结。 |
| `RuntimePlanContract` | `flagquantum/core/contracts.py` | `ExecutionPlan.to_contract()`、runtime records/result | **内部候选审计投影**。转换与计划序列化同属 `compilation/execution_plan_contract.py`，会丢失完整 program、options、environment、compiler 与 extension identity。 |
| `RuntimeSelectionPlan` | `flagquantum/runtime/execution_plan.py` | experimental planning | **内部 planner 决策报告**，不是可执行计划。 |
| `NoisyExecutionPlan`、`TrajectoryPlan`、`ParallelPlan`、`MemoryPlan` | `flagquantum/runtime/execution_plan.py`；由 `runtime/planner` 组装 | noise runtime | **过渡期子系统计划产品**，应嵌入/引用主计划扩展，不替代主计划；噪声 lowering 已归 Compiler。 |
| `DistributedStatevectorPlan`、`HybridParallelPlan`、TN/MPS/JAX plans | `flagquantum/runtime/**` | 各 backend executor | **执行引擎本地计划**。可作为 Provider 内部 lowering 结果；跨领域只暴露主计划和证据。 |

短期权威必须仍是受保护的 `ExecutionPlan`。在没有 API Change Proposal 前，不得用
`RuntimePlanContract` 替换其公开类型、改变 schema 或修改 identity inputs。Core
`RuntimePlanContract` 可暂时保留为只读审计投影，但应明确命名和退出条件，避免双权威。

### 3.6 ExecutionResult

| 类型 | 当前定义 | 主要消费者 | 分类与判断 |
| --- | --- | --- | --- |
| `ExecutionResult` / `MeasurementResult` | `flagquantum/runtime/result.py` | stable run/Circuit/Module/training、dynamic projection | **受保护公共契约**。summary/diagnostics 有版本；Tensor payload 的通用序列化目前明确未承诺。 |
| `ExecutionRecordContract` | `flagquantum/core/contracts.py` | `runtime/records.py`、audit | **内部候选观察记录**。与用户结果不同，适合作为持久化 evidence record，但当前命名容易与结果混淆。 |
| `RuntimeResult` | `flagquantum/_compiler/runtime_abi.py` | compiler runtime adapters/conformance | **私有 artifact ABI 结果**。bytes + identity chain，不是用户量子结果。 |
| `DynamicExecutionResult` | `flagquantum/runtime/dynamic/result.py` | dynamic runtime、Qiskit execution | **实验性原生结果**，已有 `to_execution_result()`；这是合理的边界 adapter。 |
| `DeploymentResult` | `flagquantum/deployment/cloud.py` | `QuantumProvider` 与具体 providers | **临时 Provider 结果**，counts + handle；应在 Provider 边界投影到 Core result/failure。 |
| `TargetExecutionResult` | `flagquantum/runtime/target_execution.py` | sparse target execution | **Runtime 专用结果**，应作为 native detail 或投影来源，不应成为跨领域并列权威。 |
| 各种 `*TrainingResult`、`*GradientResult`、`*TNExecutionResult` | `flagquantum/runtime/executors/**` | 对应 backend/tests/benchmarks | **算法/执行引擎本地对象**。只要不跨 Provider 边界且能无损投影，可保留。 |

建议逻辑权威为 Core 所有的 result schema，但稳定 `flagquantum.runtime.result.ExecutionResult`
实现位置在获批迁移前保持不变。新增 Core 合同应先以 adapter 和 conformance tests 证明
Simulation 与一个 QPU/remote provider 可互换，之后再讨论实现搬迁。

### 3.7 Evidence、Provenance、Metrics

| 类型 | 当前定义 | 主要消费者 | 分类与判断 |
| --- | --- | --- | --- |
| `ProvenanceContract`、`FailureContract`、`ExecutionObservation`、`OwnershipContract`、`AccuracyContract` | `flagquantum/core/contracts.py` | runtime record builder/auditor、ExecutionResult accuracy | **内部候选 Core 契约**；其中 `AccuracyContract` 已嵌入受保护结果字段，修改风险更高。 |
| `RuntimeProvenance`、`EvidenceArtifact` | `flagquantum/runtime/observability/evidence.py` | evidence creation/verification | **Runtime 生产证据信封**。字段比 Core provenance 更完整，使用 HMAC 和 scope 规则，但没有严格 from_dict 对象往返。 |
| `CapabilityEvidence` | `flagquantum/runtime/capabilities.py` | operator probe/preflight | **能力证据**，版本与证据 ID 独立。 |
| `DistributedEvidenceContract` / `DistributedTransportEvidence` | `flagquantum/runtime/audit/schema.py` | audit engine、compat API/benchmarks | **审计派生结果**，不是原始证据。 |
| `SourceProvenance` | `flagquantum/_compiler/import_models.py` | `CircuitIR` importer | **Compiler 来源元数据**，语义窄且不等同于执行 provenance。 |
| `ShadowEvidence`、`BackwardExecutionEvidence`、`KernelDispatchEvidence`、`DistributedTNMemoryEvidence` | Compiler/Runtime backend 私有模块 | 对应算法和 benchmark | **局部证据对象**。应通过 namespaced evidence sections 投影，不并入一个巨型 dataclass。 |
| `metrics: Mapping[str, Any]` 与 `MPSStepMetrics`、`MPSCanonicalizationMetrics` 等 | stable result 与 backend modules | 用户结果、training/backends | **无通用 Metrics 权威类型**。稳定结果目前只承诺映射字段；算法专用 metrics 可保留。 |

建议位置为 `flagquantum/core/evidence/`，只定义版本化 envelope、provenance identity、failure
分类和 namespaced section 规则。测量工具、HMAC key 管理、release 审核算法仍由 Runtime/Audit
负责。不要把所有 backend metrics 提升为 Core 字段。

### 3.8 Provider 能力和失败类型

当前 Provider 边界至少有三套抽象：

| 边界 | 当前类型 | 状态 |
| --- | --- | --- |
| 云/QPU execution provider | `deployment.cloud.QuantumProvider`、`ProviderTaskHandle`、`DeploymentResult` | 具体可用但返回字符串状态、内置异常和 Deployment 专用结果；不是 Core 契约。 |
| sealed artifact runtime adapter | `_compiler.runtime_abi.RuntimeAdapter`、`ExecutionHandle`、`RuntimeCallStatus`、`RuntimeDiagnostic` | 私有且 provider-neutral，状态/身份设计可复用，但错误没有映射到 Core `FailureContract`。 |
| Extension provider | `extensions.sdk.ProviderExtension`、`Extension*Error` | `flagquantum.ecosystem.extensions` 的受保护扩展协议；生命周期和 capability negotiation 与 execution provider 不同。 |
| Platform provider | `providers.platform` contracts 与 `Platform*Error` | 平台层专用，必须与完整 execution provider 分开。 |
| Compiler conformance metadata | `_compiler.provider_conformance.ProviderExtension` | 与公开 `extensions.sdk.ProviderExtension` **同名不同义**；前者只是 namespaced non-semantic metadata，应改名并最终留在 conformance 内部。 |

公共错误基类位于 `flagquantum/errors.py`：`ValidationError`、`SerializationError`、
`PlanningError`、`CompilationError`、`ExecutionError`、`CapabilityError` 均属受保护稳定扩展。
Core 另有持久化 `FailureContract`。目前 Deployment provider 广泛直接抛出内置异常，导致
相同 provider failure 在同步 API、异步 ABI 和持久化记录中分类不一致。

建议 Core 分别定义窄的 `ExecutionProvider` 与 `PlatformProvider` 协议及关闭的 failure
category；异常到 `FailureContract` 的转换由 Runtime/Provider adapter 完成。不得把两个
Provider 层合成一个含大量可选方法的接口。

## 4. 重复、转换与跨层泄漏清单

| 编号 | 重复/转换/泄漏 | 当前路径 | 处理建议 |
| --- | --- | --- | --- |
| D1 | 稳定 `ExecutionOptions` 与私有运行时绑定曾同名 | `runtime/options.py` ↔ `_compiler/runtime_abi.py` | **已完成名称消歧**：稳定类型保持不动；私有 ABI 使用 `RuntimeBindings`，且不保留同名兼容别名。后续只在 adapter 边界接收稳定请求。 |
| D2 | `ProviderExtension` 命名冲突 | `ecosystem/extensions/sdk.py`；原 `_compiler` 实现已删除 | **已消除**：扩展协议保留唯一公开定义。 |
| D3 | artifact 三轨 | Core `ProgramArtifact` ↔ compiler `SealedExecutableArtifact` ↔ Deployment `DeploymentPackage` | Core 定义 program/executable 信封；Deployment 仅做 provider wire adapter。 |
| D4 | target capability 三轨 | `_compiler.TargetCapabilities` ↔ Runtime `BackendCapabilities` ↔ Deployment `CloudBackendProfile` | Core snapshot 为跨领域权威；dynamic discovery 和 cloud profile 都显式投影。 |
| D5 | request 多轨 | `RequestedExecution` ↔ `InternalExecutionRequest` ↔ `DistributedExecutionRequest` ↔ 两套 options | 新 Core request 落地后按职责保留 importer-local request，其余 adapter 化或删除。 |
| D6 | plan 双权威风险 | stable `ExecutionPlan` → `RuntimePlanContract` | 当前转换有损。将 RuntimePlan 明确为 audit projection，禁止反向替代 stable plan；最终是否合并需 API proposal。 |
| D7 | result 多轨 | stable result ↔ Dynamic/Target/Deployment/Runtime/backend results | 保留 native 内部结果，统一在 execution provider 边界投影；删除跨层直接依赖。 |
| D8 | provenance 双轨 | Core `ProvenanceContract` ↔ Runtime `RuntimeProvenance` | 先制定字段兼容和 identity 规则，再由 Runtime 生成 Core envelope；不可直接删字段较多的一侧。 |
| D9 | evidence 多轨 | Core execution record ↔ Runtime evidence artifact ↔ audit distributed schema | 区分原始证据、观察记录和审核结论，使用引用/哈希连接，而不是合成万能 Evidence。 |
| D10 | Runtime 依赖 Compiler 计划类型 | `runtime/result.py`、`runtime/execution.py`、`runtime/plan_execution.py` 等列于 `architecture.toml` 白名单 | Core plan/artifact 合同落地后逐项减少白名单；禁止新增项。 |
| D11 | Deployment provider 原生异常泄漏 | `deployment/providers.py`、`providers/execution/braket.py`、`deployment/cloud.py` | adapter 将 provider 状态/异常映射为 Core failure category；保留原异常为 cause/private diagnostics。 |
| D12 | `Mapping[str, Any]` 跨边界 | distributed request options、ExecutionResult diagnostics/metrics/provenance | 请求语义改为 typed contract；结果的 namespaced additive diagnostics 可保留 mapping，但不得承载稳定核心字段。 |

可在契约获批和调用者清零后删除/合并的是 D2 私有同名 extension、D4
中的重复跨领域字段、D5 的 Distributed request 自由字典以及 D6 的无主双轨。专用 backend
plan/result/metrics 不是天然重复，只有在它们越过 Provider 边界时才需要删除或适配。

## 5. 序列化、哈希与拒绝行为现状

| 对象 | 往返 | 哈希/身份 | 未知字段 | 未知版本 | 重要限制 |
| --- | --- | --- | --- | --- | --- |
| `CircuitIR` | `to_dict/json` + `from_dict/json` | 完整规范 JSON SHA-256 | 顶层拒绝 | 构造时拒绝非 `1.0` | 部分集合缺省为空；Tensor 编码受支持 dtype 约束。 |
| `ProgramArtifact` | `to_dict` + `from_dict` | envelope 全字段 SHA-256 | 顶层拒绝 | 拒绝非 `1.0` | 没有 `to_json/from_json`；`version` 命名与目标文档中的 `schema_version` 不一致。 |
| `TargetCapabilities` | `to_dict` + `from_dict` | semantic dict SHA-256 | 顶层和嵌套结构拒绝 | 拒绝非 `target_capabilities_v1` | `display_label` 序列化但不参与 fingerprint，这是有意行为。 |
| stable `ExecutionOptions` | `to_dict` + `from_dict` | 无独立哈希；进入 plan options fingerprint | 未知及缺失字段拒绝 | 拒绝非 `1.0` | 所有 `None` 都被保留以维持继承语义。 |
| stable `ExecutionPlan` | dict/json 双向 | 四 fingerprint + 完整 identity 重算 | 完整顶层/section shape 拒绝 | 拒绝非 `1.0` | 受保护；pickle 不承诺；读入不重规划。 |
| `RuntimePlanContract` / `ExecutionRecordContract` | dict/json 双向 | 全部嵌套合同 SHA-256 | 顶层和嵌套拒绝 | 顶层和嵌套拒绝 | parser 对某些缺失 dataclass 字段会使用默认值，尚非全面“缺失字段拒绝”。 |
| stable `ExecutionResult` | `summary()` / `diagnostics()` 单向投影 | 无 result content hash | 不适用 | summary/diagnostics 固定 `1.0` | Tensor payload 序列化和完整反序列化未承诺；runtime projection 冲突被记录且不能覆盖核心字段。 |
| `SealedExecutableArtifact` | 无通用 from_dict | payload hash + 多级 identity | 不适用 | 构造拒绝非 schema 常量 | 只能通过 seal/verify 流程构造；这比开放反序列化更严格，但不是可移植 Core 往返合同。 |
| `EvidenceArtifact` | `summary()` + verifier | SHA-256 + HMAC | verifier 只检查所需字段，不全面拒绝未知字段 | schema 不匹配失败 | 只接受 measured production release scan；对象级 from_dict 缺失。 |

`tests/team/core/test_contract_characterization.py` 补充了跨类型特征矩阵，覆盖 `CircuitIR`、
`ProgramArtifact`、Core runtime plan、`TargetCapabilities`、稳定 `ExecutionOptions` 以及
`ExecutionResult` summary/diagnostics 冲突规则。测试只断言当前行为，不改变任何受保护实现。

## 6. 四类契约的最小收敛顺序

### 阶段 0：固定现状（本提交）

- 交付本盘点和特征测试；
- 不新增导出、不改 schema、不改哈希；
- 由集成团队确认 D1–D12 和每个旧类型的 owner/retirement condition。

### 阶段 1：ProgramArtifact / ExecutableArtifact

先建立 Core-owned artifact 身份链，因为 capability、request、plan 和 result 都需要引用同一
可执行输入。最小切片只包含：版本、kind/profile、规范 payload 或 payload hash、parent/source
identity、requirements 与 namespaced extensions。用现有 `SealedExecutableArtifact` 和一个
contract fake 证明替换，不迁入 provider handle/credentials。

### 阶段 2：TargetCapabilities + 两类 Provider 协议

把当前 `_compiler.TargetCapabilities` 的已验证语义作为输入基线，补上明确的证据引用和
execution/platform 两类能力边界。Runtime discovery 与 Cloud backend 都通过 adapter 生成
snapshot。先让 Compiler 和至少两个 provider 实现通过同一 conformance suite，再减少旧
字段和路径。

### 阶段 3：ExecutionRequest + ExecutionPlan 边界

请求组合阶段 1 artifact、阶段 2 target constraint、稳定 `ExecutionOptions` 与参数绑定。
随后消除私有同名 options 和 distributed free-form request。稳定 `ExecutionPlan` 在此阶段只
增加内部 adapter；任何公开字段、schema、identity input 或导入路径改变都另走 API Change
Proposal。`RuntimePlanContract` 明确降级为 audit projection 或在批准后合并。

### 阶段 4：ExecutionResult + Evidence / Provenance / Failure

以稳定 `ExecutionResult` 为用户投影，以版本化 execution record 为持久化证据；Simulation
和 QPU/remote provider 各实现一个 adapter。原始 evidence、audit verdict 和 native metrics
保持分层，通过 hash/reference 关联。最后统一 provider failure category，并逐项删除
Deployment/Runtime 的跨层结果和异常泄漏。

这个顺序避免 result 引用尚未稳定的 artifact/target/request identity，也避免一开始设计包含
所有未来字段的万能对象。

## 7. 批准要求

### 必须提交 API Change Proposal

- 修改 `Circuit`、`CircuitIR`、`ExecutionOptions`、`ExecutionPlan`、`ExecutionResult`、
  `MeasurementResult` 的公开导出、签名、dataclass 字段、默认值、枚举/Literal、异常行为；
- 修改上述对象的 schema/version、未知字段或版本拒绝规则、round-trip、identity/hash 输入；
- 把稳定实现移动到新导入路径、改变 class identity，或删除现有兼容入口；
- 将新的 Core request/capability/provider/evidence 类型加入稳定 root 或稳定 namespace；
- 改变 `ExecutionResult.metrics/provenance/runtime/compatibility` 的受保护行为，或承诺新的 Tensor
  payload 序列化格式。

### 必须由集成团队批准 Contract Proposal / ADR

- 在 Core 中定义新的 `ExecutableArtifact`、`TargetCapabilities`、`ExecutionRequest`、
  Execution/Platform Provider protocols 或 failure categories；
- 选择 `RuntimePlanContract` 与稳定 `ExecutionPlan` 的长期关系；
- 选择 Core provenance/evidence envelope 及 HMAC/审计职责边界；
- 改动 `contracts/`、`architecture.toml`、迁移白名单、ADR 或跨团队权威位置；
- 决定 `_compiler`、Runtime、Deployment 旧类型的兼容期、owner、移除版本和退出证据。

若上述 ADR 同时影响稳定公共行为，还必须附带 API Change Proposal；ADR 不能替代 API
owner 对受保护表面的授权。

### 可作为团队内部重构

- 在不改变输入输出和异常语义的前提下，拆分私有 helper、减少重复 canonical JSON 工具；
- 给私有 adapter 增加 conformance tests 或 contract fakes；
- 在获批 Core 契约已经合入共同基线后，将本团队拥有的私有调用者改为使用 adapter；
- 删除已经没有 importers、没有序列化兼容承诺且有明确退出记录的私有重复对象；
- 补充像本轮这样的只读盘点和现状特征测试。

即使属于内部重构，只要触及 `team-ownership.toml` 的 protected paths，也必须由集成工作区
执行，Core 团队不能以“实现等价”为由直接修改。

## 8. 兼容风险与集成验收建议

| 风险 | 严重度 | 验收要求 |
| --- | --- | --- |
| 稳定类搬迁改变 import path/class identity/pickle 行为 | 高 | 保持原实现位置或批准 API migration；检查 introspection snapshot。 |
| artifact/plan/capability 字段改名改变 canonical hash | 高 | 使用旧 fixture 和 golden hash；禁止自动更新 snapshot。 |
| Core contract parser 对缺失字段使用默认值 | 高 | 集成团队决定兼容策略；若改为拒绝，需版本迁移而非原地收紧。 |
| `ExecutionPlan.to_contract()` 有损却被当作可执行计划 | 高 | 类型/文档明确 audit projection；禁止反向执行，增加 negative conformance。 |
| 两套同名 Options/ProviderExtension 被误导入 | 中高 | 私有类型消歧；静态检查禁止新增 importer。 |
| Provider 内置异常跨实现不一致 | 中高 | 建立 failure mapping contract，保留 cause，测试 retryable/category。 |
| free-form result mappings 承载稳定字段 | 中 | namespaced sections；核心字段冲突必须可见且不覆盖。 |
| 过早合并专用 backend metrics/results | 中 | 只要求 adapter 无损投影；算法局部对象留在实现域。 |

集成验收至少应包括：旧序列化 fixture 后向读取、canonical hash golden、未知字段/版本拒绝、
稳定 `ExecutionPlan` 不重规划、Simulation/QPU provider 替换测试、Runtime 不新增 Compiler
imports，以及旧类型 importer 数量单调下降。

## 9. 本轮验证记录

- 团队范围预检和最终分支差异检查：通过；
- 架构边界检查：通过；
- 新增特征测试：6 项通过；
- 相关 IR、artifact、runtime contract、options、target capabilities、ExecutionPlan 和
  ExecutionResult 测试：86 项通过；
- 主机默认 `smoke or unit` 门禁：1804 项通过、12 项跳过、6 项失败。两项仓库检查因系统
  `/usr/bin/git` 缺少 macOS developer tools 失败，使用随附 Git 路径单独复跑后 10 项全部
  通过；两项 watchdog 测试依赖 Linux `/proc`；一项浮点抵消测试在主机 PyTorch 2.13.0
  的 `torch.dot` 结果与固定预期不同；一项 10000-gate 性能预算失败。
- 使用仓库 `flagquantum-dev:local` CPU Docker 环境复跑默认门禁：1809 项通过、10 项跳过、
  3 项失败。Linux 容器消除了 watchdog 和浮点差异；剩余两项仓库检查是 linked worktree
  `.git` 文件指向未挂载的主机绝对路径，容器中的 `git ls-files` 无法访问共享 Git 元数据；
  另一项仍为 Phase 1 import/verify 性能预算。
- 容器内单独测量性能门禁时，10/100/1000/10000-gate p95 分别为 0.358158、2.429283、
  15.866691 和 239.094925 ms；后三者中的 100/1000/10000 超过各自 1.5/12.5/120 ms
  限额，所有 case 的内存和 deterministic identity 检查均通过，但 growth gate 未通过。

上述默认门禁失败没有通过修改阈值、快照或受保护行为规避，也不作为本轮契约变更证据。
