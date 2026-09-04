# Execution Provider 现状盘点与最小合同提案

## 结论

当前仓库已经具备多条可运行的目标执行路径，但尚不存在一个可由 Runtime 在
Simulation、QPU 与 Remote Service 之间替换的 **Core-owned Execution Provider
Contract**。现有能力分散在四个边界：稳定的本地 `ExecutionResult`、本团队维护的
`DeploymentPackage`/`DeploymentResult`、`runtime/target_execution.py` 的稀疏数值结果，
以及扩展 SDK 的通用生命周期协议。它们解决的问题不同，不能把其中任意一个直接宣布为
目标合同。

本轮只冻结事实、增加特征测试并提出合同评审方案；没有改动 Stable Core、公开签名、
序列化模式、设备平台或厂商 SDK，也没有产生真实 QPU 执行证据。能力成熟度仍以
`capability-maturity.toml` 为准：Circuit packaging and cloud deployment 为
`development_evidence`，Extension SDK 为 `experimental`，没有 provider 获得
release certification。

## 盘点范围与执行分类

| 入口 | 当前职责 | 当前输出 | 目标类型 / rank 执行分类 | 声明边界 |
| --- | --- | --- | --- | --- |
| `fq.run` / `Circuit.run` / `runtime/execution.py` | 本地或分布式 Runtime 主路径，生成稳定 `ExecutionResult` | `ExecutionResult` | 本地默认是 `single_device_fast_path`；具体分布式语义由计划和 runtime evidence 决定 | 不是远程任务 API |
| `runtime/target_execution.py::run_target` | 按 full state、amplitude 或 local observable 的输出成本选择 statevector/MPS/TN | `TargetExecutionResult` | 本地 statevector/MPS/TN 为 `single_device_fast_path`；多 rank TN 稀疏切片按仓库规范应归为 `manual_sliced_tensor_contraction` | 当前内部 summary 使用更细的实现字符串，不能据此声称 `sharded_across_ranks` |
| `deployment/cloud.py::LocalSimulatorProvider` | 在无网络条件下验证 deployment package、提交句柄、counts 与身份链 | `DeploymentResult` | `single_device_fast_path` | 合同假实现/本地模拟，不是真实 QPU 证据 |
| `HttpQuantumProvider` | 通用 OpenQASM HTTP 远程任务适配 | `DeploymentResult` | `remote_service_job`（不属于 rank 分布式分类） | 后端发现失败会生成 fallback profile；这不是已验证目标能力 |
| `QuafuProvider` | Quafu HTTP 提交、轮询、取消、counts 解码、芯片信息抓取 | `DeploymentResult` | `qpu_job_candidate` | 只有 mock 合同证据；无真实任务、五态、幂等或硬件证据 |
| `AmazonBraketProvider` | Braket device profile、dry-run、OpenQASM 3 提交和结果读取 | `DeploymentResult` | `qpu_or_remote_simulator_job`，取决于 device properties | 当前测试只使用 fake device/task，不证明 AWS 或 QPU 可用 |
| `OriginQProvider` | SDK 对象适配占位 | `DeploymentResult` | `qpu_job_candidate` | 无 SDK 时 fail closed；句柄当前携带 live job object |
| `TencentQuantumProvider` | Tencent HTTP 任务提交、状态与 counts | `DeploymentResult` | `qpu_or_remote_service_job_candidate` | mock transport 证据，不代表真实设备 |
| `TianyanProvider` / `GuodunProvider` | QCIS 提交、结果矩阵转 counts | `DeploymentResult` | `qpu_job_candidate` | 状态仅以“是否已有结果项”推断，无失败/取消路径 |
| `FieldQuantumProvider` | 远程 sample 模拟服务 | `DeploymentResult` | `remote_service_job` | 静态 simulator profile；不是本地 Simulation Provider |
| `flagquantum.extensions.ProviderExtension` | 第三方扩展注册、协商、启动、隔离和清理 | `submit -> str`、`status -> str` | 扩展控制面，不是完整执行目标 | 没有 result、cancel、evidence 或标准错误合同 |

表中的 `remote_service_job`、`qpu_job_candidate` 等只是本次盘点使用的目标家族描述，不是
新的公开枚举或合同值。外部任务本身不具有 rank 分布分类；只有其背后的具体模拟执行若
暴露充分证据，才能另行标注仓库规定的 rank 语义。

`run_target(..., world_size>1, mode="tensor_network")` 在 development profile 中按 rank
模拟切片任务并本地归约；真实 process group 路径执行 slice parallel all-reduce。二者都是
稀疏张量网络切片，不等于状态/训练分片，因此本轮统一归类为
`manual_sliced_tensor_contraction`，并且 `scalability_claim_allowed=false`。所有外部云任务
不使用 rank 分布语义；“提交到云/QPU”不能被误写为分布式容量扩展。

## 当前任务生命周期

### 已有共同主干

`DeploymentPackage` 在提交前校验 provider/backend、IR→QASM、routing plan、routing
evidence digest 和 deployment artifact digest。`build_submission_receipt` 把 package identity
复制到 `ProviderTaskHandle.payload`，`build_result_metadata` 再把同一 identity 带到结果，
`validate_deployment_result` 校验 receipt-to-result identity 和 counts 总数等于 shots。

当前共同方法只有：

1. `discover_backends(n_wires)`；
2. `submit(package) -> ProviderTaskHandle`；
3. `query_status(handle) -> str`；
4. `fetch_result(handle) -> DeploymentResult`；
5. 基类 `run` 执行 submit、一次 status query、fetch result。

基类 `run` 不是完整异步驱动器：遇到 `Running` 会立即报错，不轮询。Quafu 自己实现轮询
和 timeout；Braket 的 `run` 直接在 `task.result()` 阻塞；其余 provider 沿用一次查询。这是
Provider 和 Runtime 职责重叠，也是替换时最明显的行为差异。

### 状态、取消、结果与错误矩阵

| 实现 | 状态来源与映射 | 取消 | 结果解码 | 错误映射 |
| --- | --- | --- | --- | --- |
| LocalSimulator | 内存中有结果即 `Finished`，否则 `Unknown` | 无 | 本地 `Circuit.counts` | 未知任务为 `KeyError` |
| HTTP 通用 | 原样读取 `status/state` | 无 | `counts/count/data`，数值 round→int | transport 和 payload 错误为通用异常/`RuntimeError` |
| Quafu | 部分归一为 Finished/Failed/Cancelled，保留其他原字符串 | 有，返回厂商原始响应 | 可选 reverse bits | 失败为 `RuntimeError`，超时为 `TimeoutError`，无稳定 category |
| Braket | SDK state 映射；`CANCELLED` 当前映射为 `Failed` | 未暴露 | `measurement_counts` | SDK/缺字段异常或 `RuntimeError`；保留 live task object |
| OriginQ | SDK job status 的部分映射 | 无 | `get_counts` 后固定翻转 bit string | SDK/通用异常；保留 live job object |
| Tencent | detail state 的部分映射，未知状态默认为 `Running` | 无 | 通用 counts decoder | 通用异常，无 provider error category |
| Tianyan/Guodun | 有结果项为 `Finished`，否则 `Running` | 无 | result matrix 或 count 合并 | 空结果最终成为 counts extraction `RuntimeError` |
| FieldQuantum | 常见字符串映射，未知为 `Running` | 无 | 通用 counts decoder | result 标记失败时 `RuntimeError` |

现状没有统一的 Pending/Running/Finished/Failed/Cancelled 类型、终态判定、native status
保留位置、重试建议、错误 category 或幂等 submission key。只有 Quafu 暴露 cancel；取消的
幂等性、竞态和 Cancelled 后 fetch 语义均未冻结。

### 校准与执行证据

- `QuafuProvider.fetch_chip_info` 可抓取包含 `calibration_time`、`qubits_info` 和
  `couplers_info` 的原始快照；`quafu_noise_model_from_chip_info` 可将选定物理比特转换成
  FlagQuantum `NoiseModel`。抓取、规范化、快照 hash、提交所用版本和结果 identity 尚未
  连成一条强制链。
- Braket profile 从 device properties 提取 qubit count、native gates、connectivity、ARN
  和部分 dynamic-circuit 信息；这属于 capability snapshot，不是一次执行实际采用的校准
  或物理映射证据。
- deployment 当前可靠证据是 program/routing/package identity 与 shots accounting。
  `duration_ms`、实际 backend、bit order、physical qubits、calibration/noise version、
  provider recompile 后映射、queue/execute timestamps 均不是 `DeploymentResult` 必填项。
- 厂商响应会被放入 handle payload 或 result metadata，且 SDK provider 会放入 live job
  object；这些对象不可安全序列化，不适合作为 Core handle/evidence。

## 三类 Execution Provider：共同点与不可强行统一处

### 应统一的最小语义

三类实现都应消费 Core 定义的完整 `ExecutionRequest`，在提交前按 TargetCapabilities
fail closed，返回可序列化的目标身份和 execution handle，提供标准状态，最终产生同一个
Core `ExecutionResult` 与 `Evidence`，并把 native status/error/result 保留在明确的
provider extension 区域。所有降级、provider recompilation、bit-order conversion 和
backend substitution 都必须可见。

### 必须保留的差异

| 维度 | Simulation Provider | QPU Provider | Remote Service Provider |
| --- | --- | --- | --- |
| 执行形态 | 通常进程内、可同步、可微、可返回 state/amplitude | 通常异步、shot-based、队列化、只返回硬件允许的 measurement | 通常异步且服务端可能再次规划，输出取决于服务合同 |
| 能力来源 | Simulation Engine + Platform Provider | device capability + 当前校准 + 厂商限制 | 版本化 service capability + 租户/配额/区域 |
| 失败域 | 内存、数值、kernel、设备与分布式 transport | 排队、校准过期、设备离线、shots、编译/控制失败 | 网络、鉴权、配额、服务版本、持久任务、服务端策略 |
| 结果与证据 | 精度、近似、截断、设备驻留、通信、梯度 | counts、bit order、物理映射、校准、shots、QPU execution | 服务版本、远程计划 identity、服务端 backend、账单/配额（如适用） |
| 可选能力 | checkpoint、gradient、full state | cancel、session/realtime、calibration | cancel、resume、webhook/stream、server-side workflow |

因此不应把 `state()`、gradient、calibration、cancel 或 live SDK job 塞进一个全是 optional
方法的胖接口。共同合同只表达完整请求、句柄、状态、结果和证据；取消、校准、Realtime
Session、checkpoint/gradient 应作为显式 capability 与窄扩展合同。

## 职责重叠与迁移台账

| 当前位置 | 重叠/债务 | 目标责任与退出条件 |
| --- | --- | --- |
| `deployment/cloud.py::QuantumProvider.run` | Provider 自己驱动任务生命周期；Runtime 本应拥有轮询、timeout、retry、cancel/recovery | Core 合同获批后，Runtime driver 只依赖合同；provider 仅做一次 adapter operation |
| `DeploymentResult` vs `runtime.result.ExecutionResult` | 两套结果，前者 counts-only，后者稳定 local/distributed result | Core 批准远程 measurement/evidence 映射；adapter 输出稳定 `ExecutionResult` 后退出双结果 |
| `TargetExecutionResult` vs `ExecutionResult` | cost selection 与稀疏执行产生第三种结果/summary | Runtime/Core 决定 target output 如何进入稳定 measurement/result，不由本团队私自改 public API |
| `CloudBackendProfile` vs Runtime/compiler capabilities | deployment 自有 target profile，且 provider metadata 自由扩张 | Core `TargetCapabilities` 获批并有兼容适配器与 conformance 后迁移 |
| `ProviderTaskHandle.payload` | 同时承载 receipt、原始响应和 live SDK object | Core serializable handle + provider 私有 session store；跨进程恢复测试通过后退出 |
| deployment counts helpers | 各 provider 共享部分解码，但 bit order/width/error 仍隐式 | Core measurement semantics + provider decoder conformance 获批后统一稳定部分 |
| `ProviderExtension` | 与 deployment provider 同名但只覆盖注册生命周期、submit/status | 扩展 SDK 保持唯一注册/lifecycle authority；它承载 Execution Provider 实现但不复制执行语义 |
| `runtime/target_execution.py` | 直接导入 compiler cost selection，且名称易与外部 execution target 混淆 | 保持当前内部入口；Core artifact/capability 合同落地后由 Runtime 团队消除 Runtime→Compiler 债务 |

本轮不新建 provider SPI、不修改扩展 SDK，也不把 deployment 类型复制到 Core。上述迁移均
必须遵循 contract-first：集成/Core 先批准版本化合同、假实现和 conformance，团队实现再
同步该基线。

## 结果统一风险

风险按当前阻塞程度排序：

1. **语义丢失**：把 `DeploymentResult.counts` 直接包装为 `ExecutionResult` 会缺少
   measurement wires/order、plan identity、accuracy、runtime 与 QPU evidence。
2. **位序与宽度**：通用 decoder 接受混合位宽并 round 任意数值；Quafu/OriginQ 又各自
   翻转 bit string。对称 Bell/GHZ 测试无法发现错误，必须使用非对称 fixture。
3. **身份不完整**：package digest 能证明提交前产物，但不能证明 provider 二次编译后的
   实际程序、物理映射、校准快照或实际 backend。
4. **不可恢复句柄**：Braket/OriginQ 把 SDK job object 放进 handle，无法可靠持久化、跨
   进程恢复或交给远程 Runtime driver。
5. **状态和错误不可编排**：自由字符串和通用异常使 retry、cancel、quota、bad circuit、
   device unavailable 等策略无法安全自动化。
6. **同步/异步行为漂移**：基类一次查询、Quafu 自轮询、Braket SDK 阻塞，导致同一消费
   者替换 provider 后行为变化。
7. **自由 metadata 不是合同**：当前允许额外字段但不要求 QPU evidence；不能通过“某次
   响应碰巧有字段”宣称结果已标准化。

## 交由集成/Core 评审的最小合同

建议 Core 只定义以下版本化值对象和窄协议；名称与字段均为提案，不在本团队分支实施：

```text
ExecutionProviderV1
  capabilities(target_ref) -> TargetCapabilities
  submit(ExecutionRequest, *, idempotency_key) -> ExecutionHandle
  status(ExecutionHandle) -> ExecutionStatus
  result(ExecutionHandle) -> ExecutionResult

ExecutionHandle
  schema/version, provider_id, target_id, execution_id,
  request_identity, submitted_artifact_identity

ExecutionStatus
  state = Pending | Running | Finished | Failed | Cancelled
  native_state, observed_at, error?, retry_advice?

ProviderEvidence
  provider/target identities, requested/submitted/executed artifact identities,
  timestamps, actual backend, transformations/fallbacks, provider extension
```

约束：handle 和 status 必须可 JSON 序列化且不得含凭据或 live vendor object；result 必须是
现有 Core `ExecutionResult` 的兼容演进而不是第四套结果；不 ready、terminal failure、
unsupported capability 和 identity mismatch 必须是可机器识别的 Core error category。
`cancel` 放入 `CancellableExecutionProviderV1`，校准放入 `CalibrationSourceV1`，实时会话另设
`RealtimeSession`，Simulation gradient/checkpoint 使用相应 capability，不把差异伪装成
共同必选方法。

### 替换与 conformance 测试方案

1. Core 提交合同、纯内存 `ContractFakeProvider` 和同一套 conformance；假实现覆盖五态、
   not-ready、失败、取消竞态、identity mismatch、未知 native state 与序列化恢复。
2. Runtime 编写只依赖 `ExecutionProviderV1` 的 driver，统一负责 polling、deadline、retry、
   cancellation orchestration 和 evidence assembly；测试中替换 fake provider 不改消费者。
3. 让 Local Simulation Provider 与一个无网络 Remote/QPU fake adapter 跑同一 conformance：
   同一 request identity、同一 terminal rules、同一 `ExecutionResult` accessor、同一 error
   category；结果数值可以因 provider family 不同而不同。
4. decoder conformance 使用非对称 bit-order fixture、leading-zero width、zero-count、shots
   mismatch、非整数/负 counts、重复 classical register 与 provider-native error payload。
5. evidence conformance 强制区分 submitted artifact 与 executed artifact；任何 provider
   recompilation 都记录实际 program/mapping identity，校准型目标绑定 snapshot version/hash。
6. capability conformance 明确“不支持”而不是缺方法；cancel/calibration/gradient/realtime
   分别测试 capability absent、supported 和 provider rejection。
7. 集成分支至少证明 Runtime 在 Local Simulation 与 ContractFakeRemote 之间替换时无需修改
   driver；真实 QPU adapter 只有在厂商 sandbox 和真实硬件证据到位后加入同一套测试。

## 本轮新增特征证据

`tests/team/execution/test_execution_provider_characterization.py` 固化以下当前事实：

- stable runtime、target execution 和 deployment 使用三种不同结果对象；
- LocalSimulator 完成 package→receipt→result identity chain；
- deployment result 校验只强制 identity 与 shot accounting，不强制完整 provider evidence；
- 基类 `run` 对 Running 状态只查询一次；
- cancel 只在 Quafu 具体类出现，扩展 Provider 没有 result/cancel；
- 通用 HTTP decoder 会 round 数值并接受混合 bit width，且不记录 counts bit order/width。

这些是 CPU/mock 合同证据，不是性能、分布式扩展、真实云服务或真实 QPU 证据。

## Phase 2 第二生产者替换证据

`flagquantum/deployment/synthetic_remote_target_capabilities.py` 增加了一个仅供内部
conformance 使用的 synthetic remote-style producer。它接收匿名、JSON-safe、显式提供的
fixture，直接生成 Core Target Capabilities v1 snapshot；不建立 provider registry，不提交或
轮询任务，不读取凭据，不调用网络或 vendor SDK，也不声明真实 QPU、硬件或生产能力。

- fixture 明确提供的 `device.kind` 才会成为 `verified/observed`；当前测试值为
  `synthetic_qpu`，其
  target class/provider 明确带有 synthetic 标识，不能解释为硬件观测；
- `target.class` 使用 v1 授权的静态 `verified/declared` exposure；scope、TTL、target
  identity、source/evidence reference 均由 fixture 和调用者显式给出；
- 其余省略事实仍输出 `value=null`、`unknown/not_exposed` 与按 capability 命名的 blocker，
  不从 device id、target identity、接口或 evidence 引用推断能力；
- `tests/team/execution/test_target_capability_replacement.py` 让同一个 Runtime
  `match_target_capability_candidates` 和同一个 Core RequirementSet 分别消费 CPU Platform
  snapshot 与 synthetic snapshot，并覆盖 stale、scope/evidence 引用错误、unknown/not_exposed、
  候选顺序、未授权 fallback 和 target identity 冲突。

这项证据只证明第二个 snapshot producer 可被现有 Runtime consumer 替换；它不改变默认执行
路径、plan/result/schema、failure stage 或 capability maturity。

## 真实 QPU 接入阻塞

在以下条件完成前，不应宣称真实 QPU 已接入：

- Core-owned ExecutionRequest/Handle/Status/ExecutionResult/Evidence 与 error taxonomy 获批；
- target capability 明确 shots 范围/步长、QASM/QCIS profile、basis、coupling、dynamic/realtime
  能力，并以真实 provider snapshot 验证；
- 可序列化句柄、idempotency、持久任务恢复、deadline/retry/cancel 终态规则通过测试；
- bit order、classical register、leading zeros、shot accounting 和 provider error payload 冻结；
- 校准规范化只有一个事实来源，结果绑定 calibration/noise version 与实际 physical qubits；
- provider 二次编译后的 executed program、mapping、backend 和 fallback 有可审计证据；
- 凭据解析、租户隔离、配额/付费授权和日志脱敏由实际部署环境验证；
- 厂商 sandbox 后再执行明确授权的真实任务，最后才形成真实硬件正确性与运行证据。

截至本盘点提交，所有 provider 测试均为本地模拟、fake SDK object 或 mock transport；没有
发起网络请求或付费任务，也没有真实 QPU 校准、队列、控制或 measurement evidence。

## 验证记录

验证使用仓库 `compose.dev.yaml` 与现有 `flagquantum-dev:local` Linux 开发镜像；该镜像按
仓库声明仅用于 development，不能作为性能、硬件或 release evidence。

- Execution Provider、deployment、target execution、extension protocol 定向集合：
  `81 passed`；
- 本轮新增特征测试：`6 passed`，并通过 Ruff check/format；
- `python tools/ci_tier.py pr-runtime`：`168 passed, 31 skipped`；
- `python tools/ci_tier.py pr-default`：`1811 passed, 10 skipped, 1 failed`。唯一失败为既有
  Phase 1 internal import/verify latency budget；Linux aarch64 容器中 100、1,000 与 10,000
  gate 的 p95 分别约为 7.97 ms、50.79 ms 与 661.09 ms，高于 1.5 ms、12.5 ms 与
  120 ms 预算。growth、memory 和 deterministic identity 均通过；本轮没有修改该跨团队
  性能基线或实现；
- `tools/check_team_scope.py --team execution --base
  codex/flagquantum-vnext-architecture`：通过；
- `tools/check_architecture.py`：通过。

Docker 复核消除了宿主 macOS 上缺少 `/proc`、sandbox 禁止 localhost socket、系统 Git
不可用和 PyTorch 数值环境导致的伪失败。以上仍全部是 CPU/mock 开发证据，不扩大任何
provider 成熟度或真实 QPU 声明。
