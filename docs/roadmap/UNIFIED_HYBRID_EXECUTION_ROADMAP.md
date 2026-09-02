# FlagQuantum 统一异构执行与云生态总推进路线图

状态：执行中；IR Phase 1 已正式退出，当前进入 Stage 2
适用范围：多层 IR、Quafu 纵向闭环、多量子云、经典算力云、量子—经典混合编排
当前分支：`codex/open-source-api-convergence`
原则：本路线图不授权修改 Stable Core、公共序列化 schema 或根级导出

关联文档：

- [多层 IR 总体架构](../architecture/MULTI_LEVEL_IR_ARCHITECTURE.md)
- [多层 IR Phase 0–1 实施计划](../architecture/MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)
- [IR Phase 0 退出审计](../development/IR_PHASE_0_EXIT_AUDIT.md)
- [IR Phase 1 实现授权包](../development/IR_PHASE_1_IMPLEMENTATION_APPROVAL_PACKET.md)
- [IR Phase 1 完成记录](../development/IR_PHASE_1_COMPLETION.md)
- [Quafu Provider 与门户任务契约](../development/API_CHANGE_PROPOSAL_012_QUAFU_PROVIDER_CONTRACT.md)
- [公共 API 保护规则](../development/PUBLIC_API_PROTECTION.md)
- [能力成熟度说明](CAPABILITY_MATURITY.md)
- [已知限制](../reference/KNOWN_LIMITATIONS.md)

## 1. 文档目的

本文把此前相互独立的 IR、量子云和经典算力工作合并为一条可验收的产品路线，回答：

1. 多层 IR、Quafu、多量子云和经典算力云应按什么顺序推进；
2. 哪些抽象属于 FlagQuantum 核心，哪些只属于部署接入层；
3. 如何实现“一次编程、多后端运行”，同时避免静默改变执行语义；
4. 每一阶段做到什么程度才允许进入下一阶段；
5. 如何避免用更多适配器数量掩盖统一执行闭环尚未形成的问题。

本文中的 `ExecutionTarget`、`ExecutionBinding`、`Job`、`ComputeFabricAdapter` 等名称
都是内部候选名，不构成公共 API 承诺。

## 2. 产品北极星

FlagQuantum 的目标不是成为一个量子云 SDK 集合，也不是只在 QPU 不足时提供备用
模拟器，而是形成统一的量子 AI 异构执行基础设施：

```text
fq.Circuit / fq.Module
          |
          v
 Public CircuitIR + internal multi-level IR
          |
          v
 Compiler / Planner / Execution Policy
          |
          v
 Unified Job and Execution Provenance
          |
    +-----+------------------+--------------------+
    |                        |                    |
    v                        v                    v
local CPU/GPU/NPU   classical compute fabric   quantum cloud/QPU
statevector/MPS/TN   Docker/Slurm/Kubernetes   Quafu/Braket/others
    |                        |                    |
    +------------------------+--------------------+
                             |
                             v
              normalized result + full provenance
```

目标用户体验是：

- 电路、训练代码和稳定结果读取方式不因后端变化而重写；
- 本地模拟、分布式模拟、噪声模拟和真实 QPU 使用同一程序事实源；
- 后端选择可以由用户指定，也可以按显式策略规划；
- fallback 必须获得授权，且实际执行位置始终透明；
- 量子与经典资源可以组成可恢复、可审计的混合工作流。

## 3. 当前事实基线

### 3.1 已具备

- `CircuitIR` 1.0 是受保护、可序列化、可哈希的公共电路交换表示；
- 内部 QuantumIR Phase 1 已完成：最小骨架、importer、verifier、identity、
  differential corpus 与性能门禁均已通过正式退出审查；
- `fq.run` 已统一多个本地模拟路径，`ExecutionOptions` 已包含后端、设备、shots 和
  fallback 基础字段；
- 已有 Statevector、MPS、Tensor Network、Density Matrix 及分布式执行基础；
- 已有 `CloudBackendProfile`、`DeploymentPackage`、`ProviderTaskHandle`、
  `DeploymentResult` 和部署产物身份链；
- 已有本地模拟、Quafu、OriginQ、腾讯量子、天衍、国盾、Field Quantum、Amazon
  Braket 等 Provider 实现或适配路径；
- 已有 QASM/QCIS 导出、拓扑路由、动态线路实验路径和 Quafu 校准转换；
- API 保护、能力成熟度、测试分层和 fail-closed 原则已经建立。

### 3.2 尚未具备

- `fq.run` 与 `deploy_circuit` 仍是分离的本地/云端执行入口；
- 没有生产级统一 Job、任务持久化、幂等提交、恢复和统一取消语义；
- 没有跨 Provider 的完整状态、错误、结果和 capability conformance；
- 没有按能力、队列、成本和约束选择资源的统一 Target Resolver；
- 没有通过至少两个结构明显不同的真实量子云验证通用抽象；
- 没有 Slurm/Kubernetes 等经典算力云控制面适配；
- 没有生产级 QPU—GPU 混合工作流编排；
- 当前能力矩阵没有将任何真实云 Provider 标记为 release-certified。

### 3.3 当前规划评分

以下评分只用于任务排序，不是对外能力或性能声明：

| 工作面 | 当前规划完成度 | 主要依据 |
| --- | ---: | --- |
| 公共 IR 与 API 治理 | 75/100 | 公共 IR 已认证，保护机制已建立 |
| 多层 IR 迁移准备 | 60/100 | Phase 0–1 已正式退出；Phase 2 编译能力迁移尚未启动 |
| Provider 与部署基础 | 60/100 | 多个实现存在，但未统一认证 |
| Quafu 生产闭环 | 35/100 | 候选契约和部分核心能力存在，P0 仍未闭环 |
| 用户无感后端切换 | 30/100 | 统一本地执行存在，云端仍是另一入口 |
| 经典算力云控制面 | 20/100 | 计算 Runtime 存在，资源控制面基本缺失 |
| 量子—经典混合编排 | 15/100 | 算法与执行零件存在，尚无统一可恢复工作流 |

## 4. 架构边界

### 4.1 程序平面

程序平面回答“执行什么”，包括：

- `Circuit`、`Module` 和公共 `CircuitIR`；
- 内部 ProgramIR、QuantumIR、TargetIR；
- 逻辑 qubit、测量结果、参数、控制流和目标合法化；
- program identity 与 compilation identity。

以下内容不得进入程序语义或程序 hash：

- 门户 `backend_id`；
- 供应商任务 ID；
- 访问令牌和凭证；
- 队列状态、价格和临时资源地址；
- 运行时重试次数和调度器 Pod/Job ID。

### 4.2 执行控制平面

控制平面回答“在哪里、何时、按什么约束执行”，候选内部对象包括：

```text
ExecutionRequest
ExecutionPolicy
ExecutionTarget
BackendCapabilities
ExecutionBinding
Job
ExecutionProvenance
```

`ExecutionBinding` 应在提交时形成不可变快照，至少关联：

```text
program_identity
compilation_identity
execution_identity
requested_target
actual_target
external_backend_id
provider_job_id
device_snapshot_id
idempotency_key
contract_version
```

门户 ID 只能存在于接入层和任务记录中，不能污染 Circuit/IR 或强迫 `fq.run` 接受
Quafu 特有参数。

### 4.3 执行数据平面

数据平面分为两类，不强行共用同一 Provider 接口：

```text
QuantumProvider
  discover / submit / status / cancel / result

ComputeFabricAdapter
  probe / allocate / launch / status / cancel / logs / checkpoint
```

二者共享 Job、身份、错误分类和可观测性最小语义，但保留各自特有扩展。

### 4.4 结果平面

统一结果必须区分稳定语义和 Provider 扩展：

- 稳定语义：measurement、counts/samples、shots、状态、逻辑 wire/classical bit 顺序；
- 执行事实：requested target、actual target、fallback、设备与校准快照；
- Provider 扩展：原始任务字段、厂商诊断和专有结果；
- 原始 Provider 信息不得改变稳定结果的相等性和程序身份。

## 5. 总体推进顺序

必须遵循“IR 关键合同先行、Quafu 纵向验证、再抽象多云、最后建设经典云与混合编排”。
不要求全部多层 IR 完成后才接入 Quafu，但不允许云端实现反向决定 IR 语义。

IR 不是只在 Stage 1 出现的一次性任务，而是贯穿后续阶段的持续主线。总路线采用
“IR/编译主线”和“执行/云生态主线”交错推进：

```text
IR / compiler lane                    execution / ecosystem lane

IR Phase 0 基线与决策         <----> Stage 0 基线、授权与责任边界
          |
IR Phase 1 QuantumIR骨架       <----> Stage 1 程序身份与语义边界
          |                            |
IR Phase 2 迁移现有编译能力    <----> Stage 2 Quafu最小真实纵向闭环
          |                            |
IR Phase 3 TargetIR/ABI        <----> Stage 3 Provider conformance与第二量子云
          |                            |
Phase 3完成并收敛              <----> Stage 4 统一Target/Job/执行入口
          |                            |
IR Phase 4 ProgramIR/动态线路  <----> Stage 5 经典算力云与工作流底座
          |                            |
Phase 4生产验证                <----> Stage 6 量子—经典混合编排与认证
          |
IR Phase 5 高级能力按项推进，不作为首个生产版本统一硬门槛
```

### 5.1 IR 深度与产品目标

| 产品目标 | 最低 IR 条件 | 说明 |
| --- | --- | --- |
| Quafu 最小联调 | Phase 1 | 可先使用兼容部署路径验证身份、位序和任务边界 |
| 现有静态编译统一 | Phase 2 | 编译、路由和 emitter 进入同一内部管线 |
| 多量子云与近无感切换 | Phase 3 | 依赖 TargetCapabilities、TargetIR 和 executable ABI |
| 动态线路与混合工作流 | Phase 4 | 依赖 measurement def-use、控制流和 ProgramIR |
| QIR、时序和高级硬件控制 | Phase 5 对应子项 | 按能力独立认证，不要求一次完成全部 Phase 5 |

因此，Stage 1 退出只表示“可以开始平台纵向验证”，不表示 IR 路线完成。总路线最终
验收至少要求 IR Phase 4 完成生产验证；Phase 5 仅对被声明支持的高级能力逐项设门。

## 6. 分阶段计划与退出门

### Stage 0：冻结事实、责任人与授权边界

目标：确保后续工作不会通过修改稳定合同来掩盖实现问题。

任务：

- 完成当前分支干净基线、测试基线和能力成熟度核验；
- 确认 IR、Provider、Runtime、平台接入和 API owner；
- 所有公共合同变更必须单独提交 API change proposal；
- 建立本路线图的 issue/里程碑映射，不以文档 checkbox 代替代码证据；
- 明确共享 adapter 仓只包含 OpenAPI、Schema、Mock、文档和必要薄绑定。

退出门：

- [ ] 所有工作面有 owner、证据位置、依赖和回滚方式；
- [ ] Stable Core 与候选内部 API 的边界可机器检查；
- [ ] 当前完整测试基线可在 Docker 中复现。

### Stage 1：IR Phase 0 退出与 Phase 1 内部骨架

目标：建立不会被云平台细节污染的程序事实源和身份分层。

优先任务：

1. 完成 `MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md` 的 Phase 0 退出门；
2. 完成实现 owner 复核，并取得 Phase 1 明确授权；
3. 实现 opt-in 的内部 QuantumIR 最小骨架、verifier、identity 和 importer；
4. 固定逻辑 qubit、物理 qubit、经典 bit、辅助 qubit 和测量映射边界；
5. 固定 program、compilation、execution 三类身份；
6. 用 differential tests 证明 state、measurement、gradient 和 ordering 不漂移；
7. 保持默认 `fq.run` 快路径零回归，并保留整体关闭新路径的能力。

退出门：

- [x] Phase 0 全部机器证据通过；
- [x] Phase 1 支持范围之外全部 fail closed；
- [x] 公共 API、`CircuitIR` 1.0、默认执行路径没有变化；
- [x] 同一逻辑程序可生成稳定 program identity；
- [x] Provider、credential、queue、backend ID 未进入内部程序 IR。

### Stage 2：Quafu 最小真实纵向闭环

目标：以第一个真实平台验证“程序—编译—执行—结果”边界，而不是先冻结通用 Provider
公共 API；同时启动 IR Phase 2，用真实部署 corpus 迁移现有编译能力。

链路：

```text
Circuit/CircuitIR
  -> target capability preflight
  -> deployment compilation and artifact
  -> Quafu submit/status/cancel/result
  -> counts and measurement normalization
  -> immutable execution binding and provenance
```

优先任务：

- 完成 Proposal 012 P0；
- 移除真实部署路径对已退役 `fq.experimental.QPUTwin` 的依赖；
- 明确门户 `external_backend_id`、内部 target、实际设备和 `provider_job_id`；
- 完成不对称线路位序、shot accounting、失败、取消和超时测试；
- 输出 `counts_bit_order`、program/calibration/noise-model identity 和物理比特；
- 引入幂等提交键，区分“提交成功”与“执行成功”；
- 在不复制 FlagQuantum 源码到 adapter 仓的环境中完成真实安装与端到端测试；
- 启动 IR Phase 2，将 canonicalization、decomposition、routing 和 emitter 逐项迁入
  opt-in 新管线；
- 将 Quafu 静态部署用例纳入 legacy/new compiler 差分 corpus，不允许以云端成功代替
  编译语义等价验证。

退出门：

- [ ] Issue #4 P0/P1 逐项有机器证据或明确后续 issue；
- [ ] 使用门户真实注册 backend ID 跑通提交、轮询和结果；
- [ ] QPU、理想模拟和噪声模拟结果来源不会混淆；
- [ ] fallback 默认关闭，启用时结果显式记录原因；
- [ ] 凭证未进入 IR、结果、日志或可序列化部署资产；
- [ ] IR Phase 2 迁移范围、差分 oracle、性能预算和回滚开关已经建立。

### Stage 3：Provider Conformance 与第二量子云

目标：完成 IR Phase 2，进入 IR Phase 3，以 TargetIR/executable ABI 证明抽象来自
跨平台共同语义，而不是把 Quafu API 换名后固化。

任务：

- 建立通用 Provider conformance suite；
- 统一最小状态：Pending、Running、Finished、Failed、Cancelled；
- 建立稳定候选错误分类，同时保留 Provider 原始错误；
- 统一 capability：qubit、门集、拓扑、shots、动态线路、结果类型和校准版本；
- 统一 logical-to-physical、measurement-to-classical 映射与结果归一化；
- 选择一个与 Quafu 任务、设备和结果模型差异明显的平台作为第二验证对象；
- 厂商特有字段进入明确命名空间，不扩张公共最小合同；
- 完成 IR Phase 2 的 Pass、routing、QASM/QCIS emitter 与性能门禁；
- 启动 IR Phase 3，建立内部 TargetCapabilities、TargetIR、ExecutableArtifact 和
  RuntimeAdapter 最小合同；
- 让本地 runtime、Quafu QASM 路径和第二种不同 target 共享 artifact/runtime 边界。

退出门：

- [ ] 同一 conformance suite 在本地 Provider、Quafu 和第二 Provider 通过；
- [ ] 至少两家真实平台完成受控端到端任务；
- [ ] capability 不兼容在提交前 fail closed；
- [ ] 不依赖 Provider 特有字段即可读取通用结果；
- [ ] 真实设备认证与 Mock/CPU 合同证据明确分开；
- [ ] IR Phase 2 已通过正式退出门；
- [ ] IR Phase 3 最小 target legality 和 artifact identity 已通过三类 target 验证。

### Stage 4：统一 Target、Job 与执行入口候选

目标：完成 IR Phase 3，实现代码层面的近无感切换，同时保持执行事实透明。

任务：

- 建立内部 `ExecutionTarget`、`ExecutionPolicy`、`BackendResolver` 和统一 `Job`；
- 支持 `strict`、`fallback`、`auto` 三类策略；
- 为同步模拟器和异步 QPU 提供统一状态、取消、等待和结果语义；
- 建立任务持久化、幂等键、重试、恢复和终态不可逆规则；
- 将 queue、cost、quota 和 capability 纳入 planner，但不纳入程序 identity；
- 先通过非根、候选或部署 namespace 验证，再决定是否提出 `fq.submit`；
- `fq.run` 的任何稳定签名或结果字段变化必须单独批准；
- 完成 TargetIR 合法化、ExecutableArtifact、RuntimeAdapter 和 DeploymentPackage 的
  兼容/迁移方案；
- 确保 Target/Job 只消费编译产物与执行请求，不反向成为程序 IR 的组成部分。

退出门：

- [ ] 同一 Circuit 无需重写即可在本地模拟、Quafu 和第二量子云运行；
- [ ] 用户读取通用结果的方式一致；
- [ ] 实际后端、fallback 和校准快照始终可查询；
- [ ] 网络超时重试不会重复提交付费 QPU 任务；
- [ ] 任务进程重启后可以恢复查询；
- [ ] 稳定 API 是否扩展已有真实使用证据和迁移评审；
- [ ] IR Phase 3 完整退出：local、QASM、non-QASM 三类 target 共用已验证 ABI；
- [ ] artifact 被篡改、target capability 不匹配和凭证越界均 fail closed。

### Stage 5：经典算力云控制面

目标：把现有本地/分布式计算 Runtime 接入真正的资源调度环境，同时启动 IR Phase 4，
为动态线路和混合工作流建立 ProgramIR 与经典控制语义。

顺序：

1. `LocalComputeFabric`：统一现有本地 CPU/GPU 快路径；
2. `DockerComputeFabric`：固定镜像、依赖、设备和结果身份；
3. Slurm 或 Kubernetes 二选一作为第一个真实集群适配；
4. 第二种调度环境验证适配抽象；
5. 再考虑公有云厂商特定资源和弹性策略。

任务：

- 定义资源请求：CPU、内存、accelerator、world size、节点和互联要求；
- 建立镜像、代码、数据、checkpoint 与执行身份；
- 支持 launch、status、cancel、logs、checkpoint 和恢复；
- 记录 rank ownership、通信语义、峰值内存和实际设备；
- 严格区分 sharded workload、data parallel 和 replicated execution；
- 不因云控制面引入而拖慢本地 CPU/单 GPU 快路径；
- 启动 IR Phase 4，实现 functions、block、measurement def-use、conditional、reset 和
  bounded loop 的受限内部语义；
- 建立 ProgramIR-to-QuantumIR lowering 和 OpenQASM 3 profile conformance；
- 经典算力调度 ID、Pod/Slurm Job ID 和资源状态不得进入 ProgramIR。

退出门：

- [ ] 相同执行请求可在 Local、Docker 和一个真实集群运行；
- [ ] 失败任务可以从 checkpoint 恢复或明确判定不可恢复；
- [ ] 执行身份包含镜像、代码、环境和资源拓扑；
- [ ] 分布式声明通过对应 accelerator/multinode 证据和审计；
- [ ] 成本、日志和资源释放可追踪；
- [ ] IR Phase 4 的受限动态子集通过 verifier、统计 oracle 和批量性能门禁；
- [ ] 动态程序与经典资源工作流的职责边界已经固定。

### Stage 6：量子—经典混合编排与生产认证

目标：完成 IR Phase 4 的生产验证，形成可持续运行的 QPU—GPU/NPU 闭环，而非一次性
演示脚本。IR Phase 5 的时序、QIR、梯度 lowering、分布式 lowering 和 pulse reference
按真实硬件与生态需求分别推进，不打包继承成熟度。

首批场景：

- VQE/QAOA：经典优化器与 QPU 批量测量循环；
- QPU 上机前：理想模拟、校准感知噪声模拟、真实硬件三路对比；
- 量子 AI：经典加速器训练、QPU 验证或推理；
- 误差缓解：多个 QPU 任务与经典后处理 DAG；
- 设备评测：同一逻辑程序跨设备的可比执行记录。

任务：

- 建立可恢复 workflow/DAG，区分经典节点、QPU 节点和数据转换节点；
- 支持参数晚绑定、批量任务、配额控制和部分失败；
- 明确中间数据 schema、隐私、保存周期和重放规则；
- 建立成本、队列、精度和完成时间的多目标策略；
- 完成真实生产负载、长时间运行、故障注入和版本升级测试；
- 完成静态与动态程序共享语义管线，证明 measurement feedback 和经典控制不依赖
  Provider 私有 metadata；
- 为每个进入产品声明的 IR Phase 5 子能力单独建立 schema、oracle、target evidence、
  性能基线和成熟度记录。

退出门：

- [ ] 至少一个混合算法在进程/节点故障后可恢复；
- [ ] 模拟和 QPU 结果在界面、API 与审计记录中始终可区分；
- [ ] 参数、程序、编译、设备快照和结果形成完整 lineage；
- [ ] 至少一家量子云和一种经典集群环境达到发布认证门槛；
- [ ] 对外文档只声明经过对应证据验证的能力；
- [ ] IR Phase 4 完整退出，静态和动态程序共享同一语义管线；
- [ ] 所有对外使用的 Phase 5 子能力均已独立认证，未认证子项保持实验或不可发现。

## 7. 跨阶段工作流

### 7.1 Capability 模型

需要逐步统一但不一次冻结：

```text
program formats
native gates and topology
qubit/classical-bit limits
shots range and step
dynamic circuit features
result types
gradient/parameter binding support
queue/cost/quota hints
calibration and capability snapshot identity
```

静态语义能力进入 target/compiler 合同；波动频繁的队列、成本和库存进入 resolver 快照。

### 7.2 状态与错误

状态机必须闭合，并保存原始 Provider 状态：

```text
Pending -> Running -> Finished
                   -> Failed
                   -> Cancelled
```

取消请求已受理不等于底层任务已取消。错误至少区分 validation、capability、resource、
authentication、quota、transport、provider-runtime 和 internal。

### 7.3 Fallback

- 默认 `strict/forbid`；
- 只有显式授权才能从 QPU 切换到模拟器；
- 要求真实硬件的任务绝不 fallback；
- fallback 后重新执行 capability preflight 和编译；
- `requested_target`、`actual_target`、原因和语义差异进入 provenance；
- 理想模拟、噪声模拟和 QPU 结果不得共用含糊标签。

### 7.4 安全与多租户

- 凭证只通过 secret reference 获取，不进入程序或普通 metadata；
- backend 关联至少包含 portal instance、tenant 和 external backend ID；
- 日志、异常和 trace 默认脱敏；
- 提交、取消和结果读取执行同一租户权限校验；
- 第三方 Provider payload 不直接进入核心 Runtime。

### 7.5 可观测性和可复现性

每个任务至少关联：

- program、compilation、execution identity；
- requested/actual target；
- provider job、device 和 calibration snapshot；
- shots、seed、位序与测量映射；
- 软件版本、容器镜像和依赖环境；
- world size、rank ownership、内存和通信语义；
- fallback、重试、取消和错误事件时间线。

## 8. 测试与认证矩阵

| 层级 | 必须证明 | 不能证明 |
| --- | --- | --- |
| Schema/Mock | 请求响应结构和失败形状 | 真实平台、物理精度、容量 |
| 单元测试 | 转换、状态机、位序、幂等逻辑 | 网络和真实资源行为 |
| 本地集成 | 编译到结果的闭环 | 真实 QPU 或多节点能力 |
| Provider sandbox | SDK/API 兼容和认证流程 | 生产设备性能 |
| 真实 QPU | 实际提交、状态、结果和设备身份 | 通用到其他 Provider |
| 单节点 accelerator | 本地性能和设备路径 | 多节点扩展 |
| 多节点/多卡 | 真实分片、通信与容量扩展 | QPU 物理正确性 |
| 长稳与故障注入 | 恢复、幂等、资源释放 | 科学精度本身 |

每个 Provider 至少具备以下 conformance 用例：

- 非对称线路 bit order；
- shots 守恒；
- program/deployment identity；
- unsupported gate/qubit/topology；
- missing measurement；
- Pending/Running/Finished/Failed/Cancelled；
- timeout、重试和幂等提交；
- 校准快照与实际执行设备；
- 凭证与敏感字段不泄漏。

## 9. 优先级与资源投入

当前建议投入比例：

```text
45%  IR Phase 2 编译能力迁移与差分验证
45%  Quafu 真实纵向闭环
10%  Target/Job/Provider conformance 候选设计
```

Stage 2 完成后调整为：

```text
40%  Provider conformance与第二量子云
35%  统一Target/Job和任务可靠性
25%  第一个经典算力云适配验证
```

在 Stage 3 退出前，不以“增加更多 Provider 数量”为优先成果。

## 10. 暂不推进事项

- 不立即公开 ProgramIR、QuantumIR、TargetIR；
- 不立即把 `fq.submit` 加入稳定根 API；
- 不让门户 backend ID 进入 `fq.run`、Circuit 或 IR；
- 不把 Quafu 特有状态、错误或字段定义成通用标准；
- 不同时建设多个 Kubernetes/Slurm/公有云控制面；
- 不默认静默执行 QPU 到模拟器 fallback；
- 不把 Mock、CPU 或单机测试描述成真实 QPU、多 GPU 或多节点认证；
- 不在 adapter 仓复制 FlagQuantum 模拟、噪声或调度源码；
- 不以重写现有稳定 API 换取短期接口整齐。

## 11. 主要风险与控制措施

| 风险 | 后果 | 控制措施 |
| --- | --- | --- |
| 过早冻结 Provider API | 被首个平台模型绑死 | 两个差异明显的真实 Provider 后再评审 |
| 云字段进入 IR | hash、缓存和可移植性失真 | 程序/编译/执行身份分层 |
| 静默 fallback | 科研结果来源错误 | 默认禁止、显式授权、完整 provenance |
| 状态字符串自由扩张 | 轮询和恢复不可靠 | 统一最小状态机并保存原始状态 |
| 重试重复提交 | 重复收费和重复实验 | 幂等键、持久化 binding、provider job 关联 |
| 多控制面同时开发 | 大量适配代码但无闭环 | 一个量子云、一个经典调度器逐步验证 |
| 多层 IR 拖慢本地路径 | 损害开发体验 | opt-in、缓存、bypass 和性能 gate |
| 把模拟当硬件证据 | 对外能力失真 | 分层认证与机器可审计成熟度 |

## 12. 阶段决策规则

每个阶段评审只接受以下证据：

- 合并到目标分支的实现；
- 可复现的机器测试或真实平台执行记录；
- 对应 schema、hash、日志和 provenance；
- 明确的限制、失败条件和回滚方式；
- API owner 对受保护合同变化的书面批准。

以下不视为完成：

- 只有设计文档；
- 只有 Mock 成功；
- 只有对称 Bell/GHZ 位序测试；
- 只有 Provider 类但没有真实任务；
- 只有 forward 而没有所声明的训练/梯度闭环；
- 通过放宽快照、异常或 schema 测试接受漂移。

## 13. 下一批可执行任务

按顺序进入当前分支：

1. 完成 Quafu 私有部署绑定和不对称位序端到端合同；
2. 建立 Quafu 状态、取消、错误、幂等和 metadata 最小闭环；
3. 为 IR Phase 2 固定迁移范围、差分 oracle、性能预算和回滚开关；
4. 启动 IR Phase 2，以 Quafu 静态部署 corpus 迁移 compiler、routing 和 emitter；
5. 从真实 Quafu 证据抽取 Provider conformance suite；
6. 选定第二量子云，并以其验证 IR Phase 3 TargetIR/executable ABI；
7. 在 Phase 3 和两个量子云验证后评审统一 Target/Job 候选；
8. 选定第一个经典调度环境，并同步启动 IR Phase 4 受限动态语义；
9. 只有证据稳定后，才提出稳定执行 API 的变更提案。

## 14. 最终验收定义

本路线图的目标完成，不以代码目录或 Provider 数量计，而以下列用户事实为准：

```python
circuit = build_once()

local_result = run_on_local(circuit)
cloud_result = run_on_classical_cloud(circuit)
qpu_result = run_on_qpu(circuit)
```

三条路径应满足：

1. 用户不重写 Circuit/Module；
2. 程序 identity 可证明逻辑输入一致；
3. 编译与执行差异可追溯；
4. 通用结果读取语义一致；
5. 实际资源、fallback、噪声和校准事实透明；
6. 任务可查询、取消、恢复和审计；
7. 量子与经典资源可组成可恢复的混合工作流；
8. 所有对外能力声明均有对应层级的真实证据。

此外，IR 必须满足：

- Phase 0–4 均通过各自机器可验证的正式退出门；
- Phase 1 不能被当作 IR 路线的最终完成状态；
- Phase 5 不要求一次整体完成，但每个进入产品声明的子能力必须独立认证；
- 旧编译/动态路径退出前具备明确兼容窗口、迁移证据和回滚方案。

达到这些条件后，FlagQuantum 才从“具备多个执行后端和 Provider”升级为真正的统一
量子—经典异构计算平台。
