# FlagQuantum 新一代总体架构

> 状态：候选架构，分阶段落地
>
> 适用范围：FlagQuantum 主仓库及其对外集成边界
>
> 机器可读契约：`contracts/long-horizon-architecture-v1.json`
>
> 约束优先级：`AGENTS.md`、Stable Core API 保护政策、能力成熟度政策和科研证据要求高于本文

## 1. 文档目的

本文定义 FlagQuantum 面向未来十年以上演进的稳定边界，但不承诺具体实现十年不变。
长期稳定的是领域职责、依赖方向和契约演进方式；量子编译算法、模拟算法、硬件 SDK、
通信库、智能体协议和部署方式均可替换。

架构成功的判据不是“目录看起来整齐”，而是：

1. 新编译器、新模拟器、新国产加速器或新 QPU 能通过版本化契约接入；
2. 替换某一实现时，其上游调用方无需修改；
3. 不支持的能力在执行前失败，精度降级、CPU 回退和后端替换均可审计；
4. 单机、多卡、多节点、真实 QPU 和服务化执行共享同一套核心语义；
5. 多团队可围绕稳定接口并行开发，而不依赖跨目录修改内部实现。

## 2. 设计目标与非目标

### 2.1 设计目标

- **统一语义**：线路、混合控制、目标能力、执行请求和结果证据拥有唯一权威表示；
- **编译与执行分离**：Compiler 只负责变换程序，Runtime 只负责组织一次执行；
- **运行时与算法分离**：Runtime 管资源和生命周期，Simulation 管数值方法；
- **硬件隔离**：国产加速器、QPU 和外部服务 SDK 对象不进入核心层；
- **生态可替换**：PyTorch、JAX、Qiskit、OpenQASM、QIR 等只在边界进行转换；
- **协议可替换**：MCP、REST、gRPC、CLI 不成为计算内核的依赖；
- **证据内建**：能力、精度、通信、回退和性能口径是执行结果的一部分；
- **渐进迁移**：保护现有稳定 API，每一步都保持最小端到端路径可运行。

### 2.2 非目标

- 不建立一个囊括所有未来对象的“万能 IR”；
- 不把所有现有代码一次性移动到新目录；
- 不为尚无真实用例的能力建立预防性抽象；
- 不在主仓库中复制 Compute Service 的租户、鉴权、计费和持久化任务能力；
- 不以兼容层掩盖语义差异或虚假的硬件支持。

## 3. 总体架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│ 用户与生态层                                                        │
│ Python SDK | PyTorch/JAX | OpenQASM/QIR | Qiskit | CLI             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ 边界转换
┌──────────────────────────────▼──────────────────────────────────────┐
│ Ecosystem / API                                                    │
│ 外部对象适配、公共 API、程序捕获；不产生第二套核心语义              │
└───────────────┬──────────────────────────────────┬──────────────────┘
                │                                  │
┌───────────────▼─────────────────┐  ┌─────────────▼──────────────────┐
│ Compiler                        │  │ Application Services           │
│ 校验、分析、优化、Lowering      │  │ 能力汇总、执行与部署预检       │
│ 只变换 ProgramArtifact          │  │ 仅保留有复用价值的组合流程     │
└───────────────┬─────────────────┘  └─────────────┬──────────────────┘
                │ Executable / Plan                │ ExecutionRequest
┌───────────────▼──────────────────────────────────▼──────────────────┐
│ Runtime                                                            │
│ 目标选择、资源编排、Session、分布式执行、恢复、观测和证据采集      │
└───────────────┬─────────────────────────────────────────────────────┘
                │
      ┌─────────────┬──────────────────────────┐
      │ 数值执行    │ 本进程算力               │ 外部任务控制面
┌─────▼──────────┐ ┌▼───────────────────────┐ ┌▼─────────────────────┐
│ Simulation     │ │ Compute                │ │ Remote                │
│ SV/MPS/TN      │ │ CPU | GPU/NPU | 通信   │ │ QPU | GPU/HPC | 云   │
└────────────────┘ └────────────────────────┘ └───────────────────────┘
                │                                      │ 校准与观测
                └───────────────┬──────────────────────┘
                        ┌───────▼────────┐
                        │ Noise / Twin   │
                        │ 噪声语义/设备孪生 │
                        └────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│ Core                                                               │
│ 版本化 IR、ProgramArtifact、Capabilities、Request、Result、Evidence│
└─────────────────────────────────────────────────────────────────────┘

外部控制面：MCP / REST / gRPC → 稳定公共 API 或 Application Services
```

所有箭头均表示允许的依赖或调用方向。Core 不反向依赖任何上层模块。

## 4. 九个稳定领域

| 领域 | 唯一职责 | 可以拥有 | 禁止拥有 |
| --- | --- | --- | --- |
| Core | 定义稳定语义和数据契约 | IR、程序产物、能力、请求、结果、证据、错误分类 | 硬件 SDK、调度策略、数值算法、网络框架 |
| Compiler | 将程序从一种产物变换为另一种产物 | 捕获、校验、分析、Pass、Lowering、代码生成 | 任务提交、设备生命周期、数值模拟 |
| Runtime | 组织一次执行及其生命周期 | 规划、资源、Session、分布式编排、恢复、观测 | 编译优化、状态向量/MPS/TN 算法、长期服务任务 |
| Simulation | 实现量子数值计算 | 状态向量、密度矩阵、MPS、张量网络、可微计算 | 用户策略、凭据、设备校准和集群资源治理 |
| Noise | 定义后端无关的噪声语义 | 噪声通道、设备噪声画像、读出误差 | 数值演化算法、厂商通信、Runtime调度 |
| Compute | 适配当前进程直接控制的算力 | CPU/GPU/NPU 生命周期、设备事实、通信原语 | 外部任务提交、数值算法、Runtime 调度 |
| Remote | 适配外部任务控制面 | 目标发现、提交、状态、取消、结果解码 | 本地设备生命周期、数值算法、Runtime 调度 |
| Twin | 构建并验证特定真实QPU的数字模型 | 冻结快照、校准条件化预测、漂移与真机验证 | 数值内核、厂商凭据、远程任务传输 |
| Ecosystem | 连接外部开发生态 | 框架适配、格式导入导出、插件入口 | 第二套规范 IR、Runtime 调度、设备直调 |

Application Services 是精简的应用组合层，不是另一套计算内核。只有同时组合多个稳定
API，并增加校验、策略或结构化失败语义的可复用流程才放在这里。单步编译、规划和执行
由 MCP、REST、CLI、IDE 或用户代码直接调用公共 API，不增加转发门面。

Compute 与 Remote 按控制边界划分，而不是按硬件类型划分：当前进程能直接创建张量、
选择设备和调用 Kernel 的资源属于 **Compute**；必须通过提交任务、查询状态和获取结果
使用的资源属于 **Remote**。因此同一型号 GPU 既可能作为本地 Compute，也可能位于远程
HPC 服务之后；真实 QPU 通常属于 Remote。

## 5. 目标代码结构

下列结构是目标态，不要求一次性搬迁。带 `contracts` 的目录只放接口和数据模型，
不放具体实现。

```text
flagquantum/
├── __init__.py              # fq 公共门面；只组合稳定入口，不承载领域实现
├── core/                    # 后端无关的稳定语义
│   ├── ir/
│   ├── artifacts/
│   ├── capabilities/
│   ├── execution/
│   ├── evidence/
│   └── errors/
├── compiler/                # 公共编译门面
│   ├── capture/
│   ├── analyses/
│   ├── passes/
│   ├── lowering/
│   ├── codegen/
│   └── pipelines/
├── runtime/
│   ├── planning/
│   ├── execution/
│   ├── sessions/
│   ├── distributed/
│   ├── recovery/
│   ├── observability/
├── simulation/
│   ├── statevector/
│   ├── density_matrix.py
│   ├── mps/
│   ├── tensor_network/
│   ├── differentiation/
│   └── kernels/
├── noise/                   # 后端无关的噪声通道与设备噪声画像
├── compute/                 # 当前进程直接控制的算力
│   ├── cpu/
│   ├── accelerators/        # 国产 GPU/NPU/异构加速器
│   └── communication/       # 集合通信、P2P 和异构互联
├── remote/                  # 通过外部任务控制面调用的算力
│   ├── qpu/
│   └── services/            # GPU/HPC 服务及云平台
├── twin/                    # 校准条件化、经真机验证的QPU数字模型
├── ecosystem/
│   ├── pytorch/
│   ├── jax/
│   ├── qiskit/
│   ├── openqasm/
│   ├── qir/
│   └── extensions/
├── services/                # 能力汇总和可复用的组合预检流程
├── algorithms/              # 面向用户的算法组合
├── benchmarking/            # 统一测评与证据生成
└── testing/                 # 契约、替换与一致性测试工具

contracts/                   # 契约快照与架构策略；运行时类型由 Core 所有
docs/architecture/           # 架构总纲、ADR 和专题设计
```

### 5.1 公共门面与调用顺序

`flagquantum/__init__.py` 是 `import flagquantum as fq` 的唯一公共门面。它可以组合
Compiler 与 Runtime 的稳定入口，但不定义编译、调度、数值计算或厂商适配
逻辑。普通开发者不得为了完成一次执行而手工构造内部指纹、证明或调度
对象。

Core 不是调用链上的首个“服务”，而是贯穿各阶段的共同语义。典型的
本地或国产加速器模拟任务遵循：

```text
用户 / PyTorch / JAX
  -> fq 公共门面
  -> Ecosystem 边界转换（仅当输入是外部对象时）
  -> Compiler（校验、优化、Lowering、目标合法性）
  -> Runtime（能力匹配、目标选择、计划与执行生命周期）
  -> Runtime 通过 Compute 解析设备、精度和通信资源
  -> Runtime 调用 Simulation（状态向量 / MPS / TN / 噪声 / 梯度）
  -> ExecutionResult + ExecutionEvidence
  -> Runtime
  -> Ecosystem 结果转换（如需）
  -> 用户
```

Compute 回答“本进程如何直接使用计算和通信设备”；Remote 回答“如何通过外部控制面
提交任务并取回结果”。Runtime 选择执行路径并管理生命周期，Simulation 只负责数值
计算，不直接导入 Compute，也不把远程服务伪装成本地设备。

真实 QPU 任务不经过 Simulation 或 Compute：

```text
用户 -> fq 公共门面 -> Compiler -> Runtime
     -> Remote -> 真实 QPU
     -> ExecutionResult + ExecutionEvidence -> Runtime -> 用户
```

GPU/HPC 服务与 QPU 同样位于 Remote 边界之后。Algorithms 通过公共
门面组合应用；Application Services 只通过公开契约提供有额外价值的组合预检；
协议适配器对单步操作直接调用稳定 API；Benchmarking 复用与用户相同的执行路径产生测评证据，
不建立绕过能力、安全或证据检查的专用快速路径。

### 5.2 迁移台账

机器可读的完整台账位于 `contracts/long-horizon-architecture-v1.json`。任何迁移项必须同时
声明责任团队、目标里程碑、当前权威位置、目标权威位置、适配器、完成证据、旧实现退出
条件和状态。

| 迁移项 | 当前权威位置 | 目标位置 | 完成证据 | 旧实现退出条件 |
| --- | --- | --- | --- | --- |
| 核心契约 | `core`、Runtime 执行对象、平台与执行 Provider 本地契约 | `core` | 跨领域产物、能力、请求、结果和证据均由 Core 定义并通过序列化测试 | 重复私有契约没有调用者；Provider 本地协议完成中立投影 |
| 编译器收敛 | `compiler` | `compiler` | 替换一条编译管线不修改 Runtime 和用户 API | 已删除 `_compiler` 和 `compilation` 旧入口 |
| 模拟算法抽离 | `simulation`、部分 `runtime/executors` | `simulation` | 真实引擎和契约假实现通过同一套一致性测试 | Runtime 下不再拥有数值算法 |
| 计算平台收敛 | `compute` | `compute` | 两种平台通过能力、精度、通信、回退和替换测试 | 通用代码不再导入厂商 Runtime |
| 执行目标收敛 | `runtime/executors`、`deployment` | `remote` | 模拟与 QPU/远程服务共享结果契约 | 后端选择和结果解码只存在于 Provider 后方 |
| 生态收敛 | `ecosystem` | `ecosystem` | 边界转换和往返一致性测试通过 | 外部框架对象不进入核心领域 |
| 服务/网关分离 | `services` | 主仓库组合流程；协议网关位于系统边缘 | 无 MCP SDK 时本地路径通过；适配器只调用公共 API 或组合服务 | 主仓库无生产 MCP 传输依赖和序列化转发门面 |

禁止只有目标目录而没有退出条件的迁移。一个迁移项完成后，必须删除或封闭旧权威入口，
不得让两套实现无限期并存。

## 6. 四类核心契约

四类跨领域契约全部由 Core 所有。Compiler、Runtime、Simulation 和 Provider 可以实现或
消费这些契约，但不得各自复制定义。尤其是 Runtime 只接收 Core 定义的可执行产物和
执行请求，不依赖 Compiler 包或 `compiler_contracts`。

### 6.1 ProgramArtifact

`ProgramArtifact` 是编译阶段之间的版本化信封，不取代现有 `CircuitIR`。

最小字段包括：

- `kind`：source、circuit、logical、physical、pulse、network、simulation_plan、executable；
- `schema_version`：产物模式版本；
- `payload`：该阶段的规范数据；
- `provenance`：来源、生成工具、输入摘要和父产物；
- `requirements`：精度、动态控制、通信、QEC、脉冲或网络需求；
- `extensions`：带命名空间的可选扩展，不改变核心字段含义。

编译器函数遵循：

```text
ProgramArtifact + TargetCapabilities + CompileOptions
    -> ProgramArtifact + CompileEvidence
```

### 6.2 TargetCapabilities

能力模型表达“目标真实具备什么”，而不是“用户希望它具备什么”。至少覆盖：

- 量子能力：门集、比特数、测量、动态线路、参数化；
- 加速器能力：设备类型、内存、原生精度、软件扩展精度、Kernel；
- 通信能力：P2P、集合通信、节点间通信、拓扑和带宽等级；
- 时间能力：Batch、Session、实时反馈时延范围；
- 容错能力：逻辑比特、码族、纠错周期、魔态和资源估算；
- 网络能力：多 QPU 拓扑、链路、纠缠生成和资源状态。

能力必须标注成熟度和证据来源。仅有接口声明不等于生产能力。

### 6.3 ExecutionRequest

执行请求只描述一次执行所需的稳定输入：

- 待执行产物及参数；
- 目标约束和执行模式；
- 精度、回退、近似和后端替换策略；
- 资源预算、随机种子、超时和恢复策略；
- 所需证据等级。

执行模式分为：

| 模式 | 用途 | 生命周期责任 |
| --- | --- | --- |
| Job | 一次批处理执行 | Runtime 管一次尝试，Compute Service 可管长期任务 |
| Session | 多次低开销连续执行 | Runtime 管会话内资源和状态 |
| Realtime Session | 有界时延的量子—经典反馈 | 专用 Runtime/Provider 能力 |
| Workflow | 编译、执行、训练、缓解、纠错等有向流程 | 工作流编排器组合原子契约 |

### 6.4 ExecutionResult 与 Evidence

结果不能只有数值。最小证据应包含：

- 请求、程序、编译产物、目标和环境摘要；
- 实际后端、设备、精度和执行路径；
- 计算与通信路径、拓扑、进程和设备驻留；
- 近似、截断、噪声、软件扩展精度及误差口径；
- CPU 回退、后端替换、重试和恢复记录；
- 时间、吞吐、内存等性能统计及测量方法；
- 成功、失败或部分完成的结构化原因。

任何未声明 CPU 回退均视为契约违约，而不是性能优化。

## 7. 三条端到端主路径

### 7.1 本地或分布式模拟

```text
用户程序
 -> Ecosystem/API 捕获
 -> CircuitIR / ProgramArtifact
 -> Compiler 校验与面向模拟的优化
 -> Runtime 选择执行计划和资源
 -> Simulation Provider
 -> 状态向量 / MPS / TN / Noise / Diff
 -> ExecutionResult + Evidence
```

Runtime 不实现张量收缩或量子门 Kernel；Simulation 不自行决定集群、账号和回退策略。

### 7.2 真实 QPU

```text
用户程序
 -> 规范 IR
 -> Compiler 按 QPU 能力完成逻辑/物理 Lowering
 -> Runtime 建立 Job、Session 或 Realtime Session
 -> QPU Provider 隔离厂商 SDK、凭据、校准和结果格式
 -> 统一 ExecutionResult + QPU Evidence
```

真实 QPU 因而属于 **Provider 的具体实现**；QPU 的调度生命周期属于 Runtime；
面向 QPU 的变换属于 Compiler；QPU 的能力词汇和结果模式属于 Core。

### 7.3 Application Services 与协议适配器

```text
LLM / IDE / MCP Host
 -> MCP / REST / gRPC / CLI 薄适配器
 -> 稳定公共 API（单步）或 FlagQuantum Services（组合预检）
 -> Compiler / Runtime / Deployment 公开契约
 -> 结构化结果与证据
```

- 主仓库不依赖 MCP SDK，也不包含 LLM 决策逻辑；
- Services 只保留能力汇总、执行预检和部署预检等可复用组合流程；
- 适配器负责协议解码和结果序列化，不为单步操作增加服务包装；
- Compute Service 负责租户、鉴权、配额、预算、持久化任务和协议生命周期；
- 替换 MCP 版本或网关不影响本地 SDK 和计算语义。

## 8. Runtime 与 Simulation 的边界

两者是“执行导演”和“数值演员”的关系：

| 问题 | 责任方 |
| --- | --- |
| 用哪个目标、多少设备、什么执行模式 | Runtime |
| 状态向量如何更新、MPS 如何截断、TN 如何收缩 | Simulation |
| 何时建立通信组、保存检查点、恢复任务 | Runtime |
| 前向、反向和梯度 Kernel 如何计算 | Simulation |
| 是否允许降低精度或回退 CPU | Request Policy + Runtime |
| 实际发生了何种精度、通信和回退 | Compute/Remote 采集，Runtime 汇总为 Evidence |

Runtime 组合 Simulation 与 Compute：先解析设备、精度和通信资源，再把明确的数据与
设备参数交给 Simulation。Simulation 不导入 Compute。因而数值引擎和设备适配都可独立
替换。外部任务由 Runtime 交给 Remote，不进入本地 Simulation 链路。

## 9. 国产异构算力

每一家国产 GPU/NPU/异构芯片通过独立 Compute 适配接入，不向上暴露厂商对象。
Compute 至少实现：

1. 设备发现、生命周期和内存能力；
2. Kernel 注册、编译或调用；
3. 原生精度与 Double-Single 等软件扩展精度声明；
4. 节点内 P2P、集合通信和节点间通信能力；
5. 设备拓扑、异构互联和不可用原因；
6. 实际设备驻留、Kernel、通信、精度和 CPU 回退证据；
7. 契约一致性和替换测试。

上层只依赖能力，不根据厂商名称分支。硬件选择由 Runtime Planner 根据请求约束、
能力证据和策略完成。只支持单精度的设备可以声明软件扩展双精度能力，但必须明确适用
算例、数值验证、性能开销和真实执行路径，不得等同宣称原生双精度。

## 10. 面向未来技术变化的扩展方式

### 10.1 容错量子计算

新增逻辑、物理和容错编译产物，扩展 `TargetCapabilities.fault_tolerance`，并以 Workflow
组合逻辑—物理映射、QEC 周期、魔态蒸馏和资源估算。基础 Runtime 不感知具体码算法。

### 10.2 量子—经典实时反馈

增加 `RealtimeSession` 契约，显式声明反馈时延、控制位置、可用指令和超时语义。
普通远程 Job 不伪装成实时能力；Compiler 和 Provider 分别验证动态线路语义与硬件能力。

### 10.3 多 QPU 与量子网络

新增 Network Artifact、分布式 QPU 目标描述、量子网络拓扑和纠缠资源 Provider。
Runtime 负责跨 QPU 编排，Compiler 负责程序划分，Provider 负责纠缠资源的真实操作和证据。

### 10.4 脉冲级编译与控制

新增 Pulse Artifact 和相应 Compiler Pipeline。脉冲对象不塞入通用 CircuitIR；只有声明
脉冲能力的 QPU Provider 才能接受该产物。

### 10.5 新模拟范式与 AI 编译优化

新的模拟算法实现 Simulation Contract；AI 优化器作为 Compiler Pass 或 Planner Policy
接入。模型建议必须经过确定性校验，不能绕过能力检查和语义一致性验证。

## 11. 依赖与导入规则

```text
Core <- Compiler
Core <- Runtime
Core <- Simulation
Core <- Noise
Core <- Compute + Vendor Runtime
Core + Noise + Simulation/Remote public APIs <- Twin
Core + Deployment <- Remote + External SDK
Core + Compiler/Runtime/Deployment public APIs <- Application Services
Public API + Core <- Ecosystem
Public APIs + Application Services <- Protocol Adapters
```

强制规则：

- Core 不导入 Runtime、Simulation、Compute、Remote、Ecosystem 或网关；
- Compiler 不导入 Runtime、Compute、Remote 或设备 SDK；
- Runtime 不导入 Compiler 包；二者共享的数据契约必须由 Core 所有；
- Simulation 不导入 Runtime 策略和部署代码；
- Compute 与 Remote 不得混成同一接口；
- 通用代码不导入具体 Compute 或 Remote 实现；
- Ecosystem 对象在边界完成转换，不向核心层泄漏；
- 网关不直调 Kernel、设备或编译器内部模块；
- 临时例外必须登记、设置责任人和移除条件，并由架构检查器跟踪。

这些规则应由 `architecture.toml` 和 `tools/check_architecture.py` 自动执行。

## 12. 多团队并行开发规则

按领域而不是按技术栈划分所有权：

| 团队/工作流 | 主要修改范围 | 依赖的稳定接口 |
| --- | --- | --- |
| Core/IR | `core`、契约模式 | 无下游具体实现 |
| Compiler | `compiler` | ProgramArtifact、Capabilities |
| Runtime | `runtime` | Core 中的 ExecutionRequest、ExecutionResult |
| Simulation | `simulation` | Simulation Contract、Evidence |
| Noise | `noise` | Core IR 与后端无关的噪声语义 |
| Compute | `compute` | Core 中的能力、精度和设备事实契约 |
| Remote | `remote` | Core 中的请求、结果和远程任务契约 |
| Twin | `twin` | Noise、Simulation与Remote的公共入口 |
| Ecosystem | `ecosystem` | 公共 API、ProgramArtifact |
| Application Service / Gateway | `services`、协议边缘 | Public API + composite workflow contract |

跨领域变更必须先修改契约提案和契约测试，再修改实现。禁止通过导入对方内部模块解决
短期联调问题。每个领域至少维护：所有者、公共入口、契约测试、替换用假实现和变更记录。

## 13. 能力成熟度与发布

所有能力分别标记，而不是给整个软件贴一个笼统标签：

```text
declared -> prototyped -> validated -> production
```

- **declared**：有契约和失败语义；
- **prototyped**：存在可运行实现，但尚无完整证据；
- **validated**：通过代表性环境、精度、替换和一致性测试；
- **production**：具有持续测试、文档、运维边界和发布承诺。

接口存在、Mock 通过或单机演示成功，均不能自动提升为生产能力。

## 14. 迁移路线

迁移遵循“先契约、后替换；先纵切、后横展”，不进行一次性目录重构。
当前总控顺序固定为：

```text
冻结新的横向抽象
 -> 完成并简化 Compiler 边界
 -> 打通最小 CPU 纵向链路
 -> 将已证明链路迁入目标目录
 -> 每轮删除或明确冻结历史与过渡代码
```

在最小 CPU 链路完成前，不新开其他横向架构轨道。只有当现有权威类型
确实无法支撑当前链路，且通过契约提案与“先做减法”审查时，才能增加
新的跨领域抽象。目录迁移必须跟随已运行的纵向切片，不先建立空目录或
平行实现。

### 阶段 0：冻结事实与边界

- 盘点公共 API、现有权威实现和真实能力；
- 用依赖检查固化禁止方向；
- 建立当前行为的契约测试和证据基线。

### 阶段 1：最小端到端契约

- 稳定 ProgramArtifact、TargetCapabilities、ExecutionRequest、ExecutionResult；
- 让现有 `CircuitIR` 通过兼容适配器进入新服务；
- 跑通“捕获—校验—规划—模拟—结果—证据”。

### 阶段 2：Compiler、Runtime、Simulation 解耦

- 建立各自公共门面；
- 将跨层直调替换为契约；
- 每拆出一项实现，都用替换测试证明消费者无需修改。
- 每轮记录新增、复用、删除和冻结的代码；若只增加新路径而旧权威不退出，
  该轮不得标记为迁移完成。

### 阶段 3：Compute 与 Remote 收敛

- 将本进程直接控制的设备和通信能力收敛到 Compute；
- 将 QPU、远程 GPU/HPC 服务和云平台的任务控制面收敛到 Remote；
- 逐个接入国产加速器、真实 QPU 和远程服务；
- 禁止建立第二套设备注册和能力发现系统。

### 阶段 4：生态与服务化

- 统一 PyTorch/JAX、OpenQASM/QIR 和第三方生态边界；
- 稳定少量有复用价值的 Application Services；
- 让 MCP/REST/gRPC/CLI 薄适配器直接复用公共 API 和组合流程。

### 阶段 5：未来能力插件

- 按真实项目需求增加容错、实时、多 QPU、量子网络和脉冲产物；
- 每项能力独立成熟，不改写既有基础执行模型。

## 15. 架构完成判据

一个模块只有同时满足以下条件，才算完成解耦：

1. 职责、输入、输出和失败语义已文档化；
2. 对外只暴露版本化契约，不泄漏内部或厂商对象；
3. 至少存在两个实现，或一个真实实现加一个契约假实现；
4. 替换实现时消费者代码无需修改；
5. 契约测试、一致性测试和架构依赖检查通过；
6. 不支持能力、降级和回退可以被机器识别；
7. 文档明确当前成熟度，不把目标态描述为已实现。
8. 普通功能修改原则上只涉及一个主领域的内部实现；若同一类修改反复需要
   穿越四个及以上领域，必须停止扩大实现并进行边界评审；
9. 迁移后的每个领域均有简短 README 和一条可执行的“十分钟黄金路径”，说明
   职责、禁止事项、允许依赖、公开入口和最小修改方式；
10. 公共 API 不要求使用者构造或理解内部指纹、来源记录、合法性证明、能力
    快照标识、调度对象或证据内部结构；
11. 除契约、确定性和防篡改测试外，保留可读的场景测试；新开发者能仅依据
    领域 README 和黄金路径，独立完成、测试并解释一项代表性小修改。
12. 每项非平凡变更在合入前完成“先做减法”审查：删除无当前用例的脚手架、
    透传包装、重复校验、重复表示和机械化测试；不能用当前产品行为和领域
    归属解释必要性的抽象，不得以测试通过为由合入。

新增契约、identity、verdict、registry 或中间表示时，提案必须说明现有权威
类型为何无法表达已验证需求。超大内部模块只在行为和边界稳定后按职责拆分，
不得仅为缩短文件而增加新的公共概念。“十分钟”是人工可用性验收目标，
不得伪造为无法反映真实贡献者体验的 CI 通过项。
新增 manager、registry、factory、protocol、helper 层或中间对象，必须对应
一项独立的当前职责或第二个具体用例，不得只以未来可能需要为依据。

## 16. 架构决策治理

以下变更必须提交 Architecture Decision Record（ADR）：

- 修改核心数据模型或依赖方向；
- 引入新的跨领域契约或 Provider 类型；
- 改变 Stable Core 公共 API 或序列化模式；
- 新增生产依赖、长期兼容层或跨仓库协议；
- 允许新的精度降级、CPU 回退或后端替换策略。

ADR 至少包含问题、约束、备选方案、决定、兼容性影响、迁移路径、验证方法、所有者和
退出条件。实现细节可持续演进，核心边界不得以“先这样以后再换”为依据临时突破。

## 17. 当前采用规则

本文描述目标架构，不会自动提升任何实验能力的成熟度。现有 `CircuitIR`、Runtime Plan、
数值契约、目标能力和执行结果，在各自稳定范围内仍是权威实现。目标结构通过兼容适配器
和受保护 API 流程逐步采用；每次迁移都必须具有聚焦测试、回滚边界和可验证证据。
