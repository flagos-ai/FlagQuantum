# FlagQuantum 开源前 API 质量提升与冻结计划

## 1. 文档目的

本文档用于指导 FlagQuantum 在首次公开发布前完成 API 收敛。目标不是永久冻结
所有实现细节，而是建立一套规模小、语义一致、可扩展、可测试并能长期兼容的
公开 API。

本文档覆盖：

- 当前 API 的质量判断与主要优点；
- 已确认的 API 重叠、断链和歧义；
- 建议的目标 API 与命名空间；
- 分阶段实施顺序；
- 每个阶段的验收标准；
- 首次公开版本的冻结条件。

API 冻结后的强制保护、CODEOWNERS、required checks 和变更授权流程由
[`PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md) 规定。

## 2. 结论

FlagQuantum 当前 API 主干方向正确：

```text
Circuit / Module
       ↓
      IR
       ↓
     plan
       ↓
      run
       ↓
ExecutionResult
```

其中 `fq.Circuit`、FlagQuantum IR、`fq.plan`、`fq.run`、`fq.Module`、
`fq.train` 以及统一 Result 模型值得成为长期架构基础。

当前整体 API 质量评估约为 **7.5/10**；核心主线约为 **8.5/10**。不建议把
当前 `public_api_v1.json` 中的全部 60 个 stable exports 原样作为首次公开版本的
永久承诺。首次开源前应完成一次集中式 API 收敛，目标质量为 **9/10**。

冻结原则：

1. 冻结用户心智、核心语义和兼容规则，而不是冻结所有内部实现。
2. 核心入口保持少而稳定；后端专属能力通过命名空间或 experimental 暴露。
3. 简单路径必须短，专业路径必须可检查、可复现、可控制。
4. 用户看到的执行计划必须与真正执行的计划一致。
5. 公开接口原则上只增不破；删除必须经过正式弃用周期。

## 3. 当前设计中值得保留的部分

### 3.1 统一程序模型

`fq.Circuit` 建立在版本化 FlagQuantum IR 之上，已经具备确定性序列化、内容哈希、
参数表达式、测量节点和可验证错误语义。这是后端无关生态的正确基础。

### 3.2 执行、规划和训练职责分离

- `fq.plan` 负责规划和解释；
- `fq.run` 负责一次执行；
- `fq.Module` 负责 PyTorch 参数所有权；
- `fq.train` 负责优化器循环；
- `ExecutionResult` 和 `TrainingResult` 负责稳定输出。

这种职责划分应保留。

### 3.3 PyTorch 原生体验

`fq.Module` 是普通 `torch.nn.Module`，能够参与 PyTorch 参数管理、优化器、设备迁移
和自动微分。FlagQuantum 应继续坚持 PyTorch Native 主线，不重新发明一套独立训练
对象体系。

### 3.4 已存在的 API 治理基础

当前仓库已经具备：

- 稳定 API 快照；
- executable contract tests；
- documentation source-of-truth；
- experimental 命名空间；
- DeprecationWarning 路径；
- 语义化版本和发布策略。

后续工作的重点是收缩和统一，而不是推翻重来。

## 4. 必须在开源前解决的问题

### API-001：`ExecutionPlan` 没有成为执行输入

**优先级：P0**

当前 `fq.plan(circuit)` 主要用于检查和展示，而 `fq.run(circuit)` 会再次规划。用户
显式生成的 plan 不是执行时的唯一事实来源，因此无法保证“看到的计划就是执行的
计划”。

目标语义：

```python
result = fq.run(circuit, options=options)

# 等价的专业路径
plan = fq.plan(circuit, options=options)
result = fq.run(plan)
```

验收标准：

- `fq.run(ExecutionPlan)` 直接执行该计划；
- plan 包含程序哈希、配置哈希和目标环境约束；
- 计划失效时明确报错，不允许静默重新规划；
- 自动路径与显式 plan 路径产生等价计划和结果；
- `ExecutionResult.plan` 指向实际执行的计划。

### API-002：存在多套用户可见规划接口

**优先级：P0**

当前存在 `fq.plan`、`Circuit.plan`、`Circuit.runtime_plan`、
`plan_runtime_selection` 等相近入口。它们返回的对象和回答的问题并不完全相同，
容易造成用户不知道应该使用哪一个。

目标：

- 用户文档只推荐 `fq.plan`；
- `Circuit.plan()` 最多作为 `fq.plan(circuit)` 的严格等价便捷写法；
- runtime selection 成为 `ExecutionPlan` 的一部分；
- 专项分析器进入 `fq.planning` 或 internal namespace；
- 不在根命名空间永久承诺多个相似 planner。

### API-003：存在多套相互竞争的执行入口

**优先级：P0**

当前同时存在：

- `fq.run`；
- `Circuit.run`；
- `fq.run_native`；
- `fq.run_mps`；
- `fq.run_tensor_network`；
- `fq.run_target`；
- 多个 distributed/noisy 专用 runner。

目标：

```python
fq.run(program_or_plan, options=...)
```

作为唯一推荐入口。其他入口按以下方式处理：

- `Circuit.run()` 必须与 `fq.run(circuit)` 语义严格等价；
- backend-native 执行迁入 `fq.backends.<backend>.run`；
- 尚未稳定的分布式、硬件和研究能力迁入 `fq.experimental`；
- 根级专用 runner 不进入长期 stable core。

### API-004：执行配置存在多个事实来源

**优先级：P0**

执行行为目前可能由以下位置共同决定：

- `Circuit` 构造参数；
- `RuntimeConfig`；
- `RuntimePolicy`；
- `fq.run(**options)`；
- 环境变量；
- backend-specific 参数。

这会产生覆盖顺序不清、计划难复现和文档组合爆炸。

建议引入不可变的统一配置对象：

```python
options = fq.ExecutionOptions(
    mode="auto",
    backend="auto",
    device="auto",
    batch_size=1,
    precision="complex64",
)
```

必须明确配置优先级，例如：

```text
显式 ExecutionOptions
    > Module RuntimePolicy
    > Circuit 创建时捕获的默认配置
    > 进程级默认配置
    > 框架内置默认值
```

未知选项必须立即报错，不能被无限制 `**options` 静默接收。

### API-005：`mode`、`backend`、`device` 和 `target` 概念重叠

**优先级：P0**

建议固定以下定义：

| 概念 | 含义 | 示例 |
| --- | --- | --- |
| `mode` | 数学状态或算法表示 | statevector、mps、tensor_network、density_matrix |
| `backend` | 执行实现 | pytorch、jax、triton、vendor |
| `device` | 执行设备 | cpu、cuda、npu、flagos |
| `target` | 用户需要的输出 | state、expectation、samples、amplitudes |

相同概念必须在 `plan`、`run` 和 `RuntimePolicy` 中使用相同名称、相同枚举和相同
默认值。模式别名只允许存在于有明确删除版本的兼容层。

### API-006：命名没有完全统一

**优先级：P0**

当前典型问题：

- `run(mode=...)` 与 `plan(state_mode=...)`；
- `bsz` 与用户更容易理解的 `batch_size`；
- `Circuit(n_qubits=...)`、`Circuit(n_wires=...)`、`Circuit(nqubits=...)`。

首次公开版本建议：

- 统一使用 `mode`；
- 公开接口统一使用 `batch_size`；
- Circuit 只公开 `n_qubits`；
- `wire` 仅保留为 IR、映射和内部索引术语；
- 私有阶段直接迁移内部调用，不把历史拼写带入公开兼容承诺。

### API-007：Measurement 存在两个输入来源

**优先级：P0**

测量请求既可以保存在 `CircuitIR.measurements`，也可以通过
`fq.run(..., measurements=...)` 传入。当前显式参数可能覆盖 IR 中的请求，但该行为
不够显眼。

必须选择并文档化一种模型：

1. 默认只使用 IR measurements，显式 measurements 与其冲突时报错；或
2. 显式提供 `measurement_policy="replace" | "append"`。

不允许存在隐式覆盖。

### API-008：`ExecutionResult` 的稳定边界不够严格

**优先级：P0**

当前问题包括：

- `fq.run` 的类型标注仍为 `Any`；
- `ExecutionResult.plan` 允许 `Any`；
- `MeasurementResult.value` 为 `Any`；
- 多个结果字段为 optional，用户不知道哪个一定存在；
- `ExecutionResult.__getattr__` 会向 native result 透传属性；
- `metrics/runtime/provenance` 主要是自由格式 Mapping。

目标：

```python
fq.run(...) -> fq.ExecutionResult
fq.plan(...) -> fq.ExecutionPlan
fq.train(...) -> fq.TrainingResult
```

建议增加明确访问器：

```python
result.expectation()
result.statevector()
result.samples()
result.measurement("energy")
result.native()  # 明确标记为非稳定后端对象
```

验收标准：

- 缺少对应结果时给出明确异常；
- stable 字段具有可检查类型；
- native 结果不再通过隐式属性透传污染稳定 API；
- metadata schema 带版本；
- Result 可以稳定序列化或明确声明不可序列化字段。

### API-009：stable root surface 过大

**优先级：P0**

当前 stable manifest 包含 60 个根级导出，其中混入：

- MPS 生产验收与 evidence 类型；
- backend-native runner；
- distributed 专用入口；
- 部署辅助函数；
- 便利性信息函数。

这些接口的成熟度和长期稳定性不同，不应获得同等级承诺。

建议把公开能力分成三层：

#### Stable Core

候选范围控制在约 15–25 个：

```text
Circuit, CircuitIR, Instruction
Parameter, ParameterExpression
MeasurementNode, MeasurementResult, ObservableNode
Module, RuntimePolicy
ExecutionOptions, ExecutionPlan, ExecutionResult
TrainingOptions, TrainingResult
plan, run, train
experimental
__version__
```

#### Stable Extensions

通过命名空间提供：

```text
fq.noise
fq.deployment
fq.interop
fq.backends
fq.compiler
```

#### Provisional / Experimental

包括：

- rank、shard、transport 等分布式结构；
- MPS production gate 和 benchmark evidence；
- 硬件特定控制；
- 未完成的动态线路和新后端；
- 研究阶段 planner 与优化策略。

### API-010：compatibility root 边界过于宽松

**优先级：P0**

当前 lazy `__getattr__` 和 `__dir__` 仍可能让用户发现并调用不属于 stable `__all__`
的历史接口。首次开源后，用户通常会把“能够从 `fq` 访问”理解为公开 API。

目标：

- 根命名空间只能发现 stable core 和明确标记的命名空间；
- 历史接口在首次公开前完成迁移或删除；
- 必须保留的兼容接口进入 `flagquantum.compat`；
- 不使用隐式 native attribute forwarding 扩大稳定面。

## 5. 应在开源前尽量解决的问题

### API-011：compile、plan 和 run 的职责边界不够清楚

**优先级：P1**

当前 `compile_for_backend()` 可单独产生 IR，而 `plan()` 内部又会进行编译。应明确：

```text
Circuit/IR → compile → plan → run
```

或者把 compile 定义为 plan 的内部阶段。普通用户不应被要求手动拼接两个可能重复
工作的入口。

推荐：

- `fq.plan` 自动完成规范化、编译、路由和资源规划；
- 编译专家使用 `fq.compiler.compile`；
- `ExecutionPlan` 记录编译后的 IR 和所有 transformation provenance；
- 相同输入和配置产生稳定 plan identity。

### API-012：`Module.forward`、`Module.execute` 和 `fq.run` 的关系需要固定

**优先级：P1**

当前：

- `Module.forward()` 返回 Tensor，以符合 PyTorch；
- `Module.execute()` 返回 `ExecutionResult`；
- `fq.run()` 主要接受 Circuit/IR。

这个设计可以成立，但必须明确：

- `module(inputs)` 永远返回可参与 autograd 的 Tensor；
- `module.execute(inputs)` 返回带计划和 provenance 的结果；
- `fq.run(module, ...)` 是否支持必须在开源前做唯一决定；
- 不允许不同入口对梯度、batch 或 observable 产生不同默认语义。

### API-013：`fq.Module` 构造方式较多

**优先级：P1**

当前同时支持 builder、参数化 Circuit、扁平参数、命名参数和多种初始化方式。应通过
黄金路径确认哪些组合是真正需要的公开能力。

建议：

- 保留一个最简单 builder 路径；
- 保留一个命名 Parameter 路径；
- 明确输入 batch 与参数 batch 的广播规则；
- 把复杂 compilation/deployment binding 放入配置对象或 classmethod；
- 为每个公开构造模式提供独立 contract test。

### API-014：训练入口没有完全覆盖已宣传的生命周期

**优先级：P1**

`Module` 已有 checkpoint 能力，但 `fq.train` 的公开签名主要覆盖 optimizer、objective、
steps、inputs、logging 和 callback。Checkpoint、Resume、早停、验证集和分布式生命周期
尚未通过一个统一顶层入口表达。

应在以下两种方案中选择：

1. 扩展 `TrainingOptions` 和 callback 协议，形成完整训练入口；或
2. 将 `fq.train` 定义为最小训练循环，并明确高级生命周期由用户或 Trainer 扩展负责。

文档和宣传必须与最终选择一致。

### API-015：普通训练与分布式训练入口分裂

**优先级：P1**

`train_distributed_statevector` 和 `train_distributed_mps` 暴露了实现模式。长期目标应是：

```python
fq.train(module, options=fq.TrainingOptions(execution=...))
```

如果分布式训练尚不能满足统一语义，应保留在 `fq.experimental.distributed`，而不是
提前进入 stable root。

### API-016：Noise 存在多套平行入口

**优先级：P1**

当前既可以使用 `fq.run(..., noise_model=...)`，又存在 density matrix、noisy MPS、
noisy statevector 等专用入口。

建议：

- 普通用户只使用 `fq.run(..., noise=...)`；
- mode 由 options 或 planner 决定；
- 专用模拟器迁入 `fq.backends`；
- 所有路径返回相同稳定结果契约；
- noise capability 不支持时在 planning 阶段失败。

### API-017：错误模型尚未统一

**优先级：P1**

公开 API 当前会抛出 `ValueError`、`TypeError`、`RuntimeError`、
`NotImplementedError` 以及多个领域特定异常。建议建立精简异常层级：

```text
FlagQuantumError
├── ValidationError
├── PlanningError
├── CompilationError
├── ExecutionError
├── CapabilityError
└── SerializationError
```

底层异常可以作为 `__cause__` 保留，但用户不应依赖 PyTorch、JAX、NCCL 或 provider
的偶然异常文本。

### API-018：ExecutionPlan 的可移植和可序列化边界需要定义

**优先级：P1**

需要明确 plan 是：

- 仅当前进程可执行的对象；
- 可跨进程序列化的描述；还是
- 可部署、可签名的执行包。

推荐分层：

```text
ExecutionPlan       本地可检查、可执行、带环境约束
DeploymentPackage  可序列化、可签名、可提交给硬件/provider
```

二者不能使用含糊的同一对象承担所有职责。

### API-019：扩展点需要稳定协议而不是稳定实现类

**优先级：P1**

生态建设需要允许第三方增加 backend、provider、compiler pass 和 measurement，而不
修改 FlagQuantum core。应优先冻结 Protocol 和注册机制，而不是冻结每个具体后端
类。

至少定义：

- backend capability protocol；
- planner cost/capability contribution；
- executable runner protocol；
- result normalization protocol；
- provider submission protocol；
- version negotiation 和 conformance tests。

## 6. 建议的目标 API 草案

以下只定义目标形态，不应在实现完成前直接宣布永久冻结：

```python
import flagquantum as fq

circuit = (
    fq.Circuit(n_qubits=2)
    .h(0)
    .cx(0, 1)
)

options = fq.ExecutionOptions(
    mode="auto",
    backend="auto",
    device="auto",
    batch_size=1,
)

# 最短路径
result = fq.run(circuit, options=options)

# 可检查、可复现路径
plan = fq.plan(circuit, options=options)
print(plan.summary())
result = fq.run(plan)
```

训练路径：

```python
module = fq.Module(build_circuit, parameters={"theta": ()})

training = fq.train(
    module,
    optimizer=optimizer,
    objective=objective,
    options=fq.TrainingOptions(steps=100),
)
```

后端专家路径：

```python
native = fq.backends.mps.run(circuit, options=options)
```

实验能力：

```python
plan = fq.experimental.distributed.plan(circuit, cluster=cluster)
```

## 7. 五条开源前黄金用户路径

API 冻结前必须用真实、可执行代码验证以下路径：

1. **第一个 Bell circuit**：构建、执行、读取 state/measurement。
2. **参数化 VQE**：Parameter、Hamiltonian、梯度和优化。
3. **PyTorch 模型集成**：`fq.Module` 进入普通 `nn.Module` 和 optimizer。
4. **自动后端选择**：同一代码在 Statevector、MPS、TN 之间规划。
5. **互操作与部署**：Qiskit 转换、deployment package 和硬件/provider 提交。

每条路径必须满足：

- 只导入 `flagquantum as fq` 或稳定子命名空间；
- 不访问 runtime/internal/testing/evidence；
- 示例由 CI 直接执行；
- 错误具有稳定类型和可行动信息；
- CPU 环境至少能够完成构建和 planning；
- plan、execution 和 result provenance 可以相互核对。

## 8. 实施阶段

### Phase 0：冻结现状和用户旅程

- 保存当前 `public_api_v1.json` 作为内部迁移基线；
- 使用 `contracts/public-api-v0.2-baseline.json` 记录导出、签名、默认值和
  dataclass 字段，并由 `tools/public_api_snapshot.py` fail closed；
- 建立五条黄金路径测试；
- 收集所有根级导出、函数签名和文档引用；
- 禁止在收敛期间继续增加根级 API。

完成标准：可以自动检测任何公开签名和示例变化。

实施记录（2026-08-31）：

- `contracts/public-api-v0.2-baseline.json` 已记录当前 60 个 stable exports；
- `tools/public_api_snapshot.py` 已接入 pre-commit 和 `CI / quality`；
- `tests/api_contract/test_open_source_golden_paths.py` 已覆盖 Bell circuit、
  参数化目标训练、PyTorch 组合、自动规划、本地部署和 Qiskit round trip；
- core CI 执行前五条路径，现有 Qiskit 2.0/2.5 可选依赖矩阵执行 Qiskit 路径；
- 该记录是迁移起点，不代表当前 60 个 exports 已被批准为最终 Stable Core。

### Phase 1：收缩 stable surface

- 确定 Stable Core 最终名单；
- 将后端、evidence 和验收类型迁入子命名空间；
- 删除首次开源不需要承担的历史 compatibility root；
- 更新 API manifest、文档和 import contract。

完成标准：根级 stable exports 控制在约 15–25 个，且每一个都有明确用户场景。

实施记录（2026-08-31）：

- `contracts/public-api-v1-candidate.json` 已将当前 60 个根级导出逐项且唯一分类；
- 候选 Stable Core 为 22 项，其中保留 20 项、新增 `ExecutionOptions` 与
  `ExecutionPlan`；
- 其余接口按 stable extension、experimental 和开源前移除三类给出目标位置；
- `API_CHANGE_PROPOSAL_001_STABLE_CORE.md` 已形成可审查迁移提案；
- Stable Core 分类与命名空间迁移已于 2026-08-31 获得 API owner 批准；最终 API
  freeze 仍需单独批准。
- stable extension 的 `backends`、`compiler`、`deployment`、`noise`、`operators`
  路径与 experimental 的 `distributed`、`mps`、`planning` 路径已经可导入；
- 公开黄金路径中的部署示例已停止依赖历史根级 deployment export。
- README、架构说明、主动维护的 guides/reference 和 examples 已迁移到新命名空间；
  `tools/check_legacy_root_api_usage.py` 已接入 pre-commit 与 CI，禁止回流旧根接口。
- production package、tools 和 benchmarks 已完成同一批旧根入口迁移，并纳入上述
  防回流门禁；`benchmarks/mps_stability.py` 因源码哈希绑定 A800 实测证据而保留为
  精确的只读历史例外，不能在未重新测量时仅为改导入路径而更新证据哈希；剩余迁移面
  仅为测试与由旧 manifest 生成的审计文档。
- 首批测试迁移已清除 16 个文件、21 处旧根调用；第二批进一步清除 deployment、
  distributed training、backend basic 与 algorithm 等 13 个文件、58 处旧根调用；
  第三批清除 noise 与 tensor-network 2 个测试文件、113 处旧根调用；第四批完成
  MPS、native runtime、trajectory、hybrid JAX 与专项 planner 等最后 6 个文件、
  152 处旧根调用迁移。测试侧债务已归零，
  `contracts/legacy-root-api-test-debt.json` 现为零基线，CI 禁止任何旧根接口回流。
- 正式稳定且可发现的根 API（`fq.__all__`、`dir(fq)` 和
  `docs/public_api_v1.json`）已从 60 项收缩为当前已实现的 20 项 Stable Core；
  `ExecutionOptions` 与 `ExecutionPlan` 必须等待后续语义提案批准和实现后才能加入，
  不以占位导出的方式虚增为 22 项。历史惰性属性访问仅作为未承诺的仓库兼容层暂留，
  不属于 stable manifest，后续按独立清单继续移除。
- Proposal 001 明确分类的 37 个迁移接口和 3 个开源前移除项已从根级
  `__getattr__` 关闭；错误信息直接给出规范命名空间。`flagquantum.api` 仍保存历史对象
  身份以供内部审计，但测试不再把这些对象的根级可访问性当作兼容承诺。

### Phase 2：统一命名和配置

- 引入 `ExecutionOptions`；
- 统一 `mode/backend/device/target`；
- `state_mode` 迁移为 `mode`；
- `bsz` 迁移为 `batch_size`；
- Circuit 公开构造只使用 `n_qubits`；
- 明确配置覆盖顺序。

完成标准：`plan`、`run`、`Circuit` 和 `RuntimePolicy` 不再使用冲突术语。

实施记录（2026-08-31）：

- `API_CHANGE_PROPOSAL_002_EXECUTION_OPTIONS.md` 与
  `contracts/execution-options-v1-candidate.json` 已登记；
- Proposal 002 已完成实现、根级清单授权和 default/runtime/distributed 验证；
  `ExecutionOptions` 已成为 `plan/run/Circuit/RuntimePolicy` 的统一稳定输入；
- 候选采用不可变字段级 overlay、严格优先级、无 `extras` 逃生口，以及 approximation
  和 backend fallback 默认关闭的 fail-closed 语义。

### Phase 3：建立可执行计划

设计记录（2026-08-31）：

- `API_CHANGE_PROPOSAL_003_EXECUTION_PLAN.md` 与
  `contracts/execution-plan-v1-candidate.json` 已登记；
- 当前状态为 draft pending approval，只定义 identity、序列化、stale-plan、环境约束和
  `fq.run(plan)` 目标语义，尚未授权实现或加入稳定根清单；
- `ExecutionPlan` 定位为本地可检查、可缓存、可恢复和可执行的计划，
  `DeploymentPackage` 继续承担签名、provider 提交和远程生命周期。

- `fq.plan` 返回稳定 `ExecutionPlan`；
- `fq.run(plan)` 成为正式路径；
- 实现 program/options/environment fingerprint；
- 明确 stale plan 错误；
- 消除重复编译和静默重新规划。

完成标准：自动路径与显式计划路径具有一致输出和 provenance。

### Phase 4：收紧 Result 和 Measurement

- 去除 stable API 返回值中的 `Any`；
- 定义 measurement 来源和组合规则；
- 增加稳定结果访问器；
- 隔离 backend-native 对象；
- 版本化 metadata schema。

完成标准：用户无需检查多个不明确的 optional 字段即可读取所请求结果。

### Phase 5：统一 Module 和 Training

- 固定 `forward`、`execute`、`run` 的关系；
- 固定输入 batch、参数 batch 和 observable 语义；
- 决定 checkpoint/resume 的顶层边界；
- 决定 distributed training 是统一能力还是 experimental；
- 更新 PyTorch/JAX 边界说明。

完成标准：同一 Module 在 eager、train 和显式 execute 路径中没有默认语义漂移。

### Phase 6：扩展协议与互操作

- 固定 backend/provider Protocol；
- 建立第三方 conformance suite；
- Qiskit/PennyLane 只依赖 stable core 和 extension protocol；
- 验证外部插件不需要导入 runtime internal。

完成标准：能够在不修改 core 的情况下增加一个最小第三方 backend。

### Phase 7：公开候选与冻结

- 发布 alpha API candidate；
- 用真实示例、内部应用和至少一个外部集成试用；
- beta 阶段停止任意改名，只修复 contract 问题；
- 生成最终 API manifest、typing snapshot 和迁移说明；
- 完成开源发布审计。

## 9. API 质量门禁

首次公开版本至少需要以下自动门禁：

### Surface gate

- 根级导出必须与稳定 manifest 完全一致；
- 新根级导出必须经过 API review；
- experimental 内容不得出现在 stable snapshot；
- internal 模块不得被用户示例导入。

### Signature gate

- stable 函数不得无审查增加位置参数；
- stable 返回值不得为 `Any`；
- 公共 options 必须可类型检查；
- 所有默认值必须进入 snapshot。

### Semantic gate

- `run(circuit, options)` 与 `run(plan(circuit, options))` 等价；
- stale plan fail closed；
- measurement 不允许隐式覆盖；
- unsupported capability 在 planning 阶段失败；
- backend fallback 必须在 result 中可见。

### Documentation gate

- README 第一条路径可执行；
- 五条黄金路径全部由 CI 执行；
- API 文档由 manifest 生成；
- 每个 stable export 都有一个主文档入口和 executable contract。

### Compatibility gate

- stable API 删除必须有弃用版本和删除版本；
- warning 文本包含替代接口；
- 至少跨一个公开 minor release 保留兼容；
- 序列化 IR、Plan、Result 和 DeploymentPackage 各自带 schema version。

## 10. 首次公开版本冻结条件

只有同时满足以下条件，才能宣布 API freeze：

- [ ] Stable Core 已缩减并通过逐项审查；
- [ ] `ExecutionOptions` 成为唯一推荐执行配置；
- [ ] `fq.run(plan)` 已实现并通过等价性测试；
- [ ] `plan/run/RuntimePolicy` 使用统一术语；
- [ ] Measurement 来源和覆盖规则唯一明确；
- [ ] `fq.run`、`fq.plan`、`fq.train` 返回类型不再是 `Any`；
- [ ] native backend 对象不会隐式扩张稳定 Result；
- [ ] Module 的 forward/execute/train 语义已固定；
- [ ] checkpoint/resume 的责任边界已明确；
- [ ] distributed/noise/backend 专属入口已完成分层；
- [ ] 根命名空间不存在无意暴露的 compatibility exports；
- [ ] 五条黄金路径全部通过；
- [ ] Qiskit 和 PennyLane conformance 通过；
- [ ] 一个第三方 backend 示例只使用公开扩展协议；
- [ ] 文档、typing、API snapshot 和 release notes 一致；
- [ ] alpha/beta 试用没有发现必须破坏 API 才能解决的问题。

## 11. 建议的版本策略

```text
当前私有阶段
  允许集中式破坏性 API 收敛

公开 alpha
  API candidate；允许有明确记录的调整

公开 beta
  停止任意命名变化；只修复契约缺陷

首个稳定公开版本
  Stable Core 进入兼容承诺
  后续原则上只增不破
```

项目仍低于 1.0 时可以按照语义化版本进行明确迁移，但不应把“版本号低”当作频繁
破坏用户代码的理由。首次公开后，每个 breaking change 都必须有真实收益、迁移路径
和弃用窗口。

## 12. 推荐执行顺序

最优先的实施顺序是：

1. 收缩根级 stable exports；
2. 统一 `ExecutionOptions` 和术语；
3. 实现 `ExecutionPlan → fq.run(plan)`；
4. 收紧 `ExecutionResult` 和 measurement；
5. 收敛 Module/Training 生命周期；
6. 建立 backend/provider 扩展协议；
7. 运行 alpha/beta 用户路径验证；
8. 最后冻结，而不是先冻结再修正。

完成这些工作后，FlagQuantum 的 API 将不只是“入口统一”，而会形成真正稳定的产品
契约：普通用户只需学习少量核心对象，专业用户可以检查和固定执行计划，第三方生态
可以通过协议扩展，同时内部 backend、编译器和硬件适配仍能持续演进。
