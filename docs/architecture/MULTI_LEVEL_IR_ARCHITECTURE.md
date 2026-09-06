# FlagQuantum 多层 IR 与量子编译基础设施设计

状态：架构设计草案，供 Stable Core API 收敛完成后实施
适用范围：FlagQuantum 编译器、运行时、互操作与 QPU 部署
不改变：当前公共 `CircuitIR`（序列化 schema 1.0）、Stable Core API 或已发布合同

## 1. 目的

本文定义 FlagQuantum 从当前统一电路表示演进为完整量子编译基础设施的目标架构、
边界、迁移顺序和验收标准。它回答以下问题：

1. 当前公共 `CircuitIR` 应继续承担什么职责；
2. 为什么需要 `ProgramIR -> QuantumIR -> TargetIR`；
3. OpenQASM 3、QIR、QCIS 和本地模拟器在架构中的位置；
4. 编译器与运行时如何分离；
5. 如何在不破坏 API 收敛结果的前提下迁移；
6. 如何验证语义、性能、可复现性和设备合法性；
7. Python 能完成哪些阶段，何时才需要考虑 C++ 或 MLIR。

本文中的类型名和接口名均为内部设计占位符。除非经过独立 API change proposal，
它们不应添加到 `flagquantum` 根命名空间，也不构成新的稳定公共 API。

## 2. 当前事实基线

### 2.1 已经具备的能力

FlagQuantum 当前并非没有 IR，而是已经具备一个稳定、实用的电路级交换表示：

- `Circuit.to_ir()` 生成并缓存 `CircuitIR`；
- `CircuitIR` 当前序列化 schema 版本为 1.0，包含 instruction、observable、
  measurement、dtype、shape 和 metadata；
- IR 支持校验、确定性 JSON 序列化和内容哈希；
- compiler、planner、drawer、本地执行器和分布式执行器均消费 `CircuitIR`；
- 已有门消除、相邻旋转融合、分层调度、拓扑路由和后路由优化；
- 云部署先编译 IR，再导出 OpenQASM 2/3 或 QCIS；
- 动态线路已有独立的构建、局部模拟、批量 trajectory、能力评估和 OpenQASM 3
  实验路径；
- `CircuitIR` 已在能力矩阵中列为 release-certified，并属于受保护的序列化合同。

相关实现：

- `flagquantum/core/ir.py`
- `flagquantum/circuit.py`
- `flagquantum/compiler/pipeline.py`
- `flagquantum/compiler/routing.py`
- `flagquantum/runtime/execution.py`
- `flagquantum/runtime/dynamic/`
- `flagquantum/deployment/cloud.py`
- `flagquantum/utils/qasm_exporter.py`
- `flagquantum/utils/qcis_exporter.py`

### 2.2 当前主要限制

`CircuitIR` 当前本质上是平坦的电路指令序列。它适合电路交换、模拟和训练，
但不能独立承担完整量子编译器的所有职责：

- 函数、kernel、作用域、基本块和调用关系缺少强类型表示；
- 经典值、量子值和测量结果没有统一的 def-use 数据流；
- 循环、分支和量子—经典反馈没有统一控制流模型；
- 动态线路仍与普通 `fq.run()` 主路径分离；
- 逻辑 qubit、物理 qubit、布局和时序主要依赖整数及 metadata；
- target gate set、设备能力和 QASM 子集缺少统一合法化模型；
- provider artifact 仍以顶层 `qasm` 字段为中心，QCIS 通过 metadata 补充；
- 编译 Pass 缺少统一的输入层级、输出层级、前置条件和保持属性合同；
- QASM 导出、provider 提交和运行时执行之间尚未形成通用 executable ABI。

这些限制不要求破坏 `CircuitIR`。正确方向是在其后建立内部多层 IR。

## 3. 架构决策摘要

FlagQuantum 采用以下长期结构：

```text
fq.Circuit / fq.Module / import adapters
                    |
                    v
       Public CircuitIR (schema 1.0)
        stable exchange contract
                    |
           import + legalization
                    v
              ProgramIR
     functions / classical control / calls
                    |
          partial evaluation + lowering
                    v
              QuantumIR
    qubits / gates / measurements / results
                    |
       target selection + optimization
                    v
               TargetIR
 physical layout / native gates / scheduling
                    |
       code generation / plan generation
          +---------+----------+----------+
          |         |          |          |
          v         v          v          v
       QASM 3      QIR        QCIS    Runtime Plan
          |         |          |          |
          +---------+----------+----------+
                    |
                    v
             ExecutableArtifact
                    |
                    v
              RuntimeAdapter
       local simulator / cloud QPU / controller
```

核心决策：

1. `CircuitIR` 继续是公共、稳定、可序列化的交换格式；
2. `ProgramIR`、`QuantumIR`、`TargetIR` 初始均为内部实现；
3. QASM、QIR、QCIS 是目标格式，不是 FlagQuantum 的唯一事实源；
4. 编译和任务提交分离；
5. target capability 必须在提交前 fail closed；
6. 静态结构尽早确定，数值参数尽量晚绑定；
7. 新旧执行路径并存，通过差分验证逐步迁移；
8. 第一阶段使用 Python，C++/MLIR 仅在明确收益和迁移门槛满足后评估。

## 4. 设计原则

### 4.1 一个公共事实源，多个内部层级

用户、服务和持久化系统继续通过版本化 `CircuitIR` 交换程序。内部编译器可以使用
更强的表示，但不能要求用户理解内部 dialect 或 compiler pass。

### 4.2 语义进入类型，不进入任意 metadata

以下概念应逐步成为强类型字段、operation 或 value：

- measurement result；
- classical condition；
- logical/physical qubit；
- parameter binding phase；
- target gate；
- duration 和 scheduling constraint；
- observable 和 execution request；
- result schema。

metadata 只记录扩展信息、调试信息和不影响核心语义的 provenance。

### 4.3 编译器不得静默改变科学语义

每个 Pass 必须声明它是否保持：

- state 或 density operator；
- measurement distribution；
- observable expectation；
- parameter gradient；
- approximation/error budget；
- wire/result ordering；
- dynamic feedback semantics。

任何近似、噪声注入、fallback 或结果重排都必须显式记录。

### 4.4 能力不足应在最早可知阶段失败

如果目标不支持 mid-circuit measurement、reset、实时条件控制、要求的原生门、
最大 qubit 数或 QIR profile，规划或编译阶段必须返回明确 blocker。不得提交后才依赖
provider 的模糊错误，也不得静默降级到语义不同的执行方式。

### 4.5 本地快路径不能被编译基础设施拖慢

小型 `fq.Circuit` 在 CPU 或单 GPU 上运行时应保留轻量路径。多层 IR 必须支持缓存、
惰性编译以及可选 bypass，不能强制每次本地执行都经过完整 QPU targeting 管线。

## 5. IR 层级

### 5.1 Public CircuitIR

职责：

- 稳定公共交换；
- JSON 序列化和内容寻址；
- REST/MCP/任务系统边界；
- `fq.Circuit`、interop adapter 和执行器之间的兼容合同；
- 当前静态模拟、训练和部署输入。

非职责：

- 完整 SSA；
- 通用经典控制流；
- 设备原生时序；
- provider-specific executable；
- 脉冲控制程序。

迁移要求：

- 不修改 `IR_VERSION = "1.0"` 的现有语义；
- 不把内部 IR 类型加入稳定根 API；
- 新编译器必须提供 `CircuitIR -> internal IR` importer；
- 对当前 schema 1.0 能表达的子集，应提供 internal IR 到公共 `CircuitIR` 的无损或
  显式受限 round-trip；
- richer internal IR 无法降回 v1 时必须明确报告，不得丢弃动态语义。

### 5.2 ProgramIR

ProgramIR 表示量子—经典混合程序，而不是单一电路。

建议包含：

- module；
- function/kernel；
- typed argument 和 result；
- lexical scope；
- block、branch、loop；
- function call；
- compile-time constant；
- runtime parameter；
- measurement-produced classical value；
- host-only 与 controller-eligible classical region；
- source location 和诊断上下文。

示意：

```text
func active_reset(theta: angle) -> bit {
    q = quantum.alloc
    quantum.rx(theta, q)
    m = quantum.measure(q)
    if m {
        quantum.x(q)
    }
    return quantum.measure(q)
}
```

ProgramIR verifier 至少检查：

- 符号和作用域；
- 类型一致性；
- block terminator；
- 返回类型；
- qubit/result 使用合法性；
- controller region 中是否包含目标不可能实时执行的经典操作。

### 5.3 QuantumIR

QuantumIR 是硬件无关的量子语义层。ProgramIR 中可静态求值的经典结构已经被消除，
但真正依赖测量结果的动态控制仍可以保留。

建议类型：

```text
QubitType
QubitRegisterType
MeasurementResultType
AngleType
ParameterType
ObservableType
SampleResultType
```

建议 operation：

```text
alloc / release
gate / controlled_gate
measure / reset
call
expectation / sample
barrier
conditional region
```

关键语义：

- qubit 不可复制；
- qubit 生命周期可验证；
- measurement result 具有显式 def-use；
- condition 只能消费合法的经典值；
- gate 参数区分静态值和晚绑定值；
- observable 与 measurement request 不再依赖自由格式 metadata；
- 静态线路和 adaptive 线路共享同一个基础表示。

Python 第一阶段可以使用不可变节点、显式 value ID、block argument 和 def-use index
实现受限 SSA，不要求一次实现通用编程语言的全部 SSA 能力。

#### 5.3.1 Qubit 引用与线性值模型

内部编译器采用“两阶段量子值模型”：

```text
ProgramIR：reference-oriented
  便于表达命令式 builder、作用域、函数调用和经典控制流

QuantumIR：value/linear-oriented
  显式暴露量子数据流，供验证、重写、调度和 lowering 使用
```

ProgramIR 中的 qubit reference 表示对一个量子资源的受控引用，不代表可复制的量子
状态。alias 必须被 ownership analysis 跟踪；任何可能形成两个独立可写量子状态的操作
都不合法。

ProgramIR 降到 QuantumIR 时，每个量子操作消费当前 qubit value，并产生新的 value：

```text
%q1 = quantum.h %q0
%q2, %q3 = quantum.cx %q1, %q_other
%q4, %m = quantum.measure %q2
%q5 = quantum.reset %q4
quantum.release %q5
```

这里的 value 表示量子资源在程序点上的状态版本，不表示可以复制的 statevector。
`measure` 是否返回继续可用的 qubit value 由 operation 语义显式决定；`release` 消费
最终 value 且不产生后继。

控制流合并使用 block argument 或等价的 phi-like 机制：每个前驱必须为同一逻辑
量子资源提供且仅提供一个 live value。QuantumIR verifier 至少拒绝：

- 同一 qubit value 被两个量子操作并行消费；
- gate 后继续使用旧 value；
- release 后继续使用；
- 分支只有一侧释放或重新分配而没有合法 join；
- 两个不同量子资源在 join 时被错误合并；
- qubit value 被存入不具备线性所有权语义的普通容器；
- 函数返回或捕获导致资源所有权不明确。

TargetIR 可以把已验证的线性 value 映射为物理 qubit/resource reference，但不得恢复
隐式复制语义。该模型是内部编译决策，不改变当前 `fq.Circuit` 的命令式用户体验。

#### 5.3.2 辅助比特的当前边界与演进方向

辅助比特（ancilla qubit）首先是一种资源角色，而不是不同于普通 qubit 的物理类型。
FlagQuantum 当前支持用户显式预留和操作辅助比特，但尚未实现编译器管理的辅助比特。

当前能力边界如下：

- 用户可以通过增大 `fq.Circuit(n_qubits=...)` 的 qubit 数量，把约定的部分 wire 手工
  用作辅助比特；现有 `CircuitIR` 和执行器仍将其视为普通 wire；
- `DynamicCircuit` 支持显式 `measure` 和 `reset`，因此用户可以手工表达部分辅助比特
  复位与复用流程；
- `CircuitIR` v1 只有固定 `n_wires` 和整数 wire 引用，没有 ancilla 身份、所有权、
  `alloc/release`、clean/dirty 要求或生命周期语义；
- 当前路由不得使用 `CircuitIR` 范围之外的空闲物理 qubit 作为中间路径。若路由必须
  经过此类 qubit，应 fail closed，并要求显式 layout/lowering；
- 从 Qiskit 导入命名、多重或别名 register 时，register 会被展平为稳定整数索引。
  `AncillaRegister` 对应的 qubit 可以保留，但“辅助比特”角色当前不属于稳定
  `CircuitIR` v1 语义。

因此，在公开材料和能力矩阵中应使用以下表述：

> FlagQuantum 当前支持用户手工预留和操作辅助比特，但还不具备编译器自动申请、
> 复用、释放和验证辅助比特的能力；这些能力属于多级 IR 的规划范围。

多级 IR 应明确区分四类资源：

1. 用户数据 qubit；
2. 用户显式声明或预留的 ancilla；
3. 门分解、纠错、路由等编译过程引入的临时 ancilla；
4. 目标设备提供、但尚未映射给逻辑程序的空闲物理 qubit。

演进时至少补齐以下语义与验证：

- 在 ProgramIR/QuantumIR 中使用 `alloc/release` 表达资源生命周期；
- 表达 clean ancilla（进入时必须为 `|0>`）和 dirty ancilla（允许借用未知状态但必须
  恢复）的前置与后置条件；
- 验证释放前的状态要求、所有权、分支合并和 release 后使用；
- 由 lowering、门分解和 routing 在目标容量、连接关系与校准约束内申请辅助资源；
- 支持 measure/reset/reuse，并将动态复用能力与目标 capability 显式关联；
- 默认不把内部辅助比特混入用户请求的测量结果，只有显式调试请求才暴露；
- 在规划结果中报告 `logical_qubits`、`ancilla_qubits`、`peak_physical_qubits`、
  ancilla 来源以及无法分配时的 blocker。

在上述内部语义、验证器和 lowering 稳定之前，不应仓促固定
`fq.AncillaQubit` 一类公开 API。第一阶段继续保持 `fq.Circuit` 的简单整数 qubit
体验，在内部多级 IR 中完成角色、生命周期和资源规划；是否增加公开声明语法，应在
至少两个 release cycle 的实现与互操作验证后通过独立 API change proposal 决定。

### 5.4 TargetIR

TargetIR 表示已选择目标并完成设备合法化的程序。

建议包含：

- target identity 和 capability snapshot hash；
- logical-to-physical layout；
- physical qubit type；
- native gate；
- calibrated gate/reference；
- coupling edge；
- instruction duration；
- delay、barrier、box 和 alignment constraint；
- measurement group；
- controller classical instruction；
- resource conflict；
- result ordering；
- unresolved runtime parameter schema。

TargetIR verifier 必须保证：

- 所有 operation 均属于目标支持集合；
- 所有双/多 qubit operation 满足目标连接关系；
- dynamic control 只使用目标支持的实时经典指令；
- scheduling 不存在不允许的资源重叠；
- qubit、result、shot 和参数数量满足目标限制；
- 输出格式与目标接受的 profile 一致。

### 5.5 程序语义与执行请求边界

内部多层 IR 必须区分“程序做什么”和“如何、在哪里、重复多少次执行”。推荐归属如下：

| 概念 | 内部归属 | 说明 |
| --- | --- | --- |
| gate、alloc、release、reset | ProgramIR/QuantumIR | 改变量子程序语义 |
| mid-circuit measurement | ProgramIR/QuantumIR | 结果可参与后续控制流 |
| classical condition/loop | ProgramIR | 静态部分被求值，动态部分 lower 到 QuantumIR |
| terminal measurement operation | QuantumIR | 当它是程序显式组成部分时保留 |
| observable 定义 | Typed domain object | 由 execution/differentiation request 引用 |
| expectation/sample/state 请求 | ExecutionRequest | 选择需要返回的结果，不改写源程序 |
| shots、seed、batch size | ExecutionOptions | 执行策略，不进入核心程序 identity |
| backend、device、routing strategy | CompilationPolicy | 影响 lowering 和 artifact identity |
| noise channel operation | QuantumIR | 仅当用户明确把 channel 写入程序时属于语义 |
| noise model/fallback policy | Compilation/ExecutionPolicy | 必须显式记录是否以及如何 lower |
| gradient request/method | DifferentiationRequest | 可 lower 为 adjoint、parameter-shift 或 native plan |
| credentials、quota、queue | Runtime | 不得进入 IR、artifact payload 或模型上下文 |
| physical layout/native gate | TargetIR | 目标相关编译结果 |

当前公共 `CircuitIR` 同时携带 observables 和 measurements。为保持兼容，importer 可以
读取这些字段并在内部拆分为 program semantics 与 execution request；不得因此改变公共
schema。若调用方同时通过 `fq.run(...)` 传入 execution request，组合、替换或冲突规则
必须遵守 API 收敛后的唯一公共合同，不得由 importer 隐式决定。

identity 也应分层：

```text
program_identity
  core program semantics + static parameters

compilation_identity
  program_identity + target + compiler + pipeline + capability snapshot

execution_identity
  artifact identity + runtime parameters + shots + seed + execution policy
```

这一分层避免仅改变 shots 就触发 placement/routing，也避免目标或编译管线变化时错误
复用旧 artifact。

## 6. 编译 Pass 基础设施

### 6.1 Pass 合同

每个 Pass 应声明：

```text
name
input IR level
output IR level
required analyses
preserved analyses
semantic properties preserved
possible diagnostics
determinism guarantee
```

内部 Python 接口示意：

```python
class CompilerPass(Protocol):
    name: str
    input_level: IRLevel
    output_level: IRLevel

    def run(
        self,
        module: IRModule,
        context: CompilationContext,
    ) -> IRModule: ...
```

`CompilationContext` 应包含 target capability、pipeline options、diagnostic sink、
source map、random seed、compiler version 和 cache policy。Pass 不得读取未声明的全局
环境变量来改变编译语义。

### 6.2 推荐管线

```text
Import CircuitIR/OpenQASM
  -> structural verification
  -> type and lifetime verification
  -> canonicalization
  -> constant propagation
  -> static loop unrolling
  -> dead branch elimination
  -> function/gate inlining
  -> symbolic parameter simplification
  -> ProgramIR-to-QuantumIR lowering
  -> gate canonicalization
  -> high-level gate decomposition
  -> observable/measurement lowering
  -> target capability check
  -> placement
  -> routing
  -> native gate legalization
  -> scheduling/timing resolution
  -> TargetIR verification
  -> executable emission
```

不是所有执行都必须运行完整管线。本地模拟器可以在 QuantumIR 阶段生成执行计划；
QPU 后端通常需要继续降到 TargetIR。

### 6.3 静态化策略

静态化的目标是固定程序结构和执行依赖，同时保留适合晚绑定和批处理的数值参数。

优先静态化：

- 常量表达式；
- 静态循环边界；
- 静态 qubit 索引；
- 不可达分支；
- gate/subroutine 定义；
- target-independent duration 表达式；
- 不影响结构的配置选择。

优先晚绑定：

- VQE/QML 训练角度；
- provider 支持的 parameter sweep；
- 不改变控制流和布局的 batch 参数；
- shots 等执行策略。

必须限制过度展开：大循环、递归展开、分支组合和 specialization 版本数量均需要预算。

### 6.4 AnalysisManager 与失效规则

成熟 Pass 管线不能让每个 Pass 重复扫描完整程序。内部应建立 `AnalysisManager`，把
只读分析结果与对应 IR revision、target snapshot 和分析参数绑定。

首期分析集合：

| Analysis | 主要消费者 | 失效条件示例 |
| --- | --- | --- |
| DefUseAnalysis | verifier、DCE、lowering | value/operand 改写 |
| DominanceAnalysis | SSA verifier、control-flow lowering | block/branch 改写 |
| QubitLifetimeAnalysis | ownership verifier、allocation lowering | alloc/release/control-flow 改写 |
| MeasurementDependencyAnalysis | adaptive legality、batch execution | measure/condition 改写 |
| ParameterDependencyAnalysis | specialization、cache、gradient | parameter expression/control-flow 改写 |
| InteractionGraphAnalysis | placement、routing、MPS/TN planning | gate/wire 改写 |
| CircuitCostAnalysis | optimization、backend selection | gate、layout、duration 改写 |
| TargetLegalityAnalysis | TargetIR conversion | operation/type/target snapshot 改写 |
| LivenessAndMemoryAnalysis | simulator plan、buffer reuse | value lifetime/shape 改写 |
| DistributionAnalysis | sharded runtime planning | topology、ownership、communication 改写 |

Pass 必须声明 `required_analyses` 和 `preserved_analyses`。默认规则是变换 Pass 使未显式
保留的分析全部失效；只读 AnalysisPass 不得修改 IR。分析缓存键至少包含：

```text
module revision or canonical hash
analysis name and version
analysis options
target/capability hash, when target-dependent
```

不得把可变的全局 `dict` 当成无版本 PropertySet。跨 Pass 共享的事实应是类型化、只读
的 analysis result；启发式 cost model 与语义正确性 analysis 必须分开。随机化布局或
路由必须记录 seed，保证同输入、同配置、同编译器版本可复现。

Pass instrumentation 应记录：

- Pass 前后 IR hash；
- wall time 和 peak memory；
- analysis cache hit/miss；
- gate、depth、communication 和 error-cost 变化；
- invalidated/preserved analyses；
- failure diagnostic 和最小可复现 pipeline。

## 7. 编译器与运行时边界

### 7.1 ExecutableArtifact

当前 `DeploymentPackage` 同时包含 IR 和固定 QASM 字段。目标架构应引入通用的内部
编译产物模型：

```text
ExecutableArtifact
  format
  payload or payload reference
  entrypoint
  target identity
  source IR hash
  target IR hash
  compiler version
  pass pipeline digest
  capability snapshot hash
  calibration snapshot hash, when applicable
  parameter schema
  result schema
  approximation/error metadata
```

`format` 可以是：

- `flagquantum-runtime-plan`；
- `openqasm-2`；
- `openqasm-3`；
- `qir-base`；
- `qir-adaptive`；
- `qcis`；
- provider native format。

在 API 收敛完成前，不应直接修改受保护的 `DeploymentPackage`。第一步应通过内部
adapter 包装现有对象，等独立 API change proposal 批准后再决定公共迁移方式。

### 7.2 RuntimeAdapter

运行时只负责执行编译产物，不负责重新解释源程序：

```text
submit(artifact, execution_options) -> handle
status(handle) -> status
cancel(handle) -> acknowledgement
result(handle) -> execution result
```

本地同步执行可以在内部复用相同合同。远程 provider 应实现异步生命周期。

编译器负责：

- lowering；
- optimization；
- target legalization；
- artifact generation。

运行时负责：

- 资源选择和授权；
- 参数/shot 绑定；
- 任务提交；
- 状态、取消和结果；
- receipt 和实际后端记录。

provider 如果会二次编译，应在 receipt 中区分 FlagQuantum artifact identity 与厂商
最终执行 identity，不应把远端再编译描述为 FlagQuantum 已生成脉冲程序。

## 8. TargetCapabilities

需要建立版本化、可哈希的目标能力快照，至少包含：

```text
identity and version
qubit count
native gate set and parameter domains
coupling topology
measurement and reset support
mid-circuit measurement support
adaptive classical operation subset
timing model and feedback limits
accepted artifact formats
OpenQASM/QIR profiles
maximum shots and program limits
parameter binding support
simulator/noise capabilities
```

必须区分：

1. 静态设备能力；
2. 动态校准数据；
3. 账户、配额和授权；
4. 当前可用性和排队状态。

编译缓存通常依赖前两项，任务提交还依赖后两项。不能因为账户暂时无权限而改变
目标程序语义，也不能因为设备校准更新就错误复用依赖旧 calibration 的调度结果。

## 9. OpenQASM 3 支持策略

### 9.1 不以“能输出 3.0 header”定义支持

完整支持至少需要：

- grammar parser；
- semantic/type analysis；
- scope and symbol resolution；
- ProgramIR import；
- target subset validation；
- round-trip 或明确的非对称转换合同；
- 静态与动态执行语义测试。

字符串 exporter 只是 code generation 的一部分。

### 9.2 分 profile 实施

建议内部定义能力 profile，名称在实现时再确定：

```text
Static profile
  gates, parameters, compile-time loops, terminal measurement

Adaptive profile
  mid-circuit measurement, reset, bounded conditionals

Timing profile
  duration, delay, box, stretch, scheduling intent

Pulse profile
  cal, defcal and selected calibration grammar
```

目标后端必须显式声明 profile 和更细的 feature subset。不得将局部 exporter 支持
描述为完整 OpenQASM 3 设备支持。

### 9.3 性能原则

OpenQASM 3 只参与冷路径：

```text
text -> parse once -> IR -> compile/cache -> execute many times
```

禁止：

- 每个 shot 重新解析；
- Python 逐 statement、逐 gate 解释；
- QPU 测量后返回云端 Python 再决定下一门；
- 每次训练参数更新都重新 placement/routing；
- 动态模拟每个 shot 单独触发 GPU kernel。

静态线路优化：

- 常量折叠和循环展开；
- gate cancellation/merge；
- layer scheduling；
- matrix/kernel fusion；
- 固定布局、路由和通信计划；
- workspace 预分配；
- 参数晚绑定和批量执行。

动态模拟优化：

- 编译成基本块；
- 公共前缀只执行一次或批量执行；
- measurement 在设备侧生成 branch mask；
- shots 按分支分组；
- 状态和 branch metadata 尽量保持设备常驻；
- 设置路径数量和内存预算，超限时显式选择 reference trajectory 或失败。

真实 QPU 的相干时间内反馈必须下放到设备控制器。不具备对应 capability 的目标应在
编译阶段拒绝 adaptive program。

## 10. 量子 AI、梯度和分布式执行

多层 IR 不应只服务 QPU 输出。FlagQuantum 的差异化价值来自同一量子程序在训练、
模拟和部署之间保持语义一致。

### 10.1 梯度

ProgramIR/QuantumIR 应能区分：

- parameter identity；
- parameter binding phase；
- observable；
- gradient request；
- exact/approximate gradient method；
- gradient ownership 和 aggregation。

未来可将一次梯度请求 lower 为：

- native autograd execution plan；
- adjoint plan；
- parameter-shift circuit batch；
- provider-native gradient request。

任何 gradient lowering 必须验证数值和导数语义，不得只验证 forward parity。

### 10.2 statevector、MPS 和 TN

QuantumIR 是共同输入，运行时 representation lowering 可以生成不同执行计划：

```text
QuantumIR
  +-- local statevector plan
  +-- sharded statevector communication plan
  +-- local/rank-owned MPS plan
  +-- tensor-network contraction plan
  +-- JAX kernel plan
```

这些 execution plan 不应被误称为 QPU TargetIR。它们是针对模拟表示和计算平台的
内部 target lowering，仍须保留现有 distribution semantics、fallback 和 evidence
合同。

## 11. Python 与 C++ 决策

### 11.1 初始实现使用 Python

第一阶段以 Python 实现：

- immutable IR nodes；
- block/value IDs 和受限 def-use；
- verifier；
- PassManager；
- pattern rewrite；
- target capability model；
- QASM/QCIS emitter；
- executable/runtime internal protocols；
- differential and property testing。

理由：当前语义仍需验证，且 FlagQuantum 的用户 API、PyTorch/JAX 集成和主要运行时
均在 Python。先稳定抽象比过早固化原生 ABI 更重要。

### 11.2 C++/MLIR 不是前置条件

满足以下一项或多项后再评估 C++/MLIR：

- Python 编译时间成为可复现的产品瓶颈；
- 需要深度复用 MLIR dialect conversion；
- 需要 Clang/C++ frontend；
- 需要生成和链接 LLVM/QIR binary；
- 大型程序使 Python def-use/rewrite 性能不可接受；
- 需要稳定原生 compiler plugin ABI；
- 关键厂商 SDK 只有 C/C++ 接口。

即使引入 C++，边界也应限制为 compiler library。Python 继续承载稳定用户 API、
训练接口、planner、provider 生命周期和产品层编排。

## 12. API 收敛期间的工作边界

### 12.1 现在允许做的工作

- 本文档及相关 ADR；
- 当前语义 characterization tests；
- 不公开的 experimental/internal 原型；
- `CircuitIR -> internal IR -> CircuitIR` round-trip；
- 编译性能和差分测试基础设施；
- 对当前 qasm/provider 假设进行只读盘点；
- 为 API 收敛提出兼容性约束。

### 12.2 API 收敛完成前不应做的工作

- 修改当前 `CircuitIR` schema；
- 修改稳定根导出；
- 改变 `fq.run`、`fq.plan`、`fq.compiler.compile` 的公共签名或行为；
- 直接替换受保护的 `DeploymentPackage` schema；
- 将内部多层 IR 暴露为稳定 API；
- 同时重写 compiler、runtime 和 provider 主路径；
- 更新 API snapshot 以掩盖未经批准的变化。

### 12.3 正式启动门槛

主链路迁移应等待：

- Stable Core 公共列表冻结；
- `CircuitIR` schema 兼容政策确认；
- measurement 的唯一来源/组合规则确认；
- `plan`、`run`、compile、deployment 职责确认；
- provider-specific 配置的公共边界确认；
- API contract 和行为测试稳定；
- API 收敛分支进入主线，避免长期双向同步。

## 13. 分阶段实施路线

### Phase 0：设计与基线，API 收敛期间

交付：

- 本设计文档；
- 现有 IR/compiler/runtime/deployment 行为矩阵；
- serialization 和 execution characterization tests；
- 编译与执行基准数据；
- 不影响 Stable Core 的内部原型。

退出条件：当前语义、限制和性能基线可复现。

### Phase 1：内部 QuantumIR 骨架

交付：

- Python internal IR core；
- types、values、blocks、diagnostics；
- verifier 和 PassManager；
- `CircuitIR -> QuantumIR` importer；
- 静态子集 round-trip；
- 与当前执行器的 state、expectation 和 gradient 差分测试。

退出条件：当前静态 CircuitIR 可无语义变化进入新管线。

### Phase 2：迁移现有编译能力

交付：

- identity/rotation/cancellation canonicalization；
- gate decomposition；
- topology placement/routing；
- QASM 2/3、QCIS emitter；
- pipeline digest、诊断和缓存；
- legacy compiler 与新 compiler 差分测试。

退出条件：现有静态部署用例在新管线达到功能和性能门槛。

### Phase 3：TargetIR 与 executable ABI

交付：

- TargetCapabilities schema；
- TargetIR；
- target legality；
- ExecutableArtifact internal contract；
- RuntimeAdapter internal contract；
- provider conformance tests；
- 经过批准的 DeploymentPackage 兼容/迁移方案。

退出条件：至少一个本地 runtime、一个 QASM provider 和一个非 QASM target 使用
同一 artifact/runtime 边界。

### Phase 4：ProgramIR 与统一动态线路

交付：

- functions、scope、block、control flow；
- measurement result def-use；
- reset、conditional、bounded loop；
- static partial evaluation；
- ProgramIR-to-QuantumIR lowering；
- OpenQASM 3 Static/Adaptive profiles；
- 普通执行合同与 experimental dynamic 路径的迁移方案。

退出条件：静态和动态程序共享同一语义管线，且不降低现有 batched trajectory 性能。

### Phase 5：时序、QIR 与高级 lowering

候选交付：

- timing/duration/stretch；
- QIR Base/Adaptive；
- device scheduling；
- gradient lowering；
- distributed representation lowering；
- calibration/pulse references；
- 经证据支持后再评估 C++/MLIR。

每个候选能力应独立成熟，不因属于 Phase 5 而自动获得生产声明。

### 13.6 机器可验证的阶段退出矩阵

文字退出条件必须落成 CI 检查和机器可读 evidence。各阶段至少满足：

| Phase | 必需验证 | 最小退出证据 |
| --- | --- | --- |
| 0 | 当前 API、IR、执行、部署和性能基线 | 固定 fixture、命令、环境、结果摘要与已知 blocker |
| 1 | importer、verifier、round-trip、语义差分 | 当前支持的静态 opcode 全覆盖；负向 verifier fixture；state/expectation/gradient parity；公共 API snapshot 无变化 |
| 2 | Pass 与 emitter | 每个 Pass 独立 golden/属性测试；确定性 pipeline hash；routing 后全 gate 合法；QASM/QCIS fixture 语义等价；性能不超过基线批准预算 |
| 3 | TargetIR、artifact、runtime ABI | local、QASM、non-QASM 三类 target conformance；artifact 篡改 fail closed；submit/status/cancel/result 生命周期；credential 隔离 |
| 4 | ProgramIR 与动态线路 | active reset、conditional、qubit reuse 和随机分支 oracle；OpenQASM 3 profile conformance；trajectory/batched 统计一致；批量快路径无未报告退化 |
| 5 | 每项高级 capability 独立认证 | timing/QIR/gradient/distributed/pulse 各自有 schema、oracle、target evidence、性能基线和成熟度记录，不允许打包继承认证 |

Phase evidence 必须同时覆盖五条轴：

```text
compatibility
semantic correctness
determinism and reproducibility
performance and resource bounds
operability and failure behavior
```

具体数值容差从现有 backend/dtype 合同继承，并在测试清单中显式记录；不能在架构
文档中用一个全局误差阈值替代 statevector、MPS、TN、gradient 和统计采样各自的
数值合同。性能阈值必须由 Phase 0 实测基线产生。

每个 Phase 建议维护机器可读 manifest：

```text
capability
implementation owner
input/output schema
supported profile
test command and oracle
correctness tolerance
performance budget
public API impact
rollback path
maturity level
evidence artifact
```

没有测试命令、oracle 或 evidence artifact 的条目只能保持 design/experimental，不能
因为代码存在或文档完成而升级成熟度。

## 14. 验证体系

### 14.1 结构与序列化

- parser/printer round-trip；
- stable ordering 和 content hash；
- 未知字段和未知 operation fail closed；
- schema migration fixtures；
- source location preservation。

### 14.2 verifier 负向测试

- use before definition；
- qubit 重复/越界/释放后使用；
- 非法测量依赖；
- 非法 block terminator；
- 目标不支持的 gate/control/timing；
- 参数绑定阶段错误；
- 不允许的动态反馈。

### 14.3 Pass 语义测试

- textual golden tests；
- pass idempotence，当适用时；
- deterministic output；
- statevector parity；
- measurement distribution parity；
- observable parity；
- gradient parity；
- routing result ordering；
- approximate pass error budget。

### 14.4 Differential testing

迁移期间对同一 `CircuitIR` 同时运行：

```text
legacy path
new IR path
```

比较：

- state/density；
- expectation；
- gradient；
- samples/counts；
- output bit ordering；
- selected backend 和 fallback；
- distributed semantics；
- runtime metadata。

### 14.5 Provider conformance

- capability discovery；
- artifact format negotiation；
- submit/status/cancel/result；
- parameter and shot binding；
- result ordering；
- remote error normalization；
- package/artifact identity；
- no credential leakage；
- provider second-stage compilation receipt。

## 15. 性能指标和门禁

至少记录：

```text
parse_ms
semantic_analysis_ms
lowering_ms
routing_ms
codegen_ms
peak_compiler_memory
artifact_cache_hit_rate
compiled_plan_reuse_count
gate_count_before_after
depth_before_after
kernel_launch_count
host_device_sync_count
shots_per_second
parameter_batches_per_second
dynamic_branch_count
end_to_end_latency
```

推荐基准：

- 1K、10K、100K gate 静态程序；
- 静态循环和 subroutine；
- 参数化训练循环；
- 1、10、100 个 mid-circuit measurement；
- direct `CircuitIR`、OpenQASM 2 和 OpenQASM 3 输入对比；
- statevector、MPS、TN 和 dynamic trajectory/batched 路径。

初始门禁原则：

- 缓存命中后不得重复 parse、placement、routing 和 codegen；
- 静态 QASM 3 经编译后执行吞吐应接近等价直接 CircuitIR；
- 新 IR 路径不得在逐 gate/逐 shot 热路径中引入 Python interpreter；
- 本地快路径的回归必须有明确预算和审批；
- 动态批量路径不得退化为逐 shot reference trajectory 而不报告；
- 性能声明必须区分 compile、execute、queue 和 end-to-end 时间。

具体百分比阈值应根据 Phase 0 基线确定，不应在没有测量数据时拍定。

## 16. 主要风险与控制

### 风险：内部 IR 变成第二个公共 API

控制：放入 internal/experimental namespace，不从根导出，不承诺序列化稳定性；公共
交换继续使用公共 `CircuitIR`。

### 风险：重复实现通用编译器导致维护失控

控制：Python 第一阶段只实现 FlagQuantum 所需的受限 block/value/Pass 能力；达到
明确复杂度或性能门槛后再评估 MLIR，不追求复制完整通用语言编译器。

### 风险：所有语义继续进入 metadata

控制：任何影响合法性、结果或调度的字段必须进入 typed node/value/capability；新增
metadata 语义需架构审查。

### 风险：新旧路径长期并存

控制：每个 Phase 定义迁移对象、退出条件和 legacy 删除前提；双路径只用于验证，
不能无限期成为两套产品架构。

### 风险：QASM 3 范围无限扩张

控制：采用 profile 和 feature matrix，先 Static，再 Adaptive，再 Timing/Pulse；每个
目标 fail closed。

### 风险：编译优化破坏梯度或科学语义

控制：Pass 合同包含 gradient/approximation 属性，执行 forward、gradient 和统计
差分测试，不只比较门数。

### 风险：为了 QPU 编译牺牲本地训练性能

控制：QuantumIR 阶段允许直接生成本地 execution plan；TargetIR 和 provider codegen
不是本地执行的强制步骤。

## 17. API 收敛后需要正式决策的问题

在进入 Phase 1 前形成 ADR：

1. Public CircuitIR 与内部 IR 的精确 round-trip 边界；
2. 正式确认 5.3.1 推荐的 reference-to-linear qubit lowering 模型；
3. 正式确认 5.5 推荐的程序语义、执行请求和 identity 分层；
4. dynamic circuit 如何并入普通 `fq.run()`；
5. `ExecutionPlan`、`DeploymentPackage` 与 `ExecutableArtifact` 的关系；
6. target capability schema 的稳定级别；
7. parameter late binding 和 cache key 规则；
8. provider 二次编译如何进入 receipt；
9. OpenQASM 3 profile 的首期精确范围；
10. 内部 IR 是否需要持久化，以及其兼容政策；
11. Python IR 的规模和性能退出门槛；
12. 何种证据足以启动 C++/MLIR 评估。

## 18. 完成定义

“完整量子编译 IR 基础设施”不以目录、类名或导出格式数量判断。至少满足以下条件：

- `CircuitIR` schema 保持稳定兼容；
- 静态和动态量子程序具有统一的 typed semantics；
- compiler passes 可组合、可诊断、可复现；
- target legality 在提交前 fail closed；
- 本地模拟器和 QPU provider 共享清晰的编译/运行边界；
- QASM、QIR、QCIS 是可替换 emitter；
- artifact、pipeline、target 和 result identity 可追踪；
- static specialization、cache 和 batch execution 不损害现有快路径；
- forward、measurement、gradient 和分布式语义均有差分测试；
- 所有生产能力声明与 capability maturity 和硬件证据一致。

## 19. IR API 设计候选

本章把前述架构约束落成候选 Python API。它是 API 收敛后的实施输入，不是当前已批准
的公共合同。所有新增根导出、签名、schema 和行为仍需独立 API change proposal、
兼容方案、合同测试和 API owner 审批。

### 19.1 三层 API 表面

IR 基础设施分为三层：

```text
Stable user surface
  program / plan / compile / run / result

Compiler developer surface
  internal modules / values / operations / passes / analyses

Backend extension surface
  target capabilities / legalization / emitter / runtime adapter
```

原则：

- 普通用户不需要理解 ProgramIR、QuantumIR、TargetIR；
- 编译器开发者不通过用户根 API 操作 block/value；
- provider 不直接修改公共 CircuitIR；
- runtime 不重新解释源程序；
- credentials、quota 和 queue state 不进入任何 IR；
- 内部层级演进不得迫使 Stable Core 同步版本化。

### 19.2 保留当前 Stable Core

当前稳定表面继续是：

```python
fq.Circuit
fq.CircuitIR
fq.plan(...)
fq.run(...)
```

当前 `CircuitIR` 类名保持不变，schema 版本继续存放在序列化 payload 中，不创建
`CircuitIRV1`、`CircuitIRV2` 等公共类。

现有 `compiler.compile(...)` 和 `Circuit.compile(...)` 保持其签名、返回类型和
行为。编译入口不得通过别名或静默替换改变语义；后续变更必须通过明确的迁移提案。

### 19.3 候选用户编译 API

第一阶段在具名 namespace 中试验：

```python
from flagquantum.compiler import compile
```

只有在以下条件满足后，才考虑增加稳定根入口 `fq.compile`：

- compilation input/output contract 稳定；
- artifact ABI 经过至少两个 release cycle 验证；
- `plan`、`compile`、`run` 职责没有重叠歧义；
- 公共 API contract、文档、类型和异常行为完整；
- 已批准 additive API proposal。

候选调用方式：

```python
compiled = compile(
    circuit_or_ir,
    target="quafu.baihua",
    options=CompilationOptions(
        optimization_level=2,
        routing_strategy="auto",
        parameter_binding="late",
    ),
)

result = fq.run(
    compiled.artifact,
    options=ExecutionOptions(shots=1000),
)
```

这里的 `fq.run(ExecutableArtifact, ...)` 也是候选 additive behavior；当前实现没有
自动获得该能力，必须在 API 收敛后单独批准。

### 19.4 plan、compile 与 run 的职责

三者必须保持单向、可组合关系：

```text
plan(program, execution requirements)
  -> 选择 representation、target、资源和 fallback policy

compile(program, selected target, compilation options)
  -> 生成经过验证的 ExecutableArtifact

run(program)
  -> plan + compile + execute

run(artifact)
  -> validate artifact + execute
```

约束：

- `plan` 不把规划结果描述为已经执行；
- `compile` 不申请远程资源、不读取凭据、不提交任务；
- `run(program)` 记录 requested、selected 和 actual backend；
- `run(artifact)` 不在没有报告的情况下重新 placement/routing；
- provider 二次编译必须进入 receipt；
- shots、seed、timeout 不得改变 program identity；
- target、pipeline 或 capability snapshot 改变时必须改变 compilation identity。

### 19.5 CompilationOptions

候选对象：

```python
@dataclass(frozen=True)
class CompilationOptions:
    optimization_level: int = 1
    routing_strategy: str = "auto"
    parameter_binding: str = "late"
    seed: int | None = None
    approximation_budget: float | None = None
    diagnostics: str = "errors"
```

要求：

- 字段必须有稳定枚举或 Literal，不使用任意字符串逃生口；
- 默认值必须确定且进入 pipeline digest；
- `seed` 控制启发式编译的可复现性；
- `approximation_budget=None` 表示不授权近似；
- backend-specific option 初期放入目标 namespace，不进入根对象；
- credentials、shots、queue priority、execution timeout 不属于 CompilationOptions。

如果必须支持扩展选项，应使用带 schema/version 的 typed extension，而不是
`Mapping[str, Any]`。

### 19.6 CompilationResult 与 CompilationReport

`compile(...)` 返回结果对象，而不是裸 QASM 字符串：

```python
@dataclass(frozen=True)
class CompilationResult:
    artifact: ExecutableArtifact
    report: CompilationReport
    diagnostics: tuple[Diagnostic, ...]


@dataclass(frozen=True)
class CompilationReport:
    source_ir_hash: str
    target_ir_hash: str
    pipeline_digest: str
    compiler_version: str
    target: TargetIdentity

    gate_count_before: int
    gate_count_after: int
    depth_before: int
    depth_after: int
    routing_swap_count: int

    approximation_error_bound: float | None
    elapsed_ms: float
    pass_statistics: tuple[PassStatistics, ...]
```

要求：

- report 是编译证据，不是执行证据；
- gate/depth 定义必须随 target/profile 明确；
- elapsed time 不进入 artifact content hash；
- approximation 必须区分用户预算与编译器估计；
- diagnostics 顺序确定；
- 关键字段不得藏在自由格式 metadata。

### 19.7 ExecutableArtifact

候选内部 ABI：

```python
@dataclass(frozen=True)
class ExecutableArtifact:
    format: str
    payload: bytes
    entrypoint: str

    target: TargetIdentity
    parameter_schema: ParameterSchema
    result_schema: ResultSchema

    source_ir_hash: str
    target_ir_hash: str
    compiler_version: str
    pipeline_digest: str
    capability_snapshot_hash: str
    calibration_snapshot_hash: str | None = None

    @property
    def content_hash(self) -> str: ...
```

`format` 必须来自版本化 registry，例如：

```text
flagquantum-runtime-plan
openqasm-2
openqasm-3
qir-base
qir-adaptive
qcis
provider-native:<provider>:<version>
```

Artifact 约束：

- payload 必须是确定性 bytes 或明确的内容寻址引用；
- artifact 不包含 token、credential、account ID 或 queue state；
- parameter/result schema 必须可独立校验；
- target/capability/compiler/pipeline identity 完整；
- artifact 修改后 content hash 和签名校验失败；
- runtime 不得假设 payload 一定是 QASM；
- 内联 payload 大小超过限制时使用受控 ArtifactRef，不使用任意 URL；
- artifact serialization 是否成为 Stable API，需要独立 schema proposal。

### 19.8 内部 IR 核心类型

内部不采用一个带 `level="program|quantum|target"` 的万能类。建议使用共享节点模型和
三个不同 module wrapper：

```python
@dataclass(frozen=True)
class Value:
    id: ValueId
    type: IRType


@dataclass(frozen=True)
class Operation:
    name: OpName
    operands: tuple[ValueRef, ...]
    results: tuple[Value, ...]
    attributes: FrozenAttributes
    regions: tuple[Region, ...]
    location: SourceLocation | None


@dataclass(frozen=True)
class Block:
    arguments: tuple[Value, ...]
    operations: tuple[Operation, ...]


@dataclass(frozen=True)
class Region:
    blocks: tuple[Block, ...]


@dataclass(frozen=True)
class Function:
    name: Symbol
    argument_types: tuple[IRType, ...]
    result_types: tuple[IRType, ...]
    body: Region


@dataclass(frozen=True)
class ProgramModule:
    functions: tuple[Function, ...]


@dataclass(frozen=True)
class QuantumModule:
    functions: tuple[QuantumFunction, ...]


@dataclass(frozen=True)
class TargetModule:
    functions: tuple[TargetFunction, ...]
    target: TargetIdentity
    capability_snapshot_hash: str
```

要求：

- 节点默认不可变；
- Value ID、block order 和 symbol order 确定；
- source location 不影响程序语义 hash，除非明确选择 debug artifact；
- operation schema 决定 operand/result/attribute/region 约束；
- verifier 不依赖 Python `isinstance` 链硬编码所有 provider operation；
- extension operation 必须注册 schema、verifier、parser/printer 和 legality；
- ProgramModule、QuantumModule、TargetModule 不从 `flagquantum` 根导出。

### 19.9 显式转换 API

禁止含义不明的：

```python
ir.to_ir(level="target")
ir.lower()
ir.convert("qasm3")
```

内部采用显式转换：

```python
import_circuit_ir(circuit_ir) -> QuantumModule

import_openqasm3(source, profile=...) -> ProgramModule

lower_program_to_quantum(
    program,
    context,
) -> QuantumModule

compile_quantum_to_target(
    quantum,
    target,
    context,
) -> TargetModule

emit_executable(
    target_module,
    format,
    context,
) -> ExecutableArtifact
```

每个转换必须声明：

- 输入和输出层级；
- 支持的 profile；
- required/preserved semantics；
- 是否无损；
- 可能的 diagnostics；
- target/capability 依赖；
- cache key；
- round-trip 范围。

内部 rich IR 无法降回公共 CircuitIR 时返回结构化诊断，不允许静默删除控制流、时序、
动态测量或 target 信息。

### 19.10 Pass 与 Analysis Protocol

候选内部接口：

```python
class AnalysisPass(Protocol):
    name: str
    version: str

    def run(
        self,
        module: IRModule,
        context: CompilationContext,
    ) -> AnalysisResult: ...


class TransformationPass(Protocol):
    name: str
    version: str
    input_type: type[IRModule]
    output_type: type[IRModule]
    required_analyses: tuple[AnalysisKey, ...]
    preserved_analyses: tuple[AnalysisKey, ...]

    def run(
        self,
        module: IRModule,
        context: CompilationContext,
    ) -> PassResult: ...


@dataclass(frozen=True)
class PassResult:
    module: IRModule
    diagnostics: tuple[Diagnostic, ...]
    statistics: PassStatistics
```

PassManager 候选：

```python
pipeline = PassManager(
    [
        Canonicalize(),
        ConstantFold(),
        DecomposeToBasis(),
        PlaceQubits(),
        RouteQubits(),
        LegalizeTarget(),
    ]
)

result = pipeline.run(module, context)
```

约束：

- input module 不原地修改；
- TransformationPass 默认使未声明 preserved 的 analysis 失效；
- AnalysisPass 不得修改 IR；
- Pass failure 停止后续 pipeline；
- pipeline 配置和 Pass 版本进入 digest；
- 随机化 Pass 必须读取 context seed；
- Pass/Analysis 初期只供内部使用，不作为 Stable Core 插件 API；
- 外部 Pass 插件需要 conformance 和版本协商后再开放。

### 19.11 Target 与 backend API

`TargetCapabilities` 是版本化、只读、可哈希的编译输入：

```python
@dataclass(frozen=True)
class TargetCapabilities:
    identity: TargetIdentity
    n_qubits: int
    native_gates: tuple[GateCapability, ...]
    coupling_map: tuple[tuple[int, int], ...]

    supports_mid_circuit_measurement: bool
    supports_reset: bool
    supports_adaptive_control: bool

    accepted_formats: tuple[str, ...]
    openqasm_profiles: tuple[str, ...]
    qir_profiles: tuple[str, ...]

    schema_version: str

    @property
    def content_hash(self) -> str: ...
```

Compiler backend 不处理网络生命周期：

```python
class TargetBackend(Protocol):
    def capabilities(self) -> TargetCapabilities: ...

    def legalize(
        self,
        module: QuantumModule,
        context: CompilationContext,
    ) -> TargetModule: ...

    def emit(
        self,
        module: TargetModule,
        context: CompilationContext,
    ) -> ExecutableArtifact: ...
```

Runtime adapter 只处理 artifact：

```python
class RuntimeAdapter(Protocol):
    def submit(
        self,
        artifact: ExecutableArtifact,
        options: ExecutionOptions,
    ) -> JobHandle: ...

    def status(self, handle: JobHandle) -> JobStatus: ...
    def cancel(self, handle: JobHandle) -> CancelResult: ...
    def result(self, handle: JobHandle) -> ExecutionResult: ...
```

约束：

- TargetBackend 不读取 credential；
- RuntimeAdapter 不修改 target program；
- provider-specific compile service 如果不可避免，必须返回远端 compilation receipt；
- capability snapshot、账户权限和实时可用性分开；
- backend extension 初期属于 experimental extension SDK。

### 19.12 Diagnostic 与异常 API

候选诊断对象：

```python
@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Literal["error", "warning", "remark"]
    message: str
    location: SourceLocation | None
    notes: tuple[DiagnosticNote, ...]
```

诊断 code 使用稳定前缀：

```text
FQ-IR-xxxx
FQ-PASS-xxxx
FQ-TARGET-xxxx
FQ-EMIT-xxxx
FQ-ARTIFACT-xxxx
FQ-RUNTIME-xxxx
```

异常类型保持少而稳定：

```python
IRValidationError
CompilationError
TargetLegalizationError
ArtifactCompatibilityError
```

异常必须携带结构化 diagnostics。不得为每个 gate、Pass 或 provider 创建新的公共异常
类。warning/remark 不通过 `print` 输出，交由 diagnostic sink、report 或日志策略处理。

### 19.13 推荐模块布局与稳定性

候选模块布局：

```text
flagquantum/
  compiler/                    # approved public/experimental compiler facade
    __init__.py
    options.py
    result.py
  _compiler/                   # internal implementation
    ir/
    analyses/
    passes/
    pipelines/
    targets/
    emitters/
    diagnostics/
  experimental/
    compiler/                  # early extension surface, if needed
```

稳定性分级：

| API | 初始级别 | 稳定条件 |
| --- | --- | --- |
| `fq.CircuitIR` | Stable | 保持现有 schema/version policy |
| `fq.plan`、`fq.run` | Stable | 保持 API 收敛后的合同 |
| `flagquantum.compiler.compile` | Experimental | 两个 release cycle、行为/异常合同稳定 |
| `fq.compile` | Not present | 独立 additive API proposal |
| `CompilationOptions/Result` | Experimental | 字段、默认值、serialization 冻结 |
| `ExecutableArtifact` | Internal | ABI、签名、兼容和安全审计完成 |
| `ProgramModule/QuantumModule/TargetModule` | Internal | 不计划进入根 API |
| `PassManager/AnalysisManager` | Internal | 外部插件协议另行设计 |
| `TargetBackend/Emitter` | Experimental extension | conformance、版本协商、生命周期明确 |
| internal textual IR | Debug-only | 不承诺跨版本兼容 |

### 19.14 禁止模式

禁止新增：

```python
fq.ProgramIR
fq.QuantumIR
fq.TargetIR
fq.PassManager

CircuitIRV1
CircuitIRV2

ir.to_ir(level="target")
ir.convert("provider-x")

ir.metadata["physical_qubits"]
ir.metadata["gradient_method"]
ir.metadata["runtime_credentials"]
```

禁止行为：

- 用更新 snapshot 代替 API change proposal；
- 让 `compile` 隐式提交付费 QPU 任务；
- 让 RuntimeAdapter 静默重新编译或改写 artifact；
- 让 target/provider option 泄漏进 Stable Core 根签名；
- 用 `Mapping[str, Any]` 承载影响语义、合法性或 identity 的核心字段；
- 将 debug textual IR 当成长期持久化合同；
- 因内部 IR 变化同步发布新的公共 IR 类名；
- 把远端 provider 的成功提交当成 QPU 已执行证据。

### 19.15 API 正式化门槛

任何候选 API 从 Internal/Experimental 提升为 Stable 前必须具备：

- user journey 和非目标；
- 精确签名、类型、默认值和异常；
- serialization/identity 规则；
- compatibility 和 migration policy；
- executable semantic tests；
- public API contract checker；
- docs、quick start 和 release note；
- 至少两个不同 target/runtime 的 conformance；
- 性能基线和 fallback 可见性；
- API owner 批准；
- 目标分支保护和 CODEOWNER enforcement。

如果 API 收敛结果与本章候选不同，以批准后的 Stable Core 合同为准，并通过 ADR 记录
差异和原因；不得为了遵循本设计草案而破坏已批准的公共 API。



## 20. 参考资料

仓库内：

- `AGENTS.md`
- `docs/development/PUBLIC_API_PROTECTION.md`
- `docs/reference/PUBLIC_API_POLICY.md`
- `docs/roadmap/OPEN_SOURCE_API_QUALITY_PLAN.md`
- `docs/architecture/RUNTIME_ARCHITECTURE.md`
- `docs/reference/API.md`
- `capability-maturity.toml`

外部规范与架构参考：

- MLIR Language Reference: <https://mlir.llvm.org/docs/LangRef/>
- MLIR Pass Infrastructure: <https://mlir.llvm.org/docs/PassManagement/>
- MLIR Dialect Conversion: <https://mlir.llvm.org/docs/DialectConversion/>
- OpenQASM 3 Specification: <https://openqasm.com/versions/3.0/>
- QIR Specification: <https://github.com/qir-alliance/qir-spec>
- CUDA-Q Compiler IRs:
  <https://nvidia.github.io/cuda-quantum/latest/using/extending/compiler/cudaq_ir.html>

这些外部项目用于验证通用编译器设计和互操作边界，不意味着 FlagQuantum 必须复制
CUDA-Q 的用户模型，也不构成当前后端已经支持完整 OpenQASM、QIR 或 MLIR 的声明。
