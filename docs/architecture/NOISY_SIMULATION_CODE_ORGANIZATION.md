# Noise 版本代码组织与架构审查

## 1. 结论

FlagQuantum 当前顶层分层方向合理，但复杂度已经接近需要主动治理的阶段。
Noise 版本不能继续把功能直接堆入 `simulation/noise.py`、
`runtime/execution.py` 和现有 planner，否则会复制 TN 已经出现的模块膨胀和
内部接口扩散。

架构上的核心原则是保持三个正交维度：

- Noise 是公共物理语义与编译输入；
- quantum trajectory 是共享执行策略；
- SV、density matrix、MPS、TN 是状态表示后端。

Noise 不应成为与 SV/MPS/TN 平级的状态后端，也不应通过不断增加组合模式
字符串表达所有执行方式。

## 2. 当前结构评估

### 2.1 可以保留的分层

以下目录的所有权方向是合理的：

```text
core/               IR、contracts、parameters
compilation/        分析、选择和执行计划
runtime/            调度、执行、结果和审计
runtime/backends/   SV、density、MPS、TN、JAX 后端
ops/                门语义、矩阵和低层 lowering
testing/            正确性和能力认证
benchmarking/       正式基准协议
docs/               能力、架构和 claim boundary
```

按状态表示组织 `runtime/backends/statevector`、`mps` 和
`tensor_network` 的方向应继续保留。

### 2.2 P0：高层 Circuit 反向进入底层执行

当前 `simulation/noise.py` 和 `simulation/mps_execution.py` 导入
`circuit._gate_matrix`。运行时 TN/MPS 后端还会导入 simulation 模块中的
私有 kernel。

这形成错误的依赖方向：

```text
backend/simulation → Circuit
```

正确方向应为：

```text
Circuit
   ↓
Core IR
   ↓
Compilation
   ↓
Runtime/backend
   ↓
Numerical kernels
```

共享门矩阵 lowering 应由 `ops` 或其他低层模块拥有，后端不得导入
`Circuit` 私有实现。

### 2.3 P0：Noise 执行语义在计划中丢失

当前 `mps_trajectory` 和 `noisy_mps` 最终会规范化为普通 `mps`。执行计划
无法完整表达：

- 单轨迹或多轨迹；
- trajectory count 和 batch size；
- RNG schema；
- 目标标准误；
- MPS 截断误差；
- trajectory parallel 或 state sharding；
- checkpoint 状态。

未来计划需要拆成正交维度：

```text
StateRepresentation
    statevector | density_matrix | mps | tensor_network

EvolutionSemantics
    pure | exact_mixed | quantum_trajectory

ParallelStrategy
    local | trajectory_parallel | state_sharded | hybrid

DifferentiationStrategy
    autograd | adjoint | parameter_shift | paired_trajectory
```

### 2.4 P1：中央执行入口持续膨胀

`runtime/execution.py` 当前同时承担模式分派、参数清理、noise lowering、
计划构造、执行和结果适配。继续增加 Noise 模式会产生大量新的 `elif` 和参数
组合。

目标入口应收敛为：

```python
request = normalize_request(...)
plan = planner.build(request)
executor = executor_registry.resolve(plan)
raw = executor.execute(plan, request)
return result_adapter.normalize(raw, plan)
```

旧分支可以逐个迁移，不要求一次性重写。

### 2.5 P1：TN 内部接口和文件体量膨胀

当前热点包括：

- `simulation/tensor_contraction.py` 超过 1800 行；
- `runtime/distributed/tensor_network_execution.py` 超过 1000 行；
- TN backend facade 暴露大量内部类型和 kernel；
- 多个 TN backend 模块在 600–800 行之间。

Noise 开发期间应冻结以下增长：

- 不向 `tensor_contraction.py` 加入 noisy TN；
- 不向 TN `__init__.py` 增加新的内部导出；
- 新 pair kernel 放到有明确所有权的模块；
- SV/MPS trajectory 稳定前不启动 noisy TN；
- 逐步将 `runtime/distributed/tensor_network_execution.py` 收敛为 facade。

### 2.6 P2：现有 Noise 模块混合多层职责

`simulation/noise.py` 同时包含：

- Noise domain model；
- 通道工厂；
- NoiseModel → IR lowering；
- density-matrix kernel；
- density-matrix execution；
- expectation 计算。

现阶段尚能维护，但加入 relaxation、readout、device profile、trajectory、
统计和硬件导入后会迅速成为新的单体模块。

### 2.7 P2：NoiseModel 尚不适合作为长期公共模型

当前模型是可变对象，Kraus 通道直接保存 device-bound Torch tensor，并缺少：

- 稳定 schema 与 identity；
- 显式 CPTP 验证；
- duration、idle、readout 和 reset 语义；
- 硬件校准 provenance；
- specification 与 compiled channel 的区分。

长期模型应保存可序列化的设备无关规格，编译阶段再生成 dtype/device 专属
Kraus tensor。

## 3. Noise 目标目录

### 3.1 公共噪声语义

```text
flagquantum/noise/
├── __init__.py
├── channels.py
├── model.py
├── rules.py
├── device_profile.py
├── validation.py
└── serialization.py
```

职责：

- `channels.py`：不可变、可序列化的通道规格；
- `model.py`：`NoiseModel`、规则和 placement；
- `rules.py`：门、wire、idle、readout 匹配；
- `device_profile.py`：硬件校准及 provenance；
- `validation.py`：概率、维度、CPTP 和组合规则；
- `serialization.py`：schema、digest、identity 和 round trip。

### 3.2 Noise 编译层

```text
flagquantum/compilation/noise/
├── __init__.py
├── analysis.py
├── lowering.py
├── selection.py
├── planning.py
└── estimates.py
```

`lowering.py` 是将 `CircuitIR + NoiseModel` 转换为带 ChannelInstruction 的
IR 的唯一入口。它只能依赖 core、noise domain 和 ops schema，不允许依赖
Circuit、runtime、simulation 或具体 backend。

### 3.3 Density matrix 数值实现

```text
flagquantum/simulation/density_matrix.py
```

`expand_operator`、`apply_unitary_density`、`apply_kraus_density`、
`density_matrix_from_ir` 和 density expectation 均由 Simulation 所有。
噪声 lowering、执行计划校验和 executor 分派由 `runtime/noise_registry.py`
负责，数值实现不依赖 Runtime。

### 3.4 共享 trajectory 运行时

```text
flagquantum/runtime/trajectories/
├── __init__.py
├── models.py
├── rng.py
├── statistics.py
├── scheduler.py
├── checkpoint.py
├── distributed.py
└── result.py
```

它统一拥有：

- global trajectory ID；
- world-size-independent RNG；
- Welford statistics；
- adaptive stopping；
- rank ownership；
- checkpoint/resume；
- failure accounting；
- confidence interval。

它不负责状态演化 kernel。

### 3.5 后端专属 trajectory

```text
runtime/backends/statevector/
├── trajectory.py
└── noisy_kernels.py

runtime/backends/mps/
├── trajectory.py
└── noisy_operations.py

runtime/backends/tensor_network/
└── trajectory.py       # 后续
```

每个后端只负责给定编译后的 ChannelInstruction 时如何推进一条轨迹。统计、
调度、RNG 和 checkpoint 不得在各后端重复实现。

## 4. 强制依赖规则

```text
noise
  → core
  → 禁止依赖 runtime/simulation/circuit

compilation.noise
  → core + noise + ops schema
  → 禁止依赖 backend

runtime.trajectories
  → core contracts + compilation plans
  → 禁止依赖具体 SV/MPS/TN backend

backend
  → core + compiled plan + numerical kernel
  → 禁止导入 Circuit 私有函数

api/circuit
  → runtime public entrypoint
  → 禁止被 backend 反向导入
```

应逐步消除：

```python
from flagquantum.circuit import _gate_matrix
from flagquantum.simulation.tensor_contraction import _einsum_pair_by_labels
from flagquantum.simulation.mps import _split_pair_matrix
```

## 5. Noise 开发前的架构准备

### A1：抽离 gate matrix lowering

将 `_gate_matrix` 从高层 Circuit 移到低层 `ops`。Circuit、density、MPS 和
SV 共用该实现，消除 Noise/MPS 到 Circuit 的反向依赖。

### A2：建立 Noise domain package

先迁移模型与通道工厂，不迁移执行。`simulation/noise.py` 保留兼容导出和明确
移除版本。

### A3：建立结构化 NoisyExecutionPlan

至少包含：

```python
@dataclass(frozen=True)
class NoisyExecutionPlan:
    representation: StateRepresentation
    evolution: EvolutionSemantics
    trajectory: TrajectoryPlan | None
    parallel: ParallelPlan
    error_budget: NoiseErrorBudget
    memory: MemoryPlan
```

### A4：引入 executor registry

先支持新的 Noise executor，旧模式逐步迁移：

```python
executor_registry.register(
    representation="mps",
    evolution="quantum_trajectory",
    executor=MPSQuantumTrajectoryExecutor,
)
```

### A5：限制继续扩张

- 新 Noise 文件不得申请大文件例外；
- 不在 `runtime/execution.py` 增加新的组合模式分支；
- 不复制 RNG、统计、checkpoint 和 distributed ownership；
- noisy TN 延后到共享 trajectory runtime 稳定之后。

## 6. 测试组织

```text
tests/noise/
├── test_channels.py
├── test_model.py
├── test_serialization.py
├── test_lowering.py
├── test_density_correctness.py
├── test_trajectory_statistics.py
├── test_trajectory_reproducibility.py
├── test_backend_selection.py
└── test_result_contract.py

tests/distributed/noise/
├── test_trajectory_ownership.py
├── test_world_size_invariance.py
├── test_checkpoint_resume.py
└── trajectory_runtime.py
```

约束：

- domain tests 不导入 runtime；
- lowering tests 不初始化 CUDA；
- backend correctness 以 density path 为参考；
- distributed tests 只验证调度和统计；
- 硬件 benchmark 不进入普通 unit test。

## 7. 模块规模约束

| 类型 | 建议上限 |
|---|---:|
| domain/model 模块 | 300–500 行 |
| backend execution 模块 | 600–800 行 |
| `__init__.py` facade | 150 行 |
| 单个公共 `__all__` | 30–40 个符号 |
| runtime 中央入口 | 400–500 行 |
| 单个测试文件 | 800–1000 行 |

既有超限文件可以暂时作为有 owner 和移除版本的 legacy exception，新 Noise
模块不得从一开始依赖例外。

## 8. 推荐推进顺序

```text
Architecture Preparation
    ↓
Noise domain + serialization
    ↓
Channel IR lowering
    ↓
Exact density backend
    ↓
Shared trajectory runtime
    ↓
MPS trajectory migration
    ↓
Batched SV trajectory
    ↓
Distributed trajectory
    ↓
Automatic selection
    ↓
Gradients
    ↓
MPO / noisy TN
```

禁止采用“先分别实现、最后统一”的路线。SV/MPS/TN 必须共享同一套
trajectory ID、RNG、统计、checkpoint 和 error accounting。
