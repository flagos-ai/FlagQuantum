# API Change Proposal 002: ExecutionOptions

## 状态

**Implemented and verified — 已完成实现、根级清单授权与全量验证。**

- 目标版本：首次公开 alpha；
- 影响接口：`fq.ExecutionOptions`、`fq.plan`、`fq.run`、`Circuit.run`、
  `Circuit.plan`、`RuntimePolicy`；
- 机器可读候选：`contracts/execution-options-v1-candidate.json`；
- 前置决策：已批准的 API Change Proposal 001 将 `ExecutionOptions` 列为 Stable
  Core 计划新增项，但没有批准其字段和语义。

批准记录：API owner 于 2026-08-31 授权按本提案进入实现阶段，随后授权根级导出与
Stable Core manifest 迁移。该授权不包含 `ExecutionPlan` 输入或 `ExecutionResult` 变更；
契约冻结仍以完整验证通过为前提。

## 问题

当前执行行为可能同时来自：

- `Circuit` 的 `bsz`、`device`、`dtype` 和 `runtime_config`；
- `RuntimePolicy` 的 `mode`、`backend`、fallback 和 MPS 参数；
- `fq.plan(state_mode=..., bsz=..., config=...)`；
- `fq.run(mode=..., **options)`；
- task-local `RuntimeConfig`、环境变量和 backend 专用参数。

这带来四类公开风险：

1. 相同概念使用不同名称，例如 `mode/state_mode`、`batch_size/bsz`；
2. `**options` 接受未知键，错误可能推迟到某个后端才出现；
3. 调用者无法证明最终采用了哪个来源的配置；
4. 自动选择、显式选择和 `Module` 训练可能得到不同执行语义。

## 决策摘要

新增不可变、可序列化的 `fq.ExecutionOptions`，作为 `plan` 与 `run` 唯一稳定的
执行配置输入。它是一个**字段级覆盖对象**，不是已经解析完成的运行时配置：

```python
options = fq.ExecutionOptions(
    mode="mps",
    device="cuda:0",
    precision="complex64",
    target="expectation",
    require_gradients=True,
)

plan = fq.plan(circuit, options=options)
result = fq.run(circuit, options=options)
```

所有字段默认 `None`，含义严格为“本层未指定”。合并所有来源后，框架应用内置默认值，
形成内部 resolved options。内部 resolved 类型不进入 Stable Core。

## 稳定类型

建议实现形态：

```python
@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    mode: str | None = None
    backend: str | None = None
    device: str | None = None
    target: str | None = None
    batch_size: int | None = None
    precision: str | None = None
    shots: int | None = None
    seed: int | None = None
    memory_limit_bytes: int | None = None
    require_gradients: bool | None = None
    allow_approximate: bool | None = None
    allow_backend_fallback: bool | None = None
```

字段顺序、名称、类型、默认值、`frozen=True` 和 `slots=True` 都属于稳定契约。

### 字段语义

| 字段 | 稳定含义 | 首版内置默认值 |
| --- | --- | --- |
| `mode` | 数学状态表示，不包含分布式拓扑或内核实现 | `"auto"` |
| `backend` | tensor/autograd 执行实现或注册表名称 | `"auto"` |
| `device` | 逻辑设备请求；具体设备索引属于该字符串的一部分 | `"auto"` |
| `target` | 用户需要的结果形态，用于规划而不是承载请求内容 | `"auto"` |
| `batch_size` | 程序批大小约束 | 程序声明值，否则 `1` |
| `precision` | 复数计算精度 | `"complex64"` |
| `shots` | 抽样次数；`None` 表示未请求抽样 | `None` |
| `seed` | 可复现随机执行种子 | `None` |
| `memory_limit_bytes` | 每设备/每 rank 可使用的规划内存上限 | `None` |
| `require_gradients` | 计划必须支持参数梯度 | `False` |
| `allow_approximate` | 是否允许近似表示或截断 | `False` |
| `allow_backend_fallback` | 显式 backend/device 不可用时是否允许替代 | `False` |

`allow_approximate=False` 和 `allow_backend_fallback=False` 是 fail-closed 默认值。
`mode="auto"` 本身不是 fallback 授权；它表示由 planner 做初始选择。

### 首版受控值

`mode` 只接受：

```text
auto, statevector, mps, tensor_network, density_matrix
```

`distributed_statevector`、`distributed_mps` 和 `distributed_tensor_network` 不再是
数学 mode。分布式是执行布局，由资源环境和 planner 决定，并记录在
`ExecutionPlan`。`tn`、`distributed`、`distributed_tn` 等别名不进入稳定对象。

`target` 只接受：

```text
auto, state, expectation, samples, amplitudes
```

observable、bitstring 和 measurement request 的具体内容不是 `ExecutionOptions`
字段，应继续作为程序或请求对象输入。`target="auto"` 在没有显式 measurement 时
保持当前完整状态结果语义。

`precision` 首版只接受 `complex64` 和 `complex128`。real dtype、JAX x64 和内核
精度设置必须从该值一致派生，不能成为第二事实来源。

`backend` 与 `device` 使用受验证的注册表字符串，而不是永久封闭的 `Literal`，从而
允许第三方实现。`triton` 是 kernel provider，不应冒充完整 autograd backend。

## 配置优先级

按字段执行以下固定顺序，前者覆盖后者：

```text
每次 plan/run 显式传入的 ExecutionOptions
    > Module RuntimePolicy 中的 execution_options
    > Circuit 捕获的程序约束
    > task-local / 进程级 RuntimeConfig 适配值
    > FlagQuantum 内置默认值
```

规则：

- 只有非 `None` 字段参与覆盖；
- 同一字段不能同时通过 `options` 和迁移期 legacy keyword 提供；
- 冲突必须在进入 planner 前抛出 `TypeError`；
- 环境变量只提供资源拓扑、可用设备和内部性能调优输入，不能覆盖显式数学语义、精度、
  target 或 fallback 授权；
- 所有环境影响必须进入后续 `ExecutionPlan` 的 environment/provenance fingerprint。

## 与现有对象的边界

### RuntimeConfig

`RuntimeConfig` 保留为 task-local 的内部/专家配置与序列化载体，但不再是稳定执行入口。
它必须通过单一 adapter 转换为低优先级 `ExecutionOptions`，不能由 planner 直接读取
并与显式参数竞争。

`drawing_style`、JAX matmul precision 和 operator-kernel 开关不是 Stable Core
执行语义，不进入 `ExecutionOptions`。

### RuntimePolicy

`RuntimePolicy` 应收敛为：

- `execution_options: ExecutionOptions`；
- Module/训练专用的 observable 与 correctness policy。

现有 `mode`、`backend`、`allow_backend_fallback`、`mps_max_bond` 和 `mps_cutoff`
字段应通过开源前迁移或明确兼容 adapter 处理，不能与 `execution_options` 并存为两个
等价事实来源。MPS bond/cutoff 属于 backend extension policy，不进入 Stable Core
`ExecutionOptions`。

### Circuit

`Circuit` 中的输入 batch、输入 tensor dtype/device 是程序约束，不是更高优先级的
执行偏好：

- `batch_size` 与程序固有 batch 不一致时，planning 阶段抛出 `ValueError`；
- precision/device 需要转换时必须由 planner 显式记录转换；不支持时在 planning
  阶段失败；
- 不允许 `Circuit.run` 静默注入 `bsz/device/dtype/config` 并改变显式 options；
- `Circuit.run(options=x)` 必须与 `fq.run(circuit, options=x)` 等价。

## 稳定调用签名目标

本提案批准后，Phase 2 的目标签名为：

```python
fq.plan(program, *, options: ExecutionOptions | None = None)
fq.run(
    program,
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
) -> ExecutionResult
Circuit.plan(*, options: ExecutionOptions | None = None)
Circuit.run(
    *,
    options: ExecutionOptions | None = None,
    measurements: Sequence[MeasurementNode] | None = None,
) -> ExecutionResult
```

`noise_model` 的最终稳定位置、`fq.run(ExecutionPlan)` 和 plan identity 属于后续提案，
不由 Proposal 002 顺带冻结。

## 验证与错误契约

- 构造器出现未知 keyword 或错误 Python 类型：`TypeError`；
- 值不在受控集合、整数小于合法下限：`ValueError`；
- 多来源提供同一字段：`TypeError`；
- 程序固有 batch/precision/device 与解析结果冲突：planning 阶段 `ValueError`；
- 显式 backend/device 不可用且未授权 fallback：planning 阶段失败；
- 不允许执行器静默更正、丢弃或重新解释 options。

整数规则：`batch_size >= 1`、`shots >= 1`、`memory_limit_bytes >= 1`；`seed >= 0`。
Python `bool` 不作为合法整数接受。

## 序列化

`ExecutionOptions.to_dict()` 使用：

```json
{
  "schema": "flagquantum.execution_options",
  "version": "1.0",
  "mode": null,
  "backend": null,
  "device": null,
  "target": null,
  "batch_size": null,
  "precision": null,
  "shots": null,
  "seed": null,
  "memory_limit_bytes": null,
  "require_gradients": null,
  "allow_approximate": null,
  "allow_backend_fallback": null
}
```

序列化保留 `None`，因为它代表字段级继承语义。`from_dict()` 必须拒绝未知 schema、
version 和字段；不能静默忽略未来字段。

## 明确排除

以下内容不进入 Stable Core `ExecutionOptions`：

- `world_size`、rank、node、process group 和通信 backend；
- MPS `max_bond/cutoff`、TN slicing/path、statevector kernel 开关；
- routing strategy、coupling map 和 compiler pass；
- noise model、observable、bitstrings、measurement request；
- checkpoint、optimizer、training steps；
- vendor credentials、provider job parameters；
- 任意 `extras`/`kwargs` 字典。

这些参数应属于资源环境、程序/请求、compiler policy、backend extension、deployment
或 experimental namespace。禁止通过 `extras` 绕过稳定 API 审查。

## 迁移计划

1. 审批本提案及机器可读候选，不改运行时代码；（已完成）
2. 实现 `ExecutionOptions`、严格验证、序列化和 options resolver；
3. 让 `plan/run/Circuit/RuntimePolicy` 只调用同一个 resolver；
4. 为 legacy keyword 建立仓库内迁移清单，未知键立即失败；
5. 迁移 production、tests、examples 和 docs；
6. 运行自动/显式路径、Module、noise、distributed 和 backend conformance；
7. 单独批准后，将 `ExecutionOptions` 加入 root manifest；（已完成）
8. Proposal 003 完成后再增加 `ExecutionPlan` 输入语义。

## 验收标准

- [x] API owner 批准字段、顺序、类型、默认值和 fail-closed 策略；
- [x] `ExecutionOptions` 不包含 backend-specific 或 distributed orchestration 字段；
- [x] 所有稳定配置来源由一个 resolver 按字段合并；
- [x] `plan` 与 `run` 对同一 program/options 得到同一 resolved options；
- [x] `Circuit.run` 与 `fq.run` 等价；
- [x] 未知字段、冲突字段和非法值在 planning 前失败；
- [x] 序列化 round trip 与未知字段拒绝测试通过；
- [x] production、tests、examples 与当前文档中的 legacy keyword 调用清单归零；
- [x] 完整 Stable Core、runtime、distributed 和文档契约通过；
- [x] API owner 单独批准加入 root manifest。

## 实施记录

2026-08-31 完成公开接口集成：

- `flagquantum.runtime.options.ExecutionOptions` 已实现不可变 slots dataclass、严格值验证
  和 v1 序列化；
- `flagquantum.runtime.options_resolver.resolve_execution_options` 已实现唯一字段级优先级、
  RuntimeConfig adapter、程序 batch 冲突检查和字段来源记录；
- 根级 `fq.ExecutionOptions`、`fq.plan/run`、`Circuit.plan/run` 与 `RuntimePolicy`
  已统一到该契约；backend 专用控制迁入 `fq.experimental`；
- 语义契约覆盖 plan/run 一致性、Circuit 等价性、采样可复现、旧关键字拒绝和显式后端
  fail-closed；default、runtime 与 distributed 三档验证通过后，候选契约已冻结。

## 请求批准的决策

本提案已经获得实现和根级清单授权，但**尚不等于 API freeze**，也不授权顺带实现
`fq.run(ExecutionPlan)`、改变 `ExecutionResult` 或新增其他 Stable Core 名称。
