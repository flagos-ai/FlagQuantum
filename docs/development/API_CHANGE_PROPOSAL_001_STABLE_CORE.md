# API Change Proposal 001: Stable Core 收敛

## 状态

**Approved — 进入实施，尚未冻结。**

批准记录：API owner 通过 2026-08-31 的明确用户指令批准 Stable Core 分类与
命名空间迁移，目标版本为首次公开 alpha。批准不等于 API freeze；最终冻结仍需满足
本文的验收条件。

本文定义首次公开 alpha 之前的根 API 收敛方案。它不是已冻结契约。机器可读分类位于
`contracts/public-api-v1-candidate.json`。其中原 `flagquantum.backends` 决策已由
API Change Proposal 018 取代。

## 决策摘要

当前根命名空间有 60 个 stable exports，但其中包含后端执行器、分布式研究入口、
MPS 生产验收对象、部署工具和项目介绍函数。建议将最终 Stable Core 控制为 22 项：

- 保留 20 个现有根级接口；
- 新增 `ExecutionOptions` 和 `ExecutionPlan` 两个核心类型；
- 19 个接口迁入稳定扩展命名空间；
- 18 个接口迁入 `experimental`；
- `get_version`、`hello`、`info` 在首次公开前从公开根 API 移除。

这次收敛不减少功能，只调整承诺等级和可发现位置。

## 最终 Stable Core 候选

| 领域 | 接口 |
| --- | --- |
| 构建与 IR | `Circuit`, `CircuitIR`, `Instruction`, `IR_VERSION` |
| 参数与观测 | `Parameter`, `ParameterExpression`, `ObservableNode`, `MeasurementNode`, `MeasurementResult` |
| 执行 | `ExecutionOptions`, `ExecutionPlan`, `ExecutionResult`, `RuntimePolicy`, `plan`, `run` |
| 训练 | `Module`, `TrainingResult`, `train` |
| 错误 | `IRSerializationError`, `IRValidationError` |
| 边界 | `experimental`, `__version__` |

选择标准是：是否属于多数用户的黄金路径、是否后端无关、是否能给出长期稳定语义、
是否需要从根命名空间直接发现。22 项低于 25 项预算，并覆盖
`Circuit → plan → run → ExecutionResult` 与 `Module → train → TrainingResult` 两条主线。

## 分层方案

### Stable Extensions

这些能力可以公开并稳定，但不应占据根命名空间：

| 命名空间 | 职责 |
| --- | --- |
| `flagquantum.runtime` | backend-native 执行和设备策略 |
| `flagquantum.compiler` | 专家编译入口 |
| `flagquantum.deployment` | package、provider 和 Pauli 测量部署 |
| `flagquantum.noise` | 噪声模型和专用模拟 |
| `flagquantum.operators` | gate schema 和查询 |
| `flagquantum.simulation.mps` | MPS 专用模拟 |
| `flagquantum.simulation.tensor_network` | TN 专用 amplitude/expectation 操作 |

普通用户仍通过 `fq.run` 使用这些能力；稳定扩展面向明确需要后端原生对象或专家控制的
用户。

### Experimental

以下能力尚不足以获得长期兼容承诺：

- distributed TN 和显式 distributed training；
- MPS production plan、acceptance gate、benchmark/release evidence；
- runtime-selection 和 cost-selection 专项 planner。

它们应迁入 `flagquantum.experimental.distributed`、
`flagquantum.experimental.mps` 和 `flagquantum.experimental.planning`。达到稳定门槛后，
可以升级到稳定扩展，无需扩大根 API。

### 开源前移除

`hello`、`info` 和 `get_version` 不承载量子 AI 产品能力。版本读取保留标准
`fq.__version__`，因此无需为这三个便利接口建立永久兼容承诺。

## 兼容迁移规则

由于仓库尚未公开，批准后应在首个公开 alpha 前一次性完成迁移，避免把内部历史设计
变成外部用户负担。仍需遵守以下工程顺序：

1. 先建立目标命名空间并验证新路径；
2. 将仓库代码、测试和文档迁移到新路径；
3. 运行黄金路径和完整 API contract；
4. 再收缩 `fq.__all__`、`__getattr__` 和 `__dir__`；
5. 更新最终 manifest，但保留 v0.2 baseline 作为审计记录；
6. 完成 alpha 发布审计后，才把 candidate 标记为 frozen contract。

如果批准时已经存在外部试用用户，应为迁移项增加显式 `DeprecationWarning`，而不是
直接移除。

## 明确不在本提案中完成的事项

- 不在本提案中确定 `ExecutionOptions` 的字段和配置优先级；
- 不在本提案中实现 `fq.run(ExecutionPlan)`；
- 不在本提案中收紧 `ExecutionResult` 字段或异常体系；
- 不承诺所有 stable extension 在首个 alpha 同时达到稳定等级。

这些内容分别由后续配置、可执行计划和 Result contract 提案完成，避免一个提案同时
改变命名、签名和行为而难以审查。

## 批准门槛

- [x] API owner 逐项批准 22 个 Stable Core 候选；
- [x] 每个迁移接口的目标命名空间已存在，并由 import contract 验证；
- [x] 仓库内关键黄金用户路径不再依赖历史根级扩展接口；
- [x] `ExecutionOptions` 后续提案已实现并验证（Proposal 002）；
- [x] `ExecutionPlan` 后续提案已实现并获根级批准（Proposal 003，freeze pending）；
- [x] CI 能证明 60 个当前导出全部且仅被分类一次；
- [ ] 明确首次公开 alpha 之前是否已有需要兼容的外部用户。

完成全部迁移和验收条件后，需单独批准 API freeze，不能因为本提案已批准而自动冻结。
