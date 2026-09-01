# API Change Proposal 006：公开异常与 Module 构造边界

## 状态

**Contract frozen — 候选契约已获单独冻结批准。**

- 目标版本：首次公开 alpha；
- 影响接口：`flagquantum.errors`、现有领域异常、`fq.Module` 构造器；
- 根级名称变化：无；
- 根级签名变化：从 `fq.Module` 删除 `deployment_binding`；
- 授权记录：API owner 于 2026-09-01 通过明确用户指令授权推进 Proposal 006；
- 前述实现授权本身不等于整个 FlagQuantum API 的最终 freeze。
- 冻结记录：API owner 于 2026-09-01 明确批准 review packet 所绑定的 Proposal 006
  精确契约；本冻结不等于整体首次公开 Alpha freeze。

## 问题

稳定入口此前会混合抛出 Python 内置异常、领域异常和后端偶然异常。调用方难以按
validation、planning、execution 等阶段处理失败。同时，`fq.Module` 构造器包含一个
未校验的 `deployment_binding: Mapping[str, Any]`，但该字典不参与量子执行、规划或
训练，只被高层应用模型用于部署摘要。这使 PyTorch 量子层承担了不属于它的 provider
状态。

## 公开异常决策

稳定命名空间 `flagquantum.errors` 提供：

```text
FlagQuantumError
├── ValidationError      (同时是 ValueError)
├── PlanningError        (同时是 ValueError)
├── SerializationError   (同时是 ValueError)
├── CompilationError     (同时是 RuntimeError)
├── ExecutionError       (同时是 RuntimeError)
└── CapabilityError      (同时是 NotImplementedError)
```

双重继承保留已有 Python 异常捕获兼容性。错误的 Python 参数类型和未知关键字继续使用
`TypeError`；有效类型但无效取值使用 `ValidationError`。稳定边界不得把 PyTorch、JAX、
NCCL 或 provider 的偶然异常作为公共契约，底层异常应通过 `__cause__` 保留。

现有领域异常保持原名称，同时进入统一类别：

- `IRValidationError` → `ValidationError`；
- `IRSerializationError` → `SerializationError`；
- `ExecutionPlanContractError` → `PlanningError`；
- `TrainingStateError` → `ExecutionError`；
- Result 缺失数据 → `ExecutionError`。

## Module 构造决策

保留三条已验证构造路径：

1. callable builder + flat parameters；
2. callable builder + named parameter groups；
3. parameterized Circuit template。

删除 `fq.Module(..., deployment_binding=...)`。Module 只保存 RuntimePolicy、参数和训练
状态；部署绑定由应用模型或 `flagquantum.deployment` 所有。仓库内
`HybridQuantumClassifier` 与 `VariationalEnergyModel` 将 binding 提升为自身状态，并通过
PyTorch extra-state 保存，因此高层模型 checkpoint 不丢失部署信息。

旧 Module checkpoint 中 extra-state 的 `deployment_binding` 字段继续可读但被忽略，
避免私有阶段已有 checkpoint 无法加载。首次公开版本不承诺 Module 级 binding。

## 可执行验收标准

- [x] 七个异常类型只在 `flagquantum.errors` 稳定命名空间公开；
- [x] 所有类别同时继承 `FlagQuantumError` 和对应 Python 内置异常；
- [x] IR、Plan、Result 和 training-state 领域异常进入统一类别；
- [x] stable Circuit/options/policy/Module/train 的语义取值错误使用 ValidationError；
- [x] unsupported planning/measurement/Module path 使用 CapabilityError；
- [x] Module 构造签名不再包含 deployment_binding；
- [x] Module 仍接受并忽略旧 checkpoint extra-state 中的 binding；
- [x] 高层应用模型 checkpoint 保留 binding；
- [x] API owner 单独批准 contract freeze。
