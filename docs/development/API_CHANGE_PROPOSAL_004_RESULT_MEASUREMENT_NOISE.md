# API Change Proposal 004：Result、Measurement 与 Noise 语义收敛

## 状态

**Implemented, freeze pending — 已实现核心语义与机器契约，等待最终冻结审批。**

- 目标版本：首次公开 alpha；
- 影响接口：`fq.plan`、`fq.run`、`Circuit.plan`、`Circuit.run`、
  `ExecutionResult`、`MeasurementResult` 和 `ExecutionPlan` noise extension；
- 根级名称变化：无；
- 授权记录：API owner 于 2026-09-01 通过明确用户指令授权推进 Proposal 004；
- 本授权不等于整个 FlagQuantum API 或 Result schema 的最终冻结。

## 要解决的问题

Proposal 003 建立了可序列化、可验证、可精确执行的 `ExecutionPlan`，但临时保留了
measurement 和 noise 的旁路，结果契约也仍暴露后端偶然行为：

1. IR 与 `measurements=` 同时存在时，显式参数会静默覆盖 IR；
2. `noise_model=` 直接进入 runtime，不能通过 `fq.plan` 检查、保存和恢复；
3. 带 measurement/noise 的 program 路径与 `fq.run(plan)` 不是同一条执行路径；
4. `ExecutionResult.__getattr__` 会把 backend-native 属性伪装成稳定 API；
5. 用户只能检查多个 optional 字段，缺少失败闭合的结果访问器；
6. result summary 没有 schema/version，自动化消费者无法判断结构版本。

## 决策

### 1. 所有稳定执行先形成计划

`fq.run(program, ...)` 必须等价于：

```python
plan = fq.plan(
    program,
    options=options,
    measurements=measurements,
    noise_model=noise_model,
)
result = fq.run(plan)
```

稳定路径不再保留 measurement/noise transient execution。`fq.run(plan)` 继续禁止任何
options、measurement 或 noise override，避免执行内容与 plan identity 分离。

### 2. Measurement 只有一个规范来源

- 最终 measurement requests 必须写入计划所携带的 `CircuitIR.measurements`；
- `measurements=` 只是构建计划时的便捷输入，不是执行时的覆盖层；
- program 已含 measurements 时再次传入 `measurements=` 必须报错；
- 不提供隐式 replace 或 append；
- 请求顺序进入 program fingerprint，并决定 `result.measurements` 的顺序；
- `ExecutionOptions(target="samples", shots=...)` 生成的 sample request 同样写入 IR。

这使 measurement 的序列化、identity、恢复执行和审计采用同一个事实来源。

### 3. Noise 成为可验证的计划扩展

稳定入口只接受 `flagquantum.noise.NoiseModel`。模型以一个严格扩展写入
`ExecutionPlan.extensions`：

```json
{
  "namespace": "flagquantum.noise",
  "kind": "noise_model",
  "version": "1.0",
  "identity": "<sha256>",
  "payload": {"schema": "flagquantum.noise_model.v1"}
}
```

读取计划时必须重新构造 NoiseModel 并核对 identity；未知字段、未知扩展、多个扩展或
identity 不一致均以 `extension_incompatible` 失败。首次公开 alpha 的稳定 noisy path
仅承诺 `mode="auto"` 和 `mode="density_matrix"`；其他组合在 planning 阶段失败，
而不是运行到 kernel 后才失败。

### 4. Result 使用显式访问器

稳定访问器为：

- `result.statevector()`：返回 statevector，否则明确失败；
- `result.require_samples()`：返回 samples，否则明确失败；
- `result.measurement(index_or_name)`：按位置、metadata name 或唯一 kind 选择；
- `result.expectation(...)`：只接受 expectation measurement；
- `result.native()`：显式取得不稳定的 backend-native 对象；
- `result.summary()`：返回带 schema/version 的摘要。

删除 `ExecutionResult.__getattr__` 的隐式 native 属性透传。backend-native 对象可以在
稳定访问器内部用于转换，但它的类型、方法和生命周期不属于兼容性承诺。

`MeasurementResult.value` 的候选类型收窄为：

```text
torch.Tensor | list[dict[str | int, int]]
```

分别覆盖 expectation/probability/sample 与 counts。完整 tensor payload 的跨版本 JSON
序列化本提案明确不承诺；稳定的是 plan、结果访问语义与版本化 summary，而不是把设备
tensor 或 autograd graph 隐式序列化。

## 兼容性边界

- `result.to_statevector()` 暂时保留，作为 `statevector()` 的兼容拼写；
- `value/state/samples/measurements` 字段暂时保留，便于训练和张量组合；
- `metrics/provenance/runtime/compatibility` 仍是诊断映射，不在本提案冻结其全部 key；
- `Module.execute()` 复用 `ExecutionResult`，但 Module 的 plan 类型由 Proposal 005 收敛；
- `noise_model` 参数名称本轮保留，避免把名称迁移与执行语义迁移混在一次变更中；
- 动态电路、provider job result 和 deployment result 不在本提案范围。

## 可执行验收标准

- [x] 显式 measurements 写入 plan，并通过 JSON round trip 后执行；
- [x] IR 与显式 measurements 冲突时失败，不发生隐式覆盖；
- [x] noise model payload 和 identity 写入 plan；
- [x] noisy plan 可 JSON round trip 后精确执行；
- [x] noise identity 篡改在执行前失败；
- [x] program 路径和 plan 路径共用 exact-execution path；
- [x] result accessor 对缺失和歧义数据明确失败；
- [x] backend-native 属性不再隐式泄漏；
- [x] summary 带 schema/version；
- [ ] API owner 单独批准 contract freeze。

## 后续决策

Proposal 005 已收敛 `Module`、`TrainingResult` 与 execution result 的关系：

1. `ExecutionResult.require_value()` 成为 Module/training 的失败闭合访问器；
2. diagnostics 使用带 schema/version 的 envelope，section key 允许兼容性增加；
3. `Module.forward` 返回 Tensor，`Module.execute` 返回 ExecutionResult；
4. `fq.train` 保持最小 optimizer loop，checkpoint 由 Module 所有。

`noise_model` 命名、`to_statevector()` 兼容窗口和统一异常基类仍留给后续提案，
Proposal 005 不借机扩大变更范围。
