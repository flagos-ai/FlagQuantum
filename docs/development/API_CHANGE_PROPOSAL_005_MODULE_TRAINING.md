# API Change Proposal 005：Module 与 Training 契约收敛

## 状态

**Implemented, freeze pending — 已实现核心语义与机器契约，等待最终冻结审批。**

- 目标版本：首次公开 alpha；
- 影响接口：`fq.Module`、`fq.train`、`fq.TrainingResult`、
  `fq.ExecutionResult` 和 `flagquantum.training`；
- 根级名称变化：无；
- 授权记录：API owner 于 2026-09-01 通过明确用户指令授权推进 Proposal 005；
- 本授权不等于整个 FlagQuantum API 的最终 freeze。

## 核心定位

FlagQuantum 保持 PyTorch-first，而不是重新发明一套训练框架：

```text
module(inputs)          -> autograd-compatible Tensor
module.execute(inputs)  -> ExecutionResult
fq.train(module, ...)   -> TrainingResult
```

`fq.run` 只运行 Circuit、CircuitIR 或 ExecutionPlan，不接受 Module。Module 不增加
`run()` 同义方法，避免 `forward/execute/run` 三套入口继续漂移。

## Module 契约

### forward 与 execute

- `forward()` 永远返回 `torch.Tensor`，可直接进入 PyTorch 模型和 autograd；
- `execute()` 永远返回 `ExecutionResult`；
- 相同 inputs、parameters、policy 和 module state 下，两者的 value 数值相同；
- `ExecutionResult.require_value()` 是 Module value 的失败闭合访问器；
- local PyTorch fast path 可以避免构造 diagnostics，但不得改变 observable、batch、
  dtype、gradient 或 backend 选择语义。

### inputs、parameters 与 batch

- `parameters=None` 使用 Module 拥有的参数；
- 显式 parameters 仅用于本次调用，不替换 Module parameter ownership；
- 双参数 builder 接收 `(parameters, inputs)`，单参数 builder 只接收 parameters；
- FlagQuantum 不隐式构造 input batch × parameter batch 的笛卡尔积；
- builder 和所选 backend 必须产生一致、可广播的张量形状，否则明确失败；
- observable 由 `RuntimePolicy` 唯一决定，forward 和 execute 不设置不同默认值。

## `fq.train` 的边界

选择“小而稳定的训练循环”，不在顶层 API 内复制 Lightning/Trainer：

- optimizer 由调用方创建和持有；
- `fq.train` 负责 `zero_grad → execute → objective → backward → step`；
- objective 接收 `execution.require_value()`；
- callback 每步收到 one-based step、float loss 和 detached ExecutionResult；
- `TrainingResult.last_execution` 必须 detached；
- `fq.train` 不内置 checkpoint、resume、early stopping 或 validation loop；
- 这些高级生命周期由调用方、callback 或未来独立 Trainer extension 组合。

分布式训练在达到同一语义前继续位于
`flagquantum.experimental.distributed`，不进入 stable root。

## TrainingResult 与 diagnostics

`TrainingResult` 固定四个字段：

```text
losses, completed_steps, last_execution, optimizer
```

并保证 `len(losses) == completed_steps`，提供 `final_loss`，summary 使用：

```text
schema  = flagquantum.training_result.summary
version = 1.0
```

`ExecutionResult.diagnostics()` 返回版本化 envelope：

```text
schema, version, metrics, provenance, runtime, compatibility
```

四个 section 内的 key 允许向后兼容地增加；用户业务逻辑应依赖稳定 accessor，而不是
某个 backend 偶然产生的诊断 key。

## Checkpoint / Resume 边界

Checkpoint 由 `Module.save_checkpoint/load_checkpoint` 负责，而不是 `fq.train`：

- save 接受非负整数 seed，避免公开方法依赖隐藏的 SeedContract 构造；
- 文件是带版本、原子替换的 PyTorch training-state，不宣称跨语言交换格式；
- 保存 module、optimizer、RuntimePolicy、IR/workload identity、RNG、precision 和 topology；
- workload、optimizer、precision 或 topology 不一致时 fail closed；
- restore 返回带 schema/version 的 `TrainingCheckpointRestore`；
- 普通用户类型从 `flagquantum.training` 导入，低层 save/load 函数不再暴露在根命名空间。

## PyTorch 与 JAX

- PyTorch 是稳定 autograd 与 optimizer 接口；
- JAX 是 `RuntimePolicy` 选择的可选 compiled backend，不建立第二套 Module API；
- 未授权 fallback 必须失败；
- 授权 fallback 必须记录 requested backend、selected backend 和 reason；
- callback、TrainingResult 和 checkpoint ownership 不随 backend 改变。

Module 构造器中的 deployment/provider 状态由 Proposal 006 继续收敛；Proposal 006
删除 `deployment_binding`，并将其所有权提升到应用模型或 deployment 层。

## 可执行验收标准

- [x] forward 返回 Tensor，execute 返回 ExecutionResult；
- [x] forward/execute value 在 statevector、MPS、TN 路径一致；
- [x] forward value 保持 autograd；
- [x] `fq.run(module)` 明确失败，Module 不提供 `run`；
- [x] `require_value()` 对缺失 value 失败闭合；
- [x] TrainingResult 字段、invariant、final_loss 和 summary 版本受保护；
- [x] diagnostics envelope 带 schema/version；
- [x] checkpoint restore 带 schema/version，并继续执行 mismatch preflight；
- [x] training lifecycle 类型迁入 `flagquantum.training`；
- [x] 分布式 training 保持 experimental；
- [ ] API owner 单独批准 contract freeze。
