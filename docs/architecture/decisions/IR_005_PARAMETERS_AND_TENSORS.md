# IR-005：参数、Tensor 常量与 Late Binding

状态：Approved
日期：2026-09-01
适用阶段：Phase 1 importer、identity 与 gradient differential

批准记录：API owner 于 2026-09-01 通过明确指令批准 IR-004～006。批准范围仅限内部
参数分类、binding table 和 identity 规则；不修改公共 Parameter、Module、checkpoint
或训练 API，具体 autograd 实现仍须 training owner 复核。

## 背景

CircuitIR gate 参数可以是 Python scalar、`Parameter`、`ParameterExpression` 或 torch
tensor。公共 JSON/hash 会对 tensor detach 后编码，但训练执行必须保留 autograd graph。
内部 immutable IR 不能通过复制/序列化 tensor 意外切断梯度，也不能把每次训练数值变化
错误当成程序结构变化。

## 决策

Phase 1 将参数分为三类：

1. **StaticConstant**：Python bool/int/float/complex 或明确静态的标量 tensor；
2. **SymbolicParameter/Expression**：保持名称、表达式 operation 和参数依赖；
3. **RuntimeBindingRef**：训练 tensor 和明确 late-bound 参数只在 module 中保存稳定 slot，
   实际 tensor 存放在 module 外的 binding table。

规则：

- `requires_grad=True` tensor 必须进入 `RuntimeBindingRef`，不得 detach/copy 后用于执行；
- binding table 保留原 tensor object，使测试 lowering 能连接现有 PyTorch autograd；
- internal program identity 包含 slot、shape、dtype 和表达式结构，不包含 late-bound 数值；
- execution identity 才包含本次 binding identity/value policy；Phase 1 不接入公共 identity；
- Python/static scalar 的 canonical encoding 必须区分 bool/int/float/complex；
- tensor dtype 不得静默降级，非标量 gate 参数除明确 schema 外 fail closed；
- `ParameterExpression` 只接受当前公共参数系统支持的 operation；
- binder 必须验证缺失、多余、shape、dtype 和 device policy；
- parameter binding 不得隐式改变控制流、wire、arity 或 target layout。

## 否决方案

- **所有 tensor detach 后写入 IR**：会切断 autograd，并导致训练值污染 program cache；
- **所有参数都排除 identity**：静态常量变化可能改变程序语义却错误命中缓存；
- **保存任意 Python callable**：不可确定、不可序列化且无法安全验证。

## 兼容与回滚

Phase 1 binding table 仅供内部 differential bridge 使用。公共 `Parameter`、CircuitIR JSON、
Module、checkpoint 和训练 API 均不改变。

## 验收

- Parameter/Expression 结构和依赖精确保留；
- scalar tensor 的 dtype/shape 规则明确；
- requires-grad tensor forward/gradient 与 legacy path 一致；
- late-bound 数值变化不改变 internal program identity；
- static constant 变化会改变 internal program identity；
- bind 缺失/多余/shape/dtype 错误 fail closed；
- [x] API owner 批准架构决策；
- [ ] compiler owner 在实现评审中确认 canonical parameter encoding；
- [ ] training owner 确认 binding table 与 autograd graph 保持方式。
