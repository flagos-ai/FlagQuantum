# API Change Proposal 016：统一 TargetCapabilities 精度语义

## 状态

**Approved before the first public alpha.** API owner 于 2026-09-06 明确授权
收口精度控制机制。仓库尚未公开发布，因此不保留旧值兼容或迁移层。

## 问题

TargetCapabilities 的精度字段结构已经分离，但字符串值曾混用标量 dtype 和
复数量子态 dtype。例如 `precision.effective_dtype` 同时出现过 `float64` 和
`complex128`，导致 Runtime 无法判断两者是同义表达还是不同精度层级。

## 决策

- `native_dtype`、`storage_dtype`、`parameter_dtype`、`accumulator_dtype` 只接受
  当前实际支持的标量类型 `float32`、`float64`；
- `effective_dtype` 只接受逻辑复数量子态类型 `complex64`、`complex128`；
- `software_mechanism` 保持独立机制字段，不作为 dtype 别名；
- `software_mechanism=none` 时，native、storage 和 effective 必须构成一致的
  `float32/complex64` 或 `float64/complex128` 路径；
- 软件扩展路径仍须由 Runtime 校验证据等级和适用 scope。

## 兼容性

字段集合和 TargetCapabilities schema 版本不变。过去含混的精度值尚未形成公开
兼容承诺，直接拒绝，不增加别名、自动转换或 fallback。

## 验收

- Core 在构造 requirement 和 fact 时拒绝错误的 dtype 类别；
- Core 在构造 snapshot 时拒绝无软件机制却自相矛盾的精度路径；
- Double-Single 仍可表达为 native/storage `float32`、effective `complex128`；
- Core、Platform、Runtime、序列化及 CPU 纵向链路测试通过。
