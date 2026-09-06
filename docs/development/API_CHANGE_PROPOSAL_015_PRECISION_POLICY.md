# API Change Proposal 015：收缩未实现的精度控制

## 状态

**Approved for the first public alpha.** API owner 于 2026-09-06 明确授权
在公开发布前直接收缩 `flagquantum.training.PrecisionPolicy`，不保留兼容参数或
checkpoint 迁移层。

## 问题

`PrecisionPolicy` 曾公开 `mode` 和 `accumulator_dtype`，但 Runtime、statevector、
MPS、张量网络及训练归约均不消费这两个值。它们只参与校验和 checkpoint
序列化，因而会让用户误以为混合精度累加已经生效。

## 决策

`PrecisionPolicy` 仅保留实际生效的字段：

```text
complex_dtype, parameter_dtype, allow_parameter_downcast, atol, rtol
```

仅接受 `complex64/float32` 和 `complex128/float64` 两组一致精度。删除 `mode`
和 `accumulator_dtype`，不接受旧关键字，也不增加弃用包装。设备或数值内核的
累加精度仍由 TargetCapabilities 与执行证据表达，不等同于用户控制项。

## 兼容性

这是首次公开 alpha 前的直接收敛。使用已删除关键字的源码和内部开发阶段
checkpoint 不再兼容；仓库尚无已发布用户或承诺的迁移窗口。未来只有在至少
一个执行后端真正实现、结果证据可验证且跨后端语义明确后，才可通过新的 API
提案引入混合精度控制。

## 验收

- 精确检查 `PrecisionPolicy` 字段集合；
- 拒绝不一致的复数/实数精度组合；
- checkpoint 保存和恢复保留有效精度策略；
- Module、CPU 纵向链路及公共 API 契约测试通过。
