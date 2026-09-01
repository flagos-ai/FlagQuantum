# API Change Proposal 007：扩展协议与 Alpha 冻结就绪

## 状态

**Extension contract frozen — 扩展协议契约已获单独冻结批准。**

- 目标版本：首次公开 alpha；
- 影响接口：`flagquantum.extensions`；
- 根级名称变化：无；
- 机器可读候选：`contracts/extension-protocol-v1-candidate.json`；
- 授权记录：API owner 于 2026-09-01 通过明确用户指令授权推进本轮实现；
- 本授权不等于 Proposal 001–007 或整个公开 API 已正式冻结。
- 冻结记录：API owner 于 2026-09-01 明确批准 review packet 所绑定的 Proposal 007
  精确契约；具体插件实现和整体首次公开 Alpha freeze 仍未获批准。

## 决策

### 1. 稳定扩展边界，不稳定第三方实现

`flagquantum.extensions` 作为独立的候选稳定命名空间，承诺 manifest、能力协商、
生命周期、任务局部注册、异常隔离和 conformance 入口。它不向 `flagquantum` 根级增加
名称。具体 backend、provider 或 compiler pass 默认仍为 experimental，只有单独通过
兼容性、数值、安全和维护者审查后才能获得认证。

扩展作者只需导入：

```python
from flagquantum.extensions import (
    CapabilityRequest,
    CapabilityResponse,
    ExtensionConfig,
    ExtensionManifest,
)
```

参考 backend 和 provider 位于 `examples/extensions/reference_extensions.py`。契约测试以
AST 检查它没有导入 FlagQuantum internal/runtime/core 模块，并实际运行 backend
conformance。

### 2. 保留 `noise_model` 名称

`noise_model` 已同时进入 `fq.plan`、`fq.run`、Circuit 便捷方法、ExecutionPlan 扩展
序列化、文档和测试。此名称准确表示传入的是模型对象，而不是噪声强度或一次噪声事件。
在首次公开前改成 `noise` 只会增加迁移和歧义，没有足够收益，因此候选 API 保留
`noise_model`。

### 3. ExecutionPlan 与 DeploymentPackage 继续分离

- `ExecutionPlan` 所有本地规划身份、能力约束、缓存/恢复和精确执行；
- `DeploymentPackage` 所有 provider 目标、可移植提交资产、路由证据和提交生命周期；
- plan 不暴露 `submit`、provider 凭证或 job identity；
- deployment package 不是本地 planner 的替代返回值。

二者解决的是连续但不同的生命周期。合并会同时污染本地执行的最小契约和远程部署的
安全边界。

## 候选扩展协议

协议版本为 `1.0`。扩展必须：

1. 通过 `ExtensionManifest` 声明身份、kind、API 版本和能力；
2. 在激活前完成 capability negotiation；
3. 以 `start`/`close` 明确管理生命周期；
4. 通过不可变、任务局部的 registry 安装，不修改根 API 或全局 operator 表；
5. 不把凭证放入可序列化 `ExtensionConfig`；
6. 由 boundary 包装扩展异常并保留原始 `__cause__`；compatibility 错误进入
   `CapabilityError`，lifecycle 错误进入 `ExecutionError`；
7. 使用公开 conformance 入口验证 backend/provider 行为。

## 可执行验收标准

- [x] 扩展协议有独立机器可读候选契约；
- [x] `flagquantum.extensions.__all__` 与候选契约完全一致；
- [x] 扩展名称不进入稳定根命名空间；
- [x] 第三方 backend 示例只导入公开扩展命名空间；
- [x] capability、dtype/device、gradient、异常和 cleanup conformance 通过；
- [x] `noise_model` 命名决策记录为候选稳定决策；
- [x] ExecutionPlan/DeploymentPackage 所有权由测试和文档保护；
- [x] API owner 单独批准 extension contract freeze；
- [ ] API owner 单独批准首次公开 alpha 的整体 freeze。

## 兼容策略

冻结后，SDK `1.x` 只允许兼容性增加。破坏 protocol 方法、manifest 字段或生命周期的
调整必须提升 SDK API major，提供明确诊断，并保留受支持版本窗口。具体扩展包的版本与
SDK 协议版本分开演进，不能用插件版本代替协议协商。
