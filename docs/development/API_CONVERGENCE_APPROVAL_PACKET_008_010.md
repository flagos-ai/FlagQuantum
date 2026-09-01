# FlagQuantum API 收敛审批包：Proposal 008–010

## 当前状态

**Approved by API owner on 2026-09-01 — 三项审批已经记录。**

本审批包只覆盖：

- Proposal 008：Experimental 顶层治理结构；
- Proposal 009：框架无关 Interop 协议；
- Proposal 010：DynamicCircuit 构建契约。

机器记录为 `contracts/api-convergence-review-packet-008-010.json`。其中 SHA-256
同时保留审阅前 SHA-256 和审批后 SHA-256，绑定审阅对象与最终冻结记录。后续任何受保护
内容变化都必须重新提案并取得 API owner 批准。

审批指令为 `approve 008-010`。Proposal 008 只冻结治理基线；Proposal 009、010 冻结
候选稳定契约。本审批不构成整体 first-public-alpha freeze。

## 逐项建议

| Proposal | 建议 | 批准后固定 | 明确不固定 |
| --- | --- | --- | --- |
| 008 | 批准治理基线 | 八个顶层领域、禁止扁平功能入口、实验生命周期 | 二级 experimental 符号、签名和实现 |
| 009 | 批准候选稳定契约 | `flagquantum.interop` 协议、签名、字段、错误和 conformance | Qiskit/PennyLane 实现、默认 registry 内容 |
| 010 | 批准候选稳定契约 | `flagquantum.dynamic.DynamicCircuit` 构造、IR 和错误语义 | `run_dynamic`、原生结果、路由、部署和 provider |

## 为什么 Proposal 008 不是普通 API freeze

Experimental 必须允许演进。Proposal 008 的批准只固定顶层信息架构和治理规则，不给
二级功能任何兼容性承诺。以后可以新增、重构或删除实验符号，但不能重新把几十个功能
平铺回 `fq.experimental`，也不能让实验功能无限期无 owner、无复审地滞留。

## 为什么 Proposal 009 可以稳定协议但不稳定适配器

稳定协议让第三方可以围绕 CircuitIR、conversion report、registry 和 conformance 开发；
具体 Qiskit/PennyLane 实现仍受外部依赖版本、语义覆盖和维护证据影响。协议 freeze 不会
把某个 framework 变成 FlagQuantum runtime 依赖，也不承诺任意未来版本兼容。

## 为什么 Proposal 010 不包含 DynamicExecutionResult

本地 trajectory、Qiskit Aer 和未来硬件 provider 对 final state、mid-circuit data、
statistics 与 provenance 的可用性不同。当前原生结果不适合永久固定。稳定执行结果继续
使用 `fq.ExecutionResult`；原生动态结果仅作为 experimental 适配对象。

## 已验证证据

- Proposal 008：default 942、runtime 168、distributed 376、release 113；
- Proposal 009：default 952；两个 Qiskit/PennyLane 版本窗口各 45；
- Proposal 010：default 958、runtime 168、Qiskit Aer dynamic 14；
- Ruff、format、文档 source-of-truth 与候选契约测试通过；
- 既有 Proposal 002–007 审批包和签署的 public candidate 哈希保持不变。

## 审批边界

未获批准且仍保持 experimental 的内容包括：二级 experimental 符号、Qiskit/PennyLane
适配器实现、`DynamicExecutionResult`、`run_dynamic`、路由、部署、provider，以及硬件、
性能和分布式认证。整体 first-public-alpha freeze 仍需单独审批。
