# FlagQuantum 首次公开 Alpha API 冻结审批包

## 审批结果

API owner 已于 2026-09-01 明确批准 Proposal 003–007 的候选契约冻结，并确认
Proposal 002 保持冻结。**本批准不等于整体首次公开 Alpha freeze。**

Proposal 002（`ExecutionOptions`）已经冻结，本轮只确认不重新打开。整体 freeze 仍需
当前提交通过托管 CI、核验仓库保护设置，并完成至少一轮外部 alpha 用户试用。

机器可读审批记录为 `contracts/alpha-api-freeze-review-packet.json`。其中
`reviewed_sha256` 保存获批前的精确内容，`sha256` 绑定只改变冻结状态后的现行契约；
任何后续 hash 变化都必须重新审查，不能沿用本次批准。

## 逐项签审

| Proposal | 冻结内容 | 审批结果 | 不包含 |
| --- | --- | --- | --- |
| 002 | `ExecutionOptions` 字段、默认值、优先级与序列化 | 已确认 | Plan、Result |
| 003 | `ExecutionPlan` 身份、序列化、恢复与精确执行 | 已批准 | 云提交资产、凭证、二进制 artifact |
| 004 | Result accessor、measurement 唯一来源、`noise_model` | 已批准 | native result 稳定性、tensor JSON |
| 005 | Module forward/execute、train、checkpoint 责任边界 | 已批准 | 分布式训练稳定承诺、自动 resume |
| 006 | 统一异常类别、Module 构造和 deployment binding 所有权 | 已批准 | backend 偶然异常类型 |
| 007 | 扩展 manifest、协商、生命周期、隔离与 conformance | 已批准 | 具体第三方插件稳定性或认证 |

## 批准前必须理解的边界

1. `fq.plan(...)` 返回的 `ExecutionPlan` 是本地可检查、可恢复、可精确执行的计划；它
   不是 `DeploymentPackage`，不承担 provider job、凭证或签名。
2. 执行入口继续使用 `noise_model=`，不会在首次公开后无理由改名为 `noise=`。
3. `Module.forward()` 返回 Tensor，`Module.execute()` 返回 `ExecutionResult`；
   `fq.run()` 不接受 Module。
4. PyTorch 是稳定训练主接口；JAX 是可选 compiled backend，不获得同等根级语义承诺。
5. `flagquantum.extensions` 的协议可以冻结，但插件实现默认仍是 experimental。
6. distributed、provider hardware、性能和扩展认证必须依赖各自证据，不能由 API freeze
   推导出来。

## 已完成证据

- Stable Core：22 个名称，预算上限 25；
- default 941、runtime 168、distributed CPU 376、benchmark/release 113 项通过；
- Qiskit 2.0.3/2.5.2、Aer 0.17.2、PennyLane 0.44.1/0.45.1 窗口上下界各
  51 项通过；
- 五条核心黄金路径及真实 Qiskit 第六路径通过；
- Ruff、Black、两组 mypy、文档 source-of-truth、API snapshot 通过；
- 第三方 backend 示例只导入公开扩展命名空间。

详细证据见 `docs/development/ALPHA_API_FREEZE_AUDIT.md`。

## 本次审批语义

本次明确回复“approve”承接此前唯一待决动作，采用以下审批语义：

> 我批准 review packet 所绑定 SHA-256 的 Proposal 003、004、005、006、007 候选契约
> 作为 FlagQuantum 首次公开 Alpha 的兼容性基线；确认 Proposal 002 保持冻结。本批准
> 不等于整体 Alpha freeze，不覆盖 experimental、第三方插件实现、provider hardware、
> 分布式规模或性能认证。

本审批没有扩展到其他 Proposal 或整体 freeze。

## 整体 freeze 的后续批准

只有剩余 release gates 完成后，才能另行使用以下语句：

> 我批准 FlagQuantum first-public-alpha API freeze，并授权将已批准契约和公开 API
> candidate 的冻结字段更新为 true。

没有这句独立授权，任何自动化或开发代理都不得宣布整体 API 已冻结。
