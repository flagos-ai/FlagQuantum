# Stage 2 Quafu 最小真实纵向闭环准入审计

状态：**Technical baseline complete — P0 implementation not authorized**
日期：2026-09-02
上位契约：[Proposal 012](API_CHANGE_PROPOSAL_012_QUAFU_PROVIDER_CONTRACT.md)

## 1. 已具备

- `QuafuProvider` 已实现 token 验证、backend discovery、chip-info、submit、status、cancel、
  result 和有界轮询；
- 支持标准 `DeploymentPackage` 和不允许 provider 二次编译的 sealed physical QASM；
- submission receipt、routing evidence 和 deployment artifact identity 能贯穿到结果；
- 结果执行 shot accounting，已有默认保留 portal bit order 和显式 legacy reverse 选项；
- Mock 测试覆盖 token、提交、轮询、校准读取、物理 QASM 和基本结果；
- density-matrix probability 与 readout noise 已在核心 Runtime 修复并受合同测试保护。

这些证据证明“核心链路零件存在”，不证明真实 QPU、门户兼容或生产可靠性。

## 2. 尚未闭环

| Proposal 012 P0 项 | 当前状态 | 缺口 |
| --- | --- | --- |
| 不对称位序合同 | 部分完成 | 机器 candidate 已冻结示例，但 Quafu 结果 metadata 未强制声明 bit order |
| 移除 adapter 的 `QPUTwin` 依赖 | 外部待验 | 当前仓无 adapter 源码，需真实安装环境证据 |
| 最小结果 metadata | 未完成 | 未强制 backend、duration、calibration/noise identity、physical qubits、program digest |
| 提交与终态分离 | 部分完成 | 异步轮询存在，但通用状态仍允许自由字符串 |
| 五态归一 | 未完成 | `Queued` 等状态尚未稳定映射为 `Pending`，原始状态也未统一保留 |
| 错误分类 | 未完成 | Failed/Cancelled 和平台错误仍主要表现为通用 `RuntimeError` |
| 幂等提交 | 未完成 | 没有 idempotency key 或重复提交保护 |
| capability preflight | 未完成 | shots 范围、QASM profile、basis/topology 的门户事实未统一验证 |
| 真实端到端 | 未完成 | 无真实注册 backend ID 的 submit/poll/result 证据 |

## 3. P0 建议实施边界

首次实现只允许：

- 内部状态归一函数及原始状态保留；
- Provider 边界的 counts bit-order 显式 metadata；
- 不改变稳定 dataclass 字段的最小 metadata 校验；
- idempotency key 的 provider-private transport 支持；
- timeout、Failed、Cancelled、空结果、不对称 counts 和 shot mismatch 测试；
- adapter 真实安装合同脚本或 schema，不复制 FlagQuantum 源码；
- 凭证泄漏负向测试。

以下仍需独立 API change approval：新增稳定异常层级、修改 dataclass 字段、改变根级导出、
冻结通用 Provider API 或把 provider metadata 提升为公共 schema。

## 4. 真实证据门

Stage 2 不得仅凭 Mock 退出。至少需要一条受控真实记录证明：

```text
registered external_backend_id
  -> sealed deployment artifact
  -> provider_job_id
  -> Pending/Running/terminal status
  -> normalized counts + shot accounting
  -> actual backend/calibration/program provenance
```

真实记录必须脱敏，不保存 token、用户信息或门户私有响应；backend ID 可以作为部署绑定，
但不得进入程序 IR 或 program identity。

## 5. 审批暂停点

Proposal 012 仍为 Draft。普通 `do` 不授权改变 Provider 的受保护行为。P0 实现前应生成并
批准独立机器 candidate；建议口令：

```text
approve QUAFU-PROVIDER-P0
```

该口令不授权公共 Provider API 冻结、真实付费任务提交或 adapter 仓写入。
