# FlagQuantum 首次公开 Alpha API 冻结审计

审计日期：2026-09-01  
结论：**代码与候选契约已接近冻结就绪，但尚不能宣布正式 freeze。**

## 已具备

| 项目 | 证据 | 结论 |
| --- | --- | --- |
| Stable Core | `docs/public_api_v1.json`，22 个根级名称，预算上限 25 | 就绪 |
| 执行配置 | Proposal 002，唯一 `ExecutionOptions` | 就绪 |
| 可执行计划 | Proposal 003，plan identity、序列化、stale check、`fq.run(plan)` | 就绪 |
| Result/measurement/noise | Proposal 004，唯一 measurement 来源、显式 result accessor | 就绪 |
| Module/training | Proposal 005，PyTorch 语义、checkpoint/resume 边界 | 就绪 |
| 异常与 Module 所有权 | Proposal 006，统一异常类别，部署 binding 移出 Module | 就绪 |
| 扩展协议 | Proposal 007，版本协商、隔离、conformance、外部示例 | 候选就绪 |
| 黄金路径 | `tests/api_contract/test_open_source_golden_paths.py` | 本地 CPU 路径就绪 |
| 根级防漂移 | manifest、snapshot、API proposal contract、CI gate | 就绪 |

## 本轮明确决策

- 保留 `noise_model`：语义准确，已经贯穿 plan/run/序列化，改名没有足够收益；
- 保持 `ExecutionPlan` 与 `DeploymentPackage` 分离：前者负责本地精确执行，后者负责
  provider-facing 提交资产；
- 稳定 `flagquantum.extensions` 的协议层，不承诺每个第三方扩展实现稳定；
- 不增加根级 API，扩展和部署能力继续分层；
- 候选实现完成不等于正式冻结，必须由 API owner 逐项批准。

## 冻结前仍需完成

1. API owner 分别审批 Proposal 002–007 的 contract freeze，并记录版本与日期；
2. 在安装真实 Qiskit、PennyLane 的受控 CI matrix 中通过 conformance；本地无依赖的
   adapter contract 通过不能代替真实依赖证据；
3. 完成至少一轮目标用户 alpha/beta 试用，确认没有只能通过破坏 API 解决的问题；
4. 核对发布平台上的 required checks、CODEOWNERS 和 branch protection 已实际启用；
5. 冻结当日重新运行 default、runtime、distributed、benchmark/release、typing、格式、
   文档和 API snapshot 全部门禁；
6. 冻结批准后才将各 contract 的 `candidate_is_frozen_contract` 改为 `true`，不得由生成
   工具自动改写。

## 冻结判定

当前状态应表述为“**first-public-alpha API candidate ready for owner review**”，不能表述为
“API 已永久固定”。首次公开 alpha 后，稳定入口只能兼容性增加；任何破坏性变更都必须
有独立 proposal、迁移路径、弃用窗口和 API owner 批准。

## 本分支验证记录

2026-09-01 在 CPU Docker 环境完成：

- 聚焦扩展、计划、候选清单与快照契约：32 passed；
- PR default：941 passed，10 skipped；
- PR runtime：168 passed，31 skipped；
- distributed CPU：376 passed，26 skipped；
- benchmark/release contract：113 passed，10 skipped；
- 五条核心黄金路径通过；Qiskit 可选路径因该镜像未安装 Qiskit 而跳过；
- Ruff、Black、两组 mypy、documentation source-of-truth 与 API snapshot 通过。

这些数字是本地候选回归证据，不替代多 Python 版本 CI、真实第三方 SDK matrix、硬件
认证或 API owner 冻结审批。
