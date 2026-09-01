# API Change Proposal 011：Experimental 可发现面二次瘦身

## 状态

**Implemented, pending API-owner review — 已实现，尚未冻结。**

- 目标版本：首次公开 alpha；
- 机器契约：`contracts/experimental-surface-v2-candidate.json`；
- 实施授权：API owner 于 2026-09-01 明确要求继续瘦身 experimental；
- Proposal 008 已冻结的八个顶层领域及生命周期规则保持不变；
- 稳定 API、Proposal 009 Interop 协议和 Proposal 010 DynamicCircuit 契约均不变。

## 问题

第一次整理消除了 `fq.experimental` 根下的 99 个平铺名称，但八个二级领域仍合计暴露
114 个功能符号。大量名称是证据记录、rank/shard 状态、底层 executor、kernel 统计、
conformance report 和 P0–P5 研发辅助对象。它们会让自动补全看起来像内部实现索引，而
不是供用户试用的产品入口。

## 决策

Experimental 的可发现 API 只保留任务级工作流和第三方 adapter 命名空间：

| 领域 | 原数量 | 当前数量 | 保留边界 |
| --- | ---: | ---: | --- |
| distributed | 25 | 6 | TN 查询与分布式训练工作流 |
| dynamic | 22 | 6 | 运行、评估、导出、路由与部署 |
| execution | 1 | 1 | advanced execution 入口 |
| interop | 2 | 2 | Qiskit、PennyLane adapter 命名空间 |
| mps | 9 | 2 | production planning 与 workload validation |
| numerics | 50 | 3 | 基础 split real/imag 执行与梯度 |
| planning | 3 | 1 | advanced planning 入口 |
| simulation | 2 | 1 | TEBD 工作流 |

可发现功能面由 114 个降至 22 个，减少 92 个，缩减 80.7%。返回记录仍会随工作流正常
返回，但不要求用户从 experimental 命名空间单独导入其实现类型。

## 迁移原则

被移出 `__all__` 和 `dir()` 的名称不是稳定 API。仓库内部测试、benchmark、证据工具和
实现代码已经改为直接引用所属实现模块，不再反向依赖 experimental 门面。旧内部路径
已经删除，访问任何未进入 v2 契约的名称都会抛出 `AttributeError`。

能力矩阵同步区分“公开实验入口”和“内部开发证据”。P2–P5 数值研究实现与 conformance
runner 继续保留代码、测试和证据，但不再以 Public API 名义出现在生成文档中。

## 验收标准

- [x] 八个顶层领域保持不变；
- [x] 二级可发现面与 v2 机器契约逐项一致；
- [x] 证据、底层状态、结果记录和 conformance 类型不再出现在自动补全；
- [x] stable root 与已批准稳定扩展不变；
- [x] 仓库内部调用全部迁出非公开兼容路径；
- [x] 删除临时内部解析路径；
- [ ] API owner 复核首次公开 alpha 的最终 experimental 可发现面。
