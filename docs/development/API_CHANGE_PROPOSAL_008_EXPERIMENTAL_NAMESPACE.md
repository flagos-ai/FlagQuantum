# API Change Proposal 008：Experimental 命名空间瘦身

## 状态

**Governance baseline frozen — 治理基线已批准并冻结。**

- 目标版本：首次公开 alpha；
- 影响范围：仅 `flagquantum.experimental` 的公开路由；
- 稳定根 API 变化：无；
- 功能实现删除：无；
- 机器可读契约：`contracts/experimental-namespace-v1-candidate.json`；
- 实施授权：API owner 于 2026-09-01 通过明确指令要求执行 experimental 瘦身；
- API owner 于 2026-09-01 通过 `approve 008-010` 明确批准本治理基线；
- 本批准不冻结任何二级 experimental 功能，也不改变 Proposal 002–007 的冻结状态。

## 问题

原 `fq.experimental` 同时平铺 99 个名称，混合了分布式底层记录、动态线路、第三方
互操作、TEBD 和多代数值实验。用户无法从自动补全判断能力归属，维护者也无法独立评审、
稳定或淘汰某一领域。

## 决策

顶层只公开八个领域命名空间：

```text
fq.experimental
├── distributed
├── dynamic
├── execution
├── interop
├── mps
├── numerics
├── planning
└── simulation
```

所有功能符号进入对应二级命名空间，不再从 `fq.experimental` 平铺导出。实现模块保留，
本次只改变公开路由和仓库内调用，不重新设计执行或训练语义。

典型迁移：

```python
fq.experimental.dynamic.run_dynamic(...)
fq.experimental.distributed.train_distributed_statevector(...)
fq.experimental.interop.qiskit.from_qiskit(...)
fq.experimental.numerics.run_double_single_conformance(...)
fq.experimental.simulation.run_tebd(...)
```

## 生命周期

Experimental 不提供兼容性保证，但不能无限期停留：

1. 新入口必须有 owner、范围、测试和退出条件；
2. 最迟在两个 minor 版本或六个月内评审一次；
3. 十二个月内必须稳定、明确续期、重构或删除；
4. 进入稳定 API 必须另立提案，通过语义、错误、互操作、文档和真实环境门禁；
5. 本次分类不代表其中任何功能已经具备稳定资格。

## 验收标准

- [x] `fq.experimental.__all__` 只包含八个领域命名空间；
- [x] 原有实现均可从明确的二级命名空间访问；
- [x] 仓库测试、示例、工具和文档不再使用扁平实验入口；
- [x] 稳定根 `fq.__all__` 不变；
- [x] 机器契约逐项锁定当前分类并声明不兼容承诺；
- [x] API owner 批准该分类作为首次公开 alpha 的 experimental 治理基线。

## 下一阶段：稳定化漏斗

瘦身后按“用户价值、语义完整、后端一致、维护责任、证据成熟度”逐项评审。优先候选是
已有清晰用户入口和跨版本验证的能力；开发证据类型、rank/shard 记录和硬件专属原语默认
不进入稳定层。任何晋升都单独形成提案，不能因为位于二级命名空间而自动稳定。
