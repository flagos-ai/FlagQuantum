# FlagQuantum vNext Phase 1 集成报告

> 状态：Core 契约对账、架构 ADR 收口及 Compiler 性能修复已审查、合并并完成统一验证
>
> 集成分支：`codex/flagquantum-vnext-architecture`
>
> Phase 0 基线：`99d5a92091bff35fdc573f4b401e3ebaaf5ffcbf`
>
> Phase 1 集成提交：`11c473cccbf9ef65b4a4743dd0921d582066c361`
>
> 集成完成时间：2026-09-03（Asia/Shanghai）

## 1. 本轮结论

Phase 1 完成了三个相互依赖的最小切片：

1. 对账 Core `ProgramArtifact` 的现行事实与适用边界；
2. 将长期架构中的证据分级、迁移经济性、Simulation/Runtime 协商、Agent/LLM 边界和验证路径固化为 Proposed ADR；
3. 在不改性能预算、统计方法、公共 API 和默认编译语义的前提下，修复 `CircuitIR` 重复导入与校验的性能门禁。

本轮没有创建第二套 Artifact envelope，也没有将 Proposed ADR 误作已实施契约。`ProgramArtifact` v1 仍是唯一 envelope 权威，但当前只覆盖 envelope 行为和已经运行的 `kind=circuit` 路径，不能无损替代 Compiler sealed artifact 或 Deployment package。

## 2. 审查与合并记录

| 顺序 | 切片 | 团队提交 | 集成合并提交 | 结论 |
| ---: | --- | --- | --- | --- |
| 1 | Core Artifact 对账 | `c33292fb2b27bfb6dc94cbf58c53f05dd1966043` | `288cbd0b` | 通过 |
| 2 | Phase 1 契约 ADR | `bc129bc668fa78ec4069483b90002d42e16ad291` | `23bb1c32` | 修订后通过 |
| 3 | Compiler 后继授权 | `80d3234c66bbf14f953ffc5ebb87e3807af71740` | `a7a9e8e6` | 通过 |
| 4 | Compiler 性能实现 | `cba6bad207d49fdc43acc6640e7cd67c0919c314` | `11c473cc` | 通过 |

初始 Compiler 实现因历史 SHA-256 契约约束未直接放行。Integration 先建立非追溯 successor 授权，再由 Compiler 提交精确绑定新 importer、授权、预算和测试证据的 successor candidate。旧 Phase 1 与 Deployment Bridge 记录保持不可变。

## 3. 关键决策

### Artifact 与 metadata

- `ProgramArtifact` v1 是唯一 artifact envelope 权威；未来 v2 只能沿同一契约谱系演进。
- v1 `content_hash` 是完整 envelope identity；producer、required capabilities、有序 parents、payload 和 metadata 均参与计算。
- metadata 当前值域未闭合，不能视为安全边界；闭合值代数、命名空间和容量限制仍需后续 API Change Proposal。

### 长期架构吸收项

- 证据等级采用 `basic < observable < certification`，并与支持状态、事实暴露状态正交。
- 迁移允许 `migrate`、`adapt`、`freeze_legacy`、`retire` 四种经济性处置，不强制所有遗留实现立即删除。
- Simulation 可以向 Runtime 提供类型化资源候选和成本建议；Runtime 保留资源选择与调度权。
- Agent Services 保持确定、协议无关；LLM/Reasoning 作为可选外部能力，不进入 Core/Compiler/Runtime 稳定契约。
- 验证分为 fast、standard、certification 路径，并通过机器规则追踪文档约束。

### Compiler 性能修复

- 只缓存已完整导入并通过 verifier 的成功结果；任何失败均不缓存。
- 缓存为进程内、有界、弱引用、并发保护结构，不是持久化编译缓存。
- 完整嵌套内容变化、原地 tensor 变化或同值不同可训练 tensor 身份都会失效，并重新导入、校验和绑定 autograd owner。
- 公共 API、默认路径、输入接受域、诊断、identity、round-trip、预算与统计口径均未改变。

## 4. 性能结果

标准 Linux CPU 容器中，重复导入五轮的 10,000 gate p95 从基线约 `98.304–131.938 ms` 降至 `16.310–17.444 ms`，完整门禁通过率由 `4/5` 提升到 `5/5`。

总控独立逐规模复核的 p95 约为：

| Gate 数 | p95 |
| ---: | ---: |
| 10 | 0.041 ms |
| 100 | 0.186 ms |
| 1,000 | 1.680 ms |
| 10,000 | 17.718 ms |

所有延迟、峰值主机内存、身份确定性和归一化增长判据均通过原预算。完整默认套件曾出现一次该历史性能测试的时序波动；未调整阈值，逐规模复核与随后完整复跑均通过，作为后续稳定性观察项保留。

## 5. 统一验证

- 架构边界检查：通过；
- 合并后 Core、ADR、Compiler、Agent 及导入/往返/默认路径联合测试：`121 passed`；
- Linux Docker `pr-runtime`：`175 passed, 33 skipped`；
- Linux Docker `pr-default` 最终复跑：`1867 passed, 12 skipped`；
- Compiler 团队定向语义与授权链：`110 passed`；
- Compiler 团队完整 `tests/internal_ir`：`867 passed`，大套件中一个独立 Stage 4 性能项曾波动失败，单独复核 `2 passed`。

跳过项均为当前容器缺少特定可选依赖或加速器环境，不被解释为硬件、分布式或真实 QPU 验证。

## 6. 下一轮入口

1. 收敛最小 `TargetCapabilities`，分开 requirements、discovery facts、support status 和 execution evidence；
2. 定义最小 Execution Request 与 Result/Evidence 契约；
3. 在契约稳定后批准 Platform Provider 与 Execution Provider 方法集；
4. Runtime 移除第一条纯类型 Runtime→Compiler 依赖；
5. Simulation 抽取首个可替换单设备 statevector engine；
6. 持续观察 Phase 1 与 Stage 4 历史性能门禁在共享 CPU 环境中的稳定性，但不得通过放宽预算或改变统计方法消除波动。

每个后续切片仍须遵守：单一责任团队、最小差异、先契约与特征测试、团队范围门禁、架构门禁、总控逐项合并和合并后统一复测。
