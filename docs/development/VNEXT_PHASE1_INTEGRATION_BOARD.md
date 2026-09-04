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

## 7. Phase 2 当前状态（2026-09-04）

本节是对 Runtime 后续提交的集成准入约束，不表示 Runtime matching、fallback 或 decision
record 已经成为产品能力。当前审查基线为集成提交 `c30316b9`。

| 顺序 | Phase 2 切片 | 集成提交 | 当前结论 |
| ---: | --- | --- | --- |
| 1 | Core `TargetCapabilities` v1 值对象、canonical identity 与纯 matcher | `d7fad5e4`，含后续 nullable-fact 修正 `2083dd2e`、`35f1adb0` | 已合入；保持内部、无 policy、无 fallback |
| 2 | Compiler `_compiler.TargetCapabilities` 损失记账 adapter | `1a501d8a` | 已合入；未覆盖字段仍须旧 comparator 判定，不得仅凭 Core matcher 放行 |
| 3 | CPU Platform capability snapshot adapter | `c30316b9` | 已合入；只生产独立 CPU snapshot，不发现目标、不选择 backend、不授权 fallback |
| 4 | Runtime TargetCapabilities matching seam | — | 待 Runtime 提交并按下列清单验收；尚未实现或进入默认路径 |
| 5 | 第二个独立 Platform/remote producer 替换 conformance | — | 待 matching seam 通过后实施，不得用 Runtime 私有 snapshot 类型代替 |

ARCH-003/004/005/007 仍为 **Proposed**。现有 Phase 2 机器授权只允许内部
TargetCapabilities v1、窄 adapter 和 Runtime matching seam；它不批准新的公共
Execution Request、Execution Result/Evidence、稳定 plan 字段、默认 backend、fallback 默认值或
失败阶段。

## 8. Runtime TargetCapabilities matching seam 逐项验收清单

### 8.1 输入权威与范围

- [ ] Runtime 直接消费 Core `RequirementSet`、`TargetCapabilitySnapshot` 和纯 matcher 结果；不复制、
  re-export、子类化或私建平行 capability 契约。
- [ ] Compiler 投影的 `requires_legacy_comparator` 和每条 loss 均被保留；只要该标记为真，旧
  `_compiler` comparator 未通过就不得选择候选。
- [ ] Runtime policy 只消费 requirement、snapshot、受控资源信息和显式授权；不得用用户期望、
  `world_size`、环境变量、Simulation 估算或 backend registry 布尔值补写平台事实。
- [ ] v1 延后域和未知 extension 没有登记 matcher handler 时失败关闭；不得为了接入动态、拓扑、
  通信、checkpoint 或 realtime 而扩张本切片的公共契约。

### 8.2 Mandatory fail-closed 与 preference 排序

- [ ] 每个候选先由 Core matcher 完整验证 identity、scope、freshness、证据引用和所有 mandatory
  谓词；missing、stale、scope/identity mismatch、`unknown`、`unmeasured`、`unsupported`、
  `not_exposed`、不被接受的 exposure、证据不足和未知 extension 均使该候选不可执行。
- [ ] Runtime policy 不得覆盖 mandatory blocker，不得以成本更低、目标优先级更高或仅有一个候选
  为由放行。
- [ ] preference 只对已经满足全部 mandatory 的候选排序；未满足 preference 只降低排序，不生成
  fallback 授权，也不能补偿任何 mandatory 失败。
- [ ] 若所有候选均失败，返回聚合后的 typed blockers；不得选择“最接近”的候选或静默进入现有
  默认 backend。

### 8.3 候选顺序确定性

- [ ] 同一 `RequirementSet`、同一组 snapshot、同一 policy 输入和同一评估时间产生相同的候选顺序、
  选择结果和 decision identity。
- [ ] 排序键使用版本化、显式的确定性字段：先可执行性，再 preference 满足度，再受控 policy
  priority，最后以稳定 target/snapshot identity 打破平局；禁止依赖 dict/set 顺序、并发完成顺序、
  discovery timing、对象地址或进程 hash seed。
- [ ] 输入 snapshot 顺序置换、重复发现去重和相同分数平局均有负向/变形测试；重复 identity 但语义
  不同必须失败，而不是后写覆盖。

### 8.4 CPU 独立候选与完整重匹配

- [ ] 显式请求 CPU 时，CPU 作为普通独立候选参与匹配；从非 CPU 候选转向 CPU 时，只有
  `fallback_authorizations.cpu=true` 才允许创建该 fallback 候选。
- [ ] CPU 候选使用自己的 target identity、scope、freshness、facts、evidence refs 和 blockers，
  对所有仍适用 mandatory 谓词以及必需的旧 Compiler comparator 做完整重匹配。
- [ ] 不复用原候选的 memory、precision、device count、route、evidence、租约或 matcher 成功；CPU
  memory/precision 为 nullable unknown 时，对相应 mandatory requirement 必须失败关闭。
- [ ] 设备解析失败、候选为空、probe 不可用或 CPU 在本机可用，都不能把 CPU 变成隐式 resolver
  默认值。

### 8.5 Fallback 分轴授权

- [ ] `backend`、`device`、`cpu`、`precision`、`algorithm`、`approximation` 六轴分别读取明确授权，
  缺省均为 false；一个轴的授权不得推出另一个轴。
- [ ] 每次 fallback 都创建具有自身 snapshot/约束的替代候选并完整重匹配；不得原地修改已失败候选、
  删除 mandatory requirement 或把 preference 改成授权。
- [ ] Runtime policy、provider 声明和历史宽泛 `allow_backend_fallback` 不得推导 CPU、precision、
  algorithm 或 approximation 授权。
- [ ] 实际 fallback 必须在内部 decision/attempt evidence 中可见并降低不再成立的 claim ceiling；若
  需要改变稳定 plan/result 字段、identity 或异常行为，必须停止并提交独立 API Change Proposal。

### 8.6 Typed blocker 与 decision identity

- [ ] blocker 保留 Core typed code、message、capability name 及来源上下文；Runtime 聚合使用稳定排序
  和确定性去重，不将 blocker 降为自由字符串、布尔失败或吞入日志。
- [ ] 每个被评估候选保留其 snapshot identity 与 blockers；被拒候选的失败证据不得被最终成功候选
  覆盖，preference miss 也不得误报为 fatal blocker。
- [ ] 内部 decision record 至少绑定 `requirement_set_id`、有序候选 snapshot identities、选中
  snapshot/target identity、评估时间、policy/逐轴 fallback 授权 identity、matcher blockers、排序键
  版本及实际采用的 fallback 轴。
- [ ] decision identity 由上述语义字段 canonical 计算；显示文本、运行时对象、凭据、vendor handle
  和无序容器不得参与，任何 requirement、snapshot、policy 授权或候选顺序变化都必须改变 identity。
- [ ] 本切片的 decision record 只能是内部 seam；ARCH-004/005 获批前，不得声称它是公共
  Execution Request/Result/Evidence 契约，也不得修改现有 plan/result identity。

### 8.7 公共 API、默认路径与分层保护

- [ ] `flagquantum` 根导出、稳定签名/default/Literal、序列化 schema、异常类别和失败阶段完全不变。
- [ ] 现有 `fq.run`、`fq.plan`、默认 backend/device 选择及 local CPU/single-device fast path 行为保持
  golden/兼容测试不变；新 seam 在单独批准前不接管默认路径。
- [ ] Runtime 不重写 Simulation 的算法约束或成本模型，不把估算当可用容量；Simulation 也不读取
  环境或选择物理 target。分层是所有权边界，不是阻断类型化信息交换。
- [ ] Runtime/Core/Compiler/Platform 的生产文件若超出各自 owner 或触及受保护 surface，提交必须拆回
  对应团队或先走 Integration/API proposal，不得把跨层改动夹带在 Runtime 切片中。

### 8.8 Runtime 提交必须提供的测试证据

- [ ] mandatory 状态、exposure、evidence、stale、scope、identity 与 unknown-extension 的参数化负向
  测试，证明 policy 无法越过 Core matcher。
- [ ] preference 仅排序、全候选失败、平局及输入置换/hash-seed 下候选顺序和 decision identity
  确定性测试。
- [ ] CPU 独立 full-rematch 的正向与负向测试，至少覆盖 device count、memory、native/effective
  precision、CPU 未授权、只有 backend 授权以及 unavailable CPU。
- [ ] 六个 fallback 轴的默认禁止和两两隔离测试；每个获授权 fallback 仍须证明替代候选全量匹配。
- [ ] Compiler loss-accounting/legacy comparator 共判测试，以及替换 CPU producer 为第二个 contract
  fake 后 Runtime consumer 无需修改的 conformance 测试。
- [ ] Stable API snapshot、`fq.run/plan` 默认路径和 local fast path 回归测试；任何 accelerator、QPU、
  多节点或 scalability 表述都必须保持未验证，不能由本切片测试提升。

## 9. 当前规范缺口与准入裁决

1. ARCH-004/005 只提出未来 request/decision/result/evidence 边界，现行授权没有批准稳定 decision
   record schema。Runtime 可以提交最小内部 decision record 来证明 identity 绑定，但不得写入或替换
   稳定 plan/result；若实现需要这种变化，当前切片必须暂停并转为 API Change Proposal。
2. Compiler adapter 对 gate parameter domains 与 ancilla policy 等字段仍是有损投影，并明确要求旧
   comparator。因此 Runtime 不能把 Core matcher 的 `executable=true` 等同于完整 Compiler 合法性。
3. 当前只有 CPU Platform producer 已通过实现验证，尚不满足“两个独立 producer 可替换而消费者不
   修改”的架构完成条件。第二 producer 是 Runtime seam 通过后的准入项，不得提前宣称边界已完成。
4. dynamic、checkpoint、realtime、完整 topology/communication、gradient/optimizer distribution
   仍在 v1 延后域；本清单只规定失败关闭和 extension 门槛，不授权这些能力或其发布声明。
