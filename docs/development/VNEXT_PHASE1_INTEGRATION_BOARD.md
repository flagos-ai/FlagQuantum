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

初始 Compiler 实现曾受历史 SHA-256 流程记录约束。当前以实现、场景测试和性能预算作为可执行验证依据，不再把 Deployment Bridge 的阶段审批记录视为代码契约。

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

本节记录 TargetCapabilities Phase 2 内部最小切片的集成结果，不表示 Runtime matching、
fallback、decision record 或 synthetic producer 已经成为公共、默认或生产能力。收口审查基线为
集成提交 `0f63c3b4`。

| 顺序 | Phase 2 切片 | 团队提交 | 集成提交 | 当前结论 |
| ---: | --- | --- | --- | --- |
| 1 | Core `TargetCapabilities` v1 值对象、canonical identity 与纯 matcher | `69816f42`，后续修正至 `34580bcd` | `d7fad5e4`、`2083dd2e`、`35f1adb0` | 已合入；保持内部、无 policy、无 fallback |
| 2 | Compiler `_compiler.TargetCapabilities` 损失记账 adapter | `8c3e367e` | `1a501d8a` | 已合入；未覆盖字段仍须旧 comparator 判定，不得仅凭 Core matcher 放行 |
| 3 | CPU Platform capability snapshot adapter | `38e8bf61` | `c30316b9` | 已合入；只生产独立 CPU snapshot，不发现目标、不选择 backend、不授权 fallback |
| 4 | Runtime TargetCapabilities matching seam | `935adb25`（实现系列始于 `4c8732e9`） | `b6403927` | 已合入并验证；仅为未接管默认路径的内部 policy seam |
| 5 | synthetic Execution 第二 producer replacement conformance | `110ce4a1`（实现系列始于 `6f87da7d`） | `0f63c3b4` | 已合入并验证；只证明同一 Runtime consumer 可替换 producer，不代表真实 remote/QPU |

ARCH-003/004/005/007 仍为 **Proposed**。现有 Phase 2 机器授权只允许内部
TargetCapabilities v1、窄 adapter 和 Runtime matching seam；它不批准新的公共
Execution Request、Execution Result/Evidence、稳定 plan 字段、默认 backend、fallback 默认值或
失败阶段。

## 8. Runtime TargetCapabilities matching seam 逐项验收清单

清单中的 `[x]` 表示当前内部切片有实现与测试证据；`[ ]` 表示仍延后或证据不完整。勾选不改变
capability maturity，也不产生公共、默认、硬件或生产声明。

### 8.1 输入权威与范围

- [x] Runtime 直接消费 Core `RequirementSet`、`TargetCapabilitySnapshot` 和纯 matcher 结果；不复制、
  re-export、子类化或私建平行 capability 契约。
- [ ] Compiler 投影的 `requires_legacy_comparator` 和每条 loss 均被保留；只要该标记为真，旧
  `_compiler` comparator 未通过就不得选择候选。当前 seam 尚未接入 Compiler projection，此项作为
  必要技术债延后。
- [x] Runtime policy seam 只消费 requirement、snapshot 和显式授权；不得用用户期望、
  `world_size`、环境变量、Simulation 估算或 backend registry 布尔值补写平台事实。
- [x] v1 延后域和未知 extension 没有登记 matcher handler 时由 Core matcher 失败关闭；不得为了接入动态、拓扑、
  通信、checkpoint 或 realtime 而扩张本切片的公共契约。

### 8.2 Mandatory fail-closed 与 preference 排序

- [x] 每个候选先由 Core matcher 完整验证 identity、scope、freshness、证据引用和所有 mandatory
  谓词；missing、stale、scope/identity mismatch、`unknown`、`unmeasured`、`unsupported`、
  `not_exposed`、不被接受的 exposure、证据不足和未知 extension 均使该候选不可执行。
- [x] Runtime policy 不得覆盖 mandatory blocker，不得以成本更低、目标优先级更高或仅有一个候选
  为由放行。
- [x] preference 只对已经满足全部 mandatory 的候选排序；未满足 preference 只降低排序，不生成
  fallback 授权，也不能补偿任何 mandatory 失败。
- [x] 若所有候选均失败，返回聚合后的 typed blockers；不得选择“最接近”的候选或静默进入现有
  默认 backend。

### 8.3 候选顺序确定性

- [x] 同一 `RequirementSet`、同一组 snapshot、同一 policy 输入和同一评估时间产生相同的候选顺序、
  选择结果和 decision identity。
- [x] 排序键使用版本化、显式的确定性字段：先可执行性，再 preference 满足度，再受控 policy
  priority，最后以稳定 target/snapshot identity 打破平局；禁止依赖 dict/set 顺序、并发完成顺序、
  discovery timing、对象地址或进程 hash seed。
- [x] 输入 snapshot 顺序置换、重复 candidate/snapshot/target identity 和相同分数平局已有负向或
  变形测试；重复 identity 但语义不同失败，不采用后写覆盖。独立子进程 hash-seed 测试仍列在
  8.8 的补证项，不影响当前显式排序键结论。

### 8.4 CPU 独立候选与完整重匹配

- [x] 显式请求 CPU 时，CPU 作为普通独立候选参与匹配；从非 CPU 候选转向 CPU 时，只有
  `fallback_authorizations.cpu=true` 才允许创建该 fallback 候选。
- [x] CPU 候选使用自己的 target identity、scope、freshness、facts、evidence refs 和 blockers，
  对所有 Core mandatory 谓词做完整重匹配。旧 Compiler comparator 共判仍按 8.1 延后。
- [x] 不复用原候选的 memory、precision、device count、route、evidence、租约或 matcher 成功；CPU
  memory/precision 为 nullable unknown 时，对相应 mandatory requirement 必须失败关闭。
- [x] 设备解析失败、候选为空、probe 不可用或 CPU 在本机可用，都不能把 CPU 变成隐式 resolver
  默认值。

### 8.5 Fallback 分轴授权

- [x] `backend`、`device`、`cpu`、`precision`、`algorithm`、`approximation` 六轴分别读取明确授权，
  缺省均为 false；一个轴的授权不得推出另一个轴。
- [x] 每次 fallback 都以具有自身 snapshot/约束的显式替代候选完整重匹配；不得原地修改已失败候选、
  删除 mandatory requirement 或把 preference 改成授权。
- [x] Runtime policy、provider 声明和历史宽泛 `allow_backend_fallback` 不得推导 CPU、precision、
  algorithm 或 approximation 授权。
- [ ] 已选择的 fallback 已进入内部 decision record，但 attempt evidence 与 claim-ceiling 关联尚未
  实现；若需要改变稳定 plan/result 字段、identity 或异常行为，必须停止并提交独立 API Change
  Proposal。

### 8.6 Typed blocker 与 decision identity

- [ ] blocker 已保留 Core typed code、message、capability name 及来源上下文，并按确定候选顺序聚合；
  独立的 blocker canonical dedupe 规则尚未冻结。不得将 blocker 降为自由字符串、布尔失败或吞入日志。
- [x] 每个被评估候选保留其 snapshot identity 与 blockers；被拒候选的失败证据不得被最终成功候选
  覆盖，preference miss 也不得误报为 fatal blocker。
- [ ] 内部 decision record 已绑定 `requirement_set_id`、有序候选 snapshot identities、选中
  snapshot/target identity、评估时间、逐轴 fallback 授权 identity、matcher blockers、排序键版本及
  实际采用的 fallback 轴；独立 policy/candidate provenance identity 尚未冻结，列为必要技术债。
- [x] decision identity 由上述语义字段 canonical 计算；显示文本、运行时对象、凭据、vendor handle
  和无序容器不得参与，任何 requirement、snapshot、policy 授权或候选顺序变化都必须改变 identity。
- [x] 本切片的 decision record 只存在于内部 seam；ARCH-004/005 获批前，不得声称它是公共
  Execution Request/Result/Evidence 契约，也不得修改现有 plan/result identity。

### 8.7 公共 API、默认路径与分层保护

- [x] `flagquantum` 根导出、稳定签名/default/Literal、序列化 schema、异常类别和失败阶段完全不变。
- [x] 现有 `fq.run`、`fq.plan`、默认 backend/device 选择及 local CPU/single-device fast path 行为保持
  golden/兼容测试不变；新 seam 在单独批准前不接管默认路径。
- [x] 本切片未接入 Simulation，也不重写其算法约束或成本模型，不把估算当可用容量；Simulation 也不读取
  环境或选择物理 target。分层是所有权边界，不是阻断类型化信息交换。
- [x] 本轮各实现提交通过 team scope 和 architecture 检查；Runtime/Core/Compiler/Platform 的生产文件若超出各自 owner 或触及受保护 surface，仍必须拆回
  对应团队或先走 Integration/API proposal，不得把跨层改动夹带在 Runtime 切片中。

### 8.8 Runtime 提交必须提供的测试证据

- [x] mandatory 状态、exposure、evidence、stale、scope、identity 与 unknown-extension 的 Core/Runtime
  负向测试证明 policy 无法越过 Core matcher。
- [ ] preference 仅排序、全候选失败、平局、输入置换和 decision identity 已覆盖；独立子进程
  hash-seed 测试仍需补充，属于确定性防回归技术债。
- [ ] CPU 独立 full-rematch 已覆盖 memory、身份、未授权 CPU、只有 backend 授权和 unavailable/missing
  facts；device-count 与 native/effective precision 的专项负向矩阵仍需补证。
- [x] 六个 fallback 轴的默认禁止和隔离测试已覆盖；获授权候选仍经过同一个 Core 全量匹配入口。
- [ ] synthetic 第二 producer 已证明替换 CPU producer 时 Runtime consumer 无需修改；Compiler
  loss-accounting/legacy comparator 尚未接入 Runtime seam，前半项仍延后。
- [x] Stable API、`fq.run/plan` 默认路径和 local fast path 回归边界保持；任何 accelerator、QPU、
  多节点或 scalability 表述都必须保持未验证，不能由本切片测试提升。

## 9. 当前规范缺口与准入裁决

1. ARCH-004/005 只提出未来 request/decision/result/evidence 边界，现行授权没有批准稳定 decision
   record schema。Runtime 可以提交最小内部 decision record 来证明 identity 绑定，但不得写入或替换
   稳定 plan/result；若实现需要这种变化，当前切片必须暂停并转为 API Change Proposal。
2. Compiler adapter 对 gate parameter domains 与 ancilla policy 等字段仍是有损投影，并明确要求旧
   comparator。因此 Runtime 不能把 Core matcher 的 `executable=true` 等同于完整 Compiler 合法性。
3. CPU Platform producer 与 synthetic Execution producer 已通过同一 Runtime consumer 的替换
   conformance，满足本阶段最小“两个 producer”边界证明；synthetic fixture 不是网络、真实 provider、
   QPU 或硬件证据，不能据此提升 capability maturity。
4. dynamic、checkpoint、realtime、完整 topology/communication、gradient/optimizer distribution
   仍在 v1 延后域；本清单只规定失败关闭和 extension 门槛，不授权这些能力或其发布声明。

## 10. Phase 2 最小闭环与独立验证

Phase 2 的内部最小闭环已经形成：Core contract 与纯 matcher → Compiler loss-accounted adapter →
CPU Platform producer → Runtime policy seam → synthetic Execution 第二 producer replacement
conformance。闭环只证明内部类型、匹配、排序、fallback 授权、decision identity 和 producer
替换边界可以协作；它没有接管公共 API 或默认执行路径，也没有产生实际 execution observation。

总控独立验证记录：

- Runtime 合入 `b6403927` 后，定向联合集合 `121 passed`；architecture、team scope、Ruff 和 diff
  check 均通过；
- synthetic 第二 producer 修复并合入 `0f63c3b4` 后，定向联合集合 `118 passed`；architecture、
  team scope、Ruff 和 diff check 均通过。

真实 provider/hardware、公共 API、默认路径、执行结果/evidence、claim promotion，以及 v1 deferred
domains 均未获本闭环授权。`synthetic_qpu` 只是匿名测试值，不能描述为 QPU 接入或远程执行能力。

## 11. 下一阶段建议（不自动启动）

### 11.1 必要技术债

1. **Runtime candidate provenance 与非 CPU fallback 可信绑定**：让 candidate 的来源、相对原请求的
   变化和每个 fallback 轴由可验证 diff/adapter provenance 支撑，不能永久信任调用者自报的
   `fallback_axes`；同时冻结聚合 blocker 和 policy identity 的最小内部语义。
2. **Compiler legality 共判接线**：在不改变默认路径的前提下，让 Runtime seam 消费 Compiler
   projection 的 loss record，并在 `requires_legacy_comparator=true` 时强制旧 comparator 通过。
3. **验收矩阵补证**：补充独立 hash-seed、CPU device-count、native/effective precision full-rematch
   负向测试。以上均是已批准 seam 的完整性债务，不是新产品能力。

### 11.2 未来功能与契约工作

4. **Execution observation/evidence proposal**：先收敛 attempt identity、actual target/path、fallback、
   precision、distribution 与 claim ceiling 的旁路 envelope；ARCH-004/005 仍为 Proposed，获批前不改
   稳定 plan/result。
5. **真实国产 Platform adapter 的认证前置工作**：先定义厂商无关 probe、物理设备身份、原生精度、
   kernel residency、route/no-fallback、证据 digest 与认证环境；A800 或 synthetic 结果不能替代国产
   硬件材料。真实 adapter、远程任务和能力声明须另行授权。
6. dynamic、checkpoint/realtime、完整 topology/communication 和训练分布语义继续按独立 proposal
   排队，不并入 TargetCapabilities v1。

推荐顺序为 1 → 2 → 3 → 4 → 5；第 6 项按真实用例和证据成熟度逐域启动。本看板只给出排序，
不自动创建或授权任何下一阶段实现任务。

## 12. 当前总控指令（2026-09-04）

用户已明确授权将 vNext 从“横向契约扩展”切换到“最小纵向产品链路”。即日起按
以下顺序执行，不并行开启新的横向抽象工作：

```text
冻结新的横向抽象
 -> 完成当前 Compiler 收口
 -> 打通最小 CPU 纵向链路
 -> 开始真实目录归位
 -> 每轮主动删除历史和过渡代码
```

执行门禁：

1. **Compiler 收口**：完成 legality 共判，复用现有 comparator，通过确定性、
   防篡改和场景测试，并在交付前删除无当前用例的包装与重复表示。
2. **最小 CPU 纵向链路**：只要求一条用户可见路径贯通输入、编译合法性、Runtime
   选择、CPU 模拟执行、结果、失败和基础证据；不同时建设通用工作流、远程
   任务中心、插件市场或新配置框架。
3. **真实目录归位**：只迁移上述已通过的路径，同步切换调用方、所有权和依赖
   检查；不创建无实现的目标目录，不长期保留两个权威入口。
4. **每轮做减法**：交付记录必须分别列出新增、复用、删除和冻结项，以及旧入口
   的责任人、退出条件和目标版本。只增加新路径而不处理旧路径的变更不得称为
   迁移完成。

当前唯一进行中的实现任务是 Compiler legality 收口。其通过总控审查前，不启动
最小 CPU 纵向链路的生产实现。之后的 Platform、Execution、Simulation、Ecosystem 与
Agent 工作必须从该纵向链路的真实需求中产生，不以填满目标目录或增加契约数量为
进度。
