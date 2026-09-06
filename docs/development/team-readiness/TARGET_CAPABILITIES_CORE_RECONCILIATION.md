# TargetCapabilities Core 对账与最小契约候选

状态：Core 团队提案输入，待 Integration/API owner 审议；不构成 API、schema 或 ADR 批准

日期：2026-09-03

分支：`codex/vnext-phase2-capabilities-core`

路径分类：本轮只增加盘点文档与 CPU 特征测试，不改变执行路径或能力声明；测试只能证明
现有对象的结构和 fail-closed 行为，不能形成硬件、生产、通信或可扩展性证据。

## 1. 结论与权威建议

建议 Core 最终拥有两个窄而独立、版本化且 provider-neutral 的语义对象，而不是把所有
能力信息塞进一个 `TargetCapabilities`：

1. `CapabilityRequirement`：artifact/request 对目标的闭集约束，回答“必须有什么”；
2. `TargetCapabilitySnapshot`：在给定 target、environment、来源和时间点发现的事实，回答
   “当时发现了什么、支持状态是什么、事实如何获得”。

两者通过 Core-owned 的 fail-closed comparator 相交。`EvidenceReference` 只引用证明材料，
`ExecutionObservation` 只记录一次 workload 执行事实；二者都不继承或永久提升 snapshot 的
真值。为避免与现有私有类在迁移期产生同名双权威，第一版新类型宜使用上述明确名称；只有
旧调用者清零并经 API/contract proposal 批准后，才考虑把稳定别名命名为
`TargetCapabilities`。

当前 `_compiler.TargetCapabilities` 是 requirement/编译目标字段的最强复用基线：门集、
参数域、结果、artifact profile、拓扑、控制流、limits、ancilla 与 calibration 均应保留语义，
不能另造一套不同词汇。它不是完整 discovery snapshot，也不是公开权威，不能直接搬家或
原地扩字段。Runtime `BackendCapabilities`、Deployment `CloudBackendProfile`、Platform
records、operator probes 和 extension negotiation 继续由各 owner 生产，经显式 adapter
投影到 Core snapshot；算法局部 capability 不进入 Core。

## 2. 必须严格分开的六种事实

| 概念 | 回答的问题 | 可否直接推导其他层 | 建议载体 |
| --- | --- | --- | --- |
| requirement | workload/artifact 必须满足什么 | 不能推导目标支持 | `CapabilityRequirement` |
| discovered fact | 某来源在某时刻给出的目标/环境事实是什么 | 不能因 installed、接口存在或声明而 verified | snapshot 的 typed facts |
| support status | 当前是否足以支持该 capability | `declared` 不能推导 `verified` | 每项 fact 的 `unknown/unmeasured/unsupported/verified` |
| fact exposure | 值怎样获得或为何没有值 | `not_exposed` 不能推导 false；`declared` 不能推导 observed | 每项 fact 的 `observed/declared/not_exposed/unknown/not_applicable` |
| evidence level | 允许支撑多强的 claim | claim 不得高于最低相关 evidence level | snapshot 上限及 `EvidenceReference.level` |
| execution observation | 一次具体 workload 实际走了什么路径 | 不自动成为永久 target fact | `ExecutionObservation`/evidence artifact |

支持状态和事实暴露是正交轴。合法例包括 `unsupported + observed`、`verified + observed`、
`unknown + not_exposed`、`unmeasured + declared`；非法例包括 `verified + not_exposed`、没有
blocker 的 `unknown`、以及把 `declared` 自动升级为 `verified`。`not_applicable` 必须附有
适用性依据。已知 fallback 必须进入 execution observation；route 未暴露时只能保留
`unknown/not_exposed` 和 blocker，不能反证“没有 CPU fallback”。

Evidence level 是单调上限：`basic < observable < certification`。接口、环境变量、mock、
skipped hardware test 和 CPU 分布式语义测试最多提供各自范围内的 basic/语义证据，不能
提升国产硬件、生产或 scalability claim。

## 3. 现有类型、字段、生产/消费与身份盘点

| 类型 | 当前生产者 | 主要消费者 | 字段/粒度 | 序列化与身份 | 判断 |
| --- | --- | --- | --- | --- | --- |
| `_compiler.TargetCapabilities` | fixture、provider/conformance/bridge 构造 | comparison、legalization、dry-run、artifact sealing、provider conformance | 完整编译目标语义 | 严格 `to_dict/from_dict`；`target_capabilities_v1`；semantic JSON SHA-256；未知顶层/嵌套字段和版本拒绝；`display_label` 不参与 fingerprint | 最强语义基线，但 requirement 与 available 共型且无 discovery/evidence 轴 |
| `core.CapabilityContract` | compilation contract adapter | `RuntimePlanContract`、records | backend/devices/dtypes/modes + autograd/distributed | Core versioned contract 往返与 hash；是有损 plan audit 投影 | 不足以替代 target snapshot |
| `runtime.BackendCapabilities` + `AcceleratorInfo` | backend registry、platform discovery | backend/device/mode selection、root compatibility export | tensor backend、设备、dtype、模式布尔与当前 accelerator | 无 `from_dict`、无 snapshot identity；布尔把 unknown 压成 false/缺失 | 动态 Runtime 输入，只能适配 |
| `runtime.OperatorProfile` / `OperatorRequirement` | packaged JSON profiles | operator preflight | workload 的 operator/dtype/forward/backward/determinism requirements | 严格 profile reader、canonical profile hash | requirement 的可复用来源，不是 target fact |
| `runtime.CapabilityEvidence` | operator probes/hardware CI/provider declaration | preflight | 单个 operator/dtype probe | `evidence_id` 为完整 dataclass SHA-256；只有 runtime probe/hardware CI passed 才 `is_verified` | evidence 输入；不得并入 target semantic fields |
| `providers.platform.PlatformDevice/Identity/MemorySnapshot` | CPU/CUDA/FlagOS platform provider | Runtime discovery、诊断 | 设备/运行时身份与可能未知的内存 | Identity 单向 `to_dict`；metadata 自由；无 snapshot hash | platform facts 输入；available 不等于 workload support |
| `deployment.CloudBackendProfile` | cloud/provider adapter | package creation、QPU providers | provider/name/wires/gates/coupling/formats/dynamic/simulator/metadata | 无严格 `from_dict`、无独立 identity | Deployment 临时投影；metadata 不可直接进入 Core semantic root |
| `extensions.CapabilityRequest/Response` | extension caller/provider negotiate | extension registry/conformance | required names、dtype/device/gradient；accepted/supported/blockers | 无标准 `to_dict/from_dict` 或 identity | 受保护协商消息，不是 requirement/snapshot 权威 |
| `ops.LoweringCapability` | lowering registry | compiler/backend lowering validation | backend/opcode/strategy/implementation/supported/reason | manifest 投影；registry-local | 编译实现可用性，不是运行目标支持 |
| `runtime.FlagOSWorkloadCapability` | F1/F2/F3 evidence aggregation | benchmark contract/docs | workload status、collectives、world sizes、dtype、distribution、evidence level | 专用 v1 matrix 单向 dict | 审核派生结论，不是通用 snapshot |
| Core `ExecutionObservation` 与 Runtime evidence/route/fallback records | 一次执行/审计组装 | result、record、audit | elapsed/memory/communication/completed 及 runtime 专用路径事实 | 各自 version/hash 不统一 | workload observation；只以引用连接 snapshot |

此外，MPS workspace、JAX/TN planning、kernel dispatch、noise device profile 等局部对象只回答
算法或实现内部问题。只要它们不跨 Provider 边界，就不应因名称中含 capability 而进入 Core。

### 3.1 现有 `_compiler.TargetCapabilities` 精确字段

其 semantic identity 包含：`schema_version`、`target_class`、logical/physical qubit capacity、
规范排序的 `native_gates`（含闭合参数约束）、`measurement_results`、`artifact_profiles`、
有向 coupling topology、`control_flow`、mid-circuit measurement/reset/timing/pulse/noise/
parameter-binding 六个布尔能力、shots/program operation limits、ancilla policy/capacity、
calibration hash/valid-until。`display_label` 被序列化但有意不参与 semantic fingerprint。

现有 comparator 将同一个类型同时当 `required` 与 `available`：集合采用 required subset，
capacity/limits 采用 available 不小于 required，boolean 只在 required=true 时要求 available=true，
参数域要求 available 覆盖 required，显式拓扑 edges 要求 available superset，校准引用要求相等。
它已 fail closed，但无法表达 unknown/unmeasured、snapshot freshness、精度、平台/物理 route、
communication 和 evidence claim 上限。

## 4. 最小 Core 契约候选

以下是 schema 评审输入，不是本分支实现。

### 4.1 共用闭集

- `SupportStatus = unknown | unmeasured | unsupported | verified`
- `FactExposure = observed | declared | not_exposed | unknown | not_applicable`
- `EvidenceLevel = basic | observable | certification`
- `TargetClass`、artifact format/profile、measurement result、control-flow、ancilla policy、
  gate/parameter constraint：直接复用现有 `_compiler` 已验证语义，迁入前保持值域不变。
- 新的开放词汇只能放在 namespaced extension 中；不能以自由字符串新增稳定 root capability。

### 4.2 `CapabilityRequirement` 字段集

| 组 | 字段 |
| --- | --- |
| envelope | `schema`, `schema_version`, `requirement_id` |
| target semantics | `target_classes`, `minimum_logical_qubits`, `minimum_physical_qubits`, `native_gates`, `measurement_results`, `artifact_profiles`, `topology_requirement`, `control_flow`, `ancilla_requirement` |
| execution constraints | `required_features`, `limits`, `precision_requirement`, `communication_requirement`, `distribution_requirement`, `fallback_policy` |
| proof constraint | `minimum_evidence_level`, `required_evidence_scopes` |
| evolution | `extensions` |

`requirement_id` 是除自身外完整 canonical semantic payload 的 SHA-256。Requirement 不包含
provider identity、capture time、support status、queue/calibration live state、observed metrics 或
claim verdict。`required_features` 必须使用版本化闭集 key；未知 key fail closed。

### 4.3 `TargetCapabilitySnapshot` 字段集

| 组 | 字段 |
| --- | --- |
| envelope | `schema`, `schema_version`, `snapshot_id` |
| target identity | `target_id`, `target_class`, `provider`, `provider_version`, `target_revision`, `display_label` |
| environment identity | `environment_id`, `runtime_versions`, `platform_identity`, `physical_devices`, `code_revision` |
| capture/source | `captured_at`, `valid_until`, `source_kind`, `source_id`, `source_digest`, `blockers` |
| typed facts | `qubits`, `gates`, `measurements`, `artifacts`, `topology`, `control_flow`, `features`, `limits`, `ancillas`, `calibration`, `precision`, `communication`, `distribution`, `fallback_visibility` |
| proof | `evidence_level`, `evidence_refs`, `claim_ceiling` |
| evolution | `extensions` |

每个 typed fact 使用统一 `CapabilityFact<T>`：`value`（可以 null）、`support_status`、
`fact_exposure`、`source_ref`、`blockers`、可选 `observed_at`/`valid_until`。`value=null` 不单独
表达 unknown：必须同时给出状态、暴露原因和 blocker。snapshot 的 `evidence_level` 只是所有
相关事实可共同支撑的上限；更强或更窄的材料留在 evidence references 中。

`snapshot_id` 对 canonical semantic snapshot 求 SHA-256，必须包含 target/environment identity、
capture/source、所有 typed facts、evidence refs、blockers、claim ceiling 与 semantic extensions；
排除 `snapshot_id` 自身和纯展示 `display_label`。同一 target 在不同时间或环境的 snapshot
不是同一 identity。原始 artifact URI 若会变，应由稳定 digest/ref 参与 identity，URI 仅展示。

### 4.4 extension namespace 与未知字段

- 顶层与所有 Core-owned nested object 都是闭合字段；未知字段、未知 enum 和未知 major/schema
  version 一律拒绝。
- 唯一扩展点是 `extensions: {"<reverse-dns-or-flagquantum-namespace>": section}`；namespace
  必须非空、规范小写、由 owner 登记，section 必须 JSON-safe、有限值、无 credential/vendor
  object。
- reader 可保留未知 extension section 以支持 round-trip，但 comparator 默认将 requirement
  中未知/无 handler 的 semantic extension 判为 `unknown_extension_requirement`，fail closed；
  不能静默忽略。
- extension 是否参与 identity 由 namespace registration 固定；第一版保守地全部参与 semantic
  identity。改变此规则是 schema/identity change。
- unknown fact 不等于 unknown field：前者是有类型的业务状态，后者是 reader 不认识的 schema
  成员。不得用自由 metadata 绕过闭集。

## 5. 比较与兼容规则

Comparator 输入必须是一个 requirement 和一个未过期 snapshot，不再允许 snapshot 与
snapshot、requirement 与 requirement 的偶然比较。

1. 先验证 schema、canonical identity、target/environment binding、`captured_at/valid_until`；
   stale、身份不完整或 hash 不一致直接 incompatible。
2. 对 requirement 涉及的每项 fact，只有 `support_status=verified` 可满足；`unknown`、
   `unmeasured`、`unsupported` 都产生不同 typed blocker。未要求的事实不影响兼容。
3. 集合为 required subset；minimum capacity/maximum supported limits 均要求 available 覆盖；
   gate parameter domain、topology edge、control-flow、ancilla 的覆盖规则复用现有 comparator。
4. 精度同时比较 storage/compute/accumulation、native/software、dtype 与 workload/kernel scope；
   native float64 requirement 不能由 Double-Single 满足。
5. topology requirement 比较 node/world/local-world、rank placement、物理 links 和 source；仅有
   logical device names 或 inferred topology 不能满足 observed/certification requirement。
6. communication requirement 比较 outer process-group backend、inner physical route、collective、
   dtype、device residency 与 host staging。`outer_backend=flagos` 不能推导 `inner_backend=flagcx`。
7. fallback policy 若禁止 CPU/backend substitution，则 snapshot 还必须证明 route 可观测；
   `not_exposed` 不能满足“无 fallback”要求。一次执行是否实际 fallback 由 observation 最终确认。
8. evidence level 按 `basic < observable < certification` 比较，最终 claim ceiling 取所有相关
   requirement/fact/evidence/observation 的最低值；sharding claim 还必须有 workload-specific
   observation 和完整 distribution fields。
9. comparator 返回所有 typed differences（path、required、available/status/exposure、code、
   evidence refs、blockers），禁止自动降级或 fallback。策略层可基于显式用户授权另建 requirement，
   不能篡改原 requirement 或 snapshot。

## 6. 国产加速器、精度、拓扑和通信表达

第一版必须能无歧义表达以下边界：

- `platform_identity.logical_device_type` 与 `physical_devices[].vendor/model/id` 分开；
  FlagOS-on-CUDA/NVIDIA A800 不能序列化为“国产卡”。
- precision 记录 `storage_dtype`、`compute_dtype`、`accumulation_dtype`、`mode=native|software`、
  `software_scheme`、`kernel/workload_scope`。Double-Single 是软件模式且保留 FP32 指数范围，
  可以在认证范围内形成 effective complex128，但不能别名为 native float64。
- topology 记录 `node_count/world_size/local_world_size`、每 rank 的 node/device ownership、
  links、`source=observed|provider_declared|inferred` 与 fingerprint；未知链路保持 unknown。
- communication 记录 outer backend、inner route、P2P/collective/dtype matrix、device residency、
  bytes、`host_staging_observed` 及 route verification。逻辑 process group 可用不证明 FlagCX、
  无 host staging 或物理链路。
- distribution 继续闭合区分 `single_device_fast_path`、`sharded_across_ranks`、
  `rank_local_replicated_kernel`、`manual_sliced_tensor_contraction`、`observable_term_parallel`、
  `data_parallel_replicated`、`replicated_per_rank`；只有 workload-specific sharding evidence 才能
  允许 capacity/scalability claim。

CPU fixture 可以有真实本地执行事实，但未知内存/拓扑/通信不得补零。CUDA fixture 只按具体
设备、kernel、dtype 和 workload 的材料提升。FlagOS-on-CUDA fixture 明确记录 logical FlagOS
和 physical NVIDIA；国产真实卡 fixture 在签入、审核的 attestation 与 runtime observation
出现前保持 `unknown/unmeasured`、claim ceiling 为 basic 且 blocker 非空。

## 7. Adapter 与退出台账

| Adapter | owner | 输入 → 输出 | 必须保真/禁止推断 | 退出条件 |
| --- | --- | --- | --- | --- |
| compiler target adapter | Compiler | 私有 `TargetCapabilities` → Core requirement 或 declared snapshot | 保留现有 semantic fingerprint 映射；旧 bool 不可直接 verified，无来源时为 declared/unmeasured | Compiler 消费 Core 类型，旧 importer=0，golden hash/compat fixtures 通过；目标版本由 Integration 批准 |
| runtime backend adapter | Runtime | `BackendCapabilities`/operator profile/probes → snapshot | registry bool 与 device visibility 不提升 workload support；probe evidence refs 保留 | backend selection/preflight 全部读 Core snapshot，旧跨层字段 importer=0 |
| platform discovery adapter | Platform | Device/Identity/Memory → platform fact sections | unknown memory/topology/route 保持 null+状态；vendor object 不出边界 | Platform provider 直接实现批准的 snapshot producer conformance，旧 adapter importer=0 |
| deployment profile adapter | Execution Provider | `CloudBackendProfile` → target snapshot | metadata 只进登记 namespace；provider claim 默认 declared | Provider capabilities API 返回 Core snapshot，旧 package readers 的兼容窗口完成 |
| extension negotiation adapter | Ecosystem | Core requirement ↔ 受保护 CapabilityRequest/Response | 只做可表达子集；无法表达的 requirement 必须 blocker，不能丢字段后 accepted | 新版 extension protocol 经 API proposal 发布并完成旧协议 deprecation；此前不可删除 |
| agent vocabulary adapter | Agent | alias/旧 required names ↔ Core requirement/snapshot view | 只有版本化 alias；未知名称 fail closed | Agent 只读 Core 服务，旧 alias 使用量为零且移除版本到达 |
| evidence reference adapter | Runtime/Audit | CapabilityEvidence/Runtime evidence/artifact → EvidenceReference | 保留 digest、source、workload/environment/revision；不复制原始 verdict 为 target fact | Core envelope + audit conformance 获批，所有引用可验证 |
| FlagOS workload adapter | Runtime/Audit | 专用 workload matrix → snapshot/evidence refs | 保留 NVIDIA physical device、single-node scope、claim flags false 和 blockers | 通用 schema 能无损表达现有 F1-Fn payload 且 release auditor 改读新引用 |

每个兼容 adapter 在落地提交中必须记录 owner、开始版本、支持的旧 schema、metrics、删除版本或
删除条件；不得以“稍后替换”为退出计划。迁移顺序遵循 ARCH-003：先词汇/fixtures，再 compiler，
再 Runtime/Platform，再 Deployment/Provider，最后 Agent/alias 收缩。

## 8. API/Contract change 触发条件

以下任一项必须先有 Integration 批准的 Contract Proposal/ADR：在 Core 新增上述类型或
provider protocols；选择 canonical field/value/identity；修改 `contracts/`、architecture rule、
跨团队 owner；改变旧类型兼容期或删除条件。

以下任一项还必须有 API Change Proposal 和 API owner 批准：

- 把新 capability 类型加入 `flagquantum` 根、稳定 namespace 或公共类型注解；
- 移动/重导出现有稳定 `BackendCapabilities`、`CloudBackendProfile`、Extension request/response，
  或改变 class identity、构造字段顺序/default/frozen 状态；
- 改变现有 schema、unknown field/version rejection、canonical ordering、round-trip 或任何 hash/
  fingerprint input（包括把 `display_label` 纳入 identity）；
- 改变 capability failure stage、exception category、fallback 可见性，或改变稳定 plan/result 的
  字段与序列化；
- 删除兼容入口、改变 extension negotiation accepted 语义，或开始承诺新的 stable snapshot
  backward-reading window。

仅增加本轮这种现状特征测试和提案文档不触发 API change；在共同基线已有批准契约后添加
私有 adapter，也只有在完全保持现有输入输出/异常/identity 时才可作为团队内部实现。

## 9. 验收、未决问题与本轮边界

集成实现最低验收：requirement satisfied/missing/unknown、stale snapshot、两条状态轴组合、
evidence 单调性、unknown extension fail-closed、canonical/golden identity、旧 fixture 后向读取、
CPU/CUDA/FlagOS-on-CUDA/fake QPU/国产未验证 fixtures、精度不混称、route 未暴露不反证 fallback、
以及 sharding 完整字段和 claim ceiling 的负向测试。同一 snapshot 经 Compiler、Runtime、Agent
查看时 identity 必须一致；至少替换一个 Runtime 和一个 QPU/remote implementation 而不改
消费者。

待 Integration/API owner 决定：

1. 新类型稳定命名以及是否最终接管 `TargetCapabilities` 名称；
2. requirement vocabulary 的 minor-version/additive 注册规则与 extension handler registry owner；
3. snapshot TTL/calibration staleness 的 policy owner，offline target 是否允许无 TTL；
4. workload predicate 的表达能力，避免演化成任意代码/查询语言；
5. evidence 审核签名放 Core envelope 还是外部 Audit 服务，以及 URI/digest 的保留策略；
6. `display_label`、URI 和非语义 diagnostics 的明确 identity 排除清单；
7. 旧 compiler semantic fingerprint 与新 snapshot/requirement identity 的映射是否作为稳定迁移
   字段保存；
8. Cloud/extension 现有公共类型的 deprecation 窗口和明确目标版本。

本轮没有修改任何受保护产品契约、公共导出、schema、hash、runtime behavior、capability
maturity 或 benchmark claim。
