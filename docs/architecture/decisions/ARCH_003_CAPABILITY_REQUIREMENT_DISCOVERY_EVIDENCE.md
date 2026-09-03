# ARCH-003：Capability Requirement、Discovery Snapshot 与 Evidence

状态：Proposed

日期：2026-09-03

依据：Phase 0 架构盘点与 Phase 2 Core、Platform、Runtime 三方对账；接口存在不等于能力已实现

## 决策

Core v1 只拥有以下唯一谱系，不再增加第四种跨层 capability 类型：

1. `CapabilityRequirement`：一个闭集 capability 谓词；
2. `RequirementSet`：本次请求/编译所需谓词及逐轴 fallback 授权的不可变集合；
3. `TargetCapabilitySnapshot`：一个 target/environment 在有限时间和 scope 内的事实快照。

`PlatformCapabilitySnapshotCandidate`、Runtime `BackendCapabilities`、Deployment
`CloudBackendProfile`、扩展 `CapabilityRequest/Response` 及当前私有
`_compiler.TargetCapabilities` 都只是各 owner 的现有输入。它们通过显式 adapter 进入上述
谱系，不成为 Core 别名、子类或第二事实权威。`EvidenceReference` 是快照引用的值对象，
`ExecutionObservation` 是一次 attempt 的执行记录；二者都不是 capability 类型，也不会自动
提升长期事实。

本 ADR 保持 Proposed：它批准后续内部最小实现和 adapters 的边界，但不改变 Stable Core、
公共导出、现有序列化、失败阶段、默认选择或产品实现。

## v1 最小契约

### `CapabilityRequirement`

一个 requirement 只表达一个谓词：

- `name`：v1 闭集名称；
- `operator`：`equals | at_least | at_most | contains_all | covers`；
- `value`：该名称定义的 JSON-safe 类型；
- `strength`：`mandatory | preference`；
- `source`：`user | compiler | runtime_protocol`；
- `minimum_evidence_level`：`basic | observable | certification`；
- `accepted_exposures`：非空闭集，取自 fact exposure 值域。

`mandatory` 不满足时淘汰候选；`preference` 只给已满足全部 mandatory 的候选排序，不得使
不可执行候选变得可执行，也不得形成 fallback 授权。Core v1 不提供任意表达式、回调、查询
语言或自由 capability 名称。

### `RequirementSet`

最小 envelope 为 `schema_version`、`requirement_set_id`、`requirements`、
`fallback_authorizations` 和 `extensions`。requirements 按 canonical key 去重排序；冲突的
mandatory 谓词在匹配 target 前失败。`requirement_set_id` 是除自身外语义 payload 的 canonical
JSON SHA-256。

fallback 授权分为 `backend`、`device`、`cpu`、`precision`、`algorithm`、`approximation` 六轴，
每轴显式布尔值且缺省为 false。一个轴不能推导另一个轴；偏好、Runtime policy、provider
声明和现有宽泛 `allow_backend_fallback` 都不能推导 CPU 或精度降级授权。

### `TargetCapabilitySnapshot`

最小 envelope 为：

- `schema_version`、`snapshot_id`；
- `target_identity`：`target_id`、`target_class`、`provider`、`provider_version`、
  `target_revision`、`environment_id`；
- `scope`：可空的 `device_ids`、`dtype`、`kernel`、`workload_id`、`world_size`、`node_count`；
- `captured_at`、`valid_until`；
- `facts`、`evidence_refs`、`blockers`、`extensions`。

每项 fact 恰有 `name`、`value`、`support_status`、`fact_exposure`、`source` 和 `blockers`。
`source` 只含 `kind` 与可验证 `ref`，不得放 SDK object、handle、credential 或 callable。
`snapshot_id` 是除自身外完整语义 payload 的 canonical JSON SHA-256；展示 label 和可变 URI 不
进入 identity。v1 所有 snapshot 都必须有明确 `valid_until`；到期、scope/identity 不匹配、
hash 错误或证据引用不可解析时 fail closed。是否重新发现由 Platform/Provider policy 决定，
Core 不内置全局 TTL。

`evidence_refs` 每项最小包含 `evidence_id`、`sha256`、`level` 和 `scope`。原始材料与审核
verdict 留在 Evidence/Audit owner；snapshot 只引用，不复制或改写 verdict。

## 状态、暴露和证据的正交语义

support status 回答“在此 scope 是否足以支持”：

- `unknown`：权威来源没有答案；
- `unmeasured`：存在候选接口或声明，但未测到所需阈值；
- `unsupported`：适用且权威的负向结论；
- `verified`：有匹配 identity、scope、新鲜度和值谓词的材料。

fact exposure 回答“值怎样获得或为何没有”：

- `observed`：受控探测或 workload 观测；
- `declared`：provider/spec 声明；
- `not_exposed`：权威接口没有暴露字段；
- `unknown`：来源不清、冲突或不能安全投影；
- `not_applicable`：对当前 scope 不适用，必须有 blocker 说明依据。

两轴不互相推导。`declared` 不自动成为 `verified`，`not_exposed` 不等于 false 或“未发生”，
`unsupported/observed` 是合法负向观测。`verified/declared` 只在机器授权中该闭集名称被列入
`authoritative_static_declaration_allowed` validation profile、requirement 的
`accepted_exposures` 包含 `declared`，且引用材料、scope 和 freshness 全部满足时合法；一条未验证
的声明本身默认仍是 `unmeasured/declared`。容量、precision、延迟、物理 route 和“没有 fallback”
等运行事实只接受 `observed`。`unknown`、`unmeasured`、
`not_exposed` 以及缺失 fact 都不能满足 mandatory。`not_applicable` 只能满足明确允许“不适用”的
专用谓词；v1 通用匹配默认不满足。

证据强度严格为 `basic < observable < certification`。匹配所需阈值取全部适用 mandatory
requirements 和 claim gate 中的**最强等级**；候选可用证据上限取支撑这些事实的全部不可缺
evidence references 中的**最弱等级**。只有可用上限不低于所需阈值，并且每项 fact 的 status、
accepted exposure、scope、freshness 与值比较都通过时才满足。`verified` 只表示这一完整规则下的
当前匹配成立，不等于 release certification。Mock、接口、环境变量和 CPU 分布式语义测试不能
支撑真实 accelerator、QPU、生产或 scalability 声明。

## v1 闭集与精度裁决

首个实现闭集只授权：`target.class`、`device.kind`、`device.count`、
`memory.available_bytes`、`qubits.logical_capacity`、`qubits.physical_capacity`、
`gates.native`、`measurements.results`、`artifacts.profiles`、`limits.maximum_shots`、
`limits.maximum_program_operations`、`ancillas.policy`、`ancillas.maximum_compiler`，以及以下六个
彼此独立的 precision 名称：

- `precision.native_dtype`；
- `precision.effective_dtype`；
- `precision.storage_dtype`；
- `precision.parameter_dtype`；
- `precision.accumulator_dtype`；
- `precision.software_mechanism`。

software extension 是 mechanism 轴，不是 dtype 别名。Double-Single 可以在匹配 scope 的材料
下满足 effective precision，但不能满足 native FP64/complex128；必须保留实际 storage、
parameter、accumulator dtype、mechanism 和误差/范围限制。`gates.native`、参数域、artifact、
measurement、qubit、limit 和 ancilla 的 `covers`/集合/上下界语义复用当前
`_compiler.TargetCapabilities` comparator，迁移时不得改变其 fingerprint 或旧 schema。

## 匹配与 fallback

Core 提供无 policy 的纯比较：先验证 requirement/snapshot identity、scope、freshness 与引用，
再逐项检查 status、exposure、evidence level 和 operator/value。比较结果返回全部 typed blockers，
不自动降级、选择目标或执行 fallback。

Runtime 拥有候选匹配、排序、租约与最终决策。fallback 仅在对应轴明确授权后创建替代候选；
替代候选必须使用自己的 snapshot 对所有仍适用 mandatory 谓词重新完整匹配。CPU 永远是独立
候选，不能作为设备解析失败的默认值，也不能复用 GPU 的 memory、precision、route 或 evidence。
实际 fallback 必须进入 plan/result/evidence；route `not_exposed` 时不得宣称没有 CPU/host fallback。

## 所有权

| 责任 | 唯一 owner | 边界 |
| --- | --- | --- |
| 程序与目标合法性 | Compiler | 从 IR/旧 target 产生 compiler-source mandatory 谓词；对 Runtime 选定 snapshot 做合法化；不做 discovery 或授权 fallback |
| 平台/目标发现 | Platform Provider / Execution Provider | 产生 adapter candidate、source 与 evidence refs；不决定 workload、claim 或 target 选择 |
| 候选匹配与 policy | Runtime | 消费 RequirementSet、snapshot、租约与授权，选择目标；不改 Compiler 合法性或伪造事实 |
| 算法候选与成本提示 | Simulation | 给出合法 representation、资源估算、误差和 partition 提示；不选择物理设备、route 或 fallback |
| 一次执行证据 | Runtime + executing Provider | 绑定 attempt 记录实际 target/device/precision/route/distribution/fallback；不自动回填长期 snapshot |
| claim 审核 | Audit/Release | 验证 evidence 与声明上限；Provider 或 Core comparator 不自行认证 |

## v1 明确延后

dynamic control、mid-circuit measurement/reset、checkpoint/restart、realtime/session/latency、
完整 topology/placement/link、P2P/collective/physical communication route、calibration lifecycle、
gradient/optimizer distribution 及完整 workload predicate 不进入 v1 Core root vocabulary。

确有跨层需要时只能使用登记的反向域名 namespace extension，携带明确 owner、schema version、
handler、测试和退出/晋级条件。Requirement 中出现未知 namespace 或没有 matcher handler 时必须
fail closed；snapshot 的未知 extension 可为 round-trip 保留，但不能参与满足 Core 谓词或提升
claim。稳定后通过新的 minor/major contract proposal 晋级，禁止预先把所有不稳定域塞进 v1。

## 兼容、授权与迁移

现有 `_compiler.TargetCapabilities`、`compare_target_capabilities`、Runtime/Platform/Deployment/
Extension capability 类型和稳定 API 全部保持不变。批准的下一步仅是内部 Core 值对象、strict
JSON reader/writer、canonical identity、纯 comparator、contract fakes/conformance，以及上述
现有类型的窄 adapter。adapter 必须记录 owner、损失性 blocker 和退出条件；不可表达字段继续
由旧权威处理，不得被静默丢弃。

任何根导出、稳定签名/默认值/异常、旧 schema/fingerprint、plan/result identity、默认选择或
失败阶段变化仍需独立 API Change Proposal。真实 hardware/provider 接入和能力晋级仍走
ARCH-006/008 认证路径；本决策不声明任何国产硬件已经验证。

实施顺序：Core 三类值对象与 fake → 纯 matcher → `_compiler.TargetCapabilities` 保真 adapter →
CPU Platform adapter → Runtime matching seam 与独立 CPU candidate → 第二 Platform/remote fake
替换测试 → 逐域 proposal。每一步保持旧路径行为，直到获批迁移完成。

## 验收

- strict schema、canonical SHA-256、重复/冲突谓词、stale/scope/identity mismatch 负向测试；
- mandatory/preference、五个 comparison operator、minimum evidence 与 unknown/unmeasured/
  unsupported/not_exposed fail-closed；
- precision 六轴与 Double-Single 非 native 测试；
- 六轴 fallback 隔离、CPU 独立 snapshot 全量重匹配和 route visibility 测试；
- 当前 compiler fingerprint、Stable API 与历史 contract fixture 不变；
- 至少两个 producer adapter 通过同一 conformance，消费者不修改；
- 国产硬件 fixture 保持 unknown/unmeasured、basic ceiling 和非空 blocker。

机器授权与精确字段见 `contracts/target-capabilities-v1-implementation-authorization.json`；裁决说明
见 `docs/development/VNEXT_PHASE2_CAPABILITIES_DECISION.md`。
