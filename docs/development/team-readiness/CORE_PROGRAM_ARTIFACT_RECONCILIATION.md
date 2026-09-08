# ProgramArtifact Phase 1 契约对账

> 历史说明：本文是 Phase 1 时点的特征盘点。其所述
> `AgentApplicationService` 已在发布前删除，当前 `flagquantum.services` 不消费
> 序列化 `ProgramArtifact`；协议适配器在边缘解码后调用稳定 API。以下内容仅用于
> 追溯当时的契约判断，不描述现行服务入口。

状态：Core 团队正式对账，提交 Integration 决策；不构成 API、schema 或 ADR 批准

对账基线：`99d5a92091bff35fdc573f4b401e3ebaaf5ffcbf`

团队分支：`codex/vnext-phase1-core-contracts`

日期：2026-09-03

## 1. 决策结论

现有 `flagquantum.core._artifacts.ProgramArtifact` 应继续作为仓库中唯一的 Core-owned
Program Artifact 信封权威。它已经具备严格顶层字段、封闭 `ArtifactKind`、版本拒绝、不可变
JSON 投影和确定性 envelope hash；新建第二个 Artifact 类型会违反 ARCH-001。

但该结论只覆盖当前 `flagquantum.program_artifact` 1.0 信封和已经运行的 circuit artifact
路径，不等于它现在可以无损替代 Compiler 的 `SealedExecutableArtifact`、
`SealedCircuitIRRoundTrip` 或 Deployment 的 `DeploymentPackage`：

- 当前唯一生产消费者是 `AgentApplicationService`，且只接受 `kind=circuit`；
- Compiler、Runtime、Simulation、Execution Provider 和 Ecosystem 均未直接消费
  `ProgramArtifact`；
- v1 没有一等 `provenance`、结构化 `requirements`、命名空间 `extensions`、payload profile 或
  bytes 编码；
- `required_capabilities` 是 Agent Services 使用的扁平字符串集合，不等同于编译需求；
- `producer`、`parent_hashes` 和 `metadata` 无法无损表达 Compiler 已有的带角色身份链；
- `content_hash` 是计算属性而非序列化字段，消费者不能从信封内部取得发送方声明的摘要。

因此最小方案是：冻结 v1 现状作为兼容读取基线，先批准字段语义和身份分层，再由各团队提供
显式 adapter。任何新增顶层字段、值域收紧、版本/哈希变化或稳定导出必须先走 Integration
契约决策；不得为了接入某个消费者把私有字段塞入自由 `metadata` 并宣称收敛完成。

## 2. 保护边界与事实来源

`ProgramArtifact` 定义文件 `flagquantum/core/_artifacts.py` 被 `team-ownership.toml` 明确列为
protected path。本轮只新增本文和 `tests/team/core/test_program_artifact_reconciliation.py`，
没有修改该文件、公共导出、`contracts/`、ADR、Compiler、Runtime、Deployment 或 Agent
实现。

对账依据包括：

- `ProgramArtifact` 实现和现有 Core/Agent 特征测试；
- `AgentApplicationService._decode_program()` 与 planning capability preflight；
- Compiler 的 `ImportedCircuitProgram`、`SourceIdentity`、`SourceProvenance`、
  `SealedCircuitIRRoundTrip`、`TargetIR` 和 `SealedExecutableArtifact`；
- Deployment 的 `DeploymentPackage`、routing evidence、submission receipt 和 artifact digest；
- `_compiler.deployment_compatibility` 对 legacy package 的 metadata/identity 限制；
- Phase 0 集成看板与八份 team-readiness 清单。

路径分类为契约读取和 `single_device_fast_path` 的本地 planning 特征检查。本轮没有执行或改变
分布式路径，也不产生硬件、QPU、性能或可扩展性证据。

## 3. 所有直接消费者

仓库级搜索得到以下直接定义和消费者；文档提案不计作运行时消费者。

| 所有者 | 入口 | 实际读取/生成字段 | 当前行为 | 对账结论 |
| --- | --- | --- | --- | --- |
| Core | `ProgramArtifact` | 全部 v1 字段；计算 `content_hash` | 构造、冻结、dict 往返、严格顶层读取 | 唯一 envelope 权威，仍是未稳定导出的内部候选契约 |
| Core | `from_circuit_ir()` | 生成 `kind`、`payload`、`producer`；其余默认 | 只要求对象有 `to_dict()`，不验证它确为 `CircuitIR` | 便捷 adapter，不是 per-kind payload validator |
| Agent Services | `_decode_program()` | `schema`、`kind`、`payload`、`required_capabilities` | 仅接受 circuit；由 `CircuitIR.from_dict()` 验 payload | 当前唯一生产消费链 |
| Agent Services | `validate_program()` | 间接读取 artifact，但忽略返回的 requirements | 只校验 circuit payload | 能力不足不会在 validate 阶段失败 |
| Agent Services | `plan_execution()` | `required_capabilities` | 用当前 capability manifest 的临时词汇 fail closed | 属于兼容策略，不是 Core capability matching 权威 |
| Tests | Core、Agent、long-horizon tests | round-trip、hash、未知 schema/kind、planning | 固化当前候选行为 | 本轮扩充字段级和差异级证据 |
| Compiler | 无直接 import | 无 | 使用自己的 source/target/executable artifact | 需要 adapter；不得直接替换类型 |
| Runtime / Simulation | 无直接 import | 无 | 消费 `CircuitIR`、Compiler plan 和各自内部结果 | 等 Artifact/Request 契约获批后再接入 |
| Deployment / Execution Provider | 无直接 import | 无 | 使用 `DeploymentPackage` 及独立 digest/receipt | 不是 ProgramArtifact 的现有消费者 |
| Ecosystem | 无直接 import | 无 | adapter 返回 `CircuitIR`；metadata 可能进入 IR | 应先通过 Core metadata 值域再封装 artifact |

没有发现 `ProgramArtifact` 的根导出或 `flagquantum.core` 稳定导出。外部 Compute Service 的
文档需求不能反向证明主仓库已经拥有稳定跨仓库 schema。

## 4. v1 字段逐项决策表

| 字段/属性 | 当前可观察行为 | 当前消费者 | Phase 1 决策 | 变更门槛 |
| --- | --- | --- | --- | --- |
| 固定 `schema` | `to_dict()` 写 `flagquantum.program_artifact`；缺失、未知或其他值拒绝 | Agent 用它识别 envelope | 保持 v1 精确值；不得用 `.v2` 字符串偷换版本体系 | 改值或兼容规则需契约/序列化 API 提案 |
| `version` | 唯一支持 `"1.0"`；构造器精确比较；`from_dict()` 先 `str()`，因此 JSON 数字 `1.0` 也会被接受 | Core reader、Agent 间接消费 | 冻结为兼容读取事实；未来 reader 是否停止 coercion 由 Integration 决定 | 收紧 coercion、改名 `schema_version` 或接受新版本均改变行为 |
| `kind` | 封闭八值 Enum；不按 kind 校验 payload | Agent 只支持 `circuit` | 保持 kind 权威；每个消费者声明支持子集并 fail closed | 增删/改枚举或通用 payload dispatch 需提案 |
| `payload` | 必须可投影为 mapping/list/JSON scalar；递归冻结；不接受 bytes；无 payload profile | Agent 将 circuit payload交给 `CircuitIR.from_dict()` | 保持 envelope 与 payload validator 分离；circuit adapter 可继续使用 | executable profile、bytes 表达和每-kind schema 需 ADR/版本决策 |
| `producer` | 非空检查但不 trim；参与 envelope hash；`from_dict()` 会把数值转为字符串 | 没有消费者解释其词汇 | 定义为 opaque producer label，不把它当完整 provenance 或 compiler identity | 值域、规范化和语义升级会改 hash/接受域 |
| `required_capabilities` | 排序、去重，空值拒绝；没有严格字符串类型检查；参与 hash | 仅 Agent planning | 保留为 v1 coarse compatibility hints；不改名为 `requirements`，不充当结构化编译需求 | 词汇、类型收紧、语义或 identity inclusion 变化需提案 |
| `parent_hashes` | 要求小写 SHA-256；保留顺序和重复项；不验证存在性、关系或 hash 种类；参与 hash | 无 | 保留为 ordered opaque lineage references；不得假定位置角色 | 去重、排序、角色化或验证规则会改身份，需要 ADR/API 迁移 |
| `metadata` | 参与 hash；key 被 `str()`；接受 JSON scalar/container 和任意 `to_dict()` 对象；无 namespace、深度、大小、敏感字段限制 | 当前生产消费者不读取 | v1 只作兼容数据，不承载新的必需语义、凭据、live SDK object 或权威 identity | 关闭值域、namespace/limit/unknown-key 规则均改变序列化行为 |
| `content_hash` | 对完整 `to_dict()` 的 sorted-key compact JSON 做 SHA-256；所有字段均参与；mapping 顺序无关，sequence 顺序相关；禁止 NaN 仅在取 hash 时触发 | tests；生产链未传输/比对声明 hash | 明确命名为 envelope identity；不得与 IR/payload/artifact identity 混同 | 改算法、输入字段或把摘要加入 envelope 需要版本化迁移 |

`ProgramArtifact` dataclass 字段顺序为 `kind, payload, producer, version,
required_capabilities, parent_hashes, metadata`；序列化字段顺序为 `schema, version, kind,
producer, required_capabilities, parent_hashes, payload, metadata`。JSON hash 使用 key 排序，字段
声明顺序本身不影响摘要，但 dataclass 字段、默认值和序列化 shape 仍属于受保护行为。

## 5. metadata 值域对账

### 5.1 当前实际值域

当前 `_json_value()` 接受：

- `None`、`str`、`int`、`float`、`bool`；
- mapping，key 无条件转为字符串；
- tuple/list，统一冻结为 tuple、序列化为 list；
- 任何具有 callable `to_dict()` 的对象，并递归接收其结果。

其他对象（包括 raw `bytes`）在构造时抛 `TypeError`。非有限 float 会通过构造和
`to_dict()`，但在读取 `content_hash` 时因 `allow_nan=False` 抛 `ValueError`。数字 key 与同名
字符串 key 可能在字符串化时碰撞；调用任意对象的 `to_dict()` 也不是一个封闭、纯数据的
跨信任边界。

这与 `CircuitIR` 的值域不同：IR 还拥有 Parameter、ParameterExpression、complex 和 Tensor
编码。已经序列化后的 `CircuitIR.to_dict()` 可以作为 artifact payload，但把原始 Tensor、
complex 或 Parameter 直接放进 artifact metadata 不具备同样语义。

### 5.2 与其他团队需求的差异

- Ecosystem 要求外部 SDK 对象不能经 metadata 穿透；当前 `to_dict()` duck typing 只能保证
  最终对象被投影，不能保证投影方法安全或语义受控。
- Deployment compatibility 已实现 string-only key、finite float、深度、条目数、编码大小和
  sensitive fragment 检查，但它只检查 legacy deployment package，不能反向成为 Core
  metadata 的私有事实来源。
- Platform metadata 需要 JSON-safe 身份值，但现有 `PlatformRuntime` 仍可能返回 vendor
  handle；这些值不能直接塞入 artifact。
- Agent Services 当前完全不读取 artifact metadata，因此 metadata 中存在某个 key 不能证明
  capability、provenance 或 requirement 已被执行。

### 5.3 建议

Integration 应为下一版批准一个闭合的 canonical metadata algebra：字符串 key、有限 JSON
scalar、递归 mapping/list，并明确 namespace、最大深度/条目/字节、敏感字段和碰撞拒绝。
v1 reader 行为应通过兼容 adapter 保留；不能在 1.0 原地收紧后仍声称 hash 和读取兼容。

## 6. 版本与身份分层

当前至少存在六种不同身份，不能合并成一个 `hash` 字段：

| 身份 | 当前算法/来源 | 是否可映射到 ProgramArtifact v1 | 决策 |
| --- | --- | --- | --- |
| `CircuitIR.content_hash` | canonical IR JSON SHA-256 | 可从 circuit payload 重算 | payload identity，不能替代 envelope hash |
| `ProgramArtifact.content_hash` | 完整 v1 envelope SHA-256 | 原生 | envelope identity；当前不自带声明值 |
| `ImportedCircuitProgram.internal_program_identity` | module + constraints + instruction semantics | 不能从 ProgramArtifact v1 通用重算 | Compiler 私有派生身份，直到 Core internal-program schema 获批 |
| `TargetIR.target_program_identity` | target layout/ops/result/shots 与 capability/source refs | v1 无标准 profile | 需要 executable/physical payload profile 决策 |
| `SealedExecutableArtifact.artifact_identity` | profile、media type、payload hash、source/target/capability/compilation identities | 不能无损放入 role-less parents | 不能用 metadata 临时伪装为收敛；需 ADR |
| Deployment artifact digest | name、target、shots、format、program、routing evidence | 同时混合程序、请求和目标字段 | adapter 应拆分 program artifact、execution request 和 provider receipt |

`from_circuit_ir()` 生成的 envelope hash会随 producer、requirements、parents 或 metadata
变化，即使 IR payload 完全相同。这是正确的身份分层，不应要求 envelope hash 等于 IR hash。

`parent_hashes` 目前只证明字符串形状，不证明父对象存在，也不标明 source、target、compile、
calibration 或 payload 等角色。Compiler adapter 可以继续在私有对象中保留完整角色身份链；
在 Core 决定 typed identity references 之前，不应把这些身份按约定位置压入 `parent_hashes`。

## 7. provenance 对账

现有 v1 只有三个邻近来源字段：

- `producer`：一个参与 hash 的非空 label；
- `parent_hashes`：有序但无角色的摘要；
- `metadata`：自由、参与 hash、无 provenance namespace 规则。

它们不足以替代 Compiler 的：

- `SourceIdentity(schema_version, circuit_ir_content_hash)`；
- `SourceProvenance(values)`；
- instruction-level provenance 与 semantics；
- compilation、target capability、target program 和 artifact identities。

Compiler 的 `internal_program_identity` 明确排除了 request、provenance 和 bindings，而
ProgramArtifact v1 的 `content_hash` 包含全部 metadata、producer 和 requirements。两者的
identity inclusion policy 不同，不能通过字段改名直接等同。

最小适配原则：Compiler 保留私有 provenance/identity 对象，adapter 只在已批准的 Core
profile 中投影有明确角色的引用。新增一等 `provenance` 字段、决定其是否进入 envelope
identity、或者把现有 metadata 迁入该字段，都需要 ADR 和版本化兼容方案。

## 8. requirements 与 required_capabilities 对账

| 当前概念 | 表达内容 | 消费阶段 | 与 v1 的关系 |
| --- | --- | --- | --- |
| `ProgramArtifact.required_capabilities` | 排序去重的扁平字符串 | Agent planning 前 | v1 原生，但词汇来自当前 manifest 适配规则 |
| `CircuitIR` dtype/shape/measurements/observables | 程序和请求混合的现有语义 | Compiler/Runtime | 由 circuit payload 自身携带，不应重复成字符串 |
| Compiler `ImportConstraints` | dtype、shape、batch、logical state、runtime config | import | 结构化且进入 internal identity，不能无损降为字符串 |
| `TargetIR.required_results/requested_shots` | 结果类型与 shots | target lowering | 更接近 ExecutionRequest，不应塞入 artifact capability names |
| `TargetCapabilities` | 目标事实、limits、artifact profiles | compile/preflight | 描述“目标有什么”，不是“artifact 要什么” |
| Long-horizon `requirements` | precision、dynamic、communication、QEC、pulse、network | 跨 Compiler/Runtime/Provider | v1 尚不存在 |

Agent 的临时 capability vocabulary 会收集 backend 名、`supports_*`、accelerator、device、dtype
和 contract 名称；它不是封闭 Core vocabulary。`validate_program()` 当前忽略
`required_capabilities`，`plan_execution()` 才检查并返回
`REQUIRED_CAPABILITY_UNAVAILABLE`。本轮测试明确冻结该失败阶段。

建议保留 `required_capabilities` 作为 v1 coarse hints，并只允许未来结构化 requirement
adapter 向它做有文档的保守投影；不得从字符串集合反向恢复精度、拓扑、shots、校准或动态
控制要求。正式 `requirements` 的 shape、identity inclusion、与 TargetCapabilities 的匹配和
failure category 必须由 Integration/Core ADR 决定。

## 9. 各消费者最小适配方案

### 9.1 CircuitIR 与 Agent Services

现有 circuit 路径可继续：ProgramArtifact payload 保存 `CircuitIR.to_dict()`，Agent 在消费点
用 `CircuitIR.from_dict()` 严格校验。建议只补跨仓库 fixture 和 expected envelope hash，不改
实现。能力检查阶段保持 planning-time，除非 API 提案明确改变失败阶段。

### 9.2 Compiler source/import

Compiler 可通过 adapter 接受 `kind=circuit` 并取得严格 `CircuitIR`，随后继续生成私有
`ImportedCircuitProgram` 与 `SealedCircuitIRRoundTrip`。后两者含不可通用序列化的内部
module/bindings，不能反向塞进 v1 payload。adapter owner 为 Compiler；退出条件是 Compiler
公开跨领域入口只接收 Core artifact，而内部类型不越界。

### 9.3 Compiler executable artifact

当前 v1 payload 不接受 bytes，也没有 artifact profile/media type 和六段带角色 identity。
因此不存在无损、无需决策的 adapter。Integration 必须先决定 executable payload profile、
byte encoding/外部 blob reference、identity roles 和验证算法；Compiler 的 seal/verify 仍是
实现，不迁入 Core。

### 9.4 Deployment 与 Execution Provider

`DeploymentPackage` 同时携带 compiled IR、QASM/QCIS、shots、backend profile、routing
evidence 和 provider metadata。适配时必须拆分：程序及其 payload identity 属于
ProgramArtifact，shots/measurement 属于 ExecutionRequest，target 属于 TargetCapabilities，
receipt/job 属于 Provider。不能把整个 package 当成一个 executable ProgramArtifact。

### 9.5 Runtime、Simulation 与 Ecosystem

Runtime/Simulation 在正式 ExecutionRequest/plan profile 之前无需直接消费 ProgramArtifact；
它们不应添加私有复制。Ecosystem 继续先把外部对象转换为严格 `CircuitIR`，再由 Core helper
封装；不能把外部对象或未经 canonical metadata 验证的值直接写入 artifact。

## 10. 兼容性风险

| 风险 | 严重度 | 约束/验收 |
| --- | --- | --- |
| 为加入 provenance/requirements/extensions 直接增加顶层字段 | 高 | v1 reader严格拒绝未知字段；必须新版本或兼容 envelope 方案 |
| 把 `version` 改名为 `schema_version` | 高 | 所有现有 v1 payload 会缺字段；需要双读/迁移 fixture 与 API 提案 |
| 修改 hash 输入或 canonical JSON | 高 | envelope identity 全部变化；需要 golden fixtures 和迁移 identity |
| 将 metadata 原地收紧 | 高 | 现有可读 payload可能拒绝，且 stringified key/to_dict/NaN 失败阶段变化 |
| 把 metadata 当非语义扩展 | 高 | 它当前参与 content hash；展示或 transport 字段会改变身份 |
| 把 parent hash 位置解释为固定角色 | 高 | 当前允许重复且无角色；旧 payload无法证明解释正确 |
| 把 required capabilities 当完整 requirements | 高 | 会丢 precision、limit、topology、shots、evidence 和 calibration 语义 |
| 把 envelope hash 当 payload/executable identity | 高 | producer/metadata 等会改变 envelope hash，且 digest 未序列化 |
| 直接封装 executable bytes | 中高 | 当前构造明确拒绝 bytes；私有 base64 约定会形成第二 schema |
| 依赖当前 `str()` coercion | 中 | 数字 version/producer 可被 reader 接受；未来收紧需兼容测试 |
| 非字符串 metadata key 碰撞 | 中 | 下一版应拒绝；v1 兼容 reader需保留或显式迁移 |
| 在 validate 与 plan 间改变 capability failure stage | 中 | Agent 调用者可观察；需行为提案而非测试修补 |

## 11. Integration 最小决策请求

建议 Integration 按以下顺序只批准最小必要决定：

1. **Authority ADR 补充记录**：确认现有 `ProgramArtifact` 是唯一 envelope authority；v1 先
   维持 circuit 兼容读取，不新增平行 Artifact。
2. **Identity ADR**：确认 payload identity、envelope identity、compile/target/executable
   identity 分层；决定 typed parent references 和 transmitted expected digest。
3. **Metadata ADR**：为下一兼容版本批准闭合值域、namespace、limits、敏感数据和
   non-finite/key-collision 规则，同时明确 v1 reader 迁移期。
4. **Requirements ADR**：定义结构化 requirements、Core capability vocabulary、与
   TargetCapabilities 的 fail-closed matching；保留 v1 `required_capabilities` 的单向适配。
5. **Provenance ADR**：定义 source/tool/input/parent roles、identity inclusion 和 compile
   evidence 引用，不把 Runtime measured evidence 合并进 artifact。
6. **Executable profile ADR**：仅在前五项确定后决定 bytes/blob、media type、profile version
   和 `SealedExecutableArtifact` adapter。

这些决定可以拆成多个小 ADR，但必须在 Compiler、Deployment、Runtime 或外部服务迁移前
落地相同 fixture 和 contract fake。

## 12. API Change Proposal 与内部适配边界

以下修改必须提交 API Change Proposal，并同时获得 Integration/Core 契约批准：

- 将 `ProgramArtifact`、`ArtifactKind` 或版本常量加入稳定 root/core namespace；
- 改字段名、顺序、默认值、Enum 值、schema/version、未知/缺失字段或异常行为；
- 改 metadata/producer/required capabilities 的已接受值域或 normalization；
- 改 `content_hash` 算法、identity inputs 或 parent ordering；
- 新增 provenance/requirements/extensions 顶层字段并要求现有消费者接受；
- 改 Agent validate/plan 的 capability failure stage 或 artifact kind 支持范围。

如果 Integration 明确认定当前未稳定导出的 v1 仍处于 pre-public candidate 阶段，内部重构可
不走公开弃用周期，但 protected path、已序列化 payload 和跨仓库消费者仍必须有 Contract
Change Proposal、兼容 fixture 和明确迁移记录；不能把“未根导出”等同于可任意破坏。

以下工作在契约获批后可作为各责任团队的内部 adapter 重构完成：

- circuit artifact 与 `CircuitIR` 的严格双向投影；
- Compiler 在入口解包 circuit payload，同时保持私有 IR/provenance/identity 不越界；
- Deployment 将 package 拆为 artifact、request、target 和 receipt，而不改变公共行为；
- 用当前 v1 reader读取旧 payload，再投影到获批的新内部值对象；
- 增加 golden hash、旧 fixture、fake consumer 和 replacement conformance tests。

跨团队实现仍由对应团队完成；Core 本轮只提出契约请求，不越权修改消费者。

## 13. 本轮特征测试

`tests/team/core/test_program_artifact_reconciliation.py` 固化：

- dataclass 与序列化字段 shape、requirements normalization、round-trip 和全部 identity inputs；
- schema、未知/缺失字段、版本拒绝，以及当前 numeric version/producer coercion；
- metadata 的 JSON-like/`to_dict()` 值域、key stringification、opaque object 拒绝和 NaN 的
  identity-time failure；
- parent hashes 的格式、顺序、重复和非验证语义；
- CircuitIR payload identity 与 envelope identity 分层；
- executable raw bytes 当前不可表示；
- per-kind payload 校验由消费者执行；
- Agent validate 与 plan 对 required capabilities 的不同失败阶段；
- `provenance`、`requirements`、`extensions` 不是 v1 顶层字段并会被严格拒绝。

测试只记录现状和兼容性风险，不修改或重新解释受保护行为。

## 14. 验证记录

- Core 团队范围预检：通过；
- 架构边界检查：通过；
- Ruff check 与 format check：通过；
- ProgramArtifact、Core contract、Agent、Compiler executable artifact、Deployment
  compatibility 和 static pipeline 定向交叉集合：46 项通过；
- macOS 主机 `pr-default`：1833 项通过、14 项跳过、6 项失败。失败分别为既有 Compiler
  import/verify 性能预算、两个 Linux `/proc` watchdog 测试、两个宿主系统 Git 不可用导致的
  repository 检查，以及当前主机 PyTorch 的 `torch.dot` 浮点抵消差异；
- 使用 `flagquantum-dev:local` Linux 开发镜像、只读源码与完整 Git metadata 复跑
  `pr-default`：1840 项通过、12 项跳过、1 项失败。唯一失败仍为 Phase 0 已记录的
  `test_approved_import_verify_budget_is_machine_enforced`。

本轮没有修改性能阈值、快照、受保护实现或测试预期。Linux 开发容器结果只用于排除宿主
环境差异，不构成性能、硬件或可扩展性声明。
