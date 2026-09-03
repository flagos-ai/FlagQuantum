# ARCH-002：Artifact 与 metadata 权威及兼容边界

状态：Proposed

日期：2026-09-03

依据：Phase 0 八团队盘点及 Core `ProgramArtifact` Phase 1 对账；本提案不修改公共契约或实现

## 上下文

`CircuitIR` 是唯一规范程序表示。现有 Core
`flagquantum.core._artifacts.ProgramArtifact` v1（schema
`flagquantum.program_artifact`、version `1.0`）已经是仓库唯一 artifact envelope 权威。
Compiler `SealedExecutableArtifact`、`SealedCircuitIRRoundTrip` 与 Deployment
`DeploymentPackage` 分别承载更专门且更丰富的身份、payload 和请求信息；它们不是第二个
envelope 权威，但 v1 当前也不能无损替代它们。

v1 当前唯一实际运行的生产消费链是 Agent Services 的 `kind=circuit` 路径。Compiler、
Runtime、Simulation、Execution Provider 和 Ecosystem 尚未直接消费 `ProgramArtifact`。
因此“已有唯一权威”不等于“全部消费者已经迁移”，更不等于 executable/deployment 能力已实现。

## 决策候选

1. 现有 `ProgramArtifact` v1 继续作为唯一 artifact envelope 权威。任何 v2 都是同一契约
   谱系经 API Change Proposal、兼容分析和迁移 fixture 的版本演进；不得并行创建第二套
   envelope、另一个通用 Artifact 类型或私有跨层替代 schema。
2. v1 当前权威范围只包括 envelope 字段、严格顶层读取、版本/种类拒绝、完整 envelope
   identity，以及已运行的 `kind=circuit` 消费路径。它不能无损表达 executable bytes、payload
   profile/media type、带角色 identity、结构化 requirements/provenance/extensions，亦不能无损
   替代 `SealedExecutableArtifact`、`SealedCircuitIRRoundTrip` 或 `DeploymentPackage`。
3. v1 `content_hash` 是对完整 `to_dict()` canonical JSON 重算得到的 envelope identity；
   `producer`、`required_capabilities`、有序 `parent_hashes`、`payload` 和 `metadata` 均参与现有
   identity。它不是 payload、IR、compile、target 或 deployment identity。
4. `content_hash` 当前是计算属性，不作为声明字段随 envelope 传输；接收方从接收到的完整
   envelope 重算。未来若传输 expected digest、改变 hash 输入或区分语义/展示字段，必须以同一
   契约谱系的新版本和显式迁移完成，不能静默改变旧 hash。
5. v1 metadata 行为先冻结为兼容读取事实。未来候选规则可以定义闭合值代数、namespace、
   容量限制、敏感字段和哪些新字段不参与某类语义 fingerprint；这不是 v1 当前行为，也不能
   原地改变 v1 envelope identity。

### v1 字段事实

- `producer` 只是参与 envelope hash 的非空 opaque label；它不是 provenance、tool identity
  或受控词汇。
- `required_capabilities` 是排序去重的粗粒度兼容提示；当前只有 Agent planning 消费，不能
  替代结构化编译/执行 requirements，也不能由它反推出精度、拓扑、shots 或校准要求。
- `parent_hashes` 是保留顺序和重复项的 opaque lineage；只校验小写 SHA-256 形状，不校验父
  对象存在性、hash 种类或位置角色。
- `metadata` 当前参与完整 envelope identity，但值域未闭合：key 会被字符串化，并允许调用
  任意对象的 callable `to_dict()` 后递归接收。它没有 namespace、深度、条目数、编码容量或
  敏感字段约束，不能声称为安全跨信任边界。

## 禁止事项

- 不建立第二个规范 IR、第二套 artifact envelope 或与 `ProgramArtifact` v1 并列的 v2 类型。
- 不把内部 `QuantumModule`、厂商 AST/SDK 对象、live handle、credentials、租户或队列状态
  塞入 v1 payload/metadata 来宣称收敛。
- 不把 v1 `producer` 解释为 provenance，不把 `required_capabilities` 解释为完整 requirements，
  不给 `parent_hashes` 的位置追认角色。
- 不声称 v1 metadata 已闭合或安全；不调用外部对象 `to_dict()` 后便把结果视为可信。
- 不将 v1 metadata 描述为“不参与 hash”的展示字段，不改变 canonical JSON/hash 输入，也不
  把接收方重算的 `content_hash` 误述为 envelope 中传输的发送方声明。
- 不改变现有 `CircuitIR`、`ProgramArtifact`、`ExecutionPlan` 或 Deployment schema/hash 来让
  新 fixture 通过；任何受保护行为变化另需 API Change Proposal。

## 兼容性

v1 reader、字段 shape、coercion、未知字段/版本拒绝和完整 envelope hash 保持原状。未来 v2
必须沿用 `ProgramArtifact` 契约谱系，经 API Change Proposal 决定同一 reader 的版本分派、
旧 payload 读取、identity 对照和迁移方式；不得以新类、新 namespace 或 metadata 私约绕开。

旧 v1 `content_hash` 必须作为 envelope identity 保留。即使 `CircuitIR` payload 相同，producer、
requirements、parents 或 metadata 不同也会使 envelope hash 不同；adapter 不得把它静默重算
为 payload/executable identity。

## 迁移顺序

1. 冻结 v1 golden fixture、完整 envelope hash、字段/coercion 行为和直接消费者清单。
2. 先批准 identity 分层、metadata 闭合值代数/namespace/容量和 v1 兼容读取方案。
3. 再批准结构化 requirements/provenance、typed parent identity 和 executable payload profile；
   如需 v2，作为同一契约谱系的版本演进提交 API Change Proposal。
4. Compiler 先从 `kind=circuit` adapter 解包，再保留私有内部身份；不可表示的 executable
   信息不得损失性塞回 v1。
5. Deployment 将 package 拆为 artifact、request、target 和 receipt 后分别适配；调用者归零、
   兼容窗口和替换测试齐备后才处理旧对象退出。

## 验收测试

- v1 canonical round-trip/golden hash、字段顺序无关、未知字段/版本和现有 coercion 行为；
- 明确断言 metadata、producer、required capabilities、ordered parents 和 payload 都参与 v1
  envelope hash，且 `content_hash` 不出现在 `to_dict()` 中而由接收方重算；
- `CircuitIR.content_hash` 与 `ProgramArtifact.content_hash` 分层，前者不被后者替代；
- parent 顺序/重复、opaque producer、coarse requirements 和 metadata `to_dict()` duck typing
  的当前行为由 characterization fixture 固定；
- raw executable bytes、`provenance`、`requirements`、`extensions` 在 v1 中不可表示或被拒绝；
- 文档测试阻止把 v2 写成第二权威，或把 v1 metadata/hash 误述为未来候选语义。

## 未决问题

- 同一契约谱系的 v2 如何分派、双读、传输 expected digest 并关联 v1 identity；
- executable bytes/blob reference、profile/media type 与带角色 identity 的首版范围；
- metadata 闭合值代数、namespace、深度/条目/字节上限、敏感字段和 key collision 迁移；
- 哪些未来字段进入 envelope identity，是否另设不替代 v1 hash 的语义 fingerprint。
