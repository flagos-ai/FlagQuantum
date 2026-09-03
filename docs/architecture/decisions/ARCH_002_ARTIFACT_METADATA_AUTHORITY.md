# ARCH-002：Artifact 与 metadata 权威及兼容边界

状态：Proposed

日期：2026-09-03
依据：Phase 0 八团队盘点；本提案不修改公共契约或实现

## 上下文

`CircuitIR` 是唯一规范程序表示；Core `ProgramArtifact`、Compiler
`SealedExecutableArtifact`、Deployment `DeploymentPackage` 分别承载部分身份链。Compiler、
Execution、Agent 与 Ecosystem 因而无法引用同一跨阶段 artifact。Qiskit metadata 还可能把
非 FlagQuantum 对象递归带入 `CircuitIR.metadata`。现有哈希、未知字段和版本拒绝规则不同，
不能通过改名或合并字典消除差异。

## 决策候选

1. Core 拥有版本化 artifact envelope；`CircuitIR` 保持规范语义 payload，不被 envelope
   替代。首版只表达 schema/kind/profile、payload 或其稳定引用、source/parent identity、
   requirements 和 namespaced extensions。
2. 可执行 artifact 是同一身份链上的窄 profile，而不是 Provider job、计划或结果。
3. Core 定义 metadata 的 JSON-safe 值代数、大小限制、namespace、未知键策略及 canonical
   encoding。展示字段不参与语义哈希；参与哈希的字段必须逐项声明。
4. 旧对象先经显式 adapter 投影；每个 adapter 记录 owner、输入/输出、损失性、移除条件和
   目标版本。

## 禁止事项

- 不建立第二个规范 IR，不把内部 `QuantumModule`、厂商 AST/SDK 对象或 live handle 放入 artifact。
- 不把 credentials、租户、队列状态、运行结果或自由格式 `Any` metadata 纳入身份链。
- 不改变现有 `CircuitIR`、`ProgramArtifact`、`ExecutionPlan` 或 Deployment schema/hash 来让
  新 fixture 通过；任何 Stable Core 变化另需 API Change Proposal。

## 兼容性

旧 schema 继续由原读取器处理。新 envelope 必须使用新版本并提供只读 adapter；未知 major、
未知顶层字段和非法 metadata fail closed。`ProgramArtifact.version` 与候选 `schema_version`
命名如何兼容仍未决定。旧 hash 作为 source identity 保留，禁止静默重算为新身份。

## 迁移顺序

1. 冻结旧 fixture、hash 和 importer 清单；定义 metadata 值代数。
2. 提交 Core 值对象、contract fake、往返与负向测试。
3. Compiler 适配 sealed artifact；Agent 适配 program artifact。
4. Execution/Deployment 适配 package，并把 live SDK 对象移至 provider 私有 session store。
5. 调用者归零后按台账退出重复 envelope。

## 验收测试

- canonical round-trip/hash、字段顺序无关、未知字段/版本和外部对象拒绝；
- `CircuitIR` identity 在 envelope 往返后不变；展示 metadata 不改变语义 fingerprint；
- Compiler fake 与真实 sealer 对同一输入产生兼容身份链；
- Deployment/Agent 消费者替换 adapter 时无需修改；旧 fixture 可继续读取。

## 未决问题

- payload 内联、content-addressed reference 与 tensor handle 的首版范围；
- executable profile/media type 注册与动态程序表示；
- metadata minor-version 扩展规则、容量上限及全局 phase 是否进入语义身份。
