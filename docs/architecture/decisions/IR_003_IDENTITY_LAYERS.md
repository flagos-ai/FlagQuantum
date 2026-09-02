# IR-003：Program、Compilation 与 Execution Identity 分层

状态：Approved
日期：2026-09-01
适用阶段：Phase 1 identity；Phase 2–3 cache/artifact 扩展

批准记录：API owner 于 2026-09-01 通过明确指令批准 IR-001～003。批准范围仅限 identity
分层和 Phase 1 内部 program identity 研究；不授权替换现有 ExecutionPlan、deployment、
checkpoint 或公共 result identity。Phase 2–3 使用仍需对应 owner 复核。

## 背景

当前 `CircuitIR.content_hash` 覆盖整个公共 payload，包括 measurements、metadata、dtype
和 shape；ExecutionPlan 与 deployment 也建立在现有 hash/contract 上。长期架构需要避免
仅改变 shots 就触发 placement/routing，同时必须避免 target、pipeline 或 capability
变化时复用错误 artifact。Phase 1 不能替换任何现有公共 identity。

## 决策

内部 identity 分三层：

```text
program_identity
  = canonical quantum program semantics
    + static parameter values that affect structure/semantics
    + semantic numerical constraints approved by schema

compilation_identity
  = program_identity
    + target identity
    + compiler/pipeline digest
    + compilation options
    + capability snapshot
    + calibration snapshot when required

execution_identity
  = artifact/compilation identity
    + runtime parameter bindings
    + execution request
    + shots/seed
    + execution policy
```

Phase 1 只实现内部 `program_identity`，并同时保存：

- `source_circuit_ir_hash`：现有公共 `CircuitIR.content_hash`；
- `internal_program_identity`：仅用于 differential evidence 和未来 cache 研究；
- `identity_schema_version`：内部版本，不对用户承诺兼容。

两者不得互相冒充。现有 ExecutionPlan、DeploymentPackage、checkpoint 和 public result
继续使用当前合同要求的 identity。

## Phase 1 program identity 规则

纳入：

- operation schema/version、顺序和嵌套结构；
- qubit/value dataflow；
- opcode、typed operands/results、参数和值；
- 影响量子或数值语义的 typed attributes；
- 经 ADR 明确属于 program semantics 的 dtype/precision constraint。

不纳入：

- source location、diagnostic、wall time；
- shots、seed、queue、credential、account；
- routing、target、compiler pipeline；
- 非语义 provenance/debug metadata；
- execution request，除非未来 ADR 明确某 terminal operation 是程序组成部分。

待 IR-001 metadata inventory 后仍无法分类的字段不得被静默排除。

## 确定性要求

- canonical ordering 不依赖 Python object address、dict insertion accident 或进程 seed；
- 相同输入、相同 importer/schema 版本产生相同 identity；
- semantic 变化必须改变 program identity；
- source location 和非语义 provenance 变化不得改变 program identity；
- identity 算法、schema version 和 canonical encoder 必须进入 evidence；
- Phase 1 不承诺跨版本内部 identity 稳定。

## 否决方案

### 直接复用 CircuitIR.content_hash 作为全部 identity

否决原因：当前公共 payload 混合 program、request、constraints 和 metadata，无法支持正确
的编译/执行缓存分层。

### Phase 1 立即替换 ExecutionPlan identity

否决原因：会改变已保护执行合同，且 TargetIR/artifact 尚未存在。

### 排除全部 metadata

否决原因：当前部分 metadata 影响 channel、dynamic、condition、runtime 和 routing 语义。

## 兼容与回滚

内部 identity 初期只写入测试/evidence，不进入公共 result、checkpoint、deployment 或
provider payload。删除内部 identity 不影响现有 cache key 和用户数据。

## 验收

- source hash 与 internal identity 明确并存；
- identity mutation tests 覆盖纳入/排除字段；
- 跨进程和固定平台确定性测试通过；
- 不修改现有 plan/deployment/checkpoint identity；
- [x] API owner 批准 identity 分层；
- [ ] compiler owner 在实现评审中确认 canonical encoding；
- [ ] runtime owner 在 Phase 2–3 前确认 compilation/execution identity 接入。
