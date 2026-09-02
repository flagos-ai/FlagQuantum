# IR-001：程序语义与执行请求边界

状态：Approved
日期：2026-09-01
适用阶段：Phase 1 importer 与受限 round-trip

批准记录：API owner 于 2026-09-01 通过明确指令批准 IR-001～003。批准范围仅限本文的
内部程序/执行请求边界，不修改公共 `CircuitIR`、`fq.plan`、`fq.run` 或 measurement
冲突规则；Phase 1 实现仍须满足 Phase 0 退出门。

## 背景

公共 `CircuitIR` schema 1.0 同时携带 instructions、observables 和 measurements。当前
planner/runtime 已定义唯一 request 来源和冲突规则；Interop 也明确把 observable 与
measurement 视为执行边界信息。内部 QuantumIR 需要区分程序本身与“本次执行返回
什么”，但不能修改公共 schema 或改变 `fq.plan/fq.run` 行为。

## 决策

`CircuitIR` importer 返回一个内部、不可公开的组合结果：

```text
ImportedCircuitProgram
  module: QuantumModule
  request: InternalExecutionRequest
  constraints: ImportConstraints
  source: SourceIdentity
  diagnostics: tuple[Diagnostic, ...]
```

归属规则：

- `n_wires` 和 `instructions` 进入 `QuantumModule`；
- `observables` 和 request 型 `measurements` 进入 `InternalExecutionRequest`；
- `dtype`、`shape`、batch/runtime 限制进入 typed `ImportConstraints`；
- schema version、源 content hash 进入 `SourceIdentity`；
- routing 等编译证据进入 provenance，不进入源 program semantic hash；
- instruction `is_channel`、`is_dynamic`、`condition` 等影响语义的 metadata 必须进入
  typed operation/diagnostic，不能降为普通 provenance。

Phase 1 静态范围内，dynamic instruction 返回结构化 unsupported diagnostic。显式
terminal measurement operation 与 execution request 的统一留到 ProgramIR/dynamic ADR，
Phase 1 不借机扩大能力。

## 保持的公共规则

- public `CircuitIR` 不变；
- `fq.plan(..., measurements=...)` 与 IR 内 measurements 冲突时继续失败；
- shots/seed/options 不进入 program identity；
- importer 不自行合并、替换或推断 request；
- `fq.run` 默认不调用 importer；
- 无法无损拆分时 fail closed。

## 否决方案

### 把 observables/measurements 全部留在 QuantumModule

否决原因：会继续混淆程序与执行请求，并使 shots 变化污染 program identity 和编译缓存。

### 修改 CircuitIR schema 1.0

否决原因：破坏受保护序列化合同，超出 Phase 0–1 授权。

### 用自由格式 metadata 携带 request

否决原因：无法验证类型、冲突、identity 和 round-trip。

## 兼容与回滚

该对象仅存在于 `_compiler`。删除 importer 即可回滚；公共对象、缓存、部署产物和用户
数据均不依赖它。

## 验收

- public API snapshot 不变；
- request/constraint/source 均有不可变 typed model；
- 当前 measurement 冲突测试继续通过；
- static round-trip 对支持范围精确；
- dynamic、未知语义 metadata 和 lossy 情况结构化失败；
- [x] API owner 批准架构决策；
- [ ] compiler owner 在实现评审中确认 typed model 与 importer 细节。
