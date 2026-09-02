# FlagQuantum IR Phase 1 完成记录

状态：**Approved / Complete**
日期：2026-09-02
正式授权：`approve IR-PHASE1-EXIT`

## 决议

API、compiler、runtime 与 training owner 已接受
`contracts/ir-phase1-exit-review-candidate.json` 中的技术证据，FlagQuantum 内部 IR
Phase 1 正式完成。

完成范围仅为私有 `circuit_ir_v1_static` profile：

- typed、immutable、verified QuantumIR 基础；
- 静态 CircuitIR importer 与 restricted exact round-trip；
- analysis/pass 最小合同；
- 显式测试差分桥；
- state、expectation、measurement、ordering、dtype、custom matrix 与 gradient parity；
- 已批准 CPU 性能预算和默认路径零影响证明。

## 未授权事项

本决议不授权：

- Phase 2 或 ProgramIR/TargetIR 实现；
- 公共 IR、PassManager 或插件 API；
- 默认 `fq.run/fq.plan` 或生产路径切换；
- Provider/native codegen；
- 迁移现有 optimizer/compiler/runtime；
- GPU、分布式或 QPU 性能与成熟度声明。

## 当前工程状态

Phase 1 仍位于私有 `_compiler`，用户无需迁移代码。默认路径没有依赖它；如需撤回，可删除
`_compiler` 及其内部测试和 benchmark，而不改变 Stable Core、CircuitIR 1.0、legacy
compiler/runtime 或用户数据。

进入 Phase 2 必须提出新的范围、兼容性、性能和回滚方案，并获得独立授权。
