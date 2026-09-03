# Architecture Decisions

本目录记录总体架构及多层 IR 实施过程中需要独立评审的架构决策。ADR 批准不自动改变 Stable
Core，不自动完成 Phase 退出门，也不替代实现测试、性能证据和 API change proposal。

## 总体架构

| ADR | 决策 | 状态 | 批准范围 |
| --- | --- | --- | --- |
| [ARCH-001](ARCH_001_PROVIDER_LAYERS_AND_CORE_CONTRACTS.md) | Provider 分层与 Core 契约所有权 | Approved | vNext 目标边界、迁移台账与新增依赖约束 |
| [ARCH-002](ARCH_002_ARTIFACT_METADATA_AUTHORITY.md) | Artifact 与 metadata 权威及兼容边界 | Proposed | Phase 1 契约候选；不修改 Stable Core |
| [ARCH-003](ARCH_003_CAPABILITY_REQUIREMENT_DISCOVERY_EVIDENCE.md) | Capability 的需求、发现与证据三分法 | Proposed | Phase 1 契约候选；不代表能力可用 |
| [ARCH-004](ARCH_004_EXECUTION_REQUEST_POLICY_BOUNDARY.md) | Execution Request 与 policy 边界 | Proposed | Phase 1 契约候选；不修改执行入口 |
| [ARCH-005](ARCH_005_EXECUTION_RESULT_EVIDENCE_COMPATIBILITY.md) | Execution Result 与 Evidence 兼容边界 | Proposed | Phase 1 契约候选；不新增结果实现 |
| [ARCH-006](ARCH_006_PROVIDER_ADMISSION_MODEL.md) | Platform/Execution Provider 两层模型与准入依赖 | Proposed | Phase 1 契约候选；不认证 Provider |

## 多层 IR

| ADR | 决策 | 状态 | 批准范围 |
| --- | --- | --- | --- |
| [IR-001](IR_001_PROGRAM_REQUEST_BOUNDARY.md) | 程序语义与执行请求边界 | Approved | 内部 importer 与受限 round-trip |
| [IR-002](IR_002_LINEAR_QUBIT_VALUES.md) | QuantumIR 线性 qubit value | Approved | 内部 value/verifier 模型 |
| [IR-003](IR_003_IDENTITY_LAYERS.md) | Program、Compilation、Execution identity 分层 | Approved | Phase 1 内部 program identity；不替换公共 identity |
| [IR-004](IR_004_CUSTOM_MATRIX_OPERATIONS.md) | 自定义 matrix operation 边界 | Approved | concrete unitary 验证与 round-trip |
| [IR-005](IR_005_PARAMETERS_AND_TENSORS.md) | 参数、tensor 与 late binding | Approved | 内部 binding 与 gradient 保持 |
| [IR-006](IR_006_PHASE1_REVERSIBLE_SCOPE.md) | Phase 1 可逆静态范围 | Approved | `circuit_ir_v1_static` profile |

批准记录：API owner 于 2026-09-01 通过明确指令批准 IR-001～003。实现仍须遵守
[`MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md`](../MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md)
中的 Phase 0/1 退出门。

API owner 于同日通过后续明确指令批准 IR-004～006。六份 ADR 的批准均只覆盖内部架构
决策，不自动完成 Phase 0，也不授权修改 Stable Core 或切换默认执行路径。
