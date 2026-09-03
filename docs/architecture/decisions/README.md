# Architecture Decisions

本目录记录总体架构及多层 IR 实施过程中需要独立评审的架构决策。ADR 批准不自动改变 Stable
Core，不自动完成 Phase 退出门，也不替代实现测试、性能证据和 API change proposal。

## 总体架构

| ADR | 决策 | 状态 | 批准范围 |
| --- | --- | --- | --- |
| [ARCH-001](ARCH_001_PROVIDER_LAYERS_AND_CORE_CONTRACTS.md) | Provider 分层与 Core 契约所有权 | Approved | vNext 目标边界、迁移台账与新增依赖约束 |
| [ARCH-002](ARCH_002_ARTIFACT_METADATA_AUTHORITY.md) | Artifact 与 metadata 权威及兼容边界 | Proposed | Phase 1 契约候选；不修改 Stable Core |
| [ARCH-003](ARCH_003_CAPABILITY_REQUIREMENT_DISCOVERY_EVIDENCE.md) | Capability requirement、snapshot 与 evidence 边界 | Proposed | Phase 2 最小内部实现与 adapter 授权；不代表能力可用或 Stable API 变更 |
| [ARCH-004](ARCH_004_EXECUTION_REQUEST_POLICY_BOUNDARY.md) | Execution Request 与 policy 边界 | Proposed | Phase 1 契约候选；不修改执行入口 |
| [ARCH-005](ARCH_005_EXECUTION_RESULT_EVIDENCE_COMPATIBILITY.md) | Execution Result 与 Evidence 兼容边界 | Proposed | Phase 1 契约候选；不新增结果实现 |
| [ARCH-006](ARCH_006_PROVIDER_ADMISSION_MODEL.md) | Platform/Execution Provider 两层模型与准入依赖 | Proposed | Phase 1 契约候选；不认证 Provider |
| [ARCH-007](ARCH_007_SIMULATION_RUNTIME_PLANNING_BOUNDARY.md) | Simulation 与 Runtime 的规划信息边界 | Proposed | Phase 1 契约候选；不改变当前 planner |
| [ARCH-008](ARCH_008_MIGRATION_AND_VERIFICATION_PATHS.md) | 迁移决策账本与快速/标准/认证路径 | Proposed | Phase 1 治理候选；不授权实现或退役 |

## 规则到机器门禁的追踪

本表描述当前或预期的验证落点，不把尚未实现的检查写成已启用。新增机器合同、CI 或受保护
配置仍需单独集成变更。

| 自然语言规则 | 机器事实来源 | 本地检查入口 | CI/测试入口 | 当前覆盖 |
| --- | --- | --- | --- | --- |
| 路径所有权与 ADR 由 Integration 独占 | `team-ownership.toml` | `tools/check_team_scope.py --validate` | `tests/unit/test_team_scope_policy.py`、`.github/workflows/ci.yml` | 已有；新 ADR 仍需评审 |
| 分层、禁止反向依赖及 Runtime→Compiler 债务只能减少 | `architecture.toml` | `tools/check_architecture.py` | `tests/unit/test_issue073_architecture_boundaries.py`、`.github/workflows/ci.yml` | 已有边界；迁移退出条件待补机器台账 |
| Stable Core 不得随实现或快照隐式变化 | `contracts/public-api-*`、`docs/public_api_v1.json` | `tools/public_api_snapshot.py` | 公共 API contract tests、`.github/workflows/ci.yml` | 部分启用；服务端保护仍按保护政策推进 |
| 能力成熟度与限制只有一个事实来源 | `capability-maturity.toml` | `tools/check_capability_maturity.py`、`tools/docs_source_of_truth.py --check` | `tests/unit/test_issue079_docs_source_of_truth.py`、`.github/workflows/ci.yml` | 已有；证据分级 schema 待批准 |
| 分布式/硬件声明必须有相称证据 | capability contract、benchmark result JSON | `benchmarks/audit_results.py`、`tools/check_release_evidence_environment.py` | `tools/ci_tier.py pr-distributed/gpu-scheduled/multinode-scheduled/release`、`scheduled-hardware.yml` | 已有分级门禁；ARCH-005 字段模型待实施 |
| Provider、结果和能力可替换性必须由 conformance 证明 | 拟议 Core contract fixtures | 拟议 provider/result conformance suite | 拟议 contract fake + replacement tests | 未实现；ARCH-003/005/006 的准入前置 |
| legacy 只能适配、冻结或按批准流程退役 | 拟议迁移决策台账 | 拟议 importer/caller 计数检查 | 拟议 no-new-caller、compatibility、removal tests | 未实现；ARCH-008 定义所需字段 |
| 局部优化不应被迫跨五层 | 变更分类与受保护 API diff | 团队范围、架构及最小相关测试 | `tools/ci_tier.py` 对应层级 | 原则已有；ARCH-008 明确三条路径 |

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
