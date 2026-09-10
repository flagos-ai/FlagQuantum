# Architecture Decisions

This directory records decisions requiring separate review during architecture
and multi-level IR implementation. ADR approval does not automatically change
Stable Core, complete a phase exit gate, or replace implementation tests,
performance evidence, and API Change Proposals.

## Overall Architecture

| ADR | Decision | Status | Approval scope |
| --- | --- | --- | --- |
| [ARCH-001](ARCH_001_COMPUTE_REMOTE_AND_CORE_CONTRACTS.md) | Compute, Remote, and Core contract ownership | Approved | Distinguish directly controlled compute from external task systems by control boundary. |
| [ARCH-002](ARCH_002_ARTIFACT_METADATA_AUTHORITY.md) | Artifact/metadata authority and compatibility | Proposed | Phase 1 contract candidate; no Stable Core changes. |
| [ARCH-003](ARCH_003_CAPABILITY_REQUIREMENT_DISCOVERY_EVIDENCE.md) | Requirements, snapshots, and evidence | Proposed | Minimal Phase 2 internal implementation and adapter authorization; no capability availability or Stable API change claims. |
| [ARCH-004](ARCH_004_EXECUTION_REQUEST_POLICY_BOUNDARY.md) | Execution Request and policy boundaries | Proposed | Phase 1 contract candidate; no execution entry changes. |
| [ARCH-005](ARCH_005_EXECUTION_RESULT_EVIDENCE_COMPATIBILITY.md) | Execution Result and Evidence compatibility | Proposed | Phase 1 contract candidate; no new result implementation. |
| [ARCH-006](ARCH_006_PROVIDER_ADMISSION_MODEL.md) | Platform/Execution Provider layers and admission | Superseded | Replaced by revised ARCH-001. |
| [ARCH-007](ARCH_007_SIMULATION_RUNTIME_PLANNING_BOUNDARY.md) | Planning information between Simulation and Runtime | Proposed | Phase 1 contract candidate; no current planner changes. |
| [ARCH-008](ARCH_008_MIGRATION_AND_VERIFICATION_PATHS.md) | Migration decisions and verification paths | Proposed | Phase 1 governance candidate; no implementation or retirement authorization. |

## Mapping Rules to Machine Gates

This table distinguishes existing and intended checks. Unimplemented checks are
not described as enabled. New machine contracts, CI, and protected configuration
still require separate integration changes.

| Rule | Machine source | Local check | CI/test entry | Current coverage |
| --- | --- | --- | --- | --- |
| Path ownership and integration-only ADR control | `team-ownership.toml` | `tools/check_team_scope.py --validate` | `tests/unit/test_team_scope_policy.py`, `.github/workflows/ci.yml` | Existing; new ADRs still require review. |
| Layers, forbidden reverse dependencies, and decreasing Runtime-to-Compiler debt | `architecture.toml` | `tools/check_architecture.py` | `tests/unit/test_issue073_architecture_boundaries.py`, `.github/workflows/ci.yml` | Boundaries exist; machine-readable migration exit records remain to be added. |
| Stable Core cannot change implicitly through implementation or snapshots | `contracts/public-api-*`, `docs/public_api_v1.json` | `tools/public_api_snapshot.py` | Public API contract tests, `.github/workflows/ci.yml` | Partly enabled; server-side protection follows the protection policy. |
| One source of truth for maturity and limitations | `capability-maturity.toml` | `tools/check_capability_maturity.py`, `tools/docs_source_of_truth.py --check` | `tests/unit/test_issue079_docs_source_of_truth.py`, `.github/workflows/ci.yml` | Existing; evidence-level schema awaits approval. |
| Distributed/hardware claims need proportionate evidence | Capability contracts, benchmark result JSON | `benchmarks/audit_results.py`, `tools/check_release_evidence_environment.py` | `tools/ci_tier.py pr-distributed/gpu-scheduled/multinode-scheduled/release`, `scheduled-hardware.yml` | Tiered gates exist; ARCH-005 field model awaits implementation. |
| Provider/result/capability replaceability needs conformance | Proposed Core contract fixtures | Proposed provider/result conformance suite | Proposed contract fakes and replacement tests | Not implemented; prerequisite for ARCH-003/005/006 admission. |
| Legacy is adapted, frozen, or retired through approval | Proposed migration ledger | Proposed importer/caller counting | Proposed no-new-caller, compatibility, and removal tests | Not implemented; ARCH-008 specifies required fields. |
| Local optimization need not cross five layers | Change classification and protected API diff | Team scope, architecture, and minimal relevant tests | Relevant `tools/ci_tier.py` tier | Existing principle; ARCH-008 defines three paths. |

## Multi-Level IR

| ADR | Decision | Status | Approval scope |
| --- | --- | --- | --- |
| [IR-001](IR_001_PROGRAM_REQUEST_BOUNDARY.md) | Program semantics versus execution requests | Approved | Internal importer and restricted round-trip. |
| [IR-002](IR_002_LINEAR_QUBIT_VALUES.md) | Linear QuantumIR qubit values | Approved | Internal value/verifier model. |
| [IR-003](IR_003_IDENTITY_LAYERS.md) | Program, Compilation, and Execution identities | Approved | Phase 1 internal program identity; no public identity replacement. |
| [IR-004](IR_004_CUSTOM_MATRIX_OPERATIONS.md) | Custom matrix operation boundary | Approved | Concrete unitary validation and round-trip. |
| [IR-005](IR_005_PARAMETERS_AND_TENSORS.md) | Parameters, tensors, and late binding | Approved | Internal binding and gradient preservation. |
| [IR-006](IR_006_PHASE1_REVERSIBLE_SCOPE.md) | Phase 1 reversible static scope | Approved | `circuit_ir_v1_static` profile. |

Approval record: the API owner explicitly approved IR-001 through IR-003 on
2026-09-01. Implementation remains subject to the Phase 0/1 exit gates in
[`MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md`](../MULTI_LEVEL_IR_PHASE_0_1_EXECUTION_PLAN.md).

The API owner explicitly approved IR-004 through IR-006 in a subsequent instruction
on the same day. Approval of all six ADRs covers internal architecture decisions
only. It neither completes Phase 0 nor authorizes Stable Core changes or switching
the default execution path.
