# Development

Policies and workflows for changing, testing, and releasing FlagQuantum.

- [Python engineering standard](PYTHON_ENGINEERING_STANDARD.md)
- [Testing manual](TESTING.md)
- [CPU statevector performance characteristics](CPU_STATEVECTOR_PERFORMANCE_CHARACTERISTICS.md)
- [Code organization](CODE_ORGANIZATION.md)
- [Repository governance](REPOSITORY_GOVERNANCE.md)
- [Integration workflow](INTEGRATION_WORKFLOW.md)
- [Multi-team linked-worktree development](MULTI_TEAM_DEVELOPMENT.md)
- [Multi-team handoff template](TEAM_HANDOFF_TEMPLATE.md)
- [Public API protection](PUBLIC_API_PROTECTION.md)
- [Stable Core API change proposal](API_CHANGE_PROPOSAL_001_STABLE_CORE.md)
- [Dependency policy](DEPENDENCY_POLICY.md)
- [Version and release policy](RELEASE_POLICY.md)
- [Git history recovery](GIT_HISTORY_RECOVERY.md)

The list above is the policy core. Most files in this directory are recorded
delivery artifacts rather than current policy, and they are grouped by family:

- `API_CHANGE_PROPOSAL_0*.md` (64 files, including the 001 listed above) — one
  proposal per API decision, in numeric order. Start from
  `API_CHANGE_PROPOSAL_001_STABLE_CORE.md` and follow
  the [public API protection](PUBLIC_API_PROTECTION.md) process before reusing a
  proposal as authority.
- `IR_PHASE_*.md` (34 files) — internal IR phase plans, reviews, and approval
  packets. The current state is summarized in
  [compiler and IR implementation map](IR_IMPLEMENTATION_STATUS.md).
- `HYBRID_COMPILATION_PHASE*_EVIDENCE.md` (46 files) — per-phase compilation
  evidence records. The contract they support is
  [private hybrid compilation contract](HYBRID_COMPILATION_PRIVATE_CONTRACT.md).
- `team-readiness/` — per-domain migration inventories, boundary handoffs, and
  target-capability reconciliations.

Remaining current documents:

- [Compiler and IR implementation map](IR_IMPLEMENTATION_STATUS.md)
- [CPU vertical slice](CPU_VERTICAL_SLICE.md)
- [IR metadata inventory](IR_METADATA_INVENTORY.md)
- [Private hybrid compilation contract](HYBRID_COMPILATION_PRIVATE_CONTRACT.md)
- [Interoperability adapter development](INTEROP_ADAPTERS.md)
- [Qiskit seeded differential conformance](QISKIT_DIFFERENTIAL_CONFORMANCE.md)
- [Qiskit arithmetic parameter-expression assimilation](QISKIT_PARAMETER_EXPRESSION_ASSIMILATION.md)
- [Cirq interoperability contract](CIRQ_INTEROP_CONTRACT.md)
- [Amazon Braket circuit interoperability contract](BRAKET_CIRCUIT_INTEROP_CONTRACT.md)
- [CUDA-Q export contract](CUDAQ_EXPORT_CONTRACT.md)
- [PennyLane differential conformance](PENNYLANE_DIFFERENTIAL_CONFORMANCE.md)
- [Executable artifact deployment dry run](ARTIFACT_DEPLOYMENT_DRY_RUN.md)
- [Jiuding direct program submission](JIUDING_PROGRAM_SUBMISSION.md)
- [Stage 2 Quafu minimum real vertical workflow entry audit](STAGE_2_QUAFU_ENTRY_AUDIT.md)
- [Physical resource allocation v3 completion review](PHYSICAL_RESOURCE_ALLOCATION_V3_COMPLETION_REVIEW.md)
- [Twin convergence completion](TWIN_CONVERGENCE_COMPLETION.md)
- [FlagQuantum vNext Phase 0 integration report](VNEXT_PHASE0_INTEGRATION_BOARD.md)
- [FlagQuantum vNext Phase 1 integration report](VNEXT_PHASE1_INTEGRATION_BOARD.md)
- [vNext Phase 2 target capabilities integration decision](VNEXT_PHASE2_CAPABILITIES_DECISION.md)
- [Maintainability cleanup 2026-09-09](MAINTAINABILITY_CLEANUP_20260909.md)

Read the repository-level [contribution guide](../../CONTRIBUTING.md) and
[agent operating manual](../../AGENTS.md) before making changes.

## Writing README files

Write for the reader's next decision, not a mandatory document template.

| Location | Reader's question | Include |
| --- | --- | --- |
| Repository root | What can I build, and where do I start? | Product purpose, a complete small example, and documentation links. |
| Module | What belongs here, and how do I change it? | Design intent, ownership, key entry points, and focused verification. |
| Examples and operations | How do I run this correctly? | Prerequisites, commands, expected results, and recovery or interpretation notes. |
| Evidence and decisions | What happened, and what does it establish? | Provenance, recorded facts, approval scope, and claim boundaries. |

Keep a module's shortest usage or change path in its README. Split out a guide
only when detailed protocol, migration, or experiment material would obscure
that path; do not create a second file for a short source map.

Describe unfinished capabilities as design goals and give them behavioral
acceptance criteria. Runnable examples must use existing interfaces; speculative
API sketches must be labeled non-executable. A design document neither changes
protected APIs nor establishes hardware support.

Link to the capability catalog for support levels. Preserve historical numbers
and approval records as facts. Prefer specific checks such as gradient agreement
or restart equivalence over repeated claims of rigor and completeness.
