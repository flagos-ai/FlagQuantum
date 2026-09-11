# FlagQuantum Multilevel IR Phase 0–1 Execution Plan

Status: historical implementation design; current implementation status is in
[`IR_IMPLEMENTATION_STATUS.md`](../development/IR_IMPLEMENTATION_STATUS.md).
Governing design: [`MULTI_LEVEL_IR_ARCHITECTURE.md`](MULTI_LEVEL_IR_ARCHITECTURE.md).
Scope: current semantic baseline, internal QuantumIR skeleton, differential
verification, performance baselines.
Explicitly unchanged: Stable Core, `CircuitIR` schema 1.0, `fq.plan`, `fq.run`,
deployment contracts.

Progress: P0-001/P0-002 factual inventories are complete. P0-003 has a 35-opcode
baseline corpus, metadata inventory, drift tests, and an in-progress evidence
manifest. The API owner approved IR-001–006 architecture scope. Typed metadata
destinations, legacy gradient/statistical oracles, the Phase 0 CPU performance
baseline, and Phase 1 internal budgets are complete and approved. Implementation
owner reviews remain incomplete, so Phase 1 code is not yet authorized. Technical
remediation completed the manifest oracle dispatcher, 12 positive and 14 negative
fixtures, and fixed default precision overriding CircuitIR complex128 constraints.
Remaining exit items are owner sign-off and explicit authorization. See
[`IR_PHASE_0_BASELINE.md`](../development/IR_PHASE_0_BASELINE.md),
[`IR_PHASE_0_CORPUS_DESIGN.md`](../development/IR_PHASE_0_CORPUS_DESIGN.md), and
[`decisions/`](decisions/). The original 2026-09-01 exit audit reproduced the
machine baseline while manifest oracles, owner review, and authorization remained
open; current conclusions/blockers are in
[`IR_PHASE_0_EXIT_AUDIT.md`](../development/IR_PHASE_0_EXIT_AUDIT.md). Historical
phase approvals no longer serve as code contracts.

## 1. Implementation Goal

Phase 0–1 proves that current static `CircuitIR` can enter a verifiable internal
QuantumIR without public API changes, scientific semantic changes, or unacceptable
fast-path regressions.

Creating IR classes is not completion. Completion requires:

- Reproducible behavior and performance baselines.
- Fail-closed importer support boundaries.
- Deterministic internal types, values, operations, modules, and diagnostics.
- Passing old/new state, expectation, measurement, and gradient differentials.
- Unchanged public APIs, serialization schemas, and default execution paths.
- Machine-checked Phase 1 budgets derived from Phase 0 measurements.
- A removable new path that does not affect legacy execution.

## 2. Scope

### 2.1 Included

- Inventory actual CircuitIR consumption by compiler, planner, runtime, and deployment.
- Fix a static-circuit characterization corpus.
- Establish legacy/new-path differential infrastructure.
- Build internal Python QuantumIR core, schema registry, verifier, diagnostics.
- Implement `CircuitIR -> internal QuantumIR` import.
- Provide a restricted `QuantumIR -> CircuitIR` test export for the reversible subset.
- Establish minimum PassManager/AnalysisManager contracts.
- Measure construction, verification, hashing, and round-trip time/memory.
- Produce Phase 2 entry evidence and architecture decisions.

### 2.2 Excluded

- No new `fq.compile` or stable root exports.
- No public ProgramIR, QuantumIR, TargetIR, PassManager, or AnalysisManager.
- No CircuitIR field, version, JSON, hash, or exception changes.
- No default routing of `fq.run`, `fq.plan`, or `compile_for_backend` through the new pipeline.
- No generic functions, recursion, full SSA, dynamic control flow, timing, pulses, or QIR.
- No replacement of routing, providers, deployment, or distributed runtime.
- No C++, MLIR, or new runtime dependencies.
- No debug textual IR as persistence/compatibility contract.
- No GPU, distributed, or QPU performance claims from CPU tests.

## 3. Architecture Decisions Required First

Phase 0 may proceed before these ADRs are approved; corresponding Phase 1
implementation may not.

| ADR | Required decision | Default recommendation | Blocks |
| --- | --- | --- | --- |
| IR-001 | Split CircuitIR observables/measurements | Import returns internal `ImportedProgram` with `module` and typed execution requests | Importer, round trip |
| IR-002 | Phase 1 qubit model | Linear values: each quantum operation consumes old values and produces new ones | Values, verifier, control-free lowering |
| IR-003 | Internal identity | Separate program/compilation/execution identities; implement program identity only in Phase 1 | Hashing, cache, evidence |
| IR-004 | Custom matrix operations | Preserve restricted typed custom-unitary operations; reject unverifiable cases | Importer, round trip |
| IR-005 | Parameters/tensor constants | Preserve Parameter/ParameterExpression semantics; no implicit binding or dtype downgrade | Importer, gradient parity |
| IR-006 | Phase 1 reversible scope | Current static CircuitIR subset only; dynamic metadata does not establish support | Round trip, diagnostics |

Each ADR includes context, alternatives, decision, rejected options, compatibility,
test impact, and rollback.

## 4. Target Directory Boundaries

Proposed Phase 1 internal layout. Names may change during IR-001–006 review;
dependency directions may not reverse.

```text
flagquantum/
  _compiler/
    __init__.py              # No implementation-symbol exports
    diagnostics.py
    identity.py
    ir/
      __init__.py
      types.py
      values.py
      operations.py
      modules.py
      schemas.py
      verifier.py
      printer.py             # Debug-only canonical printer
    importers/
      circuit_ir.py
    exporters/
      circuit_ir.py          # Restricted round trips and tests only
    analyses/
      base.py
      def_use.py
      qubit_lifetime.py
    passes/
      base.py
      manager.py
```

Tests:

```text
tests/internal_ir/
  test_types_and_values.py
  test_operation_schema.py
  test_verifier_negative.py
  test_circuit_ir_import.py
  test_circuit_ir_round_trip.py
  test_identity_determinism.py
  test_pass_manager.py
  test_semantic_differential.py
  test_gradient_differential.py
  test_performance_budget.py
```

Rules:

- No `_compiler` exports from the root, stable facades, or `__all__`.
- Runtime may consume internal IR under explicit experimental controls; internal
  IR does not depend on Runtime.
- Importers may depend on public `core.ir`, never the reverse.
- No providers, credentials, queues, or network clients in `_compiler/ir`.
- Do not update public API snapshots to accept new symbols.

## 5. Phase 0: Factual Baseline

### P0-001 Public Contract Protection Inventory

Record in `docs/development/IR_PHASE_0_BASELINE.md`:

- Stable root exports and protected candidate contracts.
- `IR_VERSION` and CircuitIR serialization/deserialization behavior.
- Content hashes and failure behavior for unknown fields/opcodes and invalid wires.
- Current `fq.plan`, `fq.run`, `compile_for_backend` signatures and key semantics.
- Protected contract files and baseline hashes.

Acceptance: existing API contracts, snapshots, and IR tests pass. The baseline
records facts without updating protected snapshots.

### P0-002 Consumer Matrix

Record field reads, writes, and assumptions for:

- Circuit and parameters.
- Compiler canonicalization.
- Planner, routing, execution-plan builders.
- Statevector, MPS, TN, noise, distributed runtime.
- Drawer and QASM/QCIS emitters.
- Dynamic import/export.
- Deployment packages and providers.

Classify each as `semantic`, `execution_request`, `planning_hint`, `provenance`, or
`legacy_metadata_dependency`. Unclassified metadata dependencies block Phase 1.

### P0-003 Characterization Corpus

Fixed deterministic static circuits cover at least:

- Every currently supported canonical opcode.
- Parameterized one-/two-qubit gates and ParameterExpression.
- Custom matrices.
- Observables, terminal measurements, shots, wire order.
- Batch shapes and complex64/complex128.
- Empty metadata, valid provenance metadata, rejection cases.
- Topology-sensitive circuits.
- VQE/QML gradient circuits.
- Serialization round trips.

Machine-readable fixtures are authoritative. Do not generate the sole truth
transiently from random tests. Seed random property tests and keep them separate.

### P0-004 Differential Oracles

Define a common comparison protocol:

| Output | Comparison |
| --- | --- |
| CircuitIR round trip | Exact canonical dict/JSON and content hash |
| Statevector | Dtype contracts with explicit global-phase treatment |
| Expectation | Existing backend/dtype tolerances |
| Samples/counts | Fixed-seed determinism contract or preapproved statistical test |
| Gradient | Both forward values and parameter gradients; forward-only parity is insufficient |
| Wire/result order | Exact equality |
| Fallback/backend | Exact requested/selected/actual values and blockers |

Do not invent one global tolerance for every backend.

### P0-005 Performance and Resource Baseline

Internal benchmarks separately measure 10, 100, 1K, and 10K gates:

```text
circuit_to_ir_ms
legacy_compile_ms
legacy_plan_ms
legacy_run_cold_ms
legacy_run_warm_ms
peak_host_memory_bytes
serialized_ir_bytes
```

Each record includes environment, Python, Torch, FlagQuantum commit, CPU/device,
dtype, warmup, iterations, and seed. Phase 0 establishes budget inputs, not claims
that new IR is faster.

### P0-006 Phase 1 Performance Budgets

Derive internal budgets from repeated reproducible P0-005 measurements and obtain
human approval. Constrain separately:

- Small-circuit fixed overhead.
- Importer growth with gate count.
- Verifier growth with values/operations.
- Canonical hash determinism/time.
- Round-trip peak memory.
- No impact of opt-in paths on default `fq.run`.

Without approved measured budgets, the Phase 1 gate remains unsatisfied. Arbitrary
percentages are not substitutes.

### Phase 0 Exit Gate

- [ ] P0-001–P0-006 complete.
- [ ] IR-001–IR-006 approved.
- [ ] Corpus covers every current static canonical opcode.
- [ ] Legacy semantics/performance reproduce in clean Docker.
- [ ] Public APIs and serialization unchanged.
- [ ] Blockers, owners, dates, and rollback recorded.
- [ ] Explicit API-owner authorization for Phase 1.

## 6. Phase 1: Internal QuantumIR Skeleton

### P1-001 Types, Values, and Modules

Implement minimum immutable models:

- `IRType` and required concrete types.
- Deterministic `ValueId` and `ValueRef`.
- `Operation`, `Block`, `Region`, `QuantumModule`.
- Frozen attributes.
- Module revision/program identity.
- Source locations separate from semantic hashes.

No provider-specific base classes or mutable global dictionaries for semantic state.

### P1-002 Operation Schema Registry

Each schema declares:

- Operand/result types and counts.
- Attribute names, types, requirements, defaults.
- Region counts.
- Effect/linearity constraints.
- Parser/printer.
- Verifier.
- Whether it lowers back to CircuitIR v1.

Unknown operations/core attributes and incompatible versions fail closed.

### P1-003 Verifier and Diagnostics

Initially reject:

- Use before definition.
- Repeated consumption of linear qubit values.
- Use of old values after gates.
- Out-of-range wires, arity/parameter mismatch.
- Invalid measurement/result types.
- Invalid block terminators.
- Use after release.
- Unregistered operations or unknown semantic attributes.
- Claimed-supported CircuitIR that cannot be represented losslessly.

Failures produce structured code, message, location, and notes, not printed text.

### P1-004 CircuitIR Importer

The importer must:

- Accept CircuitIR without requiring user-built internal objects.
- Preserve opcodes, wire order, parameter identity, dtype, batch shape.
- Separate semantics from requests under IR-001.
- Build one linear value chain per logical wire.
- Explicitly check custom matrices, metadata, and dynamic markers.
- Return diagnostics and source mappings.
- Leave inputs unchanged.
- Produce identical QuantumIR identities for identical inputs.

### P1-005 Restricted Round Trip

Provide a test export for the reversible static subset:

- Exact canonical payload equality after CircuitIR -> QuantumIR -> CircuitIR.
- Diagnose structures unrepresentable in schema 1.0; no lossy export.
- Keep the exporter out of public namespaces.
- Do not use it as a premature provider codegen replacement.

### P1-006 Minimum Analysis and Pass Contracts

Implement:

- `DefUseAnalysis`.
- `QubitLifetimeAnalysis`.
- Analysis caches bound to module revisions.
- Transformations invalidate analyses unless explicitly preserved.
- PassResult, diagnostics, statistics.
- Pipeline digests containing pass names, versions, order, options, seed.

One semantics-preserving canonicalization example proves the infrastructure.
Phase 1 does not migrate existing optimizers.

### P1-007 Old/New Differential Bridge

Add a test/development-only execution bridge:

```text
CircuitIR
  +-- legacy execution
  +-- import QuantumIR -> verified test lowering -> existing executor
```

Constraints:

- Default `fq.run` does not read its control switch.
- No undeclared environment variables changing public behavior.
- The bridge must not present QuantumIR as a new runtime.
- Differential reports include failure, fallback, dtype, device, result order.

### P1-008 Correctness and Gradient Gates

Require:

- Full Phase 0 corpus import/verification.
- Exact round trips.
- State, expectation, measurement, wire-order differentials.
- Parameter/ParameterExpression identity.
- Complex64/complex128 coverage.
- PyTorch forward/gradient parity.
- Fixed-seed determinism.
- Property/fuzz negative tests.
- Unchanged public API snapshots.

### P1-009 Performance Gates

Machine-check approved P0-006 budgets:

- No unexpected superlinear importer/verifier growth.
- Small-circuit fixed overhead within budget.
- 10K-gate construction, verification, hashing within time/memory budgets.
- Record hit/miss when repeated structural imports safely use caches.
- Merely installing internal IR adds no measurable path to default legacy `fq.run`.
- Record budget violations as blockers; never loosen budgets to make CI pass.

### Phase 1 Exit Gate

- [ ] P1-001–P1-009 complete.
- [ ] Entire supported static CircuitIR corpus passes.
- [ ] Every verifier negative fixture passes.
- [ ] State, expectation, measurement, gradient, ordering differentials pass.
- [ ] Deterministic identity, printer, pipeline digests on supported platforms.
- [ ] Approved P0-006 performance/memory budgets met.
- [ ] Stable Core, CircuitIR 1.0, default run/plan paths unchanged.
- [ ] Internal modules absent from root, stable namespaces, user completion.
- [ ] Single-switch rollback and complete removal plan.
- [ ] API/compiler owners approve Phase 1 evidence.
- [ ] No automatic Phase 2 entry without a new proposal.

## 7. Recommended Order

```text
P0-001 ─┐
P0-002 ─┼─> P0-003 -> P0-004 -> P0-005 -> P0-006
ADR 001–006 ┘                         |
                                      v
                              Phase 0 approval
                                      |
                 +--------------------+-------------------+
                 v                                        v
             P1-001                                    P1-002
                 +--------------------+-------------------+
                                      v
                                   P1-003
                                      |
                                      v
                                   P1-004
                                      |
                           +----------+----------+
                           v                     v
                        P1-005                P1-006
                           +----------+----------+
                                      v
                                   P1-007
                                      |
                           +----------+----------+
                           v                     v
                        P1-008                P1-009
                           +----------+----------+
                                      v
                              Phase 1 evidence review
```

Phase 0 fact gathering may run in parallel; ADR decisions precede importer/value
implementation. Phase 1 avoids multiple long-lived product paths. Complete each
milestone before expanding semantic scope.

## 8. Merge Requirements per Work Package

Every PR/commit includes:

- Work-package ID and explicit scope.
- Behavior/architecture change description.
- Public API impact of `none`; otherwise stop for an API proposal.
- Focused tests.
- Required negative tests.
- Performance impact or evidence that the change is outside hot paths.
- Rollback method.
- Known blockers.
- Capability maturity impact; Phase 0–1 does not promote public maturity by default.

Minimum checks:

```bash
python tools/public_api_snapshot.py
python tools/check_architecture.py
python tools/check_repository_hygiene.py
python tools/docs_source_of_truth.py --check
python -m pytest tests/internal_ir -q
python tools/ci_tier.py pr-default
```

Run `pr-runtime` for existing compiler/runtime adaptations. Run `pr-distributed`
under AGENTS.md for distributed metadata or execution semantics. CPU results are
not real scalability evidence.

## 9. Evidence Manifests

Phase 0 and Phase 1 each maintain a machine-readable manifest containing at least:

```text
schema_version
phase
source_commit
environment
public_api_contract_hashes
circuit_ir_schema_version
corpus_hash
supported_opcode/profile
test_commands
test_results
correctness_tolerances_by_backend_and_dtype
performance_budget
performance_results
known_blockers
rollback_path
owners
approval
```

Reference reconstructable evidence only. Do not commit credentials, accounts,
internal addresses, machine identities, or unbounded raw benchmark data.
Unapproved manifests cannot claim release certification.

## 10. Rollback Strategy

Phase 1 must remain structurally reversible:

1. Default production paths do not depend on `_compiler`.
2. Importer, verifier, and differential bridge can be removed together.
3. Legacy compiler/runtime is neither migrated nor deleted.
4. Public serialized data remains unchanged.
5. Caches use separate namespaces/versions.
6. Evidence/debug textual IR does not become a user data dependency.
7. If semantics, performance, or maintenance costs are unacceptable, retain
   Phase 0 baselines/ADRs and withdraw Phase 1 code.

## 11. Suggested First Batch

Initially implement only:

1. P0-001 public contract protection inventory.
2. P0-002 consumer matrix.
3. Core ADRs IR-001, IR-002, IR-003.
4. P0-003 corpus design without immediately creating IR types.

After review, collect P0-005 performance baselines and approve P0-006 budgets.
Begin Phase 1 `_compiler/ir` implementation only after every Phase 0 exit condition
passes and the API owner explicitly approves.
