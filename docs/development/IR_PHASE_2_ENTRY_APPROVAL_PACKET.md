# FlagQuantum IR Phase 2 Entry and Batch A Authorization Packet

Status: **Ready for owner review — implementation is not authorized**
Date: 2026-09-02
Prerequisite: Phase 1 complete; current results in [IR Implementation Status](IR_IMPLEMENTATION_STATUS.md).

## 1. Objective

Phase 2 migrates existing static compilation capabilities while remaining
internal, explicitly opt-in, and fully reversible. The first batch migrates only
the smallest canonicalization subset of `optimize`. It does not connect to default
`fq.run`, `fq.plan`, `compile_for_backend`, or deployment paths.

This round neither redesigns public compiler APIs nor targets legacy compiler removal.

## 2. Baseline Facts

The existing static compilation chain is:

```text
CircuitIR
  -> remove identity / zero rotations
  -> cancel adjacent self-inverse gates
  -> merge adjacent rotations
  -> optional topology routing
  -> post-routing optimization
  -> QASM/QCIS emission
  -> DeploymentPackage
```

Phase 1 provides immutable QuantumIR, a verifier, pass contracts, restricted
lowering, a differential bridge, and performance gates. Its current canonicalization
pass is only a no-change scan; it does not implement the legacy optimizations above.

## 3. Phase 2 Batches

| Batch | Scope | Exit evidence |
| --- | --- | --- |
| A | Identity/zero-rotation removal, self-inverse cancellation, adjacent rotation merging | Independent goldens, idempotency, state/expectation/gradient parity, deterministic identities. |
| B | Target gate-set decomposition | Exact gate-set contracts per rewrite; parameter and gradient differentials. |
| C | Placement/routing | All gates legal, logical/physical mapping, asymmetric bit order, independent performance baseline. |
| D | OpenQASM 2, restricted OpenQASM 3 static profile, QCIS emission | Parsing/goldens, character-level determinism, semantic fixtures. |
| E | Pipeline cache, diagnostics, compilation identity | Cache keys, invalidation, error paths, observability. |
| F | Legacy/new deployment corpus and Quafu static cases | Complete differentials, performance gates, no default-path impact, Phase 2 exit review. |

Batches must not start early in parallel. Routing and emitters establish their own
baselines only after preceding rewrite semantics stabilize.

## 4. Permitted Batch A Scope

Implementation is limited to these private surfaces:

```text
flagquantum/_compiler/passes/
flagquantum/_compiler/testing/
tests/internal_ir/
tests/fixtures/internal_ir/
benchmarks/internal/
docs/development/
contracts/
```

Batch A must:

- Explicitly reconnect linear qubit values; skipping verification cannot stand in
  for valid removal.
- Fail closed for parameters, tensors, custom unitaries, channels, or measurements
  whose safe handling cannot be proved.
- Preserve wires, parameter identity, dtype, batch shape, measurements, and requests.
- Treat numerical rotations only according to legacy `_is_zero`; do not fold
  trainable tensors.
- Preserve autograd when merging parameter expressions.
- Give each pass an independent descriptor, deterministic digest, and statistics.
- Compare structure and scientific semantics with stable `optimize` after restricted lowering.
- Keep Phase 2 passes out of imports and execution on default public paths.

## 5. Explicit Prohibitions

This packet does not authorize:

- Stable Core, root exports, `CircuitIR` 1.0, or public serialization schema changes.
- Default behavior changes to `fq.run`, `fq.plan`, `compile_for_backend`, or
  `DeploymentPackage`.
- Silent compiler switches through environment variables or import side effects.
- Removing, forwarding, or deprecating the legacy compiler.
- Starting Batches B-F.
- Migrating routing, QASM/QCIS emitters, or Provider codegen.
- TargetIR, ProgramIR, public PassManager, or new stable APIs.
- Quafu/backend IDs, credentials, queues, jobs, or pricing in program IR.
- Changing public snapshots, Phase 0 baselines, or approved budgets to pass tests.

## 6. Differential Oracles

Batch A must verify at least:

| Axis | Oracle |
| --- | --- |
| Structure | Exact canonical instruction sequences after legacy/new lowering. |
| State | Existing backend/dtype tolerances with explicit global-phase handling. |
| Expectation | Existing dtype contracts. |
| Gradient | Compare both forward values and parameter gradients. |
| Ordering | Exact wire, measurement, and result ordering. |
| Identity | Identical inputs/pipelines produce identical program/pipeline identities. |
| Failure | Structured rejection of unsupported inputs without fallback. |
| Default path | `fq.run/fq.plan/compile_for_backend` do not load new passes. |

Do not introduce one loose tolerance covering every backend and dtype.

## 7. Performance Budget Candidates

Current machine regression budget:
`tests/fixtures/internal_ir/phase2_batch_a_performance_budget.json`.

It covers import + verify + approved passes + restricted lowering + lowered-result verification:

| Gates | p95 limit | Peak host memory limit |
| ---: | ---: | ---: |
| 10 | 1.25 ms | 1.25 MiB |
| 100 | 2.0 ms | 1.25 MiB |
| 1,000 | 15 ms | 2 MiB |
| 10,000 | 140 ms | 14 MiB |

Routing, emission, and deployment construction cannot reuse this budget. Each needs
an independent baseline before its batch starts. Budget failures require
optimization, reduced scope, or blockers; limits cannot be relaxed automatically.

## 8. Rollback

Batch A remains opt-in. If any semantic, gradient, identity, performance, or
default-path gate fails:

1. Stop before the next batch.
2. Keep the legacy compiler as the sole default path.
3. Remove or disable Phase 2 passes and their opt-in pipeline.
4. Preserve failing fixtures, diagnostics, and audit records.
5. Do not change public contracts to absorb differences.

## 9. Approval Semantics

Ordinary `do`, `continue`, or Phase 1 authorization does not authorize Phase 2.
After accepting the machine candidate, differential scope, performance budgets,
and rollback conditions, an authorized owner may use:

```text
approve IR-PHASE2-ENTRY-BATCH-A
```

This authorizes Batch A only. Batches B-F, public-path switching, and legacy
retirement require separate subsequent evidence and authorization.
