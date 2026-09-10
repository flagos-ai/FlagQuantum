# API Change Proposal 012: Quafu Provider and Portal Task Contracts

## Status

**Draft — core capabilities are being implemented on the current API consolidation branch; protected contract changes still require API-owner approval.**

- Target version: first public alpha.
- Machine-readable contract: `contracts/quafu-provider-contract-v1-candidate.json`.
- Inputs: `flagquantum-quafu-adapter` Issue #4 and PR #3.
- Stable root-level API changes: none.
- `ExecutionResult`, `ExecutionPlan`, `CircuitIR`, and `DeploymentPackage` schemas:
  unchanged.
- FlagQuantum execution, noise, and result-conversion code belongs only in the
  current FlagQuantum branch. The shared adapter repository contains OpenAPI,
  schemas, mocks, documentation, and necessary thin bindings; it must not copy
  FlagQuantum implementation source.
- New stable types, fields, or exception contracts require separate approval.

## Problem

The API consolidation branch distinguishes local `ExecutionPlan` objects from
provider-facing `DeploymentPackage` objects and establishes an extension protocol
and deployment artifact identity chain. Five gaps remain in Quafu integration:

1. FlagQuantum local counts follow measurement wire order, while the portal uses
   Qiskit classical-string order.
2. Provider tasks expose string statuses and generic exceptions without stable
   terminal states, error categories, or result failure semantics.
3. Backend capabilities do not express valid shot ranges or QASM subset versions.
4. Result metadata has no uniform requirements for the executor, duration,
   calibration snapshot, or noise model identity.
5. The real Quafu adapter path still depends on `fq.experimental.QPUTwin`, which
   was removed from the API consolidation branch.

The old implementation also writes the deprecated top-level `compile` boolean
into task requests and fails to bind the logical circuit to an ordered physical
qubit mapping, leaving compiled-circuit submission semantics incomplete.

## Decisions

### 0. Compiled-Circuit Submission Contract

OpenQASM emitted by a local compiler such as QSteed must retain contiguous logical
indices `q[0]...q[N-1]`. Physical qubit IDs must not be written into QASM. Submission
requires all three of the following:

- `circuit`: complete OpenQASM 2.0 using logical indices.
- `options.compiler=None`: explicitly request no additional cloud compilation.
- `options.target_qubits`: an ordered list of N physical qubits; item i maps to
  logical `q[i]`.

The top-level `compile` boolean is deprecated. FlagQuantum no longer sends,
accepts, or records it. Missing or duplicate physical mappings, incorrect mapping
lengths, or inconsistent QASM register widths must fail before a network request.

### 1. Name Each of the Three Bit-Ordering Layers

Do not use the unqualified term "bit order." The candidate contract distinguishes:

| Layer | Convention |
| --- | --- |
| FlagQuantum sample/counts measurement | `measurement_wires_left_to_right`: string positions follow the requested wires |
| OpenQASM classical register | `classical_msb_left`: the string runs from the highest classical bit to `c[0]` |
| Quafu portal `result.counts` | `classical_msb_left` |

For `x q[0]`, `measure q[0] -> c[0]`, and `measure q[1] -> c[1]`:

- FlagQuantum local counts for wires `(0, 1)` contain `"10"`.
- Quafu portal counts contain `"01"`.

The provider boundary must convert explicitly; these conventions do not permit
silent inconsistencies. Quafu deployment result metadata must include
`counts_bit_order="classical_msb_left"`. Contract tests must use asymmetric
circuits: symmetric Bell/GHZ circuits cannot establish bit ordering.

This proposal does not change local `MeasurementResult` value semantics frozen
by Proposal 004. Promoting a bit-order field into a stable `MeasurementResult` or
`DeploymentResult` field would require separate approval.

### 2. Migrate Private Deployments to the Consolidated Execution Entry Points

The real FlagQuantum simulator implementation must not enter the shared adapter
repository or depend on `fq.experimental.QPUTwin` or any internal name outside
the discoverable experimental surface. After installing the current FlagQuantum
branch in the deployment environment, the real simulation path is:

```text
normalized Quafu calibration
  -> flagquantum.deployment.quafu_noise_model_from_chip_info(...)
  -> fq.run(circuit, measurements=(probabilities,), noise_model=model)
  -> provider-boundary bit-order normalization
  -> portal counts
```

The shared adapter forwards validated requests to the private FlagQuantum service
and returns results according to the shared contract. It must not copy noise
models, density-matrix executors, GPU scheduling, or other FlagQuantum source.
Use the noise model identity as `noise_model_version` and the stable hash of
normalized calibration content as `calibration_version`.

### 3. Provider Task States and Error Categories

The target states are `Pending / Running / Finished / Failed / Cancelled`.
Platforms may expose more native states, but adapters must preserve the original
state and map it to these five. Successful `submit` means acceptance, not
successful execution.

Candidate error categories use the portal's existing names:

```text
precheck.badCircuit
precheck.noMeasurement
precheck.shotsOutOfRange
precheck.tooDeep
precheck.notTranspiled
precheck.qubitUnavailable
precheck.edgeUnusable
precheck.gateUnsupported
precheck.simTooLarge
runtime.deviceUnavailable
runtime.quotaExhausted
runtime.rejected
runtime.internal
```

Before stable exception types are approved, adapters may expose `error_category`
in the HTTP schema. Core `QuafuProvider` must not bypass the freeze process by
changing existing public exception types.

### 4. Minimum Result and Capability Contract

Quafu portal result metadata should contain at least:

- `backend`: the actual executor.
- `duration_ms`.
- `calibration_version`.
- `noise_model_version`.
- `counts_bit_order`.
- The physical qubits actually used.
- Deployment artifact identity or program digest.

Backend capabilities should subsequently express `min_shots`, `max_shots`, shot
increments, QASM profile, basis gates, coupling map, and dynamic-circuit support.
The first implementation may use provider-specific metadata, but must not present
free-form metadata as a permanently stable schema.

### 5. Calibration Policy Has One Source of Truth

The adapter normalizes raw Quafu payloads into snapshots with provenance; the core
converter consumes those snapshots:

- Invalid or frozen qubits/couplers are excluded from the available set.
- If `T2 > 2*T1` is clipped, record the original value, corrected value, and rule version.
- Estimates used for missing readout data must be marked `estimated`, never
  represented as measured calibration.
- Every result must be traceable to the normalized snapshot hash.
- The adapter and core paths must not apply different corrections to the same snapshot.

## Phased Implementation

### P0: Before the First Mock/Real-Backend Integration Exercise

- [x] Freeze portal counts ordering using asymmetric circuits.
- [x] Support density probability measurements with readout noise on the current
  FlagQuantum branch.
- [ ] Remove the private deployment binding's dependency on `fq.experimental.QPUTwin`.
- [ ] Emit `counts_bit_order`, `calibration_version`, and `noise_model_version`
  from the private service.
- [ ] Keep submission success separate from terminal execution status.
- [ ] Run real end-to-end contract tests without copying source into the adapter repository.

P0 verification found that the measurement layer misidentified density-matrix
output as a statevector, causing probability request reshaping to fail. The API
consolidation branch fixed this without changing public signatures: density
probability reads the density matrix diagonal, applies NoiseModel readout
confusion, and produces marginal probabilities in measurement wire order. An API
contract test protects this behavior.

### P1: Before the Portal Contract Freeze

- [ ] Freeze `/result` status codes and error bodies for Failed/Cancelled tasks.
- [ ] Add the `error_category` schema.
- [ ] Freeze the cancellation state-transition graph.
- [ ] Freeze the QASM subset and register constraints.
- [ ] Align declared shot capabilities with actual gateway limits.

### P2: Provider Productization

- [ ] Implement a shared provider conformance suite.
- [ ] Normalize native platform states while preserving their original values.
- [ ] Use frequent initial polling followed by backoff for longer-running tasks.
- [ ] Record the actual mapping and calibration snapshot after provider recompilation.
- [ ] Persist tasks, results, quotas, and audit records.

## Acceptance Boundaries

- Passing mock contracts does not establish noise accuracy or hardware capability.
- CPU contract tests do not establish production throughput or capacity.
- Provider counts must pass shot accounting and bit-order conformance first.
- Verify the adapter/core package compatibility window through real installation tests.
- Do not change `docs/public_api_v1.json`, stable root signatures, or protected
  serialized schemas without approval.
