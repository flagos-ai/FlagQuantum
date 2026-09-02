# FlagQuantum IR Phase 3 entry and Batch A approval packet

Status: **Ready for owner review — implementation not authorized**

Date: 2026-09-02

Prerequisite: IR Phase 2 is technically complete under
`contracts/ir-phase2-exit-authorization.json`.

## 1. Objective

Phase 3 introduces a private target and executable ABI so local simulation, QASM targets
and non-QASM targets can share one verified compilation-product boundary. It does not
make the new compiler public and does not switch `fq.run`, `fq.plan`, deployment or cloud
submission to a new default path.

The intended internal flow is:

```text
verified QuantumIR
  -> TargetCapabilities snapshot
  -> target legalization
  -> TargetIR
  -> sealed ExecutableArtifact
  -> RuntimeAdapter
  -> execution binding / receipt / result
```

## 2. Identity and ownership boundary

Phase 3 must keep these identities separate:

| Identity | Owns | Must not own |
| --- | --- | --- |
| Program identity | User-visible program semantics | Backend, queue, credential or price |
| Compilation identity | Program + pipeline + target capability/topology/calibration snapshots + compile options | Token or mutable job state |
| Target semantic fingerprint | Canonical capabilities, topology and calibration content | Provider credential or queue |
| Artifact identity | TargetIR/code payload, format/profile and compilation identity | Runtime token or job status |
| Execution binding | Actual dispatch locator, request policy and immutable artifact identity | Program semantics |
| Submission receipt | Provider/job identifiers and artifact binding | Compiler rewrites |

A real external backend ID is allowed only in an execution binding or receipt after target
resolution. It must not enter ProgramIR, QuantumIR, TargetIR, compilation identity,
portable artifact payloads, cache keys, logs used as public evidence, or test fixtures.

## 3. Planned batches

| Batch | Scope | Required exit evidence |
| --- | --- | --- |
| A | Immutable `TargetCapabilities` snapshot, closed vocabularies, semantic fingerprint and capability-diff diagnostics | Schema/verifier negatives, canonical hash, mutation isolation, anonymous local/QASM/non-QASM fixtures, performance baseline |
| B | Private TargetIR model and QuantumIR-to-TargetIR legalization | Gate/type/layout legality, deterministic lowering, unsupported capability fail-closed, state/order/gradient differential where applicable |
| C | Artifact profile registry and sealed `ExecutableArtifact` | Payload/profile/content/target/compilation identity chain, tamper rejection, credential and dispatch-locator exclusion |
| D | Internal `RuntimeAdapter` protocol and execution-binding contract | Local sync lifecycle plus offline async mock lifecycle, terminal-state and result-identity conformance |
| E | Provider-neutral conformance suite | Same suite for local runtime, synthetic QASM target and synthetic non-QASM target; provider extensions namespaced |
| F | Explicit opt-in shadow comparison harness | Legacy remains authoritative, mismatch taxonomy, bounded overhead, privacy-safe evidence, kill switch |
| G | Deployment compatibility and canary-readiness proposal | `DeploymentPackage` mapping, rollback, migration and public/default impact review; no canary activation |
| H | Aggregate Phase 3 exit review | Three target classes share the verified artifact/runtime boundary and every batch budget passes |

Batches are sequential. Each performance budget is derived from an independent baseline
and requires owner approval before becoming a gate.

## 4. Batch A allowed scope

Batch A may add only private, provider-free types and tests under:

```text
flagquantum/_compiler/
tests/internal_ir/
tests/fixtures/internal_ir/
benchmarks/internal/
docs/development/
contracts/
```

The minimal capability model may represent:

- target class: local runtime, QASM text, or non-QASM text/binary;
- logical and physical qubit capacity;
- native operation set and parameter constraints;
- directed topology identity;
- supported result and measurement classes;
- static/adaptive, reset, timing, pulse and noise support flags;
- artifact formats and versions;
- calibration snapshot identity and validity metadata;
- auxiliary-qubit capacity/policy as an explicit unsupported or bounded capability.

Batch A must use closed vocabularies for semantic fields, canonical immutable encoding and
structured differences. Unknown semantic values fail closed. Provider-specific raw fields
remain outside the schema or in a clearly non-semantic extension namespace.

## 5. Explicit exclusions

This entry proposal does not authorize:

- TargetIR, ExecutableArtifact, RuntimeAdapter, conformance or shadow implementation;
- Batch B through H;
- changes to Stable Core, `CircuitIR` 1.0, public exports or protected schemas;
- changes to `fq.run`, `fq.plan`, `compile_for_backend` or `DeploymentPackage` behavior;
- import hooks, environment switches or default-path shadow execution;
- adapter-repository source, provider SDK calls, discovery or remote submission;
- credentials, tokens, real backend IDs, queues, jobs, quota, price or retry policy;
- public compiler APIs, legacy retirement or Phase 3 completion;
- C++/MLIR introduction.

## 6. Batch A verification axes

| Axis | Minimum evidence |
| --- | --- |
| Compatibility | Stable API manifest and public baseline unchanged |
| Correctness | Accepted values round-trip exactly; invalid combinations fail closed |
| Determinism | Equal snapshots have equal canonical bytes and semantic fingerprints across hash seeds |
| Isolation | Caller mutation cannot alter a constructed snapshot |
| Identity | Topology/calibration/format/capability changes alter the fingerprint; display labels do not |
| Privacy | Credential, token, URL, real backend ID and job fields are rejected or excluded |
| Performance | 10/100/1K/10K capability-entry baseline measured separately; no budget before measurement |
| Default path | Importing and using `fq.run`/`fq.plan` does not load or execute Batch A code |

## 7. Shadow-mode safety rule

Phase 3 shadow comparison is not ordinary logging. Even when it does not control the
result, it consumes CPU/memory, may process sensitive deployment data and can change
latency. Therefore only Batch F may propose an explicit opt-in harness. Attaching it to a
default path, environment switch, provider submission or production request requires a
separate authorization after overhead and privacy evidence exists.

## 8. Rollback

Batch A is removable without user migration because it remains private and unreachable
from default paths. Any schema ambiguity, non-deterministic fingerprint, privacy leakage,
performance regression or public/default import causes the batch to stop. Existing Phase
2 artifacts and the legacy compiler remain authoritative.

## 9. Requested decision

Approve Phase 3 entry and only Batch A provider-free target capability/identity work with:

`approve IR-PHASE3-ENTRY-BATCH-A`

This command must not authorize later batches, TargetIR/artifact/runtime implementation,
shadow/default integration, provider work, public APIs, legacy retirement or Phase 3 exit.
