# Runtime / TargetCapabilities Requirement Mapping

Status: Phase 2 Runtime proposal and characterization tests, not a public contract implementation.

Baseline: `vnext-phase1-contract-foundation`

Scope: Runtime planning, environment matching, preflight, and execution evidence.

## Conclusion

Runtime needs two distinct inputs: what a workload requires and what an available
target has demonstrably provided. These should become `RequirementSet` and
`CapabilitySnapshot`. They cannot share a boolean capability object. Runtime must
not turn user expectations, environment variables, or algorithm estimates into
hardware facts.

Runtime selects executable candidates from requirements, snapshots, leases, and
explicit policies, runs an attempt, and records actual routes and results.
Compiler owns program legalization and compilation requirements; Platform/Provider
owns discovered facts; Simulation owns algorithm candidates and cost hints.
None makes Runtime's final resource selection.

This proposal changes no Stable Core API or serialization contract. Type names
below describe shapes requiring Integration approval.

## 1. Existing Entries and Implicit Requirements

| Concern | Current entry | Representation | Risk or gap |
| --- | --- | --- | --- |
| User intent | `runtime/options.py::ExecutionOptions` | Mode, backend, device, target, batch, precision, shots, seed, memory limit, gradients, approximation, backend fallback | Precision mixes storage, effective, and native precision; no separate CPU fallback, dynamic-circuit, recovery, or real-time requirement |
| Option resolution | `runtime/options_resolver.py` | Defaults -> config -> program constraints -> runtime policy -> call | IR dtype becomes precision without source/strength in matching evidence |
| Public planning | `flagquantum/api.py`, `runtime/planner::plan`, `runtime/execution.py` | Resolve options, world size, backend; call planner | Planning, environment inference, and execution lack an explicit requirement/snapshot seam |
| Plan environment | `compilation/execution_plan_contract.py` | Backend, device kind, precision, world size, distribution semantics, gradients, approximation | Infers `sharded_across_ranks` from `world_size > 1`; lacks actual devices, memory, topology, communication, shots/dynamic, checkpoint/realtime |
| Backend discovery | `runtime/backend_registry.py` | Devices, dtypes, autograd, distributed, statevector/density/MPS, preferred device, accelerator memory | Mostly booleans/static declarations; no unknown/unmeasured/not_exposed; broad PyTorch declarations can be mistaken for target facts; `flagos` bypasses device-support checks |
| Compiler capabilities | `_compiler/target_capabilities.py` | Gates, results, formats, topology, dynamic/timing/pulse, shot/program limits | Same type for required/available; `False`, `None`, and absence cannot distinguish fact status and exposure |
| Capability comparison | `_compiler/capability_comparison.py` | Required/available comparison | Comparison tends to fail closed, but types cannot distinguish unknown, unmeasured, unsupported, and hidden facts |
| Device/dtype checks | `runtime/backend_registry.py::backend_execution_options`, `compilation/execution_plan_contract.py::validate_plan_environment` | Device, dtype, mode, world size | Compiler imports Runtime discovery; dtype checks inspect declared sets without native/software mechanisms |
| Distributed policy | `runtime/distributed/backend_policy.py` | Profiles, JAX/Torch backend, local/effective world size, torchrun/GPU policy | Environment/process groups prove orchestration state, not physical topology, communication, or sharding; development LocalTensor is simulated evidence |
| Memory and costs | `runtime/planner/backend_selection.py`, `candidates.py`, `candidate_plans.py` | SV/MPS/TN estimates, costs, memory/communication/gradient plans | Workload estimates can be mistaken for available memory; division by world size does not prove allocation or communication feasibility |
| Dynamic circuits | `runtime/dynamic/conformance.py`, `runtime/dynamic/deployment.py` | Mid-circuit measurement/reset, feed-forward, results, provider support | Local checks are not integrated into RequirementSet/Snapshot matching |
| Shots/trajectories | `ExecutionOptions.shots`, `runtime/trajectories/**` | Shots, ownership, failure, statistics, checkpoints | Shots are requirements; trajectory completion is attempt evidence, not a target capability |
| Training precision | `runtime/training_state.py::PrecisionPolicy`, training engines | Complex/parameter/accumulator dtype, mixed/full precision, downcast, gradients, optimizer state | Execution options and training policy duplicate precision representation; no shared software-extension mechanism |
| Checkpoint/recovery | `runtime/training_state.py`, `runtime/trajectories/checkpoint.py`, training engines | Schema/version, topology, IR hash, RNG, atomic commit, space preflight, writer lease, resume | Recoverability is absent from generic requirements; storage/compatibility discovery has no shared snapshot |
| Real-time sessions | `_compiler/TargetCapabilities` realtime/adaptive fields and some provider/dynamic paths | Booleans and latency bounds | No generic `RealtimeSession` lifecycle; unmeasured latency is not a satisfying fact |
| Fallback/routes | `runtime/fallback.py`, `runtime/routing.py`, result/evidence fields | Policy, route categories, events, host debug policy | Backend, CPU, algorithm, and precision fallback need independent authorization; absent events do not prove absent fallback |
| Post-execution evidence | `runtime/observability/evidence.py`, backend result records | Actual provenance, fallback, distributed/memory/communication/result metadata | Evidence is scattered; must bind to attempts/resources and cannot automatically become a permanent capability |

### 1.1 Required Dimensions

Use eight sparse groups. Fields irrelevant to an execution need not appear.

1. **Workload:** mode/representation, program features, exact/approximate semantics,
   permitted truncation and error bounds.
2. **Numeric:** storage dtype, effective precision, native-precision requirement,
   parameter/accumulator dtype, tolerances, permitted software mechanisms.
3. **Measurement:** shots, result types, dynamic circuits, mid-circuit measurement/
   reset, feed-forward, timing, pulse, noise.
4. **Compute:** device kind, minimum/exact device count, per-device/aggregate memory,
   user budget, workspace.
5. **Fabric:** nodes/local devices, placement/topology, communication primitive/
   backend/dtype, route visibility.
6. **Training:** gradients, reverse path, optimizer/update distribution semantics,
   parameter binding, batches.
7. **Continuity:** checkpoints, storage consistency, restart compatibility, recovery
   limits, seed/determinism, batch/session/realtime, latency bounds.
8. **Authorization/evidence:** approximation and backend/device/CPU/precision/
   algorithm fallback permissions, required evidence and claim level.

This does not require a permanent giant configuration object. A minimal
`RequirementSet` contains relevant predicates, provenance, mandatory/preference
strength, and authorization. Missing authorization means prohibited.

### 1.2 Preserve User and Compiler Provenance

Both contribute to the sparse `RequirementSet`, but predicates retain their own
sources. Compiler deductions are not user authorization; user preferences are
not program-legality constraints.

| Source | May supply | Must not supply |
| --- | --- | --- |
| User/request adapter | Target, shots, budgets, preferred device/backend, precision goal, deadline, realtime/checkpoint needs, per-axis approximation/fallback authorization | Native gates, algorithm feasibility, allocated devices/memory, actual routes |
| Compiler | IR dtype/shape, native gates/results/control flow, parameter binding, legal topology predicates, gradient/output semantics, workspace/artifact requirements | CPU/precision/algorithm fallback permission, platform availability, measured capacity/latency, final placement |
| Runtime protocol | Required lease, route visibility, checkpoint compatibility, evidence level, cleanup conditions | Revised user authorization, relaxed Compiler legality, platform facts |

Intersect mandatory constraints, retain both sources, and fail before matching
if they are incompatible. Policy must not guess a resolution. Current
`ExecutionPlan` stores resolved options and decisions but loses original field
provenance. Test adapters can label their output only as `source=compiler` plan
projections, not reconstructed user requests. The final request-to-plan contract
must preserve source and authorization chains.

### 1.3 Requirements, Facts, and Evidence by Dimension

| Dimension | Requirement examples | Snapshot facts/thresholds | Attempt evidence |
| --- | --- | --- | --- |
| dtype/precision | Storage, parameter, accumulator, effective, native-required, software mechanisms | Supported dtypes/mechanisms; separate native/effective axes; scoped error validation | Actual dtypes, mechanism, error bounds, downcasts |
| Memory | State/workspace/peak, per-device/aggregate units, budget | Available/reserved memory for the same lease/scope; unmeasured does not meet hard capacity | Actual peak/reserved, OOM/preflight, estimate differences |
| Device count | Minimum/exact, local devices, nodes | Verified allocated lease count; world size/environment are not device facts | Actual ranks/devices/nodes and ownership |
| Topology | Coupling, placement, interconnect/NUMA, visibility | Scoped topology/routes; unknown/not_exposed fail closed | Actual placement, links, routes, deviations |
| Communication | Primitive/backend/dtype, host-staging policy | Verified support for the combination; interface existence is only declared/unmeasured | Collectives/P2P, bytes, time, backend, staging/fallback |
| Shots | Exact/min/max, batching, seed/determinism | Provider maximum, batch limits, sampling/RNG | Requested/completed shots, losses/retries, seeds, statistics |
| Dynamic | Measurement/reset, feed-forward, timing/pulse, latency | Independent fact states; one boolean cannot establish adaptive realtime | Branch/measurement/reset trace, control latency, actual provider route |
| Checkpoint/restart | Necessity, frequency/consistency, format, restart topology/dtype, retry count | Storage/atomicity/format/compatibility; local helper existence is insufficient | Checkpoint identity/commit/base attempt, recovery outcome |
| Realtime/session | Lifecycle, end-to-end/control latency, duration, exclusivity | Session support and workload-matched latency measurement; unmeasured fails closed | Session/lease IDs, latency distribution, timeouts, reconnect/degradation |

`estimated_memory_bytes`, structural scores, and candidate `available` in
`runtime/planner/backend_selection.py` describe algorithm feasibility under
assumptions, not platform capacity. A plan's `world_size` is a requirement or
decision, not proof of allocation or sharding.

## 2. Four Information Layers

| Layer | Authority | Content | Prohibited use |
| --- | --- | --- | --- |
| `RequirementSet` | User + Compiler; Runtime adds execution protocol constraints only | Program semantics, numeric needs, resource minima, continuity, authorization | Available memory, actual topology/devices, measured latency |
| `CapabilitySnapshot` | Platform/Provider discovery and controlled probes where needed | Scoped, timestamped target/lease facts with source, status, exposure, freshness | Fabricating hardware facts from user device/world-size wishes, Compiler estimates, Runtime defaults |
| Runtime policy/decision | Runtime | Filtering, priorities, leases, placement, partition, fallback, blockers | Rewriting Compiler legality or Simulation costs; calling a decision a platform capability |
| Execution evidence | Runtime + owning backend/provider | Actual devices/routes/ownership, memory/communication, fallback, checkpoints, results/failures | Automatic promotion to permanent capabilities; planned intent substituted for actual behavior |

Requirements retain source. Snapshots retain target and platform/provider identity,
collection time, version, scope, and probe/declaration source. Decisions identify
the requirements and snapshots used. Evidence binds to an execution attempt.

## 3. Minimum Matching Semantics

### 3.1 Status and Exposure Are Independent

Each fact has a support status:

- `unknown`: the authority has no answer. Mandatory requirements fail closed;
  preferences may lower ranking while retaining a blocker.
- `unmeasured`: an interface/declaration suggests support but lacks measurement
  meeting this threshold. Hard capacity, performance, realtime latency, effective
  precision, communication, and no-fallback predicates fail closed. Explicitly
  authorized development/debug degradation may permit an attempt with a lower
  claim ceiling.
- `unsupported`: the authority explicitly rejects support. Eliminate the
  candidate; policy cannot override it. Continue only with another candidate and
  corresponding explicit fallback/approximation authorization.
- `verified`: evidence meets scope, freshness, and predicate requirements. This
  permits the match, not automatic release or scalability certification.

Exposure is a separate axis: `observed`, `declared`, `not_exposed`, `unknown`, or
`not_applicable`. `not_exposed` does not mean false, zero, or unsupported. It cannot
prove routes, absence of CPU fallback, communication paths, available memory, or
latency. Debug degradation needs authorization, blockers, and a non-release claim
ceiling. `not_applicable` requires a reason.

### 3.2 Matching Order

For each candidate:

1. Validate requirement provenance and consistency; do not turn requirements into facts.
2. Check snapshot identity, scope, version, and freshness. Stale or mismatched
   snapshots cannot provide proof.
3. Require a fact for every mandatory predicate and enforce its own evidence
   threshold. Immutable specifications may accept authoritative `declared`
   evidence; capacity, latency, routes, and no-fallback usually need `observed`.
   A `verified` status does not raise inadequate exposure.
4. Compare values and mechanisms: set inclusion, numeric bounds with units, and
   explicit topology/route predicates rather than string equality alone.
5. Separate native/effective precision. Software extension may satisfy
   `precision.effective=complex128`, never `precision.native=float64`. Evidence
   records mechanism, storage/accumulator dtype, and error bounds. Native,
   storage, parameter, and accumulator use scalar dtypes; effective uses logical
   complex quantum-state dtype.
6. Reject unsupported candidates immediately. Unknown, unmeasured, and hidden
   mandatory facts produce reasoned blockers.
7. Preferences rank executable candidates; they cannot make an invalid candidate executable.
8. Before fallback, check its independent authorization and rematch every still
   applicable predicate using the replacement's own snapshot/lease. Do not reuse
   original facts or infer CPU, precision, or algorithm permission from
   `allow_backend_fallback`.
9. Return target/lease, backend, mode, placement/partition, snapshot, degradation,
   blockers, and claim ceiling.
10. Compare actual evidence with the decision. Route/distribution deviations must
    fail or become explicitly authorized, recorded fallback events.

### 3.3 CPU Fallback

CPU is a separate candidate, not a device resolver default. Selection requires:

- explicit `cpu_fallback` authorization from the user or approved policy;
- full workload matching against a CPU snapshot;
- plan/decision records identifying the original device/route and CPU replacement;
- result/evidence records of actual device, reason/event, precision, and distribution changes;
- removal or reduction of GPU, distributed-capacity, and performance claims.

A `not_exposed` route cannot prove absence of CPU fallback. Runtime therefore
cannot publish GPU-execution or no-fallback evidence for that route.

## 4. Simulation Candidates and Costs

Simulation may supply:

- legal representation/mode candidates derived from IR/workload;
- SV/MPS/TN memory, communication, and computation estimates;
- partition/slicing/sharding candidates and algorithm prerequisites;
- cost-model version, assumptions, confidence, error range, unknowns;
- unsupported-algorithm, incomplete-gradient, approximation, and truncation blockers.

Simulation does not select physical devices/providers, target identities, leases,
actual topology/transport, CPU fallback, or release claims. Its
`required_memory_bytes` estimates workload needs rather than
`available_memory_bytes`. Candidate `world_size` is not discovered device count;
a communication plan is not a verified interconnect.

Runtime combines candidates with requirements, snapshots, leases, and policies to
select backend, mode, placement, and partition. It cannot rewrite algorithm
constraints or treat low cost as feasibility proof. Compiler turns the selected
legal candidate into a target artifact without importing Runtime discovery.

## 5. Match-to-Attempt Lifecycle

```text
User intent + Compiler requirements
                  |
                  v
          Sparse RequirementSet
                  |
Simulation candidates/cost hints -----+
                                      |
Platform/Provider CapabilitySnapshot --+--> Runtime match/policy
                                                |
                                                v
                                       Decision + resource lease
                                                |
                                       Compile/execute one attempt
                                                |
                                                v
                                    Actual result + execution evidence
```

A persistent Compute Service task may queue, retry, migrate, and contain multiple
attempts. Snapshots and leases may change between them. Each attempt freezes its
requirements, snapshot, decision, artifact/IR identity, checkpoint base, and
evidence. Service retries cannot replace an unsupported outcome with unauthorized
CPU fallback or overwrite failure evidence. Checkpointing is explicit continuity
between attempts, not an inherent capability of a persistent task.

## 6. Characterization Tests

`tests/team/runtime/test_target_capabilities_runtime_requirements.py` consumes
existing `fq.plan` output and uses minimal test fakes for the proposed seam. It
does not introduce public types. Coverage includes:

- plan mode/device/precision/count/memory/gradient/fallback interpreted as
  requirements, not hardware facts;
- unknown/unmeasured topology failing mandatory requirements;
- unsupported realtime sessions remaining blocked under policy;
- software extension satisfying effective complex128 but not native float64;
- explicit CPU permission, complete rematching with a separate snapshot, and events;
- broad backend fallback not authorizing CPU fallback;
- hidden routes not proving absence of CPU fallback.

After approval, migrate these into contract fake/conformance tests. Prove
replacement using at least two snapshot providers or fakes.

## 7. Decisions Required from Integration

1. Core ownership, versions, and serialization of `RequirementSet`,
   `CapabilitySnapshot`, status, and exposure vocabulary; no Runtime-private copies.
2. Whether to split `_compiler.TargetCapabilities` required/available roles or
   retain it only for compiler target descriptions, with compatibility migration.
3. Compatible separation of `ExecutionOptions.precision` into storage/effective/
   native, parameter/accumulator, and software mechanisms.
4. Independent backend, CPU, device, precision, algorithm, and approximation
   permissions; defaults fail closed.
5. Snapshot target/lease identity, TTL/freshness, Platform/Provider precedence,
   and predicates satisfied by declarations.
6. Whether Compute Service, Platform, or Provider supplies leases and allocation;
   Runtime consumes valid leases only.
7. Removal of the inference from `world_size > 1` to `sharded_across_ranks`;
   evidence needed for actual distribution and scalability claims.
8. Migration of Compiler `validate_plan_environment` imports of Runtime discovery
   toward Runtime matching of Compiler requirements.
9. Shared dynamic-circuit, checkpoint/restart, and realtime-latency requirements,
   facts, scopes, and thresholds.
10. Debug claim ceilings for hidden/unmeasured facts in plans, results, and audit payloads.
11. Minimal versioned Simulation candidate/cost envelope; Runtime owns final ranking.
12. API proposals, compatibility analysis, and conformance tests for stable plan
    identity, failure-stage, or result-evidence schema changes.

## 8. Recommended First Integration Slice

Approve an internal, non-public contract fake first: one sparse requirement
predicate, one fact with status/exposure/provenance, and a pure matcher. Cover only
`device.kind`, `device.count`, `precision.effective/native`,
`memory.available_bytes`, and independent `cpu_fallback` permission. Reuse the
existing plan adapter without changing Stable APIs.

Acceptance requires missing facts to fail closed, software extension to remain
distinct from native precision, CPU fallback evidence in both plan and result,
and replacement of a Platform fake without changing the Runtime consumer.
Then extend topology/communication, dynamic/shots, and checkpoint/realtime.
