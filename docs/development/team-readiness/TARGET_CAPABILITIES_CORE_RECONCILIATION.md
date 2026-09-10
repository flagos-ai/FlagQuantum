# TargetCapabilities: Core Reconciliation and Minimal Contract Candidate

Status: Core proposal input awaiting Integration/API owner review; not API,
schema, or ADR approval.

Date: 2026-09-03

Branch: `codex/vnext-phase2-capabilities-core`

Path classification: this work adds inventory documentation and CPU
characterization tests only. It changes no execution path or capability claim.
Tests establish object structure and fail-closed behavior, not hardware,
production, communication, or scalability evidence.

## 1. Recommendation and Authority

Core should eventually own two narrow, independent, versioned, provider-neutral
objects instead of putting every capability concept in `TargetCapabilities`:

1. `CapabilityRequirement`: closed artifact/request constraints describing what
   a target must provide.
2. `TargetCapabilitySnapshot`: facts discovered for a target and environment at
   a particular time, including source, support status, and acquisition method.

A Core-owned fail-closed comparator matches them. `EvidenceReference` points to
supporting material; `ExecutionObservation` records one workload's actual
execution. Neither inherits nor permanently raises a snapshot's truth status.
Use these distinct names during migration to avoid competing authorities. A
stable `TargetCapabilities` alias should be considered only after old consumers
are gone and an API/contract proposal approves it.

Existing `_compiler.TargetCapabilities` provides the strongest reusable baseline
for requirements and compiler target semantics. Preserve gates, parameter domains,
results, artifact profiles, topology, control flow, limits, ancillas, and
calibration vocabulary. It is neither a complete discovery snapshot nor a public
authority; do not simply move or extend it in place. Runtime
`BackendCapabilities`, Deployment `CloudBackendProfile`, Platform records,
operator probes, and extension negotiation remain owner-produced inputs that
explicit adapters project into Core snapshots. Algorithm-local capabilities stay
outside Core.

## 2. Six Distinct Concepts

| Concept | Question | Inference restriction | Proposed representation |
| --- | --- | --- | --- |
| Requirement | What must the workload/artifact satisfy? | Cannot establish target support | `CapabilityRequirement` |
| Discovered fact | What did a source report about this target/environment at this time? | Installation, interface existence, and declarations do not establish verification | Snapshot typed facts |
| Support status | Is support sufficiently established? | Declared does not imply verified | Per-fact `unknown/unmeasured/unsupported/verified` |
| Fact exposure | How was a value obtained, or why is it absent? | Hidden does not mean false; declared does not mean observed | Per-fact `observed/declared/not_exposed/unknown/not_applicable` |
| Evidence level | How strong a claim is permitted? | Claims cannot exceed the weakest relevant evidence | Snapshot ceiling and `EvidenceReference.level` |
| Execution observation | Which path did one workload actually take? | Does not become a permanent target fact automatically | `ExecutionObservation`/evidence artifact |

Support and exposure are independent. Valid combinations include
`unsupported + observed`, `verified + observed`, `unknown + not_exposed`, and
`unmeasured + declared`. Invalid cases include `verified + not_exposed`, unknown
without a blocker, and automatic promotion from declared to verified.
`not_applicable` requires an applicability reason. Known fallback belongs in
execution observations. Hidden routes remain unknown/not_exposed with blockers;
they cannot prove absence of CPU fallback.

Evidence has the ordered ceiling `basic < observable < certification`.
Interfaces, environment variables, mocks, skipped hardware tests, and CPU
distributed-semantics tests establish only their scoped basic/semantic evidence.
They do not raise domestic-hardware, production, or scalability claims.

## 3. Existing Types, Consumers, and Identities

| Type | Producer | Consumers | Fields/granularity | Serialization/identity | Assessment |
| --- | --- | --- | --- | --- | --- |
| `_compiler.TargetCapabilities` | Fixtures, provider/conformance/bridge construction | Comparison, legalization, dry runs, sealing, provider conformance | Complete compiler-target semantics | Strict `to_dict/from_dict`; `target_capabilities_v1`; semantic JSON SHA-256; rejects unknown top-level/nested fields and versions; excludes `display_label` from fingerprint | Strong baseline, but required/available share a type and lack discovery/evidence axes |
| `core.CapabilityContract` | Compilation contract adapter | `RuntimePlanContract`, records | Backend/devices/dtypes/modes, autograd/distributed | Versioned Core round trip/hash; lossy plan audit projection | Insufficient as target snapshot |
| `runtime.BackendCapabilities` + `AcceleratorInfo` | Registry/platform discovery | Backend/device/mode selection, root compatibility export | Tensor backend, devices, dtype, mode booleans, current accelerator | No `from_dict` or snapshot ID; unknown collapses to false/absence | Dynamic Runtime inputs requiring adaptation |
| `runtime.OperatorProfile` / `OperatorRequirement` | Packaged JSON profiles | Operator preflight | Workload operator/dtype/forward/backward/determinism | Strict reader, canonical profile hash | Requirement source, not target facts |
| `runtime.CapabilityEvidence` | Operator probes, hardware CI, provider declarations | Preflight | Operator/dtype probe | `evidence_id` hashes full dataclass; `is_verified` requires passing runtime probe/hardware CI | Evidence input, separate from target semantic fields |
| `compute.PlatformDevice/Identity/MemorySnapshot` | CPU/CUDA/FlagOS providers | Runtime discovery, diagnostics | Device/runtime identity and possibly unknown memory | Identity has one-way `to_dict`, free metadata, no snapshot hash | Platform facts; availability is not workload support |
| `deployment.CloudBackendProfile` | Cloud/provider adapters | Packages, QPU providers | Provider/name/wires/gates/coupling/formats/dynamic/simulator/metadata | No strict `from_dict` or independent identity | Temporary projection; metadata cannot enter Core semantic root directly |
| `extensions.CapabilityRequest/Response` | Extension callers/provider negotiation | Registry/conformance | Required names, dtype/device/gradient; accepted/supported/blockers | No standard round trip or identity | Protected negotiation messages, not requirement/snapshot authority |
| `compiler.operator_lowering.LoweringCapability` | Lowering registry | Compiler/backend lowering validation | Backend/opcode/strategy/implementation/supported/reason | Registry-local manifest projection | Compiler implementation availability, not execution target support |
| `runtime.FlagOSWorkloadCapability` | F1/F2/F3 evidence aggregation | Benchmark contracts/docs | Workload status, collectives, world sizes, dtype, distribution, evidence level | Specialized v1 matrix, one-way dict | Reviewed derived conclusion, not generic snapshot |
| Core `ExecutionObservation`, Runtime evidence/route/fallback records | Attempt/audit assembly | Results, records, audits | Elapsed/memory/communication/completion and Runtime path facts | Separate, nonuniform versions/hashes | Workload observations connected by references only |

MPS workspace, JAX/TN planning, kernel dispatch, and noise device profile objects
answer local algorithm or implementation questions. A capability-like name does
not justify moving them into Core when they do not cross provider boundaries.

### 3.1 Exact Existing Compiler Fields

The semantic identity of `_compiler.TargetCapabilities` includes:
`schema_version`, `target_class`, logical/physical qubit capacities, canonically
sorted `native_gates` with closed parameter constraints, `measurement_results`,
`artifact_profiles`, directed coupling topology, `control_flow`, six booleans for
mid-circuit measurement/reset/timing/pulse/noise/parameter binding, shot and
operation limits, ancilla policy/capacity, calibration hash and validity.
`display_label` is serialized but deliberately excluded from the fingerprint.

The comparator uses the same type for required and available. It requires set
inclusion, sufficient capacities/limits, true available booleans when required,
parameter-domain coverage, topology-edge inclusion, and equal calibration
references. It fails closed but cannot represent unknown/unmeasured states,
freshness, precision, platform/physical routes, communication, or claim ceilings.

## 4. Minimal Core Contract Candidate

These are schema review inputs, not implementations on this branch.

### 4.1 Shared Closed Vocabularies

- `SupportStatus = unknown | unmeasured | unsupported | verified`
- `FactExposure = observed | declared | not_exposed | unknown | not_applicable`
- `EvidenceLevel = basic | observable | certification`
- Reuse validated `_compiler` meanings for `TargetClass`, artifact formats/profiles,
  measurement results, control flow, ancilla policy, and gate/parameter constraints.
  Preserve accepted values before migration.
- New open vocabulary belongs only in namespaced extensions, not free-string
  additions to stable root capabilities.

### 4.2 CapabilityRequirement

| Group | Fields |
| --- | --- |
| Envelope | `schema`, `schema_version`, `requirement_id` |
| Target semantics | `target_classes`, `minimum_logical_qubits`, `minimum_physical_qubits`, `native_gates`, `measurement_results`, `artifact_profiles`, `topology_requirement`, `control_flow`, `ancilla_requirement` |
| Execution constraints | `required_features`, `limits`, `precision_requirement`, `communication_requirement`, `distribution_requirement`, `fallback_policy` |
| Proof constraints | `minimum_evidence_level`, `required_evidence_scopes` |
| Evolution | `extensions` |

`requirement_id` is SHA-256 of the complete canonical semantic payload excluding
itself. Requirements exclude provider identity, capture time, support status,
live queue/calibration state, measured metrics, and claim verdicts.
`required_features` uses versioned closed keys; unknown keys fail closed.

### 4.3 TargetCapabilitySnapshot

| Group | Fields |
| --- | --- |
| Envelope | `schema`, `schema_version`, `snapshot_id` |
| Target identity | `target_id`, `target_class`, `provider`, `provider_version`, `target_revision`, `display_label` |
| Environment identity | `environment_id`, `runtime_versions`, `platform_identity`, `physical_devices`, `code_revision` |
| Capture/source | `captured_at`, `valid_until`, `source_kind`, `source_id`, `source_digest`, `blockers` |
| Typed facts | `qubits`, `gates`, `measurements`, `artifacts`, `topology`, `control_flow`, `features`, `limits`, `ancillas`, `calibration`, `precision`, `communication`, `distribution`, `fallback_visibility` |
| Proof | `evidence_level`, `evidence_refs`, `claim_ceiling` |
| Evolution | `extensions` |

Each `CapabilityFact<T>` contains nullable `value`, `support_status`,
`fact_exposure`, `source_ref`, `blockers`, and optional `observed_at`/`valid_until`.
Null alone does not mean unknown: status, exposure reason, and blocker are also
required. Snapshot evidence level is the ceiling jointly supported by relevant
facts; stronger or narrower material remains in evidence references.

`snapshot_id` hashes canonical semantic content including target/environment,
capture/source, typed facts, evidence references, blockers, claim ceiling, and
semantic extensions. Exclude only the ID itself and presentation-only
`display_label`. Different capture times or environments produce different IDs.
Use stable digests/references for mutable artifact URIs; keep URI as display data.

### 4.4 Extension Namespaces and Unknown Fields

- Top-level and Core-owned nested objects are closed. Reject unknown fields,
  enums, and major/schema versions.
- The sole extension point is
  `extensions: {"<reverse-dns-or-flagquantum-namespace>": section}`. Namespaces
  are nonempty, canonical lowercase, and owner-registered. Sections contain
  finite JSON-safe data without credentials or vendor objects.
- Readers may preserve unknown sections for round trips. Comparators reject
  unhandled semantic requirement extensions as `unknown_extension_requirement`;
  they must not silently ignore them.
- Namespace registration fixes identity inclusion. Conservatively include all
  extensions in the first version; changing this is a schema/identity change.
- An unknown fact is a typed business state. An unknown field is an unrecognized
  schema member. Free metadata must not bypass the closed schema.

## 5. Comparison and Compatibility

The comparator accepts one requirement and one unexpired snapshot. It must not
accidentally compare two snapshots or two requirements.

1. Validate schema, canonical identity, target/environment binding, and
   `captured_at/valid_until`. Stale, incomplete, or mismatched identities are incompatible.
2. Only `support_status=verified` satisfies a requested fact. Unknown, unmeasured,
   and unsupported produce distinct typed blockers. Unrequested facts do not
   affect compatibility.
3. Reuse subset, capacity/limit coverage, gate parameter, topology edge,
   control-flow, and ancilla rules from the existing comparator.
4. Compare storage/compute/accumulation, native/software mode, dtype, and workload/
   kernel scope. Double-Single cannot satisfy native float64.
5. Compare node/world/local-world sizes, rank placement, physical links, and
   sources. Logical device names or inferred topology cannot meet observed/
   certification requirements.
6. Compare outer process-group backend, inner physical route, collectives, dtype,
   residency, and host staging. `outer_backend=flagos` does not imply
   `inner_backend=flagcx`.
7. A prohibition on CPU/backend substitution requires observable routes. Hidden
   routes cannot establish no fallback. Execution observations confirm whether
   fallback actually occurred.
8. Compare `basic < observable < certification`. The final claim ceiling is the
   minimum relevant requirement/fact/evidence/observation level. Sharding claims
   also need workload-specific observations and complete distribution fields.
9. Return all typed differences: path, required value, available value/status/
   exposure, code, evidence references, and blockers. Do not degrade or fall back
   automatically. Policy may construct a separate explicitly authorized
   requirement; it must not alter the original requirement or snapshot.

## 6. Domestic Accelerators, Precision, Topology, and Communication

The first version must distinguish:

- Logical `platform_identity.logical_device_type` from physical
  `physical_devices[].vendor/model/id`. FlagOS-on-CUDA/NVIDIA A800 is not a
  domestic physical accelerator.
- Precision `storage_dtype`, `compute_dtype`, `accumulation_dtype`,
  `mode=native|software`, `software_scheme`, and `kernel/workload_scope`.
  Double-Single is software precision retaining the FP32 exponent range. It may
  provide effective complex128 within a certified scope, never native float64.
- Topology `node_count/world_size/local_world_size`, rank-to-node/device ownership,
  links, `source=observed|provider_declared|inferred`, and fingerprint. Unknown
  links stay unknown.
- Communication outer backend, inner route, P2P/collective/dtype matrix, residency,
  bytes, `host_staging_observed`, and route verification. A working logical
  process group does not prove FlagCX, absence of host staging, or physical links.
- Closed distribution categories: `single_device_fast_path`,
  `sharded_across_ranks`, `rank_local_replicated_kernel`,
  `manual_sliced_tensor_contraction`, `observable_term_parallel`,
  `data_parallel_replicated`, and `replicated_per_rank`. Capacity/scalability
  claims require workload-specific sharding evidence.

CPU fixtures can contain actual local observations, but unknown memory/topology/
communication must not become zero. CUDA evidence is scoped to device, kernel,
dtype, and workload. FlagOS-on-CUDA records logical FlagOS and physical NVIDIA.
Real domestic-device fixtures remain unknown/unmeasured with basic claim ceilings
and nonempty blockers until checked-in, reviewed attestations and Runtime
observations exist.

## 7. Adapter and Retirement Ledger

| Adapter | Owner | Input -> output | Preserve / do not infer | Exit condition |
| --- | --- | --- | --- | --- |
| Compiler target | Compiler | Private `TargetCapabilities` -> Core requirement or declared snapshot | Preserve fingerprint mapping; old booleans do not become verified; absent provenance remains declared/unmeasured | Compiler uses Core types, zero old importers, golden/compatibility fixtures pass; Integration approves target version |
| Runtime backend | Runtime | `BackendCapabilities`, profiles, probes -> snapshot | Registry booleans/device visibility do not establish workload support; preserve probe references | Selection/preflight consume Core snapshots; zero old cross-layer field importers |
| Platform discovery | Platform | Device/Identity/Memory -> fact sections | Unknown memory/topology/routes remain null plus state; no vendor-object leakage | Approved snapshot producer conformance; zero old adapter importers |
| Deployment profile | Execution Provider | `CloudBackendProfile` -> target snapshot | Metadata enters registered namespaces only; provider claims default to declared | Core snapshots from provider capability API; old reader compatibility window complete |
| Extension negotiation | Ecosystem | Core requirement <-> protected CapabilityRequest/Response | Project expressible subset; block unrepresentable requirements rather than accepting after dropping fields | New protocol approved/released and old protocol deprecated; retain adapter until then |
| Agent vocabulary | Agent | Aliases/old names <-> Core requirement/snapshot views | Versioned aliases only; unknown names fail closed | Agent reads Core services only; alias usage zero and removal version reached |
| Evidence reference | Runtime/Audit | CapabilityEvidence/Runtime evidence/artifact -> EvidenceReference | Preserve digest, source, workload/environment/revision; do not copy raw verdicts into target facts | Core envelope/audit conformance approved; references verifiable |
| FlagOS workload | Runtime/Audit | Workload matrix -> snapshot/evidence references | Preserve NVIDIA physical identity, single-node scope, false claim flags, blockers | Generic schema losslessly represents F1-Fn payloads; release auditor consumes new references |

Each adapter implementation records owner, start version, supported old schemas,
metrics, and removal version or condition. “Replace later” is not an exit plan.
Follow ARCH-003: vocabulary/fixtures, Compiler, Runtime/Platform,
Deployment/Provider, then Agent/alias retirement.

## 8. API and Contract Change Triggers

Integration-approved Contract Proposals/ADRs are required for new Core types or
provider protocols, canonical fields/values/identities, `contracts/` changes,
architecture rules, cross-team ownership, and compatibility/retirement changes.

API Change Proposals and API owner approval are additionally required for:

- new capability types in root/stable namespaces or public annotations;
- moving/re-exporting stable `BackendCapabilities`, `CloudBackendProfile`, or
  extension messages, or changing class identity, field order/defaults/frozen state;
- schema, unknown-field/version rejection, canonical ordering, round-trip, or
  fingerprint-input changes, including identity inclusion of `display_label`;
- capability failure stages, exception categories, fallback visibility, or stable
  plan/result fields and serialization;
- removal of compatibility entries, changed negotiation acceptance, or a new
  stable snapshot backward-reading commitment.

Characterization tests and proposal documents alone do not trigger API changes.
After shared contracts are approved, private adapters count as internal work only
when they preserve inputs, outputs, exceptions, and identity completely.

## 9. Acceptance, Open Questions, and Scope

Minimum integration tests cover satisfied/missing/unknown requirements, stale
snapshots, both state axes, evidence monotonicity, unknown extensions, canonical/
golden identity, old fixtures, CPU/CUDA/FlagOS-on-CUDA/fake-QPU/unverified-domestic
fixtures, precision distinctions, hidden routes and fallback, complete sharding
fields, and negative claim-ceiling cases. Compiler, Runtime, and Agent must see
the same snapshot identity. Replace at least one Runtime and one QPU/remote
implementation without changing consumers.

Integration/API owner decisions remain:

1. Stable names and eventual ownership of `TargetCapabilities`.
2. Minor/additive requirement-vocabulary registration and extension-handler registry ownership.
3. TTL/calibration-staleness policy ownership and whether offline targets may omit TTL.
4. Workload predicate expressiveness without becoming arbitrary code or a query language.
5. Evidence review signatures in Core envelopes versus external Audit, and URI/digest retention.
6. Explicit identity exclusions for labels, URIs, and nonsemantic diagnostics.
7. Whether old Compiler fingerprints map to new identities through stable migration fields.
8. Cloud/extension public-type deprecation windows and target versions.

This work changes no protected product contract, public export, schema, hash,
Runtime behavior, capability maturity, or benchmark claim.
