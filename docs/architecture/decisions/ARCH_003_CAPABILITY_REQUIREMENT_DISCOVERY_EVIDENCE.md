# ARCH-003: Capability Requirements, Discovery Snapshots, and Evidence

Status: Proposed

Date: 2026-09-03

Basis: Phase 0 architecture inventory and Phase 2 Core/Platform/Runtime
reconciliation; an interface's existence does not establish implemented capability.

## Context

Compiler, Platform, Runtime, Deployment, and Extension each have local capability
types. These represent compilation legality, platform discovery, runtime policy,
or integration capabilities and cannot stand in for one another. Phase 2 needs
one verifiable set of cross-layer capability semantics without changing existing
public behavior.

## Decision Candidates

1. Merge all existing team types into one oversized capability object.
2. Keep parallel contracts that Runtime interprets temporarily.
3. Establish a Core-owned requirement/fact lineage with explicit adapters for
   existing types.

Choose option 3 to fix semantics and ownership without prematurely freezing
unstable future domains.

## Decision

Core v1 owns exactly this lineage, without a fourth cross-layer capability type:

1. `CapabilityRequirement`: one predicate from a closed capability vocabulary.
2. `RequirementSet`: an immutable collection of predicates required for this
   request/compilation and fallback authorization for each axis.
3. `TargetCapabilitySnapshot`: facts about one target/environment within a
   bounded time and scope.

`PlatformCapabilitySnapshotCandidate`, Runtime `BackendCapabilities`, Deployment
`CloudBackendProfile`, extension `CapabilityRequest/Response`, and private
`_compiler.TargetCapabilities` remain inputs owned by their respective domains.
Explicit adapters connect them to this lineage; they do not become Core aliases,
subclasses, or second fact authorities. `EvidenceReference` is a value referenced
by a snapshot, and `ExecutionObservation` records one attempt. Neither is a
capability type or automatically promotes a fact to long-term validity.

This ADR remains Proposed. It authorizes boundaries for subsequent minimal internal
implementations and adapters, without changing Stable Core, public exports,
existing serialization, failure stages, default choices, or product implementations.

## Minimal v1 Contracts

### `CapabilityRequirement`

Each requirement expresses one predicate:

- `name`: a name from the closed v1 vocabulary;
- `operator`: `equals | at_least | at_most | contains_all | covers`;
- `value`: the JSON-safe type defined for that name;
- `strength`: `mandatory | preference`;
- `source`: `user | compiler | runtime_protocol`;
- `minimum_evidence_level`: `basic | observable | certification`;
- `accepted_exposures`: a nonempty closed subset of fact exposure values.

Failure to satisfy `mandatory` eliminates a candidate. A `preference` only ranks
candidates already satisfying all mandatory predicates; it cannot make an
unexecutable candidate executable or authorize fallback. Core v1 does not provide
arbitrary expressions, callbacks, query languages, or free-form capability names.

### `RequirementSet`

The minimal envelope contains `schema_version`, `requirement_set_id`,
`requirements`, `fallback_authorizations`, and `extensions`. Requirements are
deduplicated and sorted by canonical key. Conflicting mandatory predicates fail
before target matching. `requirement_set_id` is the SHA-256 of canonical JSON for
the semantic payload excluding the ID itself.

Fallback has six independent axes: `backend`, `device`, `cpu`, `precision`,
`algorithm`, and `approximation`. Each has an explicit boolean, defaulting to
false. One axis cannot imply another. Preferences, Runtime policy, provider
declarations, and the existing broad `allow_backend_fallback` cannot imply
permission for CPU fallback or precision reduction.

### `TargetCapabilitySnapshot`

The minimal envelope contains:

- `schema_version` and `snapshot_id`;
- `target_identity`: `target_id`, `target_class`, `provider`, `provider_version`,
  `target_revision`, and `environment_id`;
- `scope`: nullable `device_ids`, `dtype`, `kernel`, `workload_id`, `world_size`,
  and `node_count`;
- `captured_at` and `valid_until`;
- `facts`, `evidence_refs`, `blockers`, and `extensions`.

Each fact contains exactly `name`, `value`, `support_status`, `fact_exposure`,
`source`, and `blockers`. A `source` contains only `kind` and a verifiable `ref`,
never SDK objects, handles, credentials, or callables. `snapshot_id` is the SHA-256
of canonical JSON for the complete semantic payload excluding the ID itself.
Presentation labels and mutable URIs do not enter identity. Every v1 snapshot
requires an explicit `valid_until`. Expiry, scope/identity mismatch, incorrect
hashes, or unresolvable evidence references fail closed. Platform/Provider policy
decides whether to rediscover; Core has no global TTL.

Each `evidence_refs` item contains at least `evidence_id`, `sha256`, `level`, and
`scope`. Raw material and review verdicts remain with Evidence/Audit owners;
snapshots reference them without copying or rewriting verdicts.

## Orthogonal Status, Exposure, and Evidence Semantics

Support status answers whether the available basis establishes support in this scope:

- `unknown`: the authoritative source has no answer;
- `unmeasured`: an interface or declaration exists, but the required threshold has
  not been measured;
- `unsupported`: an applicable, authoritative negative conclusion;
- `verified`: supporting material matches identity, scope, freshness, and value
  predicates.

Fact exposure answers how a value was obtained or why it is absent:

- `observed`: a controlled probe or workload observation;
- `declared`: a provider/specification declaration;
- `not_exposed`: the authoritative interface does not expose the field;
- `unknown`: the source is unclear, conflicting, or cannot be projected safely;
- `not_applicable`: inapplicable in this scope, with a blocker explaining why.

These are two orthogonal axes; neither implies the other. `declared` does not
become `verified` automatically. `not_exposed` is neither false nor proof that an
event did not occur. `unsupported/observed` is a valid negative observation.
`verified/declared` is valid only when the closed-set name is listed in the
machine-authorized `authoritative_static_declaration_allowed` validation profile,
the requirement accepts `declared`, and referenced material, scope, and freshness
all satisfy the rules. An unverified declaration remains `unmeasured/declared` by
default. Runtime facts such as capacity, precision, latency, physical routes, and
absence of fallback require `observed`. `unknown`, `unmeasured`, `not_exposed`,
and missing facts cannot satisfy mandatory requirements. `not_applicable` can
satisfy only a specialized predicate explicitly permitting inapplicability;
generic v1 matching rejects it by default.

Evidence strength is strictly `basic < observable < certification`. The required
threshold is the **strongest level** among all applicable mandatory requirements
and claim gates. The candidate's available ceiling is the **weakest level** among
all indispensable evidence references supporting those facts. Matching succeeds
only when that ceiling reaches the required threshold and every fact passes
status, accepted exposure, scope, freshness, and value comparisons. `verified`
means a current match under this complete rule, not release certification.
Capability or release claim levels must not exceed the available evidence ceiling
and must not hide known or observed blockers, fallback, or degraded paths. Mocks,
interfaces, environment variables, and CPU distributed semantic tests cannot
support real accelerator, QPU, production, or scalability claims.

## Closed v1 Vocabulary and Precision Rules

The first implementation authorizes only `target.class`, `device.kind`,
`device.count`, `memory.available_bytes`, `qubits.logical_capacity`,
`qubits.physical_capacity`, `gates.native`, `measurements.results`,
`artifacts.profiles`, `limits.maximum_shots`, `limits.maximum_program_operations`,
`ancillas.policy`, `ancillas.maximum_compiler`, and six independent precision names:

- `precision.native_dtype`;
- `precision.effective_dtype`;
- `precision.storage_dtype`;
- `precision.parameter_dtype`;
- `precision.accumulator_dtype`;
- `precision.software_mechanism`.

Software extension is a mechanism axis, not a dtype alias. With scope-matched
material, Double-Single can satisfy effective complex128 but cannot satisfy
native float64. Record actual storage, parameter, and accumulator dtypes,
mechanism, and error/range limits. The `covers`, set, and bound semantics for
`gates.native`, parameter domains, artifacts, measurements, qubits, limits, and
ancillas reuse the current `_compiler.TargetCapabilities` comparator. Migration
must not change its fingerprint or old schema.

Use these exact precision meanings; `float64` and `complex128` are not synonyms:

- `native_dtype`: the scalar type executed natively by the numerical kernel, such
  as `float32` or `float64`;
- `effective_dtype`: logical complex quantum-state precision valid for users and
  algorithms, such as `complex64` or `complex128`;
- `storage_dtype`: scalar type of each component actually stored in the state
  representation;
- `parameter_dtype`: scalar type of trainable parameters;
- `accumulator_dtype`: scalar type used for reductions and accumulation;
- `software_mechanism`: software that raises native/storage precision to effective
  precision, or `none` when unused.

An ordinary double-precision statevector path therefore uses `native=float64`,
`storage=float64`, `effective=complex128`, and `mechanism=none`. Double-Single uses
`native=float32`, `storage=float32`, `effective=complex128`, and an explicitly
recorded mechanism name.

## Matching and Fallback

Core provides pure comparison without policy. Validate requirement/snapshot
identity, scope, freshness, and references first, then check each fact's status,
exposure, evidence level, and operator/value. Return all typed blockers; do not
automatically degrade, select targets, or execute fallback.

Runtime owns candidate matching, ranking, leases, and final decisions. It creates
fallback candidates only with explicit authorization on the relevant axes. Each
alternative must fully rematch all still-applicable mandatory predicates using
its own snapshot. CPU is always an independent candidate, never a default after
device-resolution failure. It cannot reuse GPU memory, precision, routes, or
evidence. Record actual fallback in plans/results/evidence. A `not_exposed` route
cannot justify claiming that CPU/host fallback did not occur.

## Ownership

| Responsibility | Sole owner | Boundary |
| --- | --- | --- |
| Program and target legality | Compiler | Produce compiler-source mandatory predicates from IR/old targets and legalize against Runtime's selected snapshot; no discovery or fallback authorization. |
| Platform/target discovery | Compute / Remote | Produce adapter candidates, sources, and evidence references; do not decide workloads, claims, or target selection. |
| Candidate matching and policy | Runtime | Consume RequirementSet, snapshots, leases, and authorizations to select targets; do not alter Compiler legality or invent facts. |
| Algorithm candidates and cost hints | Simulation | Supply legal representations, resource estimates, errors, and partition hints; do not select physical devices, routes, or fallback. |
| Evidence from one execution | Runtime + executing Provider | Bind actual target/device/precision/route/distribution/fallback to an attempt; do not automatically backfill long-term snapshots. |
| Claim review | Audit/Release | Validate evidence and claim ceilings; Providers and Core comparators do not self-certify. |

## Explicitly Deferred from v1

Dynamic control, mid-circuit measurement/reset, checkpoint/restart,
realtime/session/latency, complete topology/placement/link models,
P2P/collective/physical communication routes, calibration lifecycle,
gradient/optimizer distribution, and full workload predicates do not enter the
v1 Core root vocabulary.

A demonstrated cross-layer need may use only a registered reverse-domain namespace
extension with an explicit owner, schema version, handler, tests, and exit or
promotion conditions. Unknown requirement namespaces or missing matcher handlers
fail closed. Unknown snapshot extensions may be retained for round-trip but
cannot satisfy Core predicates or raise claims. Promote stable extensions through
new minor/major contract proposals; do not preemptively include unstable domains
in v1.

## Prohibited Practices

- Do not infer fallback authorization from preferences, declarations, or default policy.
- Unobserved, hidden, or expired facts are not verified capabilities.
- Do not change historical contracts, fingerprints, default choices, or failure
  stages to avoid adapters.
- Do not present uncertified hardware, QPU, multinode, or performance results as
  verified claims.

## Compatibility

### Authorization and Migration

Existing `_compiler.TargetCapabilities`, `compare_target_capabilities`, Runtime/
Platform/Deployment/Extension capability types, and stable APIs remain unchanged.
The authorized next step is limited to internal Core values, strict JSON
readers/writers, canonical identities, pure comparators, contract fakes/conformance,
and narrow adapters for those existing types. Adapters must record owners,
information-loss blockers, and exit conditions. Unrepresentable fields remain
with their old authority and must not be silently discarded.

Root exports, stable signatures/defaults/exceptions, old schemas/fingerprints,
plan/result identities, default choices, and failure-stage changes still require
separate API Change Proposals. Real hardware/provider integration and capability
promotion still follow ARCH-006/008 certification paths. This decision does not
claim verification of any domestic hardware.

Implementation order: three Core value types and fakes; pure matcher; faithful
`_compiler.TargetCapabilities` adapter; CPU Platform adapter; Runtime matching seam
and independent CPU candidate; replacement tests with a second Platform/remote
fake; domain-specific proposals. Preserve old-path behavior at every step until
approved migration is complete.

## Acceptance Tests

- Strict schemas, canonical SHA-256, duplicate/conflicting predicates, and negative
  stale/scope/identity mismatch cases.
- Mandatory/preference behavior, five comparison operators, minimum evidence, and
  fail-closed unknown/unmeasured/unsupported/not_exposed behavior.
- Six precision axes and Double-Single's non-native precision.
- Six-axis fallback isolation, full rematching against an independent CPU snapshot,
  and route visibility.
- Unchanged current compiler fingerprints, Stable API, and historical fixtures.
- At least two producer adapters pass identical conformance without consumer changes.
- Domestic-hardware fixtures retain unknown/unmeasured status, a basic ceiling,
  and nonempty blockers.

Exact fields are defined in `flagquantum/core/target_capabilities.py` and verified
by capability matching tests across Core, Runtime, Compiler, and Provider. The
rationale is in `docs/development/VNEXT_PHASE2_CAPABILITIES_DECISION.md`.

## Open Questions

- When deferred domains meet stability and replaceability conditions for the Core
  closed vocabulary.
- Which adapters may enter the default runtime after a second independent
  producer passes conformance.
- How claim gates bind evidence to external evaluations and hardware certification.
