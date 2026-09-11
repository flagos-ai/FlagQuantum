# vNext Phase 2 Target Capabilities Integration Decision

> Commit references below have been mapped to the publication history.
> Recorded outcomes and approval status are unchanged.

Status: Integration decision; authorizes subsequent minimal internal implementation,
not a Stable API release.

Date: 2026-09-03

## Final Decision

Consolidate the three team proposals into one Core-owned model family:
`CapabilityRequirement` represents one predicate; `RequirementSet` groups
predicates and fallback authorizations; `TargetCapabilitySnapshot` records facts
with identity, scope, and validity. The Platform candidate is adapter input, not
a fourth retained type. Evidence references and execution observations describe
evidence and individual execution records, respectively; neither is a capability type.

Use sparse predicates and a generic fact envelope instead of merging Core,
Platform, and Runtime fields into one large dataclass. Preserve the verified gate,
parameter-domain, result, artifact, capacity, limit, and ancilla coverage semantics
of `_compiler.TargetCapabilities`, while deferring unstable dynamic, checkpoint,
realtime, topology, and communication domains.

## Exact v1 Shape

`CapabilityRequirement`: `name`, `operator`, `value`, `strength`, `source`,
`minimum_evidence_level`, `accepted_exposures`.

`RequirementSet`: `schema_version`, `requirement_set_id`, `requirements`,
`fallback_authorizations`, `extensions`.

`TargetCapabilitySnapshot`: `schema_version`, `snapshot_id`, `target_identity`,
`scope`, `captured_at`, `valid_until`, `facts`, `evidence_refs`, `blockers`, `extensions`.

Each fact: `name`, `value`, `support_status`, `fact_exposure`, `source`, `blockers`.
Each source: `kind`, `ref`. Each evidence reference: `evidence_id`, `sha256`, `level`,
`scope`.

Target identity contains exactly `target_id`, `target_class`, `provider`,
`provider_version`, `target_revision`, `environment_id`. Scope contains nullable
`device_ids`, `dtype`, `kernel`, `workload_id`, `world_size`, `node_count`. All
Core-owned objects reject unknown fields, enum values, and versions. `extensions`
is the only extension point.

The v1 operators are `equals`, `at_least`, `at_most`, `contains_all`, and `covers`.
Evidence levels are ordered `basic < observable < certification`. Support and
exposure are strictly orthogonal:
`unknown/unmeasured/unsupported/verified` states the support conclusion;
`observed/declared/not_exposed/unknown/not_applicable` states where a value came
from or why it is absent.

The required evidence threshold is the strongest level among applicable mandatory
requirements and claim gates. The candidate evidence ceiling is the weakest level
among all indispensable evidence references. A mandatory requirement passes only
when the ceiling meets the threshold and status, accepted exposure, freshness,
scope, and value comparisons all pass. `verified/declared` is reserved for
authoritative static specifications explicitly listed in the machine
authorization's `authoritative_static_declaration_allowed` profile. Bare
declarations default to `unmeasured/declared`. Capacity, precision, latency,
physical routes, and absence of fallback require observed evidence. Preferences
rank executable candidates only. Missing, stale, unknown, unmeasured, unsupported,
not-exposed, unsupported not-applicable, and unknown extension-handler cases all
fail closed.

## Precision and Fallback

Precision is not one string. v1 has six predicates: native, effective, storage,
parameter, accumulator dtype, and software mechanism. Double-Single is a software
mechanism: narrowly scoped evidence may support effective complex128, but it can
never represent native float64. Native, storage, parameter, and accumulator use
scalar dtypes; effective precision uses the logical complex quantum-state dtype.

Fallback authorization has six axes: backend, device, CPU, precision, algorithm,
and approximation. All default to prohibited. Every fallback creates a new
candidate and fully rematches its own snapshot. CPU is an independent normal
candidate, not a resolver default. Broad backend fallback, preferences, or policies
cannot imply CPU fallback authorization.

## Ownership

- Compiler: program/target legality and compiler-source requirements; no platform
  discovery or fallback authorization.
- Platform/Execution Provider: discover platform or target facts and provide
  source/evidence references; no selection.
- Runtime: combine sources without rewriting them; own matching, ranking,
  leases, placement, and fallback decisions.
- Simulation: legal algorithm candidates and resource/cost/error estimates;
  estimates must not become platform facts.
- Runtime and executing Provider: actual routes, precision, devices, fallback,
  and distribution evidence for one attempt.
- Audit/Release: claim eligibility. Interfaces, mocks, and provider declarations
  cannot substitute for certification.

## v1 Closed Set and Deferred Domains

The first closed set covers only target/device/count/memory,
qubit/gate/measurement/artifact/limit/ancilla, and the six precision axes. Defer
dynamic execution, checkpoint/restart, realtime/session, full topology/placement/
links, P2P/collectives/physical routes, calibration lifecycle, gradient/optimizer
distribution, and a complete workload language.

Deferred domains may initially use registered namespaced extensions with an owner,
version, matcher, tests, and promotion/exit criteria. Unknown handlers fail closed.
Once stable, domains enter the Core closed set through subsequent contract
proposals. This preserves an evolution path without turning v1 into a general
rule engine for every future scenario.

## Authorization and Prohibitions

This decision authorizes subsequent commits implementing internal Core value
objects, strict serialization, canonical identity, a pure matcher, contract
fakes/conformance, and narrow adapters for `_compiler.TargetCapabilities`, CPU
Platform, Runtime backends, and Deployment/Execution Providers. Old fields an
adapter cannot express must retain blockers and remain under the old authority.

It does not authorize product implementation changes in the decision commit;
Stable Core exports; changes to existing public APIs, schemas, fingerprints,
plans/results, default backend/fallback behavior, failure stages, or capability
maturity. It does not authorize real network, provider, or hardware integration
and establishes no domestic-hardware, FlagCX, multinode, QPU, production, or
scalability validation.

## Subsequent Implementation Order

1. Core implements the three internal value-object types, strict readers/writers,
   identities, and contract fakes.
2. Core implements a pure comparator covering mandatory/preferences, all five
   comparisons, evidence, and stale-data rejection.
3. Compiler adds a faithful adapter for existing `_compiler.TargetCapabilities`
   without changing old fingerprints/comparators.
4. Platform integrates CPU first, then demonstrates replacement conformance with
   a second fake/remote producer.
5. Runtime integrates the matching seam, independent CPU candidates, six-axis
   fallback, and decision/evidence association.
6. Deferred domains submit individual contract proposals. Real hardware
   capabilities follow a separate certification path.

Implementation authority resides in `flagquantum/core/target_capabilities.py` and
the Core, Runtime, Compiler, and Provider matching/replacement tests. Historical
implementation authorization records no longer serve as code contracts.

## Phase 2 Integration Closure Status (2026-09-04)

The minimum internal workflow above completed at integration commit `f25c4bc4faa6cce8618f84bbb75542f4b51adf6c`:
Core contracts/matcher, Compiler adapter, CPU Platform producer, Runtime policy
seam, and synthetic Execution second-producer replacement conformance were merged
and passed targeted integration verification. Verification reported `121 passed`
after Runtime integration and `118 passed` after the second-producer fix. Both
runs passed architecture, team scope, Ruff, and diff checks.

Completion here covers only the internal boundaries authorized by this decision.
The Runtime seam is not on the default execution path; its internal decision
record is not stable Execution Request/Result/Evidence. The synthetic producer
uses no network, provider SDK, or real hardware. Public API changes, default
backend/fallback changes, plan/result changes, actual execution evidence, real
provider/hardware integration, and all deferred domains remain unauthorized.

The next phase should first resolve Runtime candidate provenance/trusted non-CPU
fallback binding, joint use of the Compiler legacy comparator, and remaining
determinism/CPU full-rematch test debt. Only then should execution observation/
evidence proposals and certification prerequisites for real domestic Platform
adapters receive separate review. This status record does not start those tasks
or alter the decision's machine authorization boundary.
