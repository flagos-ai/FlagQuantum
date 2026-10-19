# API Change Proposal 070: Research capability services

Decision support: see
[`API_CHANGE_PROPOSAL_070_APPROVAL_PACKET.md`](API_CHANGE_PROPOSAL_070_APPROVAL_PACKET.md)
for the exact owner checklist and verified cross-repository evidence. That
packet is not itself an approval.

## Status

**Approved and implemented at production-supported maturity on 2026-10-13.** The user
supplied the proposal's exact authorization token as the authority for both the
API-owner and Services-domain-owner roles. This
proposal defines a protocol-independent application-service contract for
discovering and executing scientific workflows such as the existing
provider-owned ground-state VQE service. It does not add a Stable Core root
export, change `fq.run` or `fq.plan`, alter an existing serialized schema, or
authorize an MCP dependency in this repository. The approval permits a thin
external adapter to call this public namespace, but it does not authorize MCP
tool/resource registration before the separate maturity and release gates pass.

The machine-readable candidate is
[`research-capability-services-v1-candidate.json`](../../contracts/research-capability-services-v1-candidate.json).
The candidate records the approved contract boundary and the first three
operation identities. All six operations and the P0 local slice are implemented;
incompatible changes to their approved shape require a new reviewed proposal.

The owner decision approved the exact namespace operations, signatures, value
shapes, schema identities, fail-closed behavior, and initial experimental
binding recorded in the machine-readable candidate. A subsequent explicit
release/maturity decision promoted only this bounded local slice to
`production_supported`. An earlier record selected `0.3.0rc3` as the first
compatible release, but the published `0.3.0rc3` and `0.3.0` wheels omit this
service API. The first compatible released version and MCP dependency range
remain unset until an installed-wheel check passes. Neither decision authorizes remote execution,
automatic provider substitution beyond the explicit local unavailable-only
policy below, MCP dependencies in this repository, or an MCP execution tool
before the separate adapter gate is satisfied.

## 1. Measured gap

FlagQuantum already has three different things that use the word capability:

1. `capability-maturity.toml` records product maturity and claim boundaries;
2. `flagquantum.core.target_capabilities` records target requirements, observed
   target facts, and evidence-qualified matching;
3. `flagquantum.services.capabilities()` aggregates installed backend facts and
   the bounded ground-state VQE service description.

None of these identifies a scientific operation independently of its provider.
The existing `run_ground_state_vqe(...)` path is executable and returns
provider-owned evidence, while `exact_sector_reference(...)` supplies a separate
reference calculation. A caller still cannot ask one deterministic FlagQuantum
surface to:

- find the operation by scientific task;
- inspect its input, applicability, validation, and provider limits;
- estimate before execution;
- distinguish execution completion from scientific validation;
- receive one transport-neutral evidence envelope.

An edge adapter can fill these gaps only by duplicating scientific semantics.
That would make MCP, REST, Python, and future adapters disagree. The reusable
composition therefore belongs in FlagQuantum Services; protocol decoding and
serialization remain out of tree.

## 2. Contract identity and naming

The proposed family uses **research capability** consistently. This distinguishes
an executable scientific operation from a target capability fact in ARCH-003 and
from a maturity claim in `capability-maturity.toml`.

The first operation identities are:

```text
ground_state.solve.vqe
ground_state.solve.exact
validate.energy_variance
```

An operation identity describes scientific meaning, not an implementation.
Provider identity is an independent manifest field. The native implementation
uses `flagquantum-native`; future implementations of the same operation retain
the operation ID and declare their own provider and adapter versions.

The proposed namespace API is deliberately not exported from `flagquantum` root:

```python
from flagquantum import services

services.search_research_capabilities(query)
services.describe_research_capability(capability_id, version=None)
services.estimate_research_capability(capability_id, request, target=None)
services.validate_research_capability_input(capability_id, request)
services.execute_research_capability(capability_id, request, options=None)
services.validate_research_evidence(validator_id, evidence)
```

The names state their domain directly and avoid collision with target-capability
matching. No generic manager, mutable global plugin registry, natural-language
router, or transport facade is proposed.

Two read-only schema-resource operations accompany, but do not expand, those six
scientific operations:

```python
services.list_research_schemas()
services.describe_research_schema(schema_id)
```

They return the installed FlagQuantum JSON Schema Draft 2020-12 documents for
manifests, requests, results, validation reports, structured errors, and evidence
bundles. They exist so an out-of-tree adapter can expose resources without
copying or redefining FlagQuantum-owned shapes. Unknown schema IDs fail closed,
and returned documents are isolated copies.

## 3. Relationship to existing authorities

### 3.1 Target requirements and facts

This proposal does not create another hardware or backend capability model.
Applicability that depends on target facts projects into ARCH-003's existing
`RequirementSet` and is matched against a `TargetCapabilitySnapshot`. Research
manifests may reference those requirements; they do not copy target facts or
implement a second comparator.

### 3.2 Execution request and policy

User intent, budgets, seed, target constraints, and fallback authorization must
remain compatible with ARCH-004. A research-capability request is a typed
scientific payload plus a compatible execution-policy projection. It does not
contain credentials, tenant state, provider SDK objects, or an LLM plan.

### 3.3 Result and evidence

`ResearchCapabilityEvidenceBundle` is an application-service aggregation, not a
replacement for stable `ExecutionResult`, compilation evidence, runtime evidence,
or the future ARCH-005 Execution Record. It references or embeds their public
projections, preserves their identities, and cannot raise their evidence or claim
level.

The first local VQE slice may record its existing provider result directly while
the ARCH-005 record remains proposed. The bundle must describe that limitation and
must not present a deterministic digest as runtime certification.

### 3.4 Product maturity

`capability-maturity.toml` remains the sole maturity authority. A research
manifest binds to a maturity entry by ID after that entry exists. The manifest
does not independently promote itself. Until the binding and its evidence gate
exist, an external protocol adapter may expose description but must not claim the
operation is production supported.

## 4. Minimal v1 values

### `ResearchCapabilityManifest`

The minimal manifest records:

- `capability_id`, semantic `version`, provider, and adapter version;
- task domain, task name, description, and tags;
- strict input and output schema identities;
- applicability and known limitations;
- dependency, target-requirement, and resource-estimation declarations;
- validator IDs and fallback candidates;
- provenance, license, implementation status, maturity binding, and test status.

Schemas are closed and versioned. Units, basis order, precision, and coefficient
domains are input semantics, not prose-only documentation.

### `ResearchCapabilityCandidate`

A candidate records manifest identity, applicability, deterministic match score,
resource estimate when available, validators, fallbacks, and structured reasons.
Mandatory incompatibility eliminates a candidate. Preferences rank only candidates
that already satisfy mandatory constraints.

### `ResearchValidationReport`

A validation report records `passed`, validator identity, checks, metrics,
warnings, and blockers. Input validation, execution completion, and scientific
validation remain distinct stages. An optimizer returning normally does not make
scientific validation pass.

### `ResearchCapabilityError`

The error contract records code, stage, retryability, message, context, evidence
already generated, and suggested actions. The v1 stages are:

```text
catalog_match
input_validation
resource_estimation
environment_preflight
compilation
execution
result_parsing
scientific_validation
artifact_persistence
```

### `ResearchCapabilityEvidenceBundle`

The bundle records run identity, capability/provider identity, status, result,
metrics, artifacts, execution facts, validation reports, attempts, warnings, and
provenance. The v1 statuses are:

```text
succeeded
unsupported
validation_failed
failed
```

`succeeded` requires every mandatory scientific validator to pass. A provider
success with a failed mandatory validator is `validation_failed`, not success.

## 5. Ownership boundary

| Responsibility | Owner |
| --- | --- |
| Backend-neutral manifest and evidence value semantics | Core, after separate contract approval |
| Scientific operation implementation and validators | Algorithms or the owning scientific domain |
| Reusable discovery, validation, estimation, execution, and evidence composition | Services |
| Target matching, target ranking, leases, and authorized fallback | Runtime |
| Directly controlled target facts | Compute |
| Remote job and QPU provider lifecycle | Remote |
| Protocol request decoding, tool registration, and JSON response encoding | Out-of-tree MCP/REST/CLI adapter |
| Natural-language interpretation and research decisions | External client/Compute Service |
| Tenant, authentication, quota, billing, and durable task state | External Compute Service |

The main repository must remain usable with no MCP SDK installed. MCP tool names,
prompts, resources, transport annotations, and protocol lifecycle do not enter
this contract.

## 6. P0 vertical slice

The first implementation extends the existing two-qubit ground-state workflow;
it does not generalize VQE before the vertical path is complete.

1. `ground_state.solve.vqe` adapts `run_ground_state_vqe(...)`.
2. `ground_state.solve.exact` adapts `exact_sector_reference(...)`.
3. `validate.energy_variance` computes the final-state Hamiltonian variance
   independently from the optimizer's convergence condition.
4. The executor validates input and estimates resources before numerical work.
5. VQE execution automatically invokes the two mandatory validators when their
   applicability conditions hold.
6. The evidence bundle distinguishes unsupported input, execution failure,
   completed-but-invalid science, and validated success.
7. Backend, device, CPU, precision, algorithm, and approximation fallback remain
   prohibited. Provider fallback is disabled by default and requires an explicit
   ordered provider list plus `allow_provider_fallback=True`.

The fixed-budget optimizer does not expose an early-stop signal. The service
therefore records a bounded diagnostic rather than inventing one: optimizer
convergence is established only when the energy span over the final five
objective evaluations is at most `5e-6` Hartree. Fewer than two observations or
a wider span records `OPTIMIZER_CONVERGENCE_NOT_ESTABLISHED`; when the configured
evaluation limit was also consumed, `optimizer_budget_exhausted` is true. This
diagnostic never substitutes for the mandatory exact-energy and variance
validators.

The P0 scope remains real two-qubit Pauli sums, the declared basis convention,
Hartree coefficients, fixed Hamming weight one, the existing single-excitation
ansatz, bounded Adam, and local statevector execution. Broader Hamiltonians,
ansatzes, optimizers, noisy execution, shots, remote targets, and QPUs remain
unsupported until separate slices earn them.

### 6.1 M3 first provider-routing slice

`ResearchExecutionOptions.provider_order` defaults to the single native provider,
so existing calls retain exactly their previous selection. More than one provider
fails before execution unless `allow_provider_fallback` is explicitly true. Even
when authorized, the first slice advances only when an adapter is absent or its
local availability preflight fails. An ordinary execution exception, unsupported
scientific input, or failed scientific validator does not trigger substitution.

Every considered provider is retained in `attempts`, including unknown and
unavailable provider IDs. `execution.provider_order`, `fallback_used`, call
counts, timing, the selected provider identity, and the complete attempt history
are covered by the bundle content digest. Provider objects remain an immutable,
process-local dependency seam rather than a public mutable registry.

The provider-owned vertical slices install `qiskit-aer` and
`pennylane-lightning` as optional adapters without changing the native default.
Their availability probes do not import optional SDKs, and a missing dependency
is a preflight-unavailable condition rather than an execution failure. Vendor
objects stay inside their respective `flagquantum.ecosystem` packages; the
service receives only an owned VQE result. Shared FlagQuantum mechanics build
the bounded ansatz, use exact parameter-shift gradients with the request's Adam
settings, and assemble provider-neutral scientific evidence.
`objective_evaluations` retains the optimizer's reported objective points,
while `backend_executions` separately records the additional external
statevector executions used by shifted gradients.

The service also records a deterministic `metrics.provider_route` explanation.
It assesses every requested adapter, preserves caller order, assigns only a
rank-derived score, and selects the first available provider without automatic
reordering. Each candidate records installation and availability state, adapter
version, preflight reason/error code, selection decision, and its adapter-owned
backend-execution estimate. This route plan is covered by the Evidence Bundle
content digest. It is an explanation of explicit policy, not an autonomous
cost/performance optimizer.

## 7. Compatibility

This proposal is additive and namespace-scoped. It changes no root export,
existing signature, result field, exception category, default backend, default
provider selection, default fallback behavior, or maturity status. The candidate
Evidence Bundle schema now closes the execution-observability object and adds
provider-order fields before API approval rather than changing a released schema.

The first implementation must remain experimental and internal to
`flagquantum.services` until:

1. strict schemas and negative fixtures pass;
2. a provider fake and the native implementation pass the same conformance test;
3. evidence tampering and identity mismatch fail closed;
4. the maturity binding is reviewed;
5. an API owner approves the namespace surface for external adapters.

The out-of-tree MCP server may add tools only after the corresponding FlagQuantum
surface is approved and versioned. It must call the public surface rather than
reimplementing validation, estimation, VQE, exact diagonalization, variance, or
evidence assembly. Schema resources must likewise be read from
`describe_research_schema(...)`; the adapter cannot ship a second authoritative
copy.

## 8. Acceptance tests

- Exact P0 IDs, versions, statuses, error stages, and owners are machine checked.
- Every manifest schema identity resolves through the installed schema catalog,
  and representative runtime values match the catalog's closed shapes.
- Unknown manifest fields and schema versions fail closed.
- Input units, basis order, coefficient domains, and applicability are checked
  before execution.
- Resource estimates are produced before execution and compared with actual use.
- VQE success requires exact-reference and energy-variance validation when both
  validators apply.
- Unsupported input, execution failure, validation failure, and success each
  produce an evidence bundle with the same top-level shape.
- Evidence identity changes when any semantic result, validation, attempt, or
  provenance field changes.
- Target constraints reuse ARCH-003 matching rather than a service-local
  comparator.
- Local native and a provider fake pass replacement conformance without changing
  the service consumer.
- Multiple providers fail closed without authorization; authorized fallback
  advances only across unavailable providers and retains every attempt.
- Execution and scientific-validation failures never cause provider substitution.
- The package imports and the complete P0 workflow run with MCP absent.

## 9. Explicitly deferred

- Natural-language query parsing and autonomous research planning;
- tenant state, authentication, quotas, billing, and durable task persistence;
- remote polling, cancellation, realtime sessions, and paid-resource approval;
- general provider ranking learned from historical success;
- fallback on execution or scientific failure, automatic algorithm substitution,
  and approximation;
- general chemistry, excited-state, QML, QEC, pulse, photonic, and network
  manifests;
- Stable Core root exports or replacement of existing execution results.

These capabilities require later vertical slices and do not become available by
being named here.
