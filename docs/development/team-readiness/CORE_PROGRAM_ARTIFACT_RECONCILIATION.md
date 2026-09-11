# ProgramArtifact Phase 1 Contract Reconciliation

> Commit references below have been mapped to the publication history.
> Recorded outcomes and approval status are unchanged.

> Historical note: this document characterizes the Phase 1 implementation.
> `AgentApplicationService` was removed before release. The current
> `flagquantum.services` does not consume serialized `ProgramArtifact` objects;
> protocol adapters decode at the boundary and call stable APIs. The analysis
> below records the contract decisions at that time, not the current service API.

Status: formal Core reconciliation submitted for Integration decisions; this is
not API, schema, or ADR approval.

Baseline: `3292078950b515333d422e383d1e5ca8039daa0a`

Team branch: `codex/vnext-phase1-core-contracts`

Date: 2026-09-03

## 1. Decision

The existing `flagquantum.core._artifacts.ProgramArtifact` should remain the
repository's sole Core-owned Program Artifact envelope authority. It already
provides strict top-level fields, a closed `ArtifactKind`, version rejection,
immutable JSON projection, and a deterministic envelope hash. A second Artifact
type would violate ARCH-001.

This conclusion covers only the current `flagquantum.program_artifact` 1.0
envelope and working circuit artifact path. It does not mean that the envelope
can replace Compiler's `SealedExecutableArtifact` or `SealedCircuitIRRoundTrip`,
or Deployment's `DeploymentPackage`, without losing information:

- The only production consumer is `AgentApplicationService`, which accepts only
  `kind=circuit`.
- Compiler, Runtime, Simulation, Execution Provider, and Ecosystem do not directly
  consume `ProgramArtifact`.
- v1 has no first-class `provenance`, structured `requirements`, namespaced
  `extensions`, payload profile, or byte encoding.
- `required_capabilities` is the flat string collection used by Agent Services,
  not a representation of compilation requirements.
- `producer`, `parent_hashes`, and `metadata` cannot fully express Compiler's
  existing identity chain with named roles.
- `content_hash` is computed rather than serialized. The envelope does not carry
  a sender-declared digest for the consumer to compare.

The smallest viable approach is to freeze v1 as the compatibility reader baseline,
approve field semantics and identity layers, and then have each owner provide an
explicit adapter. New top-level fields, narrower accepted values, version or hash
changes, and stable exports require an Integration contract decision first.
Putting private required fields in free-form `metadata` does not establish a
shared contract.

## 2. Protected Boundaries and Sources

`team-ownership.toml` explicitly protects `flagquantum/core/_artifacts.py`.
The original reconciliation added only this document and
`tests/team/core/test_program_artifact_reconciliation.py`. It did not change that
implementation, public exports, `contracts/`, ADRs, or Compiler, Runtime,
Deployment, and Agent implementations.

Sources include:

- `ProgramArtifact` and existing Core/Agent characterization tests;
- `AgentApplicationService._decode_program()` and planning capability preflight;
- Compiler's `ImportedCircuitProgram`, `SourceIdentity`, `SourceProvenance`,
  `SealedCircuitIRRoundTrip`, `TargetIR`, and `SealedExecutableArtifact`;
- Deployment's `DeploymentPackage`, routing evidence, submission receipt, and
  artifact digest;
- `_compiler.deployment_compatibility` restrictions on legacy package metadata
  and identity;
- the Phase 0 integration board and eight team-readiness inventories.

The work consists of contract inspection and local planning characterization on
`single_device_fast_path`. It neither executes nor changes distributed paths and
provides no hardware, QPU, performance, or scalability evidence.

## 3. Direct Consumers

Repository searches identified these definitions and consumers. Proposals in
documentation are not runtime consumers.

| Owner | Entry | Fields read or generated | Current behavior | Reconciliation |
| --- | --- | --- | --- | --- |
| Core | `ProgramArtifact` | All v1 fields; computes `content_hash` | Construction, freezing, dict round trips, strict top-level reading | Sole envelope authority; still an internal candidate contract without stable exports |
| Core | `from_circuit_ir()` | Generates `kind`, `payload`, `producer`; defaults the rest | Requires only `to_dict()`, not an actual `CircuitIR` | Convenience adapter, not a per-kind payload validator |
| Agent Services | `_decode_program()` | `schema`, `kind`, `payload`, `required_capabilities` | Accepts only circuit; validates payload through `CircuitIR.from_dict()` | Only production consumption chain |
| Agent Services | `validate_program()` | Reads the artifact indirectly but ignores returned requirements | Validates only the circuit payload | Missing capabilities do not fail validation |
| Agent Services | `plan_execution()` | `required_capabilities` | Fails closed using temporary vocabulary from the current capability manifest | Compatibility policy, not Core capability matching authority |
| Tests | Core, Agent, long-horizon tests | Round trip, hash, unknown schema/kind, planning | Characterizes current candidate behavior | This reconciliation adds field-level and difference-focused evidence |
| Compiler | No direct import | None | Uses its own source, target, and executable artifacts | Requires an adapter; no direct type replacement |
| Runtime / Simulation | No direct import | None | Consumes `CircuitIR`, Compiler plans, and internal results | Integrate after Artifact/Request contract approval |
| Deployment / Execution Provider | No direct import | None | Uses `DeploymentPackage` and separate digests/receipts | Not an existing ProgramArtifact consumer |
| Ecosystem | No direct import | None | Adapters return `CircuitIR`; metadata may enter IR | Validate Core metadata values before wrapping an artifact |

No root export or stable `flagquantum.core` export of `ProgramArtifact` was found.
External Compute Service requirements do not prove that the main repository
already has a stable schema for use across repositories.

## 4. v1 Field Decisions

| Field or property | Observable behavior | Current consumer | Phase 1 decision | Change threshold |
| --- | --- | --- | --- | --- |
| Fixed `schema` | `to_dict()` writes `flagquantum.program_artifact`; missing, unknown, and other values are rejected | Agent identifies the envelope | Preserve the exact v1 value; do not replace the version system with a `.v2` suffix | Value or compatibility changes require a contract/serialization API proposal |
| `version` | Only `"1.0"` is supported; constructor compares exactly; `from_dict()` calls `str()`, accepting JSON number `1.0` | Core reader, indirectly Agent | Preserve this compatibility fact; Integration decides whether a future reader stops coercion | Narrower coercion, renaming to `schema_version`, or new versions change behavior |
| `kind` | Closed eight-value Enum; does not validate payload by kind | Agent supports only `circuit` | Preserve kind authority; each consumer declares its supported subset and fails closed | Enum changes or generic payload dispatch require a proposal |
| `payload` | Must project to mappings/lists/JSON scalars; recursively frozen; bytes rejected; no payload profile | Agent passes circuit payload to `CircuitIR.from_dict()` | Keep envelope validation separate from payload validation; retain circuit adapter | Executable profiles, bytes, and per-kind schemas require ADR/version decisions |
| `producer` | Checked for nonempty value without trimming; included in hash; reader converts numbers to strings | No consumer interprets vocabulary | Opaque producer label, not complete provenance or compiler identity | Value rules, normalization, or richer semantics alter hash/accepted inputs |
| `required_capabilities` | Sorted and deduplicated; empty values rejected; no strict string type check; included in hash | Agent planning only | Coarse v1 compatibility hints; do not rename to `requirements` or treat as structured compilation requirements | Vocabulary, typing, semantics, or identity inclusion changes require a proposal |
| `parent_hashes` | Lowercase SHA-256 required; preserves order and duplicates; does not validate existence, relationships, or hash kind; included in hash | None | Ordered opaque lineage references; no positional roles | Deduplication, sorting, roles, or validation changes require ADR/API identity migration |
| `metadata` | Included in hash; keys converted with `str()`; accepts JSON scalars/containers and arbitrary `to_dict()` objects; no namespace, depth, size, or sensitive-field limits | Production consumers do not read it | v1 compatibility data only; no new required semantics, credentials, live SDK objects, or authoritative identity | Narrower values, namespaces, limits, or unknown-key rules change serialization behavior |
| `content_hash` | SHA-256 of complete `to_dict()` as compact sorted-key JSON; includes all fields; mapping order irrelevant, sequence order relevant; NaN rejection occurs only when hashing | Tests; production chain does not transmit/compare a declared hash | Envelope identity, distinct from IR/payload/artifact identity | Algorithm, inputs, or adding the digest to the envelope requires versioned migration |

Dataclass field order is `kind, payload, producer, version,
required_capabilities, parent_hashes, metadata`. Serialization order is
`schema, version, kind, producer, required_capabilities, parent_hashes, payload,
metadata`. Sorted JSON keys make declaration order irrelevant to the digest,
but dataclass fields, defaults, and serialized shape remain protected behavior.

## 5. Metadata Values

### 5.1 Accepted Values

`_json_value()` currently accepts:

- `None`, `str`, `int`, `float`, and `bool`;
- mappings, with unconditional conversion of keys to strings;
- tuples/lists, frozen as tuples and serialized as lists;
- any object with callable `to_dict()`, recursively projecting its result.

Other objects, including raw `bytes`, raise `TypeError` during construction.
Non-finite floats pass construction and `to_dict()`, but `content_hash` raises
`ValueError` because it uses `allow_nan=False`. Numeric and string keys can
collide after conversion. Invoking arbitrary `to_dict()` methods is not a closed,
pure-data operation across a trust boundary.

`CircuitIR` supports a different value set, including Parameter,
ParameterExpression, complex, and Tensor encodings. Serialized
`CircuitIR.to_dict()` output can be an artifact payload; putting raw Tensor,
complex, or Parameter objects in artifact metadata does not have those semantics.

### 5.2 Differences from Other Teams' Requirements

- Ecosystem requires external SDK objects to stop at the boundary. Duck-typed
  `to_dict()` ensures projection, not that the method is safe or semantically
  controlled.
- Deployment compatibility checks string-only keys, finite floats, depth, entry
  count, encoded size, and sensitive fragments. These checks apply to legacy
  deployment packages; they are not a private authority for Core metadata.
- Platform metadata needs JSON-safe identity values. Existing `PlatformRuntime`
  implementations may still return vendor handles, which cannot enter artifacts
  directly.
- Agent Services does not read artifact metadata. A key's presence does not prove
  that capabilities, provenance, or requirements have been enforced.

### 5.3 Recommendation

Integration should approve a closed canonical metadata algebra for the next
version: string keys, finite JSON scalars, recursive mappings/lists, explicit
namespaces, depth/entry/byte limits, sensitive-field rules, and collision
rejection. Preserve v1 reader behavior through compatibility adapters. Narrowing
1.0 in place cannot be described as preserving hash and reader compatibility.

## 6. Version and Identity Layers

At least six identities currently exist. They cannot share one undifferentiated
`hash` field.

| Identity | Algorithm or source | Mapping to ProgramArtifact v1 | Decision |
| --- | --- | --- | --- |
| `CircuitIR.content_hash` | SHA-256 of canonical IR JSON | Recomputable from circuit payload | Payload identity, not envelope identity |
| `ProgramArtifact.content_hash` | SHA-256 of complete v1 envelope | Native | Envelope identity; no embedded declared digest |
| `ImportedCircuitProgram.internal_program_identity` | Module, constraints, and instruction semantics | Not generically recomputable from v1 | Compiler-private derived identity until Core approves an internal-program schema |
| `TargetIR.target_program_identity` | Target layout/ops/results/shots and capability/source references | No standard v1 profile | Requires executable/physical payload profile decision |
| `SealedExecutableArtifact.artifact_identity` | Profile, media type, payload hash, source/target/capability/compilation identities | Cannot be represented losslessly by parents without roles | Requires ADR; temporary metadata does not establish convergence |
| Deployment artifact digest | Name, target, shots, format, program, routing evidence | Mixes program, request, and target fields | Adapter separates program artifact, execution request, and provider receipt |

The envelope hash from `from_circuit_ir()` changes with producer, requirements,
parents, or metadata even when the IR payload is identical. This is valid
identity layering; envelope and IR hashes need not match.

`parent_hashes` validates string shape, not parent existence or source, target,
compile, calibration, and payload roles. Compiler adapters may retain complete
role-aware identity chains privately. They should not encode those roles through
positions in `parent_hashes` before Core approves typed identity references.

## 7. Provenance

v1 has only three related fields:

- `producer`: nonempty label included in the hash;
- `parent_hashes`: ordered digests without roles;
- `metadata`: free-form data included in the hash, without provenance namespaces.

They cannot replace Compiler's:

- `SourceIdentity(schema_version, circuit_ir_content_hash)`;
- `SourceProvenance(values)`;
- instruction-level provenance and semantics;
- compilation, target capability, target program, and artifact identities.

Compiler's `internal_program_identity` explicitly excludes requests, provenance,
and bindings. ProgramArtifact v1's `content_hash` includes all metadata,
producer, and requirements. Renaming fields cannot reconcile these different
identity inclusion policies.

The minimum adapter keeps Compiler provenance/identity objects private and
projects references with explicit roles only into approved Core profiles. A
first-class `provenance` field, its inclusion in envelope identity, or migration
from metadata requires an ADR and versioned compatibility plan.

## 8. Requirements and required_capabilities

| Concept | Content | Consumer stage | Relationship to v1 |
| --- | --- | --- | --- |
| `ProgramArtifact.required_capabilities` | Sorted, unique flat strings | Before Agent planning | Native v1; vocabulary comes from current manifest adaptation |
| `CircuitIR` dtype/shape/measurements/observables | Existing mix of program and request semantics | Compiler/Runtime | Carried by circuit payload; do not duplicate as strings |
| Compiler `ImportConstraints` | dtype, shape, batch, logical state, runtime config | Import | Structured and included in internal identity; lossy as strings |
| `TargetIR.required_results/requested_shots` | Result types and shots | Target lowering | Closer to ExecutionRequest than artifact capability names |
| `TargetCapabilities` | Target facts, limits, artifact profiles | Compile/preflight | Describes what a target has, not what an artifact requires |
| Long-horizon `requirements` | Precision, dynamic execution, communication, QEC, pulse, network | Compiler/Runtime/Provider | Not present in v1 |

The temporary Agent vocabulary collects backend names, `supports_*`, accelerator,
device, dtype, and contract names. It is not a closed Core vocabulary.
`validate_program()` ignores `required_capabilities`; `plan_execution()` checks
it and returns `REQUIRED_CAPABILITY_UNAVAILABLE`. Characterization tests preserve
this failure stage.

Keep `required_capabilities` as coarse v1 hints. A future structured requirement
adapter may make a documented conservative projection into them. Strings cannot
recover precision, topology, shots, calibration, or dynamic-control requirements.
Integration/Core ADRs must define the structured shape, identity inclusion,
TargetCapabilities matching, and failure categories.

## 9. Minimum Consumer Adapters

### 9.1 CircuitIR and Agent Services

The circuit path can continue using `CircuitIR.to_dict()` as payload and strict
`CircuitIR.from_dict()` validation at consumption. Add shared repository fixtures
and expected envelope hashes without changing implementation. Capability checks
remain at planning time unless an API proposal explicitly changes that stage.

### 9.2 Compiler Source/Import

A Compiler adapter can accept `kind=circuit`, obtain strict `CircuitIR`, and
produce private `ImportedCircuitProgram` and `SealedCircuitIRRoundTrip` objects.
Their internal modules/bindings are not generically serializable and cannot go
back into v1 payloads. Compiler owns the adapter. The exit condition is that
Compiler's public cross-domain entry accepts Core artifacts while internal types
remain private.

### 9.3 Compiler Executable Artifact

v1 lacks bytes, artifact profiles/media types, and the six role-aware identities.
No lossless adapter exists without contract decisions. Integration must first
define executable payload profiles, byte encoding/external blob references,
identity roles, and verification algorithms. Compiler keeps seal/verify
implementation; it does not move into Core.

### 9.4 Deployment and Execution Provider

`DeploymentPackage` combines compiled IR, QASM/QCIS, shots, backend profile,
routing evidence, and provider metadata. Split program and payload identity into
ProgramArtifact, shots/measurement into ExecutionRequest, target into
TargetCapabilities, and receipt/job into Provider. Do not treat the whole package
as an executable ProgramArtifact.

### 9.5 Runtime, Simulation, and Ecosystem

Runtime/Simulation need not consume ProgramArtifact directly before formal
ExecutionRequest/plan profiles exist. They should not create private copies.
Ecosystem first converts external objects to strict `CircuitIR`, then uses a Core
helper to wrap them. External objects and metadata values without canonical
validation cannot enter artifacts directly.

## 10. Compatibility Risks

| Risk | Severity | Constraint or acceptance condition |
| --- | --- | --- |
| Adding top-level provenance/requirements/extensions | High | v1 rejects unknown fields; requires a new version or compatible envelope strategy |
| Renaming `version` to `schema_version` | High | Existing payloads lack the new field; requires dual reading, migration fixtures, and API proposal |
| Changing hash inputs or canonical JSON | High | Changes envelope identities; requires golden fixtures and migration identities |
| Narrowing metadata in place | High | Previously readable payloads may fail; changes string-key, `to_dict()`, and NaN failure stages |
| Treating metadata as nonsemantic extensions | High | Metadata affects the hash; presentation or transport fields change identity |
| Assigning fixed roles to parent positions | High | Existing duplicates and absent roles cannot prove the interpretation |
| Treating required capabilities as complete requirements | High | Loses precision, limits, topology, shots, evidence, and calibration semantics |
| Equating envelope hash with payload/executable identity | High | Producer/metadata change envelope identity; digest is not serialized |
| Wrapping executable bytes directly | Medium-high | Constructor rejects bytes; private base64 conventions create a second schema |
| Depending on current `str()` coercion | Medium | Reader accepts numeric version/producer; narrowing requires compatibility tests |
| Colliding non-string metadata keys | Medium | Next version should reject collisions; v1 requires preservation or explicit migration |
| Moving capability failures between validate and plan | Medium | Observable to Agent callers; requires a behavior proposal, not a test adjustment |

## 11. Minimum Integration Decisions

Approve only the necessary decisions, in this order:

1. **Authority ADR addendum:** existing `ProgramArtifact` is the sole envelope
   authority; preserve v1 circuit compatibility reading without a parallel type.
2. **Identity ADR:** distinguish payload, envelope, compile, target, and executable
   identities; define typed parent references and transmitted expected digests.
3. **Metadata ADR:** approve closed values, namespaces, limits, sensitive-data
   rules, non-finite/key-collision handling, and the v1 reader migration period.
4. **Requirements ADR:** define structured requirements, Core capability
   vocabulary, and fail-closed TargetCapabilities matching; retain one-way v1
   `required_capabilities` adaptation.
5. **Provenance ADR:** define source/tool/input/parent roles, identity inclusion,
   and compilation evidence references; keep Runtime measured evidence separate.
6. **Executable profile ADR:** after the first five decisions, define bytes/blob,
   media type, profile version, and the `SealedExecutableArtifact` adapter.

These may be separate small ADRs. Shared fixtures and contract fakes must be in
place before Compiler, Deployment, Runtime, or external service migration.

## 12. API Proposals and Internal Adapters

The following require an API Change Proposal and Integration/Core approval:

- stable root/core exports of `ProgramArtifact`, `ArtifactKind`, or version constants;
- changes to field names/order/defaults, Enum values, schema/version,
  unknown/missing fields, or exceptions;
- changes to accepted metadata/producer/required-capability values or normalization;
- changes to `content_hash`, identity inputs, or parent ordering;
- new top-level provenance/requirements/extensions required of existing consumers;
- changes to Agent validate/plan capability failure stages or supported kinds.

If Integration explicitly classifies the unexported v1 as a pre-public candidate,
internal refactoring may avoid a public deprecation cycle. Protected paths,
serialized payloads, and consumers in other repositories still need a Contract
Change Proposal, compatibility fixtures, and migration records. Absence of a root
export does not permit arbitrary breakage.

After approval, owners may implement these internal adapters:

- strict bidirectional circuit artifact/`CircuitIR` projection;
- Compiler circuit unpacking while keeping IR/provenance/identity private;
- Deployment package separation into artifact, request, target, and receipt,
  preserving public behavior;
- reading old payloads with the v1 reader and projecting approved internal values;
- golden hashes, legacy fixtures, fake consumers, and replacement conformance tests.

Each owner implements its consumer changes. This Core reconciliation requests
contract decisions and does not modify other teams' consumers.

## 13. Characterization Tests

`tests/team/core/test_program_artifact_reconciliation.py` records:

- dataclass/serialization shape, requirements normalization, round trips, and all
  identity inputs;
- schema, unknown/missing fields, version rejection, and numeric version/producer coercion;
- JSON-like/`to_dict()` metadata, stringified keys, opaque object rejection, and
  NaN failure at hashing time;
- parent hash format, order, duplicates, and absence of relationship validation;
- separate CircuitIR payload and envelope identities;
- rejection of raw executable bytes;
- consumer-owned per-kind payload validation;
- different required-capability failure stages in Agent validate and plan;
- strict rejection of `provenance`, `requirements`, and `extensions` as v1 fields.

These tests characterize behavior and compatibility risks without changing or
reinterpreting protected contracts.

## 14. Historical Validation Record

- Core team scope precheck: passed.
- Architecture boundary check: passed.
- Ruff check and format check: passed.
- Focused ProgramArtifact, Core contract, Agent, Compiler executable artifact,
  Deployment compatibility, and static pipeline selection: 46 passed.
- macOS `pr-default`: 1833 passed, 14 skipped, 6 failed. Failures covered the
  existing Compiler import/verify performance budget, two Linux `/proc` watchdog
  tests, two repository checks affected by unavailable host Git, and a host
  PyTorch `torch.dot` floating-point cancellation difference.
- `pr-default` in `flagquantum-dev:local` Linux with read-only source and complete
  Git metadata: 1840 passed, 12 skipped, 1 failed. The remaining failure was the
  previously recorded `test_approved_import_verify_budget_is_machine_enforced`.

The original reconciliation did not change performance thresholds, snapshots,
protected implementations, or test expectations. The Linux result distinguishes
host environment differences; it is not performance, hardware, or scalability
certification.
