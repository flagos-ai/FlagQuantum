# API Change Proposal 022: ProgramArtifact v2 circuit and executable profiles

## Status

**Approved on 2026-09-10; Phase 32 Core implementation complete.**

Date: 2026-09-10

Approval token: `approve API_CHANGE_PROPOSAL_022_PROGRAM_ARTIFACT_V2`

The API owner supplied the exact approval token on 2026-09-10. Phase 32 added
the strict Core v2 model, explicit v1/v2 dispatcher, circuit-profile
constructor, executable-profile validator, and bounded v1 circuit migrator.
Compiler construction, Runtime consumption, Deployment adaptation, and public
exports remain outside Phase 32 and require their documented follow-on gates.

## Problem

The current compilation path can now produce and independently check
deterministic OpenQASM 2, OpenQASM 3, and QCIS text. The output is bound to the
final circuit, target-capability snapshot, legalization, logical schedule,
emission, and conformance identities. Those facts currently live in private
Compiler dataclasses and have no approved cross-layer serialization contract.

`ProgramArtifact` v1 cannot carry this information losslessly:

- it has no payload profile, media type, encoding, or declared payload digest;
- its opaque `parent_hashes` have no roles;
- its free-form metadata has no closed value algebra or security limits;
- its computed `content_hash` is not transmitted for receiver comparison;
- its flat `required_capabilities` cannot represent structured target
  requirements;
- raw executable bytes are rejected and executable payloads have no per-kind
  validator.

Encoding mandatory executable facts into v1 metadata would create an informal
second contract while changing neither the v1 reader nor its guarantees.
Creating an unrelated `ExecutableArtifact` envelope would violate ARCH-002.

## Decision

Add version `2.0` to the existing `flagquantum.program_artifact` schema lineage.
The first v2 release has two profile families: canonical Core `CircuitIR` and
fully bound static target text that has passed target legalization, logical
scheduling, deterministic emission, and strict conformance.

Version dispatch remains owned by Core. Core constructs the circuit profile;
Compiler constructs executable profiles through an explicit adapter only after
Phase 30 succeeds. Runtime and Deployment consume them only through separately
approved adapters.

### Initial envelope

The exact top-level fields are:

```text
schema
version
kind
producer
profile
payload
payload_sha256
circuit_content_hash
requirements
target
compilation
parameter_schema
result_schema
artifact_identity
```

Normative fixed values are:

```text
schema   = "flagquantum.program_artifact"
version  = "2.0"
kind     = "circuit" | "executable"
```

Unknown and missing top-level fields fail closed. Strings are not coerced from
numbers or arbitrary objects.

### Profile and payload

`profile` is a closed record. The circuit profile is:

```text
name       = "circuit-ir-1.0"
media_type = "application/vnd.flagquantum.circuit-ir+json;version=1.0"
encoding   = "canonical-json"
kind       = "circuit"
```

Its payload is the exact `CircuitIR.to_dict()` mapping and must round-trip
through `CircuitIR.from_dict()`. Its payload bytes are the canonical JSON
encoding used by `CircuitIR.content_hash`; consequently `payload_sha256` and
`circuit_content_hash` have distinct roles but equal values for this profile.

The executable profiles are:

```text
name       = "openqasm-2.0" | "openqasm-3.0" | "qcis-1.0"
media_type = exact value registered for the selected name
encoding   = "utf-8"
```

For executable profiles, `payload` is canonical text, not bytes and not a URL.
For the circuit profile it is a canonical JSON mapping. The encoded payload
must be nonempty and at most 16 MiB. `payload_sha256` is the lowercase SHA-256
of the exact profile-defined bytes. The reader recomputes and compares it
before parsing.

Large blob references, QIR/native binary payloads, signatures, compression,
and encryption require later profiles and are not inferred from this schema.

### Compilation identity chain

`circuit_content_hash` is the payload CircuitIR hash for `circuit-ir-1.0` and
the final legalized Core `CircuitIR.content_hash` for executable profiles. It
is not renamed to source identity because the current vertical slice does not
retain the pre-specialization source-program identity through every stage.

For executable profiles, `target` contains only:

```text
snapshot_id
```

`compilation` contains:

```text
target_legalization_identity
schedule_identity
emission_identity
conformance_identity
```

For `circuit-ir-1.0`, both `target` and `compilation` are null because an
uncompiled circuit is not target-bound. Every populated identity is a lowercase
64-character SHA-256 digest. These are
role-named identities, not positional interpretations of v1 parent hashes.
Target provider names, backend locators, accounts, and queue state are not
artifact identity.

### Requirements, parameters, and results

For executable profiles, `requirements` is the canonical serialized Core
`RequirementSet` used during target legalization, including its fallback
authorizations. For `circuit-ir-1.0` it is null; target-independent circuit
semantics remain in CircuitIR. The receiver does not treat v1 flat capability
strings as equivalent.

Executable profiles use:

```text
binding = "fully_bound"
parameters = []
```

The circuit profile derives a sorted parameter-name list from CircuitIR and
uses `binding="fully_bound"` or `binding="symbolic"`. Parameter values and
expressions remain authoritative in CircuitIR; the schema is a validated index,
not a duplicate binding store. Unbound runtime parameters remain outside the
initial executable profiles.

Executable profiles use:

```text
kind = "samples"
wires = [0, ..., n_wires - 1]
shots_source = "execution_request"
```

Shots are deliberately absent from the artifact. Requested repetitions,
seeds, mitigation policy, timeouts, priorities, and result-delivery options
belong to the execution request and must not change artifact identity.
For `circuit-ir-1.0`, `result_schema` is null because CircuitIR measurements and
observables remain authoritative and may not yet be executable-profile legal.

### Artifact identity

`artifact_identity` is transmitted and verified. It is the lowercase SHA-256
of the canonical JSON encoding of all other top-level fields:

```text
UTF-8
sorted keys
separators=(",", ":")
ensure_ascii=true
allow_nan=false
artifact_identity field omitted
```

This does not change v1 `content_hash`. The v1 property remains a locally
recomputed hash of the complete v1 `to_dict()` result and remains absent from
v1 serialization.

## Closed data and security policy

The first v2 profiles have no free-form metadata or extension map.
All records use exact string keys and finite JSON scalars, lists, and maps
defined by the schema. Duplicate keys, key coercion, unknown keys, non-finite
numbers, and callable `to_dict()` projection are rejected at the serialized
boundary.

Initial limits are:

- payload UTF-8 bytes: 16 MiB;
- complete envelope UTF-8 bytes: 18 MiB;
- maximum nesting depth: 8;
- maximum total mapping/list entries: 4096;
- maximum ordinary string length other than payload: 4096 UTF-8 bytes.

The following content is prohibited in structured envelope fields outside the
strictly profile-validated payload: credentials, tokens, secrets, private keys,
account identifiers, tenant state, provider SDK objects, live handles, job IDs,
queue positions, callback URLs, local file paths, arbitrary remote URLs, and
mutable calibration objects. Payload text must pass its closed grammar; the
fixed standard-library include literal emitted by an approved OpenQASM profile
is program syntax rather than an artifact locator. The target snapshot digest
is allowed; snapshot contents remain in their own authority.

This proposal adds hashing and validation, not signing or trust. Authenticity,
authorization, encryption, and secure transport remain separate concerns.

## v1 compatibility and migration

- Preserve the exact v1 class behavior, field defaults, coercions,
  serialization, and `content_hash` algorithm.
- Add explicit version dispatch; do not make the v1 constructor accept v2.
- An old v1-only reader must reject v2 exactly as an unsupported version.
- A new reader accepts v1 through the unchanged v1 path and v2 through the
  strict profile validator.
- New artifact writes use v2; v1 is a read-only compatibility path.
- A deterministic v1-to-v2 migration is allowed only when v1 has
  `kind="circuit"`, empty metadata, empty `required_capabilities`, empty
  `parent_hashes`, and a payload accepted by `CircuitIR.from_dict()`.
- Any other v1 artifact remains readable but fails automatic migration with a
  typed reason. No v1 metadata or opaque parent role is guessed.
- Do not downgrade v2 artifacts to v1.
- Do not reinterpret v1 `parent_hashes`, `required_capabilities`, or metadata.
- Freeze the Phase 31 v1 compatibility fixture before implementation and run
  it in every v2 reader change.

No stable root or `flagquantum.core` export is authorized by this proposal.
Export and public lifecycle decisions require a separate approval after the
internal adapters have conformance evidence.

## Ownership and adapters

- Core owns schema/version dispatch, canonical encoding, circuit-profile
  construction, v1 circuit migration, limits, and identity verification.
- Compiler owns construction from verified Phase 29/30 results and must not
  submit or execute artifacts.
- Runtime owns compatibility checks against an execution request and target;
  it must not reinterpret Compiler identities.
- Deployment/Remote own provider conversion, credentials, submission receipts,
  task lifecycle, and returned provider evidence; those facts stay outside the
  artifact.
- Simulation does not consume target-text artifacts in the normal execution
  path; its use in Phase 30 remains bounded test evidence.

## Rejected alternatives

- Mandatory v1 metadata conventions: not a closed or versioned shared contract.
- A new generic `ExecutableArtifact`: duplicates the ARCH-002 authority.
- Keeping v2 executable-only: would leave the repository's only demonstrated
  v1 circuit use case on the legacy writer without a technical need.
- Putting shots or provider target locator in the artifact: mixes program,
  request, and deployment identities.
- Storing credentials or job state: creates a security and lifecycle violation.
- Treating payload hash as artifact identity: loses profile, target,
  requirements, and compilation binding.
- Reusing v1 `content_hash` semantics under version 2: cannot provide a
  transmitted recomputable identity without changing behavior.
- Supporting arbitrary binaries or URLs initially: exceeds the verified
  Phase 29/30 text profile.

## Implementation gates

1. **Complete in Phase 32:** Core implements the strict v2 value model, circuit profile, bounded v1
   circuit migration, and version dispatcher while the v1 compatibility fixture
   remains byte/hash stable.
2. **Complete for the Core construction seam in Phase 32:** new Core circuit
   artifacts use the v2 constructor while the exact v1 reader remains available
   as a read-only compatibility path.
3. **Complete:** Compiler adds a one-way adapter from successful target-text
   conformance into the v2 executable profile.
4. Round-trip, unknown-field, limits, non-finite, sensitive-content, payload
   tamper, identity tamper, and hash-seed tests pass.
5. **Complete:** Runtime adds read-only compatibility preflight without
   submission.
6. **Complete:** the separately documented Deployment dry-run adapter binds a
   verified artifact to a resolved target and execution request without any
   provider call or legacy-package conversion. See
   `ARTIFACT_DEPLOYMENT_DRY_RUN.md`.

## Acceptance

- The machine-readable candidate contract agrees with every field and limit in
  this proposal.
- The pinned v1 fixture is accepted by the unchanged v1 reader and retains its
  existing content hash.
- The fixture payload is also accepted by `CircuitIR.from_dict()` and migrates
  to the pinned v2 circuit candidate without semantic or identity ambiguity.
- Candidate tests prove payload and artifact identities use separate inputs.
- Candidate tests prove shots, credentials, task state, and provider locators
  are excluded.
- The approval token is recorded and Core implementation evidence is captured
  in `HYBRID_COMPILATION_PHASE32_EVIDENCE.md`.
- No public export, Runtime path, Deployment path, or provider behavior changes
  are included in the Phase 32 implementation.
