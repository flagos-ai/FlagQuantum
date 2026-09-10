# API Change Proposal 022: ProgramArtifact v2 executable-text profile

## Status

**Proposed; not approved and not implemented.**

Date: 2026-09-10

Approval token: `approve API_CHANGE_PROPOSAL_022_PROGRAM_ARTIFACT_V2`

This proposal follows ARCH-002. Until the API owner explicitly approves it,
`flagquantum.core._artifacts.ProgramArtifact` remains a v1-only internal
candidate, Phase 29/30 results remain in-process Compiler records, and Runtime
and Deployment must not consume the v2 candidate.

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

## Decision proposed

Add version `2.0` to the existing `flagquantum.program_artifact` schema lineage.
The first v2 profile is deliberately narrow: a fully bound, static,
UTF-8-encoded target-text executable that has passed target legalization,
logical scheduling, deterministic emission, and strict conformance.

Version dispatch remains owned by Core. Compiler constructs the executable
profile through an explicit adapter only after Phase 30 succeeds. Runtime and
Deployment consume it only through separately approved adapters.

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

Normative fixed values for the initial profile are:

```text
schema   = "flagquantum.program_artifact"
version  = "2.0"
kind     = "executable"
encoding = "utf-8"
```

Unknown and missing top-level fields fail closed. Strings are not coerced from
numbers or arbitrary objects.

### Profile and payload

`profile` is a closed record:

```text
name       = "openqasm-2.0" | "openqasm-3.0" | "qcis-1.0"
media_type = exact value registered for the selected name
encoding   = "utf-8"
```

`payload` is canonical text, not bytes and not a URL. Its UTF-8 encoding must
be nonempty and at most 16 MiB. `payload_sha256` is the lowercase SHA-256 of
those exact bytes. The reader recomputes and compares it before parsing.

Large blob references, QIR/native binary payloads, signatures, compression,
and encryption require later profiles and are not inferred from this schema.

### Compilation identity chain

`circuit_content_hash` is the final legalized Core `CircuitIR.content_hash`.
It is not renamed to source identity because the current vertical slice does
not retain the pre-specialization source-program identity through every stage.

`target` contains only:

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

Every identity is a lowercase 64-character SHA-256 digest. These are
role-named identities, not positional interpretations of v1 parent hashes.
Target provider names, backend locators, accounts, and queue state are not
artifact identity.

### Requirements, parameters, and results

`requirements` is the canonical serialized Core `RequirementSet` used during
target legalization, including its fallback authorizations. The receiver
reconstructs and validates it with the Core schema; it does not treat v1 flat
capability strings as equivalent.

The initial `parameter_schema` is:

```text
binding = "fully_bound"
parameters = []
```

Unbound runtime parameters are outside the first profile.

The initial `result_schema` is:

```text
kind = "samples"
wires = [0, ..., n_wires - 1]
shots_source = "execution_request"
```

Shots are deliberately absent from the artifact. Requested repetitions,
seeds, mitigation policy, timeouts, priorities, and result-delivery options
belong to the execution request and must not change artifact identity.

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

The first v2 executable profile has no free-form metadata or extension map.
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
  strict executable-profile validator.
- Do not automatically upgrade v1 metadata into v2 fields.
- Do not downgrade v2 executable artifacts to v1.
- Do not reinterpret v1 `parent_hashes`, `required_capabilities`, or metadata.
- Freeze the Phase 31 v1 compatibility fixture before implementation and run
  it in every v2 reader change.

No stable root or `flagquantum.core` export is authorized by this proposal.
Export and public lifecycle decisions require a separate approval after the
internal adapters have conformance evidence.

## Ownership and adapters

- Core owns schema/version dispatch, canonical encoding, limits, and identity
  verification.
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
- Putting shots or provider target locator in the artifact: mixes program,
  request, and deployment identities.
- Storing credentials or job state: creates a security and lifecycle violation.
- Treating payload hash as artifact identity: loses profile, target,
  requirements, and compilation binding.
- Reusing v1 `content_hash` semantics under version 2: cannot provide a
  transmitted recomputable identity without changing behavior.
- Supporting arbitrary binaries or URLs initially: exceeds the verified
  Phase 29/30 text profile.

## Implementation gates after approval

1. Core implements the strict v2 value model and version dispatcher while the
   v1 compatibility fixture remains byte/hash stable.
2. Compiler adds a one-way adapter from successful Phase 30 conformance into
   the v2 executable profile.
3. Round-trip, unknown-field, limits, non-finite, sensitive-content, payload
   tamper, identity tamper, and hash-seed tests pass.
4. Runtime adds read-only compatibility preflight without submission.
5. Deployment dry-run adaptation is proposed separately before any provider
   call.

## Acceptance

- The machine-readable candidate contract agrees with every field and limit in
  this proposal.
- The pinned v1 fixture is accepted by the unchanged v1 reader and retains its
  existing content hash.
- Candidate tests prove payload and artifact identities use separate inputs.
- Candidate tests prove shots, credentials, task state, and provider locators
  are excluded.
- No implementation, existing schema, protected Core code, public export,
  Runtime path, Deployment path, or provider behavior changes before explicit
  approval.
