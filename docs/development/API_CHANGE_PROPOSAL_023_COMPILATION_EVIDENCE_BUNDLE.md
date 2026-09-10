# API Change Proposal 023: Compilation evidence bundle

## Status

**Approved on 2026-09-10; evidence handoff implementation complete.**

Date: 2026-09-10

Approval token:
`approve API_CHANGE_PROPOSAL_023_COMPILATION_EVIDENCE_BUNDLE`

The API owner supplied the exact token on 2026-09-10. Core now owns the strict
value model, canonical JSON, version dispatch, limits, nested validation, and
bundle identity. Compiler constructs and verifies bundles only from the actual
`ArtifactCompilationResult`, retained `PhysicalCircuitPlan`, target snapshot,
source artifact or binding result, and executable artifact. Runtime and
Deployment provide read-only verification and side-effect-free carrying. No
provider submission behavior is included.

## Problem

Compiler now produces a deterministic `PhysicalCircuitPlan` that joins source
instruction lineage, topology mapping, native-gate decomposition, and dependency
scheduling. `ArtifactCompilationResult` retains this evidence in process and
includes the plan identity in its compilation identity. The final
`ProgramArtifactV2`, however, intentionally contains only the four compilation
identities approved by Proposal 022.

Consequently, a deployment or third-party verifier that receives only the
executable artifact cannot reconstruct or audit the physical plan. Adding fields
to the closed ProgramArtifact v2 `compilation` record would change an approved
serialized schema. Encoding the plan into free-form metadata would create an
unbounded second contract.

## Decision

Introduce a separate Core-owned, versioned
`flagquantum.compilation_evidence_bundle` schema. It accompanies a circuit and
an executable artifact but is not itself executable. The bundle records how a
verified final `CircuitIR` was derived; it never becomes another circuit IR and
does not duplicate circuit parameters, matrices, measurements, target payload,
or provider lifecycle state.

The initial version is `1.0`. Core owns canonical encoding, limits, identity,
and strict reading. Compiler constructs the bundle from an
`ArtifactCompilationResult`. Runtime and Deployment may verify it through
separately tested read-only adapters. No root export is included in the first
implementation.

## Closed envelope

The exact top-level fields are:

```text
schema
version
producer
source
target
physical_plan
output
bundle_identity
```

`schema` is `flagquantum.compilation_evidence_bundle`; `version` is `1.0`.
Unknown or missing fields fail closed. Canonical JSON uses sorted keys, compact
separators, UTF-8, `ensure_ascii=true`, and rejects non-finite numbers and
duplicate keys.

## Source and output lineage

`source` contains only role-named identities:

```text
source_artifact_identity
circuit_artifact_identity
binding_identity | null
source_circuit_hash
final_circuit_hash
```

`output` contains:

```text
profile
payload_sha256
emission_identity
conformance_identity
executable_artifact_identity
artifact_compilation_identity
```

Every identity and hash is a lowercase SHA-256 digest. The bundle does not
embed either artifact. Verification requires the caller to provide the actual
source circuit artifact or binding result, target snapshot, and executable
artifact. Detached strings alone never authorize execution.

## Target binding

`target` contains:

```text
snapshot_id
target_legalization_identity
```

The target snapshot itself remains in the Core capability-evidence lifecycle.
Provider locators, accounts, credentials, calibration objects, leases, job
identifiers, and queue state are prohibited.

## Physical plan evidence

`physical_plan` contains the deterministic Phase 36 evidence:

- `plan_identity`, source and physical circuit hashes;
- optional topology identity and an optional undirected coupling record with
  wire count and normalized edges;
- initial, pre-restore, and final logical-to-physical layouts;
- ordered mapping transitions with source index, routing phase, physical SWAP
  wires, and before/after layouts;
- ordered final instruction records with source and routed instruction indexes,
  native replacement ordinal, origin, opcode, logical and physical wires,
  dependency layer, predecessors, and dependency kinds;
- topology and native-gate legalization identities;
- dependency schedule identity, depth, maximum parallel width, and one
  deterministic critical path.

The records are evidence about the authoritative Core `CircuitIR`. They do not
carry gate parameters, custom matrices, measurement definitions, observable
coefficients, or executable text. A consumer must compare hashes against the
provided artifacts and reconstruct the circuits through their existing schema.

## Semantic limits

Version 1.0 supports only the verified current profile:

- static fully bound circuit artifacts;
- optional undirected coupling topology;
- identity initial and final logical layout;
- deterministic topology routing and native-gate decomposition;
- unit-time dependency layers, not hardware timing.

It does not claim directed-edge synthesis, physical ancilla allocation, gate
durations, pulse scheduling, crosstalk, calibration-aware optimization,
fault-tolerant resource expansion, numerical correctness, or hardware
performance. Later profiles require new proposals rather than optional
free-form fields.

## Limits and security

- maximum canonical bundle size: 16 MiB;
- maximum physical instruction records: 4096;
- maximum mapping transitions: 4096;
- maximum predecessor references per instruction: 4096;
- maximum nesting depth: 8;
- maximum non-identity string size: 256 UTF-8 bytes;
- no URLs, paths, credentials, opaque provider objects, binary blobs, code,
  callbacks, task state, execution requests, or mutable calibration data.

The reader validates limits before allocating large nested structures. It
recomputes `bundle_identity` over every other top-level field and then verifies
all nested identities and structural invariants.

## Ownership and dependency direction

- Core owns the immutable value model, canonical JSON, version dispatch,
  limits, and bundle identity.
- Compiler owns construction from verified compiler-stage objects.
- Runtime owns optional read-only preflight against an execution request and
  target snapshot; it does not rewrite evidence.
- Deployment owns packaging or transport of the bundle beside an executable
  artifact; it does not reinterpret compiler records.
- Remote/provider code owns submission receipts and lifecycle facts, which are
  excluded from this bundle.
- Simulation does not consume the bundle as numerical input.

## Compatibility

ProgramArtifact v1 and v2 remain byte- and behavior-compatible. No field is
added to either envelope. Existing executable artifacts remain valid without a
bundle. A workflow that explicitly requires auditable physical compilation may
fail closed when the bundle is absent; that requirement must be expressed by
the consuming workflow, not retroactively inferred by the artifact reader.

## Rejected alternatives

- Add `physical_plan_identity` to ProgramArtifact v2: changes the approved
  closed compilation record and still does not provide replayable evidence.
- Put the complete plan in ProgramArtifact payload or metadata: mixes execution
  content with compilation provenance and duplicates contract authority.
- Serialize `PhysicalCircuitPlan` with `dataclasses.asdict`: leaks internal
  objects and makes refactors accidental schema changes.
- Serialize another TargetIR: creates a second circuit-semantic authority.
- Store only a plan hash: detects a known plan but cannot support independent
  inspection or replay.

## Implementation gates

1. **Complete:** record the exact candidate schema and limits before implementation.
2. **Complete:** receive the exact approval token from the API owner.
3. **Complete:** implement the strict Core value model and duplicate-key-safe reader without
   changing ProgramArtifact v1 or v2.
4. **Complete:** add Compiler construction from `ArtifactCompilationResult`; no stage may be
   reconstructed from detached hashes.
5. **Complete:** add verification against the actual source artifact or binding result,
   target snapshot, final executable artifact, and physical plan.
6. **Complete for Core and Compiler:** add deterministic round-trip, unknown-field, limit, non-finite, duplicate,
   sensitive-content, nested-tamper, and hash-seed tests.
7. **Complete:** add Runtime and Deployment read-only adapters in separate bounded phases.
8. **Complete:** keep the type internal until a later public naming and lifecycle review.

## Acceptance

- Candidate contract, proposal, and implementation agree exactly.
- Canonical serialization is deterministic across hash seeds.
- Every physical record remains linked to a valid source and final instruction.
- Mapping transitions replay from initial to final layout.
- Coupling, legalization, schedule, artifact, and bundle identities are checked.
- A valid bundle can be independently verified with the required external
  artifacts and target snapshot.
- Any missing external object, mismatch, unknown field, or tamper fails closed.
- ProgramArtifact fixtures and identities remain unchanged.
- No public export, execution behavior, provider submission, default-path
  change, timing claim, or performance claim is introduced.

## Approval

The API owner approved the proposal on 2026-09-10 with the exact token:

```text
approve API_CHANGE_PROPOSAL_023_COMPILATION_EVIDENCE_BUNDLE
```

This approval authorizes the gated implementation described above; it does not
authorize a public root export or changes to ProgramArtifact v1/v2.
