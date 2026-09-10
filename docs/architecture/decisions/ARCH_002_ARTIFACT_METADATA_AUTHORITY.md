# ARCH-002: Artifact and Metadata Authority and Compatibility

Status: Proposed

Date: 2026-09-03

Basis: Phase 0 inventories from eight teams and Phase 1 reconciliation of Core
`ProgramArtifact`; this proposal changes neither public contracts nor implementations.

## Context

`CircuitIR` is the sole canonical program representation. Existing Core
`flagquantum.core._artifacts.ProgramArtifact` v1 (schema
`flagquantum.program_artifact`, version `1.0`) is already the repository's sole
authoritative artifact envelope. Compiler `SealedExecutableArtifact`,
`SealedCircuitIRRoundTrip`, and Deployment `DeploymentPackage` carry more
specialized, richer identities, payloads, and request information. They are not
second envelope authorities, but v1 cannot yet replace them losslessly.

When this proposal was written, v1's only active production consumer chain was the
then-existing Agent Services `kind=circuit` path. That unreleased facade was later
removed. Compiler, Runtime, Simulation, Remote, and Ecosystem did not thereby
become direct `ProgramArtifact` consumers. Having one authority does not mean all
consumers have migrated or that executable/deployment support has been implemented.

## Decision Candidates

1. `ProgramArtifact` v1 remains the sole artifact envelope authority. Any v2 must
   evolve within the same contract lineage through an API Change Proposal,
   compatibility analysis, and migration fixtures. Do not create a second parallel
   envelope, another generic Artifact type, or a private cross-layer schema.
2. v1 currently governs envelope fields, strict top-level reading, version/kind
   rejection, complete envelope identity, and the previously exercised
   `kind=circuit` consumer path. It cannot losslessly express executable bytes,
   payload profiles/media types, identities with roles, or structured
   requirements/provenance/extensions. It cannot losslessly replace
   `SealedExecutableArtifact`, `SealedCircuitIRRoundTrip`, or `DeploymentPackage`.
3. v1 `content_hash` is an envelope identity recomputed from canonical JSON of the
   complete `to_dict()` output. `producer`, `required_capabilities`, ordered
   `parent_hashes`, `payload`, and `metadata` all participate in the existing
   identity. It is not a payload, IR, compile, target, or deployment identity.
4. `content_hash` is currently a computed property, not a declared field
   transmitted with the envelope. The receiver recomputes it from the complete
   received envelope. Transmitting an expected digest, changing hash inputs, or
   separating semantic and presentation fields requires a new version in the
   same contract lineage and explicit migration; old hashes must not change silently.
5. First freeze v1 metadata behavior as compatibility facts. Future rules may
   define a closed value algebra, namespaces, capacity limits, sensitive fields,
   and new fields excluded from particular semantic fingerprints. This is not
   current v1 behavior and must not change v1 envelope identity in place.

### v1 Field Facts

- `producer` is only a nonempty opaque label included in the envelope hash. It is
  neither provenance nor a tool identity or controlled vocabulary.
- `required_capabilities` is a sorted, deduplicated set of coarse compatibility
  hints. At proposal time, only old Agent planning consumed it. It cannot replace
  structured compilation/execution requirements or imply precision, topology,
  shots, or calibration requirements.
- `parent_hashes` is opaque lineage preserving order and duplicates. Validation
  checks lowercase SHA-256 shape only, not parent existence, hash kind, or
  positional roles.
- `metadata` participates in complete envelope identity, but its value domain is
  not closed: keys are stringified, and arbitrary objects with callable `to_dict()`
  methods are accepted recursively after conversion. There are no namespace,
  depth, entry-count, encoded-size, or sensitive-field constraints. It cannot be
  described as safe across trust boundaries.

## Prohibited Practices

- Do not create a second canonical IR, artifact envelope, or v2 type alongside
  `ProgramArtifact` v1 as another authority.
- Do not put internal `QuantumModule`, vendor AST/SDK objects, live handles,
  credentials, tenant state, or queue state into v1 payload/metadata to claim
  convergence.
- Do not interpret v1 `producer` as provenance or `required_capabilities` as
  complete requirements, or retroactively assign roles to parent positions.
- Do not claim v1 metadata is closed or safe. Calling an external object's
  `to_dict()` does not make its output trusted.
- Do not describe v1 metadata as presentation-only or excluded from hashing.
  Preserve canonical JSON/hash inputs, and do not describe receiver-computed
  `content_hash` as a sender declaration transmitted in the envelope.
- Do not change existing `CircuitIR`, `ProgramArtifact`, `ExecutionPlan`, or
  Deployment schemas/hashes to pass new fixtures. Protected behavior changes
  require a separate API Change Proposal.

## Compatibility

Preserve the v1 reader, field shapes, coercion, unknown-field/version rejection,
and full envelope hash. A future v2 must remain in the `ProgramArtifact` contract
lineage. An API Change Proposal must define reader version dispatch, old payload
reading, identity correspondence, and migration. New classes, namespaces, or
private metadata conventions cannot bypass this process.

Retain old v1 `content_hash` as envelope identity. Identical `CircuitIR` payloads
still produce different envelope hashes when producer, requirements, parents, or
metadata differ. Adapters must not silently recompute it as payload/executable
identity.

## Migration Sequence

1. Freeze v1 golden fixtures, complete envelope hashes, field/coercion behavior,
   and the direct-consumer inventory.
2. Approve identity layering, closed metadata values/namespaces/capacity, and v1
   compatibility reading first.
3. Approve structured requirements/provenance, typed parent identities, and
   executable payload profiles next. If needed, propose v2 as an evolution of the
   same contract lineage.
4. Compiler first unwraps a `kind=circuit` adapter, retaining its private internal
   identities. Executable information that v1 cannot represent must not be forced
   back into v1 with information loss.
5. Deployment separates packages into artifact, request, target, and receipt for
   individual adaptation. Retire old objects only after callers reach zero and
   compatibility windows and replacement tests are complete.

## Acceptance Tests

- v1 canonical round-trip/golden hashes, independence from field order, rejection
  of unknown fields/versions, and existing coercion behavior.
- Explicit assertions that metadata, producer, required capabilities, ordered
  parents, and payload all affect v1 envelope hashes; `content_hash` is absent
  from `to_dict()` and recomputed by the receiver.
- Separate `CircuitIR.content_hash` and `ProgramArtifact.content_hash`; the latter
  does not replace the former.
- Characterization fixtures preserve parent order/duplicates, opaque producers,
  coarse requirements, and current metadata `to_dict()` duck typing.
- Raw executable bytes, `provenance`, `requirements`, and `extensions` cannot be
  represented in v1 or are rejected.
- Documentation tests prevent a second v2 authority and prevent future candidate
  metadata/hash semantics from being presented as current v1 behavior.

## Open Questions

- v2 dispatch, dual-version reading, expected-digest transport, and links to v1
  identity within the same contract lineage.
- Initial scope for executable bytes/blob references, profiles/media types, and
  identities with roles.
- Closed metadata values, namespaces, depth/entry/byte limits, sensitive fields,
  and key-collision migration.
- Which future fields enter envelope identity, and whether a separate semantic
  fingerprint is needed without replacing the v1 hash.
