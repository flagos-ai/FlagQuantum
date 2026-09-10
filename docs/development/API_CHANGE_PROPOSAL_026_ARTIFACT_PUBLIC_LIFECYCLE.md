# API Change Proposal 026: Artifact public naming and lifecycle

## Status

**Approved on 2026-09-10; experimental read-only preview complete.**

Approval token:
`approve API_CHANGE_PROPOSAL_026_ARTIFACT_PUBLIC_LIFECYCLE`

The API owner supplied the exact token on 2026-09-10. The approved implementation
adds the six-symbol read-only preview, frozen branded role views, strict Core
dispatch, canonical dumping, bounded text input/output, and lazy experimental
discovery. It does not add a stable export or workflow operation.

## Problem

Proposals 022–025 established three strict ProgramArtifact envelope versions,
three compilation-evidence versions, Compiler construction, Runtime verification,
and a Deployment dry-run. Their concrete classes and adapters intentionally remain
internal. The private profile is technically complete, but directly exporting
`ProgramArtifactV2`, `ProgramArtifactV3`, or `CompilationEvidenceBundleV3` would
make wire-format revisions part of the user-facing type hierarchy.

There is also an existing internal version-1 class named `ProgramArtifact`.
Reusing that name as an alias for a newer concrete class would create ambiguous
construction, `isinstance`, migration, and documentation semantics. Exporting all
versions would force ordinary users to dispatch on schema versions that Core
already owns.

## Decision

Introduce a read-only preview at `flagquantum.experimental.artifacts`. Do not add
root, Core, Compiler, Runtime, or Deployment exports. The preview exposes stable
role names rather than versioned implementation classes:

```text
ProgramArtifact          frozen read-only role view
CompilationEvidence      frozen read-only role view
load_program_artifact
dump_program_artifact
load_compilation_evidence
dump_compilation_evidence
```

`load_*` accepts JSON text and delegates to the existing strict Core version
dispatcher. `dump_*` accepts only the matching branded role view and
returns its existing canonical JSON. Neither function migrates, repairs, coerces,
recompiles, revalidates against external objects, or changes an identity.

The views expose only the common portable envelope surface:

```text
ProgramArtifact:
  version
  kind
  producer
  payload_sha256 | null
  artifact_identity | content_hash
  to_dict()
  to_json()

CompilationEvidence:
  version
  producer
  bundle_identity
  to_dict()
  to_json()
```

Because the legacy v1 artifact uses `content_hash` while v2/v3 use
`artifact_identity`, the role view provides a read-only `identity` property through
the preview adapter rather than declaring either storage name portable. Each
loaded Core value is held by a frozen role view; concrete internal class names are
not returned as the public contract. Dumpers accept only these branded view types,
not arbitrary objects that happen to have similarly named attributes.

## Why an experimental module first

The schema bytes and readers are strict, but public usage patterns are not yet
known. A preview is needed to validate naming, inspection needs, typing, error
messages, documentation, and whether users require construction or compilation
entry points. Experimental placement makes that uncertainty explicit without
weakening the underlying serialized identities.

The preview module is domain-specific and lazily discoverable from
`flagquantum.experimental`. Its six names are the only proposed additions. No
symbol is re-exported from `flagquantum` or another stable namespace.

## Lifecycle

The preview has three explicit states:

1. `experimental_read_only`: this proposal; parsing and exact canonical dumping.
2. `stable_read_only_candidate`: requires usage evidence, naming review, typing
   review, security review, and a separate approval.
3. `stable_workflow_candidate`: constructors, binding, compilation, verification,
   Runtime, or Deployment operations require an additional proposal after the
   target-capability public contract is settled.

Experimental removal or rename requires release-note notice and one minor-release
compatibility alias once the project begins publishing the preview. Promotion does
not change artifact JSON, identities, or concrete Core ownership.

## Error and security contract

- Invalid JSON, duplicate keys, unsupported versions, unknown fields, identity
  mismatches, size violations, and nested semantic violations propagate as the
  existing deterministic `TypeError` or `ValueError` classes.
- Loaders accept text only. They do not accept paths, URLs, bytes, streams, network
  locations, provider handles, or callbacks.
- Dumpers return the exact Core canonical JSON and reject lookalike mutable
  mappings or objects that merely define similarly named attributes.
- Preview views are frozen and retain the underlying immutable Core value without
  copying payloads into a second authority.
- No credentials, provider identifiers, job state, execution results, arbitrary
  code, or binary payload support is added.

## Compatibility

- ProgramArtifact v1/v2/v3 and compilation-evidence 1.0/2.0/3.0 bytes, identities,
  dispatch, limits, and validation remain unchanged.
- Existing stable exports and the public API snapshot remain unchanged.
- `flagquantum.experimental.__all__` gains only the `artifacts` domain module after
  approval; feature symbols are not re-exported from that package.
- Default compile, plan, run, remote, and deployment behavior remains unchanged.

## Excluded scope

This proposal does not authorize:

- public artifact constructors, parameter binding, compilation, evidence
  construction, Runtime verification, or Deployment preparation;
- concrete versioned class exports or public `isinstance` promises;
- target capability snapshot public APIs;
- provider qubit binding, credentials, submission, receipts, jobs, cancellation,
  polling, or result retrieval;
- general ancilla, QEC, FTOC, calibration, noise-aware allocation, timing, pulse,
  fidelity, latency, capacity, or performance claims;
- stable namespace or root promotion.

## Implementation gates

1. **Complete:** record this proposal and exact machine-readable candidate.
2. **Complete:** receive the exact approval token from the API owner.
3. **Complete:** add the reserved `experimental.artifacts` domain and only the six approved
   symbols.
4. **Complete:** add frozen role views and strict load/dump delegation without changing Core
   serialization.
5. **Complete:** pin v1/v2/v3 round trips, identity preservation, duplicate/unknown/size failure,
   lookalike rejection, lazy import behavior, and the stable public API snapshot.
6. **Complete:** publish preview documentation that states the lifecycle and excluded operations.
7. **Complete for this proposal:** keep stable promotion and workflow operations
   behind later proposals.

## Acceptance

- Users can load and canonically dump every supported artifact and evidence version
  through role-named APIs without importing versioned concrete classes.
- Loading then dumping returns exactly the canonical Core JSON and preserves
  identity.
- Unsupported, malformed, oversized, tampered, or lookalike inputs fail closed.
- Importing `flagquantum` does not eagerly import the preview implementation.
- Stable package exports and default execution behavior do not change.
- No compiler, Runtime, Deployment, provider, or QEC/FTOC capability becomes
  public by implication.

## Approval

The API owner approved this bounded implementation with exactly:

`approve API_CHANGE_PROPOSAL_026_ARTIFACT_PUBLIC_LIFECYCLE`
