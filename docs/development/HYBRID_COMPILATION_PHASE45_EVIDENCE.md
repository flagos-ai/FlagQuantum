# Hybrid compilation Phase 45 evidence

Date: 2026-09-10

Status: **complete — approved ProgramArtifact 3.0 and result-projection slice**

Authorization:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

## Delivered boundary

Phase 45 adds the private Core `ProgramArtifactV3` executable envelope authorized
by Proposal 025. It is intentionally not a replacement source-circuit format:
input circuits and parameter binding remain ProgramArtifact v2, while only an
allocated physical plan can produce the version-3 executable result.

The artifact has strict closed fields, canonical JSON, duplicate-key rejection,
finite bounded structured data, sensitive-locator rejection, deterministic payload
and artifact hashes, a target snapshot binding, and a fully-bound parameter schema.
Its compilation record binds target legalization, physical plan, allocation,
schedule, emission, and conformance identities.

## Logical result projection

The result schema declares dense logical wires and an equally sized unique list of
physical result slots in logical-wire order. Compiler construction verifies those
slots against the actual physical-plan width and final result placement.

For allocated OpenQASM, the quantum register retains physical width while the
classical register has logical width. OpenQASM 2.0 emits one
`measure q[physical] -> c[logical]` statement per logical result. OpenQASM 3.0
emits the corresponding indexed assignments. Existing full-register OpenQASM
bytes remain unchanged when no physical allocation exists.

The conformance parser independently checks both register widths, exact terminal
measurement order, physical wire ranges, operation text, payload identity, and the
reconstructed physical circuit. QCIS allocation fails because the current profile
does not encode ordered result projection.

## Compatibility and preserved limits

- ProgramArtifact v1/v2 dispatch, golden fixtures, constructors, canonical bytes,
  and identity semantics remain unchanged.
- Equal-capacity artifact compilation continues to return ProgramArtifact v2.
- ProgramArtifact v3 supports allocated executable OpenQASM only and has no public
  root export or default-path effect.
- Compilation-evidence v3, Runtime result-width validation, provider identifiers,
  submission, and Deployment handoff are not implemented in this phase.
- Physical workspace remains standard-zero unitary routing storage and carries no
  QEC/FTOC claim.

## Verification

The bounded suite covers Core v3 construction, immutability, round trips, explicit
reader dispatch, duplicate fields, identity and compilation tampering, ambiguous
projections, unsupported profiles, both OpenQASM versions, physical/classical
register widths, terminal mapping order, conformance reconstruction, and the
existing v1/v2 artifact, target-emission, target-conformance, and artifact-
compilation corpus.
