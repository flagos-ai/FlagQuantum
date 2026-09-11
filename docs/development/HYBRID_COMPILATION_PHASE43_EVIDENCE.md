# Hybrid compilation Phase 43 evidence

Date: 2026-09-10

Status: **complete — approved private Compiler allocation slice**

Authorization:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

## Delivered boundary

Phase 43 implements physical routing workspace for the private directed-topology
path when a target exposes more dense physical slots than the source `CircuitIR`
has logical wires. It does not implement the protected physical-plan, artifact,
evidence, Runtime, or Deployment version-3 contracts.

The route validates an injective logical-to-physical placement, builds a complete
nullable physical-to-logical occupancy, maps source operations onto occupied
slots, and may exchange logical state with a null slot while routing a two-wire
operation. Every generated SWAP records both mappings before and after the
transition. Cleanup reverses the routing SWAP sequence and verifies the declared
logical-result placement and null occupancy of all remaining workspace.

The resulting authoritative `CircuitIR` has the physical slot count as its wire
width. Terminal sampling still names only the physical slots corresponding to
dense logical-wire order. This is compiler evidence for the later authenticated
result projection; existing static emission and ProgramArtifact v2 do not accept
the allocated program.

## Compatibility and fail-closed behavior

- Equal logical and physical capacity continues through the unchanged routing-plan
  version-1 branch and retains version-2 physical-plan semantics.
- Undirected excess-capacity routing is not inferred; the first slice requires a
  `DirectedCouplingMap`.
- Physical-plan construction rejects an allocated program with an explicit
  `physical-circuit-plan 3.0` requirement.
- Observables, partial/non-sample logical measurements, dynamic instructions,
  reset, measurement instructions, conditional operations, and channels fail.
- Additional slots are standard-zero routing workspace, not public ancillas or
  fault-tolerant/QEC resources.

## Verification

The bounded suite covers idle-slot routing, nullable-occupancy transitions,
workspace cleanup, logical result-wire preservation, physical-state embedding,
gradient preservation, deterministic identities, invalid injections, unsupported
undirected allocation, excluded observables, explicit physical-plan blocking, and
the complete pre-existing directional-topology and physical-plan suites.

The repository formatting, lint, language, architecture, and relevant test gates
must pass before the phase commit is recorded. No provider call or hardware claim
is part of this evidence.
