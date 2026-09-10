# API Change Proposal 025: Physical resource allocation and logical result projection

## Status

**Approved on 2026-09-10; Runtime v3 verification slice complete.**

Date: 2026-09-10

Approval token:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

The API owner supplied the exact token on 2026-09-10. The first bounded
implementation adds private sparse placement, nullable physical occupancy,
deterministic idle-slot routing, inverse-SWAP workspace cleanup, logical-result
slot metadata, and explicit rejection by the still-version-2 physical-plan path.
ProgramArtifact 3.0, physical-plan 3.0, compilation-evidence 3.0, target emission,
Runtime verification, and Deployment handoff remain separate gated slices.

The second bounded implementation adds private physical-circuit-plan version 3.0.
It replays both logical layouts and nullable physical occupancy from the routed
instructions, binds allocation and result-slot facts into the plan identity, and
retains the exact version-1/2 identity payloads for equal-capacity programs.

The third bounded implementation adds Core ProgramArtifact version 3.0 for
allocated executable OpenQASM, Compiler construction from the actual version-3
physical plan, ordered partial-register emission, and strict conformance parsing.
Its immutable result schema binds dense logical-wire order to unique physical
result slots.

The fourth bounded implementation adds strict Core compilation-evidence version
3.0 and Compiler construction and verification from the actual retained plan and
artifact compilation. The evidence independently replays nullable occupancy,
logical layouts, cleanup, allocation identity, result projection, topology,
direction legality, instruction lineage, and schedule facts. Runtime and
Deployment consumption remain unimplemented.

The fifth bounded implementation adds Runtime preflight and immutable handoff
verification for matching ProgramArtifact v3 and compilation-evidence 3.0 values.
Runtime cross-checks physical-plan and allocation identities, logical result wires,
physical result slots, ordering, and shot ownership. Returned sample tensors must
already have logical width; Runtime rejects physical-width or scalar results and
does not infer or repair a mapping. Deployment v3 handoff remains unimplemented.

## Problem

The approved directional-topology slice deliberately requires equal logical and
physical capacity. Its layout is a complete permutation and routing restores the
identity layout before a full-register terminal measurement. Those invariants let
ProgramArtifact v2, compilation-evidence 2.0, and the existing target emitters keep
one dense wire range without confusing logical results with physical resources.

A real target may have more available physical wires than a source program has
logical wires. Routing through an initially idle target wire can make an otherwise
impossible or unnecessarily long interaction legal. This is not expressible by the
current contracts: an idle slot has no logical-wire integer, a sparse placement is
not a permutation, and increasing `CircuitIR.n_wires` causes ProgramArtifact v2 to
expose every physical slot as a user-visible result.

Silently relaxing any one of those checks would make existing version-2 identities
and result semantics ambiguous. Physical allocation therefore requires explicit
new versions rather than reinterpretation of v2.

## Decision

Add a bounded, initially private physical-resource-allocation profile with:

- compilation-evidence bundle version 3.0;
- physical-circuit-plan version 3.0; and
- ProgramArtifact version 3.0 for allocated executable target text.

ProgramArtifact v1/v2 and compilation evidence 1.0/2.0 remain byte- and
semantics-compatible. The authoritative source program remains Core `CircuitIR`;
no second instruction authority or TargetIR is introduced.

The first implementation supports idle routing workspace only. Every additional
physical slot begins in the standard zero state, may participate in compiler-added
SWAP routing, and must be idle again at the result boundary. Such a slot is not a
fault-tolerant ancilla, syndrome qubit, reusable live-range resource, or user-visible
logical wire.

## Four distinct wire domains

The implementation must not use the word `wire` without an owning domain:

1. **Logical wire:** the dense source `CircuitIR` wire range and user result order.
2. **Physical slot:** a dense compilation-local index in the selected target
   snapshot, used by routing, legalization, scheduling, and emitted instructions.
3. **Provider qubit identifier:** an adapter-owned device identifier. Version 3.0
   does not serialize it in Core evidence and does not infer it from a logical wire.
4. **Result position:** the ordered output column returned to the user. It is
   projected from declared physical result slots into dense logical-wire order.

The initial profile requires physical slots to be `[0, ..., P-1]`. Translation to
provider identifiers remains a target-adapter responsibility outside the Core
evidence bundle. Sparse or textual provider identifiers require a later separately
reviewed deployment-binding contract.

## Allocation and routing semantics

For `L` logical wires and `P` physical slots, version 3.0 requires `P >= L`.
`initial_logical_to_physical` has length `L`, contains unique slots in
`[0, P)`, and need not cover all slots. `physical_to_logical` occupancy has length
`P`; each entry is either one logical-wire integer or null.

Each routing transition records the complete occupancy before and after a SWAP.
Replaying a transition exchanges exactly two physical-slot occupants, including
the case where one occupant is null. The logical-to-physical projection derived
from every occupancy must agree with the recorded layout.

The first profile retains mandatory restoration: the final logical-to-physical
layout equals the declared `logical_result_physical_slots`, and all other physical
slots have null occupancy. This restores the additional workspace to its standard
zero state for the supported unitary SWAP profile. It does not claim general
ancilla cleanup after measurement, reset, noise, or non-unitary operations.

## ProgramArtifact 3.0 result contract

An allocated executable payload may declare `P` physical slots while returning
only `L` logical results. Its closed result schema is:

```text
kind = "samples"
logical_wires = [0, ..., L-1]
physical_result_slots = [slot_for_logical_0, ..., slot_for_logical_L-1]
ordering = "logical_wire_order"
shots_source = "execution_request"
```

`physical_result_slots` must be unique, in range, and equal the final physical-plan
layout. Emitters measure exactly these slots in this order, or fail closed if their
target format cannot preserve that ordering. Runtime validates the returned width
as `L`; it must never expose the unused `P-L` slots or infer projection from payload
text.

ProgramArtifact 3.0 compilation identity additionally binds the physical-plan
identity and allocation identity. The target snapshot identity remains mandatory.
No provider credentials, provider task identifiers, mutable calibration records,
or execution results enter the artifact.

## Evidence-bundle 3.0

Version 3.0 retains explicit source, target, output, lineage, native legality,
direction legality, and schedule evidence from version 2.0. Its physical plan adds:

```text
logical_wire_count
physical_slot_count
initial_physical_to_logical
pre_restore_physical_to_logical
final_physical_to_logical
logical_result_physical_slots
allocation_identity
```

Mapping transitions add complete nullable occupancy before and after the routed
SWAP. Existing logical-to-physical fields remain length `L`. Instruction physical
wires are physical slots in `[0, P)`; instruction logical wires remain source
logical wires. Plan and bundle identities cover every added field.

Readers dispatch only on an explicit supported version. A 1.0 or 2.0 bundle is
never promoted to 3.0 by inference. Version 2.0 continues to mean equal capacity,
complete-permutation placement, identity restoration, and full-register results.

## Ownership and compatibility

- Core owns immutable ProgramArtifact 3.0 and evidence 3.0 values, strict fields,
  canonical JSON, limits, version dispatch, and result-projection validation.
- Compiler owns allocation, occupancy transitions, routing, cleanup proof,
  legalization, scheduling, emission, and construction from actual retained
  compilation objects.
- Runtime verifies the artifact and evidence against the actual source and target,
  validates logical result width/order, and does not repair mappings.
- Deployment may carry only verified immutable objects. Provider-slot translation
  is not authorized by this proposal.
- Existing `CouplingMap`, `DirectedCouplingMap`, ProgramArtifact v1/v2,
  compilation-evidence 1.0/2.0, public exports, and default execution do not change.

## Limits and excluded claims

The first version-3 profile excludes:

- user-controlled ancillas or ancilla values other than the standard zero state;
- mid-circuit measurement, reset, conditional control, or dynamic ancilla reuse;
- QEC syndrome extraction, logical-qubit expansion, code deformation, decoding,
  magic-state factories, and any fault-tolerant resource claim;
- arbitrary initial quantum states on physical workspace;
- observable remapping and expectation-valued target artifacts;
- sparse/textual provider identifiers and provider submission;
- calibration-aware allocation, noise-aware routing, gate durations, pulses,
  crosstalk, fidelity, capacity, latency, or QPU performance claims.

The version-2 limits for nesting, sizes, instructions, transitions, predecessors,
strings, duplicate keys, finite numbers, and sensitive fields remain minimum
requirements and may only become stricter.

## Implementation gates

1. **Complete:** record this proposal and the exact machine-readable candidate.
2. **Complete:** receive the exact approval token from the API owner.
3. **Complete:** implement private sparse placement and nullable-occupancy routing without
   changing the existing equal-capacity path.
4. **Complete:** add physical-plan 3.0 validation, cleanup proof, deterministic identity, and
   bounded state/gradient differential tests.
5. **Complete:** add strict Core ProgramArtifact 3.0 and compilation-evidence 3.0 while retaining
   pinned v1/v2 fixtures and explicit reader dispatch.
6. **In progress:** target emission, conformance, and Runtime verification are
   complete; Deployment dry-run handoff remains a separate bounded slice.
7. Keep all new entry points non-root until naming, lifecycle, and provider-binding
   review.

## Acceptance

- A two-logical-wire interaction can route deterministically through an idle slot
  on a three-or-more-slot directed path and restore the declared result placement.
- Nullable occupancy and logical layout replay exactly for every transition;
  duplicate, missing, out-of-range, disconnected, or insufficient allocations fail.
- The final occupancy contains every logical wire exactly once and null elsewhere.
- State and trainable gradients match the source over the bounded unitary oracle
  corpus, and restored workspace is verified in the zero state.
- Emitters return only declared logical results in logical-wire order; no workspace
  output is silently exposed or discarded without an authenticated projection.
- Every final operation remains target-native and direction-legal, and all lineage,
  identities, schedules, artifact hashes, and evidence bindings replay.
- Canonical JSON and identities are deterministic across hash seeds.
- Existing v1/v2 fixtures, bytes, identities, public exports, and default execution
  behavior remain unchanged.

## Approval

The API owner approved this proposal with the exact token:

`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

The approval authorizes only the gated profile above. It does not authorize public API
promotion, provider submission, general ancillas, or fault-tolerant/QEC claims.
