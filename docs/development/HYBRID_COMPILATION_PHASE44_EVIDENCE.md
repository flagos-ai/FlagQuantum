# Hybrid compilation Phase 44 evidence

Date: 2026-09-10

Status: **complete — approved physical-circuit-plan 3.0 slice**

Authorization:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

## Delivered boundary

Phase 44 adds a private version-3 `PhysicalCircuitPlan` for the excess-capacity
directed-routing profile delivered in Phase 43. It remains a verified view over
the authoritative Core `CircuitIR`; it is not another instruction IR.

The plan records the logical-wire and physical-slot counts, initial, pre-restore,
and final nullable physical occupancy, ordered logical-result physical slots, and
allocation identity. Each routing transition records both the logical layout and
physical occupancy before and after its SWAP. Construction derives these records
from actual routed instructions rather than trusting a detached summary.

## Validation and identity

Plan validation reconstructs the coupling topology and requires its size to equal
the physical program width. It replays every routing SWAP, including exchanges
between a logical occupant and a null workspace slot, and verifies transition
continuity against instruction metadata. The replay must reach both the declared
final logical layout and final occupancy; all non-result slots are null.

Physical-plan identity version 3.0 covers every new allocation field and nullable
occupancy transition as canonical JSON input. Equal-capacity undirected and
directed plans omit those fields and retain their existing version-1.0 and
version-2.0 identity payloads. They fail closed if allocation evidence is attached.

## Preserved boundary

This phase does not add ProgramArtifact 3.0, compilation-evidence 3.0, result
projection at target emission, Runtime verification, provider binding, or
Deployment handoff. Allocated programs therefore remain unable to cross the
existing version-2 executable-artifact boundary. Physical workspace remains
standard-zero unitary routing storage, not a QEC/FTOC ancilla claim.

## Verification

The suite covers deterministic version-3 plan construction, nullable-occupancy
replay, pre-restore and cleanup state, result-slot binding, allocation and
transition tampering, state and gradient preservation, topology legality, and the
existing version-1/2 physical-plan and directional-routing corpus. Core evidence
and artifact compatibility suites pin the unchanged earlier contracts.
