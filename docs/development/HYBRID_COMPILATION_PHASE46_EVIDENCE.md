# Hybrid compilation Phase 46 evidence

Date: 2026-09-10

Status: **complete — approved compilation-evidence 3.0 slice**

Authorization:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

## Delivered boundary

Phase 46 adds private Core `CompilationEvidenceBundleV3`,
`PhysicalPlanEvidenceV3`, and `MappingTransitionEvidenceV3` values. The JSON
reader dispatches version 3.0 explicitly; it never promotes a version-1/2 bundle
by inference. Compiler construction chooses v3 only for an actual allocated
physical plan and verification reconstructs the complete expected bundle from
retained compilation objects.

## Allocation proof

The evidence distinguishes logical-wire count from physical-slot count. Every
logical layout is an injection into physical capacity and every nullable occupancy
covers all physical slots with each logical wire exactly once. Each transition
must exchange exactly two occupancy entries, and its derived logical layout must
match the recorded layout before and after the SWAP. Replay authenticates the
initial, pre-restore, and final states.

The final logical layout must equal the ordered physical result slots. The
allocation identity is independently recomputed from the initial placement,
initial occupancy, standard-zero workspace assumption, inverse-SWAP cleanup, and
result projection. Plan and bundle identities cover all added state.

## Preserved limits

- Compilation evidence 1.0/2.0 and ProgramArtifact v1/v2/v3 remain unchanged.
- v3 retains strict topology, direction-rewrite, native-lineage, scheduling,
  canonical encoding, duplicate-field, size, and identity validation.
- No public root export or default-path behavior changes.
- Runtime and Deployment do not accept v3 in this phase.
- Provider identifiers, submission, general ancillas, QEC, and FTOC claims remain
  excluded.

## Verification

The bounded suite covers canonical round trips, immutable mappings, explicit v3
dispatch, nullable occupancy replay, logical result projection, allocation and
bundle identities, unknown fields, record limits, actual Compiler construction,
and exact verification against allocated OpenQASM artifact compilation. Existing
v2 evidence tests remain green.
