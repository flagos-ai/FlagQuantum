# Physical resource allocation v3 completion review

Updated: 2026-09-10

Status: **private technical exit complete**

Authorization:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

## Executive conclusion

Proposal 025 is complete within its approved private scope. FlagQuantum can compile
a dense logical circuit onto a larger directed physical-slot domain, route through
standard-zero idle workspace, restore that workspace, emit an executable with an
authenticated logical result projection, serialize the proof, verify it in Runtime,
and prepare a side-effect-free Deployment handoff.

This is not a production or fault-tolerant milestone. No v3 type or operation is a
stable export or default execution path. No provider identifier is bound, no
credential is read, and no task is submitted.

## Completed chain

| Boundary | Delivered invariant |
| --- | --- |
| Compiler allocation | Injective logical placement, nullable occupancy, deterministic idle-slot routing and inverse-SWAP cleanup |
| Physical plan 3.0 | Dual layout/occupancy replay, allocation identity, native/direction legality and schedule lineage |
| ProgramArtifact 3.0 | Physical quantum width, logical classical width and authenticated ordered result projection |
| Compilation evidence 3.0 | Closed canonical proof of allocation, transitions, plan, artifact, target and output lineage |
| Runtime | Matching v3 handoff, target/capability preflight and logical-width result validation without mapping repair |
| Deployment dry run | Exact immutable artifact/evidence carriage without provider translation, credentials, task state or submission |

## Acceptance result

The bounded oracle routes two logical wires through a three-slot directed path and
restores the initial placement and null workspace. State and trainable gradient
results match the logical source. OpenQASM 2.0 and 3.0 expose two ordered logical
results from selected physical slots while preserving physical program width.

Strict readers reject unknown or duplicate fields, invalid capacity, occupancy,
transition, topology, direction, schedule, projection and identity data. Bundle
identities are stable across Python hash seeds. Existing ProgramArtifact v1/v2,
compilation-evidence 1.0/2.0, public API snapshots and default paths remain intact.

The machine-readable conclusion is recorded in
`contracts/physical-resource-allocation-v3-exit-audit.json`.

## Residual boundaries

- The v3 profile is internal and deliberately absent from stable package exports.
- Physical slots are compilation-local indices, not provider qubit identifiers.
- Deployment stops before credentials, network access, submission, receipt or job
  lifecycle.
- Workspace is standard-zero unitary SWAP storage only; there is no reset,
  mid-circuit reuse or arbitrary ancilla lifecycle.
- There is no QEC syndrome, logical-qubit expansion, decoder, code-distance,
  magic-state or other fault-tolerant resource claim.
- Calibration/noise-aware placement, timing, pulse, crosstalk, fidelity, capacity,
  latency and QPU performance remain unimplemented and unclaimed.

## Next decision

Proposal 025 authorizes no further expansion. Public API naming/lifecycle,
provider-qubit binding and actual submission each require a new scoped proposal,
compatibility analysis, tests, rollback definition and explicit owner approval.
