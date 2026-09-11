# Hybrid compilation Phase 47 evidence

Date: 2026-09-10

Status: **complete — approved Runtime v3 verification slice**

Authorization:
`approve API_CHANGE_PROPOSAL_025_PHYSICAL_RESOURCE_ALLOCATION`

## Delivered boundary

Runtime preflight now accepts Core ProgramArtifact v3 and applies the existing
target snapshot, profile, capability, and execution-request shot checks. Runtime
handoff verification accepts compilation-evidence 3.0 only with ProgramArtifact
v3 and preserves the v1/v2 pairing rules.

For v3, verification binds the actual source lineage and target snapshot to the
executable artifact and then cross-checks the physical-plan identity, allocation
identity, schedule, logical wires, ordered physical result slots, result kind,
ordering, and shot-source ownership against Core evidence.

## Result boundary

`validate_executable_result_samples` validates the adapter-normalized tensor's
last dimension against the artifact's declared logical result width. It returns
the same tensor without gathering, truncating, or reordering it. The Compiler's
OpenQASM emitter owns `q[physical] -> c[logical]`; a Runtime adapter must expose
those classical positions in logical-wire order. A scalar result or a tensor with
physical rather than logical width fails closed.

## Preserved limits

- Existing ProgramArtifact v2 and compilation-evidence 1.0/2.0 behavior remains
  unchanged.
- Runtime does not parse target text, repair mappings, submit work, or bind
  provider-qubit identifiers.
- Deployment does not accept ProgramArtifact/evidence v3 in this phase.
- The physical workspace is still standard-zero routing storage, not a QEC/FTOC
  ancilla claim.

## Verification

The allocated OpenQASM 2.0 and 3.0 integration paths now pass Runtime preflight,
artifact/evidence handoff verification, and logical-width sample validation.
Tests reject mixed artifact/evidence versions, physical-width results, scalar
results, and non-tensor results while retaining the earlier v1/v2 regression
suite.
