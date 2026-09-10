# Phase 37 compilation-evidence Core and Compiler evidence

Date: 2026-09-10

## Outcome

Proposal 023 was approved with the exact token. Its machine-readable contract
and Core implementation define a separate, versioned
`CompilationEvidenceBundle`. The bundle preserves Phase 36
mapping, native-gate, and dependency evidence across process boundaries without
changing ProgramArtifact v1/v2 or introducing another circuit IR.

Compiler construction and verification consume actual retained compilation
objects and the exact target snapshot. Runtime verifies the decoded bundle
against the actual fully bound source or binding result, target snapshot, and
executable artifact. Deployment dry-run carries only a verified bundle and
does not submit work or acquire provider state.

## Decisions pinned

| Decision | Candidate result |
| --- | --- |
| Serialization owner | Core; implemented |
| Construction owner | Compiler; implemented |
| Runtime and Deployment role | Implemented read-only verification and dry-run carrying |
| ProgramArtifact v1/v2 | Unchanged |
| Circuit authority | Core `CircuitIR` |
| Evidence payload | Mapping, native lowering, dependency schedule, and role-named identities |
| Provider and execution state | Prohibited |
| Timing, pulse, calibration, FTOC, and performance claims | Excluded |
| Public export | Not proposed for the initial implementation |

## Verification

The combined hybrid compiler, Core artifacts, compilation evidence,
artifact-to-artifact compiler, Runtime/Deployment adapters, local execution,
and private-contract suites passed **300 tests**. ProgramArtifact compatibility
remained unchanged.

## Approval record

The API owner supplied:

```text
approve API_CHANGE_PROPOSAL_023_COMPILATION_EVIDENCE_BUNDLE
```
