# Phase 37 compilation-evidence contract proposal evidence

Date: 2026-09-10

## Outcome

Proposal 023 and its machine-readable candidate contract define a separate,
versioned `CompilationEvidenceBundle`. The proposed bundle preserves Phase 36
mapping, native-gate, and dependency evidence across process boundaries without
changing ProgramArtifact v1/v2 or introducing another circuit IR.

Implementation is not authorized. The exact approval token remains an explicit
gate before changes to Core, Compiler, Runtime, or Deployment.

## Decisions pinned

| Decision | Candidate result |
| --- | --- |
| Serialization owner | Core |
| Construction owner | Compiler |
| Runtime and Deployment role | Read-only verification and transport adapters |
| ProgramArtifact v1/v2 | Unchanged |
| Circuit authority | Core `CircuitIR` |
| Evidence payload | Mapping, native lowering, dependency schedule, and role-named identities |
| Provider and execution state | Prohibited |
| Timing, pulse, calibration, FTOC, and performance claims | Excluded |
| Public export | Not proposed for the initial implementation |

## Verification

The candidate-contract, private hybrid-contract, and unchanged ProgramArtifact
v2 candidate suites passed **55 tests**. Repository policy checks must continue
to protect ProgramArtifact compatibility during implementation.

## Approval gate

Implementation requires the exact API-owner response:

```text
approve API_CHANGE_PROPOSAL_023_COMPILATION_EVIDENCE_BUNDLE
```
