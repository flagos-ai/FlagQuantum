# Phase 31 ProgramArtifact-v2 proposal evidence

Date: 2026-09-10

## Result

API Change Proposal 022 and a machine-readable candidate define the first
executable-text profile in the existing `flagquantum.program_artifact`
contract lineage. The proposal is explicitly unapproved and no implementation
path is enabled.

The candidate covers only fully bound static OpenQASM 2, OpenQASM 3, and QCIS 1
text that has passed the Phase 29/30 path. It separates payload, final circuit,
target, compilation-stage, and envelope identities; keeps shots in the
execution request; and excludes credentials and provider lifecycle state.

## Verification summary

| Gate | Result |
| --- | --- |
| Formal proposal and machine candidate identify the same approval token | pass |
| Candidate status is proposed and implementation is unauthorized | pass |
| Three exact UTF-8 text profiles are closed | pass |
| Payload and envelope identities are distinct | pass |
| Size, nesting, entry, and string limits are explicit | pass |
| Request and sensitive provider state are excluded | pass |
| v1 fixture remains accepted by the unchanged reader | pass |
| v1 generic-envelope fixture retained its then-pinned hash | superseded by Phase 31.1 consumable-circuit fixture |
| Phase 31 candidate and v1 compatibility tests | 7 passed |
| Hybrid compiler, proposal, and private-contract focused suite | 222 passed |

## Claim boundary

This phase produces a proposal, candidate schema, and compatibility evidence.
It does not implement ProgramArtifact v2, modify Core, add a Compiler or Runtime
adapter, change Deployment, submit provider work, add public exports, alter the
default path, or make a performance claim.
