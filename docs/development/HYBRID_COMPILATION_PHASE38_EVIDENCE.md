# Phase 38 directional-topology and layout proposal evidence

Date: 2026-09-10

## Result

API Change Proposal 024 and a machine-readable candidate define how the proven
historical directed-routing semantics can migrate into the current
single-`CircuitIR` target pipeline. The proposal is explicitly unapproved and
no implementation path is enabled.

The design preserves the existing undirected `CouplingMap`, compilation
evidence 1.0, and ProgramArtifact v1/v2 contracts. It proposes a distinct
private directed topology type, explicit full-permutation initial placement,
direction-correct CX legalization, identity final-layout restoration, complete
instruction lineage, and compilation-evidence version 2.0.

## Verified migration basis

The historical Phase 2 Batch C code and tests establish reusable semantics for
canonical directed graphs, stable weak-connectivity paths, reverse-only CX
synthesis, non-identity placement, identity restoration, state equivalence,
gradient preservation, invalid-layout rejection, disconnected-topology
failure, and exhaustive three-wire placement/operand cases. The proposal uses
those results as an oracle but explicitly forbids restoring the removed
parallel `_compiler` tree.

## Verification summary

| Gate | Result |
| --- | --- |
| Proposal and candidate use the same exact approval token | pass |
| Candidate status is unapproved and implementation is disabled | pass |
| Existing `CouplingMap` semantics remain undirected | pass |
| Evidence 1.0 remains strict read-only compatible | pass |
| ProgramArtifact v1/v2 remain unchanged | pass |
| Directed CX and initial/final layout semantics are explicit | pass |
| Missing edge/native basis and invalid layout fail closed | pass |
| Physical ancillas and broader hardware claims are excluded | pass |
| Removed private compiler tree is not restored | pass |

## Claim boundary

This phase produces only a proposal, candidate contract, migration analysis,
and governance tests. It does not add directed topology code, change Core
serialization, alter compilation behavior, add Runtime or Deployment adapters,
export a public API, change the default path, or make a hardware/performance
claim.
