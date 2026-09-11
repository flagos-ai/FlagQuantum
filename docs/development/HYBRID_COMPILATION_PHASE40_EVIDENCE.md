# Phase 40 directional compilation-evidence Core and Compiler evidence

Date: 2026-09-10

## Result

Core now owns a strict `flagquantum.compilation_evidence_bundle` version 2.0
model for directed-CX topology, explicit initial placement, identity final
layout, native and direction-rewrite lineage, and dependency scheduling.
Compiler constructs it only from the actual retained artifact compilation and
physical plan, and verifies decoded evidence by reconstructing the expected
bundle from those objects.

Version 1.0 remains a separate class with unchanged construction, canonical
identity payload, and strict undirected/identity-layout semantics. The shared
JSON reader performs explicit 1.0/2.0 dispatch and rejects unknown versions and
duplicate keys.

## Verification

- Directed coupling edges are canonical, ordered, unique, in range, and bound
  to a deterministic topology identity.
- Initial and pre-restore layouts are complete permutations; final layout is
  identity and mapping transitions replay exactly.
- Native instruction groups are dense, contiguous, and ordered.
- Reverse-CX groups must contain exactly `h, h, cx, h, h`, ordinals 0–4,
  consistent source/topology/native lineage, and direction-correct wires.
- Every final CX follows an ordered edge; a native SWAP requires a weak physical
  link; other two-wire operations fail closed.
- Schedule depth, maximum width, dependencies, critical path, plan identity,
  bundle identity, closed records, size limits, record limits, and immutable
  role-named maps are verified.
- Canonical round trips reconstruct the correct versioned type, and bundle
  identities are stable across Python hash seeds.
- An actual directed artifact compilation produces evidence 2.0 whose physical
  plan identity exactly matches the retained Compiler plan.

The combined hybrid-compiler, artifact, Core evidence 1.0/2.0, proposal, and
private-contract suite passes 303 tests.

## Boundary

Runtime and Deployment do not yet accept version 2.0. Their read-only handoff
verification is the next bounded phase. No ProgramArtifact change, public root
export, default-path change, provider submission, physical ancilla allocation,
or hardware-performance claim is included.
