# Phase 39 directional-topology Compiler evidence

Date: 2026-09-10

## Result

The approved Proposal 024 Compiler slice is complete. The current
single-`CircuitIR` target path now supports a private canonical directed-CX
topology, complete-permutation initial placement, weak-connectivity routing,
mandatory identity-layout restoration, target-native legalization,
direction-correct reverse-CX synthesis, dependency scheduling, physical-plan
2.0 lineage, deterministic target emission, and executable artifact creation.

The existing public expert `CouplingMap` remains undirected and unchanged. The
removed `_compiler` IR and pass tree were not restored.

## Correctness evidence

- All 36 combinations of three-wire initial layouts and ordered CX operands
  pass on an asymmetric directed line.
- Statevectors match the source up to the simulator's numerical tolerance.
- Trainable parameter values retain object identity and gradients match the
  source circuit.
- Reverse-only CX uses H conjugation and every emitted CX follows an authorized
  ordered edge.
- Physical records retain source, topology, native, direction-rewrite, mapping,
  and dependency-schedule lineage.
- Invalid layouts, unequal capacity, disconnected graphs, unavailable ordered
  edges, absent native H/CX basis, expansion overflow, and plan tampering fail
  closed.
- An artifact compiles through OpenQASM 3 conformance without a TargetIR.

The combined hybrid-compiler, artifact compilation/deployment, Core evidence
1.0 compatibility, proposal, and private-contract suite passes 297 tests.

## Compatibility boundary

Undirected physical plans continue to use the exact 1.0 identity payload.
Directed plans use physical-plan identity version 2.0. Compilation-evidence 1.0
does not reinterpret directed plans and now rejects them with an explicit
version-2 requirement. Core evidence 2.0, Runtime verification, and Deployment
carrying are the next implementation slice.

No public root export, default-path change, provider submission, physical
ancilla allocation, calibration-aware optimization, pulse/timing claim,
fault-tolerant expansion, or hardware-performance claim is included.
