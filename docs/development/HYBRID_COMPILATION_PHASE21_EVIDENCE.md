# Phase 21 verified Program IR normalization evidence

Date: 2026-09-10

## Accepted profile

The private hybrid compiler now runs a bounded verified normalization stage
before static specialization and dynamic lowering. It computes constant-value,
whole-program SSA-use, and operation-count facts. The default pipeline folds
supported constant-only arithmetic and removes unused constants. It preserves
the existing `HybridProgram` representation and lowers through the existing
Core `CircuitIR` handoff.

Every input and transformed program is verified. A pass returning another type,
a malformed program, or a sequence longer than 32 entries fails closed. Each
lowered result records the source and optimized semantic identities and the
operation-count transition for every pass. The private `optimize=False` option
provides the differential oracle used by tests.

## Evidence

| Gate | Result |
| --- | --- |
| Constant-only arithmetic is folded | pass |
| Unused constants are removed after folding | pass |
| Representative operation count decreases from 9 to 4 | pass |
| Repeated capture produces identical optimized identities and records | pass |
| Re-optimizing an optimized program is an identity transformation | pass |
| Static optimized and unoptimized `CircuitIR` artifacts agree | pass |
| Dynamic optimized and unoptimized `CircuitIR` artifacts agree | pass |
| Runtime tensor binding and autograd edge remain attached | pass |
| Invalid pass interface, result, and malformed IR fail closed | pass |
| Existing and new hybrid compiler tests pass | 117 passed |
| Full unit suite under the repository fallback-git environment | 1,230 passed, 14 skipped |

## Claim boundary

This establishes a real analysis-and-transformation stage, not a complete
optimizing compiler. Dead-operation elimination is intentionally limited to
unused constants so runtime failure behavior is not erased. There is no new
Target IR, dialect legalization, target-aware optimization, scheduling,
general pass plugin API, MLIR/LLVM/QIR integration, compile-time speedup, runtime
speedup, accelerator, distributed, provider, capacity, or fault-tolerance
claim.
