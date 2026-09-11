# Phase 23 bounded constant-loop unroll evidence

Date: 2026-09-10

## Accepted profile

`bounded_loop_unroll` handles direct entry-block loops whose lower, upper, and
nonzero step are compile-time integer facts. The default ceiling is eight
iterations per loop and 256 expanded operations across the pass. Above-budget
loops remain as `scf.for` operations for existing lowering.

Every expanded iteration receives an explicit induction constant and fresh SSA
identities for cloned definitions, including nested region definitions.
Classical loop-carried values and the linear quantum effect flow through each
cloned yield. The number of removed loop iterations is recorded in `PassRecord`
and carried into specialization and dynamic-lowering counters, preserving the
caller's original unroll-limit semantics.

## Evidence

| Gate | Result |
| --- | --- |
| Three-iteration parameterized loop is removed | pass |
| Three ordered RX gates target wires 1, 0, and 1 | pass |
| Optimized and unoptimized parameter values agree | pass |
| Gradient of parameter sum with respect to theta is 3 | pass |
| Gradient of parameter sum with respect to delta is 6 | pass |
| Fresh cloned SSA definitions pass the ordinary verifier | pass |
| Repeated normalization is an identity transformation | pass |
| Three pre-expanded iterations remain visible in audit evidence | pass |
| A caller limit of two still rejects the three-iteration program | pass |
| Dynamic metadata retains all three pre-expanded iterations | pass |
| Dynamic optimized and unoptimized `CircuitIR` artifacts agree | pass |
| Iteration and operation budget overflow preserve `scf.for` | pass |
| Hybrid compiler tests | 121 passed |
| Full unit suite under the repository fallback-git environment | 1,242 passed, 14 skipped |

## Claim boundary

The pass does not expand nested loops independently, runtime-dependent bounds,
zero-step loops, loops above either budget, or arbitrary control-flow graphs.
It establishes semantic preservation for the checked profile, not compile-time
or runtime acceleration. Target scheduling, vectorization, MLIR/LLVM/QIR,
accelerator, provider, distributed, capacity, and fault-tolerance claims remain
outside this phase.
