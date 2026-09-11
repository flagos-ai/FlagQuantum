# Phase 22 structured-control-flow simplification evidence

Date: 2026-09-10

## Accepted profile

The private normalization pipeline now includes
`structured_control_flow_simplification` after constant folding and before dead
constant elimination. A constant `scf.if` is replaced by its selected region;
operation results are redirected to the selected `scf.yield` values. A
constant-bound `scf.for` proven to execute zero times is removed and its results
are redirected to the initial carried values. Both transformations include the
linear quantum effect as well as classical SSA values.

Nonconstant structured control is retained. In particular, Compiler does not
evaluate measurement-dependent predicates. The transformed program passes the
same verifier used at capture and pass boundaries before either static or
dynamic lowering.

## Evidence

| Gate | Result |
| --- | --- |
| Constant true branch retains only its selected quantum operations | pass |
| Unselected branch quantum operations are absent | pass |
| Empty loop body is absent | pass |
| Classical carried wire value is rewired through branch and loop removal | pass |
| Linear quantum effect remains single-use and ordered | pass |
| Measurement-dependent dynamic branch remains executable | pass |
| Static optimized and unoptimized `CircuitIR` artifacts agree | pass |
| Dynamic optimized and unoptimized `CircuitIR` artifacts agree | pass |
| Representative Program IR operation count decreases from 19 to 5 | pass |
| Hybrid compiler tests | 118 passed |
| Full unit suite under the repository fallback-git environment | 1,239 passed, 14 skipped |

## Claim boundary

This is structured canonicalization, not a complete control-flow optimizer.
Nonempty loop unrolling, loop fusion, loop-invariant code motion, speculative
execution, dynamic-predicate resolution, target legalization, MLIR/LLVM/QIR
integration, compile-time or runtime speedup, accelerator, provider,
distributed, capacity, and fault-tolerance claims remain outside the accepted
profile.
