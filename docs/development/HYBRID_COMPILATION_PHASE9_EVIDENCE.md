# Phase 9 loop-carried classical-state evidence

Date: 2026-09-09

## Accepted profile

The source program initializes a scalar angle and an index wire, updates both
through two iterations of a bounded `range`, applies RX with the current values,
uses the final scalar in a post-loop RY, measures one wire, and executes through
the existing local batched dynamic trajectory path.

Capture emits the scalar, index, and quantum effect as explicit `scf.for`
operands and results. The body receives iteration, scalar, index, and effect
arguments and yields the updated scalar, index, and effect. No mutable runtime
cell or hidden side table is introduced.

## Evidence

| Gate | Result |
| --- | --- |
| Verified loop exposes scalar/index/effect operands and results | pass |
| Body arguments and yield signature match the carried tuple | pass |
| Each unrolled iteration consumes the previous scalar value | pass |
| Each unrolled iteration consumes the previous index value | pass |
| Post-loop RY consumes the explicit final scalar result | pass |
| Accumulated RX angles are pi and 2pi for the deterministic oracle | pass |
| Carried index selects wires 1 then 0 | pass |
| Carried bool selects a post-loop branch, including zero-trip behavior | pass |
| Measurement and final sample observe the expected wire state | pass |
| Tensor carried state fails during capture | pass |
| Loop-target shadowing fails during capture | pass |
| Nested-loop rebinding of an outer value fails during capture | pass |
| Existing hybrid compiler suite remains green | pass |

## Claim boundary

This evidence covers direct loop-body rebinding of existing scalar, index, and
bool locals. It does not cover tensor state, branch-carried values, nested-loop
carry, augmented assignment, loop measurement, dynamic loop bounds derived
from measurement, finite-shot gradients, graph capture, accelerators,
distributed execution, capacity, or performance.
