# Phase 10 branch-carried classical-state evidence

Date: 2026-09-09

## Accepted profile

The source program initializes a scalar angle, index wire, and bool selector;
updates all three directly in both sides of an input-dependent branch; uses the
merged bool to select RX or RY; and consumes the merged scalar and index in the
selected gate and final measurement.

Capture emits one `scf.if` with predicate, scalar, index, bool, and quantum
effect operands. Both blocks receive scalar, index, bool, and effect arguments
and yield the same tuple. Results replace the outer bindings without a mutable
runtime cell or side table.

## Evidence

| Gate | Result |
| --- | --- |
| `scf.if` exposes scalar/index/bool/effect operands and results | pass |
| Both branch argument signatures match the carried tuple | pass |
| Both branch yields match the result tuple | pass |
| True path selects RX(2pi) on carried wire 1 | pass |
| False path selects RY(pi) on carried wire 0 | pass |
| Final measurement consumes the carried wire | pass |
| Both deterministic trajectory outcomes match the specialized circuit | pass |
| Existing one-sided scalar assignment uses unchanged-value pass-through | pass |
| Tensor carried state fails during capture | pass |
| Nested branch rebinding fails during capture | pass |
| Measurement-dependent carried state fails during dynamic lowering | pass |

## Claim boundary

This evidence covers direct branch-body rebinding under predicates resolved by
specialization. It does not cover tensor state, nested-region carry, augmented
assignment, classical values escaping measurement-dependent branches, general
runtime classical expressions, finite-shot gradients, graph capture,
accelerators, distributed execution, capacity, or performance.
