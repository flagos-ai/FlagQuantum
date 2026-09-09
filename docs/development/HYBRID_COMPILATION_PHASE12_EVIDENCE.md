# Phase 12 measurement-boolean predicate evidence

Date: 2026-09-09

## Accepted profiles

The first program measures two independently prepared qubits and applies X to
a third qubit only when `first and not second` is true. The second program
compares one measurement with `False` and applies complementary gates in its
true and false branches.

Compiler retains measurement booleans as symbolic classical-bit literals and
emits the existing Core instruction `conditions` metadata. Runtime and
Simulation are unchanged.

## Evidence

| Gate | Result |
| --- | --- |
| Program IR verifies `arith.not` and `arith.and` as bool operations | pass |
| Two measurement values remain distinct classical SSA literals | pass |
| `first and not second` lowers to `((0, 1), (1, 0))` | pass |
| All four two-bit measurement combinations occur under the seeded oracle | pass |
| Controlled output equals `first * (1 - second)` shot by shot | pass |
| Reference and batched trajectory strategies satisfy the same predicate | pass |
| `bit == False` lowers to classical bit 0 equal to zero | pass |
| `bit != True` lowers to the same literal and complement | pass |
| Comparison else branch lowers to the complementary bit value | pass |
| Static bool `not` and `and` continue to specialize correctly | pass |
| Boolean disjunction fails during capture | pass |
| Inline measurement composition fails during capture | pass |
| Negated conjunction fails during dynamic lowering | pass |
| Quantum work in a conjunction else branch fails during lowering | pass |
| Measurement-to-measurement comparison fails during lowering | pass |

## Claim boundary

This evidence covers conjunctive predicates over already assigned measurement
booleans. It does not cover disjunction, arbitrary boolean normal forms,
measurement-to-measurement comparison, classical arithmetic on measurement
results, measurement-dependent carried state, loop measurement, finite-shot
gradients, graph capture, accelerators, distributed execution, capacity, or
performance.
