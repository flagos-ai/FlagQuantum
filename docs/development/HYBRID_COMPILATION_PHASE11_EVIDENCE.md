# Phase 11 nested structured-state evidence

Date: 2026-09-09

## Accepted profiles

The first source program carries scalar angle, index wire, and bool selector
state through an outer bounded loop and an inner parity branch. It then uses the
updated values to select RX or RY and returns a measurement from the final
carried wire.

The second source program carries a scalar through an outer input-dependent
branch and an inner two-iteration loop, then consumes the selected final value
in a post-branch RX and measurement.

## Evidence

| Gate | Result |
| --- | --- |
| Recursive assignment discovery finds writes below nested regions | pass |
| Outer loop carries scalar/index/bool/effect state | pass |
| Inner branch carries the same required state with pass-through | pass |
| Loop-around-branch lowers to RX(pi), RY(2pi), then measurement | pass |
| Final carried wire is used by the post-loop measurement | pass |
| Outer branch carries a scalar updated by its inner loop | pass |
| True branch accumulates two pi/2 increments and measures one | pass |
| False branch passes through zero angle and measures zero | pass |
| Existing cumulative loop-unroll ceiling remains authoritative | pass |
| Measurement-dependent state escape remains rejected | pass |
| Tensor carried state and loop measurement remain rejected | pass |

## Claim boundary

This evidence covers nesting composed from the restricted `if` and bounded
`for` vocabulary with scalar, index, and bool carried values. It does not cover
tensor state, while loops, break/continue, exceptions, arbitrary mutation,
measurement-dependent classical state escape, loop measurement, finite-shot
gradients, graph capture, accelerators, distributed execution, capacity, or
performance.
