# Phase 15 fixed-round syndrome-feedback evidence

Date: 2026-09-09

## Accepted profile

The private dynamic compiler supports measurement, immediate Boolean feedback,
and unconditional reset inside statically bounded loops. The accepted oracle
is a three-round ancilla-syndrome program with a deterministic injected data
error and same-round correction.

## Evidence

| Gate | Result |
| --- | --- |
| Program IR verifies `quantum.reset` as index, effect -> effect | pass |
| Three loop iterations lower to three ordered measurement instructions | pass |
| Measurements receive dense classical bits 0, 1, and 2 | pass |
| The final loop-carried syndrome returns classical bit 2 | pass |
| The first syndrome detects the injected data error | pass |
| Same-round conditional X clears the data error | pass |
| The second and third syndromes remain zero | pass |
| Every round emits and executes one unconditional ancilla reset | pass |
| Final data and ancilla samples are both zero | pass |
| Reference and batched trajectories produce the same outcome | pass |
| A configured measurement ceiling below three fails before Runtime | pass |

## Claim boundary

This evidence establishes one fixed-round syndrome-feedback control slice. It
does not establish a full stabilizer-code implementation, decoder integration,
logical error rates, threshold behavior, measurement-dependent termination,
conditional measurement/reset, finite-shot gradients, provider execution,
real-time latency, capacity, or performance.
