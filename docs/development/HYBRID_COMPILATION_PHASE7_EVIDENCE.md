# Phase 7 dynamic measurement-session evidence

Date: 2026-09-09

## Accepted profile

The private source program prepares a three-wire state, measures wire 0 into a
boolean SSA value, applies X to wire 1 when the bit is one and X to wire 2 when
the bit is zero, and returns the measured bit. Compiler lowers the program to
one Core `CircuitIR` containing H, measure, and the two classically conditioned
X instructions.

Runtime executes 128 independent local CPU trajectories per test. Each
trajectory owns its collapsed state and classical register until the final
sample is produced. The same lowered artifact is tested with the reference
trajectory strategy and the batched statevector strategy.

## Evidence

| Gate | Result |
| --- | --- |
| Measurement is represented as a boolean SSA result | pass |
| True and false regions lower to conditions on one classical bit | pass |
| Core `CircuitIR` JSON round trip preserves dynamic metadata | pass |
| Both measurement outcomes are observed | pass |
| Wire 0 final sample equals the measured bit | pass |
| True-branch target equals the measured bit | pass |
| False-branch target equals the complement | pass |
| Reference and batched trajectory strategies satisfy the same oracle | pass |
| Equal integer seeds reproduce measurements and samples | pass |
| Zero shots, read-before-measurement, and conditional measurement fail closed | pass |
| Trainable tensors fail under the stochastic-gradient policy | pass |

## Claim boundary

This evidence covers a local semantic vertical slice only. It does not cover
runtime tensor inputs, nested or loop-carried measurement, reset, arbitrary
classical expressions, durable sessions, gradients, PyTorch graph capture,
accelerators, distributed execution, remote hardware, capacity, or
performance.
