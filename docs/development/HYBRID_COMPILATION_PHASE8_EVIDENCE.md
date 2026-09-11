# Phase 8 parameterized dynamic-session evidence

Date: 2026-09-09

## Accepted profile

The source program accepts one non-trainable `float64` scalar angle and one
integer loop count. It specializes a bounded `range`, applies RX in each
iteration, measures wire 0, conditionally applies X to wire 1, and returns the
measured bit. The test circuit uses `complex128` statevector execution.

Compiler emits a reusable `CircuitIR` template with ordered Core `Parameter`
slots and a separate binding map. Runtime receives only the bound Core
artifact and executes the existing local batched dynamic trajectory path.

## Evidence

| Gate | Result |
| --- | --- |
| Scalar and index inputs receive shape/dtype validation | pass |
| One loop iteration emits one RX parameter slot | pass |
| Binding retains the original scalar tensor object | pass |
| Different scalar values preserve the template hash | pass |
| Different scalar values preserve the input-signature identity | pass |
| RX(0) produces only zero measurement and feedback bits | pass |
| RX(pi) produces only one measurement and feedback bits | pass |
| Bool input selects RX versus RY before measurement | pass |
| `float32`/`complex64` parameterized execution | pass |
| Configured unroll ceiling rejects excessive trip count | pass |
| Trainable runtime scalar fails before lowering | pass |
| Unbound parameterized template fails before execution | pass |
| Tensor input and measurement inside a loop fail closed | pass |

## Claim boundary

The parameter changes finite-shot outcome probabilities but has no supported
gradient. This evidence does not cover tensor inputs, loop-carried classical
values, measurement inside loops, dynamic loop bounds after measurement,
durable sessions, PyTorch graph capture, accelerators, distributed execution,
capacity, or performance.
