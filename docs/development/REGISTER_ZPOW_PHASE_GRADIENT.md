# Constant Z powers through phase kickback

The shared emitter ports the locked Qualtran v0.7.0
ZPowConstViaPhaseGradient graph. The concrete angle is the unsigned
n-bit fractional encoding of exponent/2. It truncates towards zero
and wraps at the register modulus. For four bits, exponents 0.375,
0.37, 2 and 3 load 3, 2, 0 and 8 respectively. A negative source
exponent is rejected by the pinned unsigned format. The exact
adjoint instead keeps the same loaded word and reverses the addition.

The input gradient is caller-owned. With the source's negative
gradient exponent, the target acquires phase
`exp(2π i k / 2^n)` on its one branch, where k is the quantized word.
The original controlled XOR loads k into n clean wires, and the
second controlled XOR returns them to zero. The shared measured
adder consumes n-1 clean carry wires and matching classical slots.
The source AddIntoPhaseGrad leaf has a classical action and a
resource model. The native leaf uses the pinned executable Add and
Add adjoint under the same input and gradient binding.

Tests compare the original forward and inverse allocation graph,
load values and wires, and full classical map. They also compare
the emitted adder gate stream against the original Add components, including measurement feedback. The tested basis
support has all-history And certificates. All raw histories are
forced through four gradient bits, and eight-bit runs check full
complex trajectories. The gradient eigenstate and work are restored.

| Gradient, exponent | Physical wires | Literal T-family | Literal CX | Literal measurements | Source model Toffoli / qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, 0.5 | 3 | 0 | 1 | 0 | -1 / 4 |
| 4, 0.375 | 12 | 12 | 37 | 3 | 2 / 9 |
| 4, 0.37 | 12 | 12 | 35 | 3 | 2 / 9 |
| 8, 1.99 | 24 | 28 | 95 | 7 | 6 / 17 |

The original leaf model has zero measurement cost and reports a
negative Toffoli count at one bit. The literal counts come from the
native gate stream and include the measured carry implementation.
