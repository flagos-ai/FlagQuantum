# Hamming-weight phasing with a phase gradient

The emitter binds the locked Qualtran v0.7.0
`HammingWeightPhasingViaPhaseGradient` graph to shared register
operations. It computes the Hamming weight, applies QvrPhaseGradient
to the unsigned integer count with coefficient `exponent / 2`,
then runs measured count erasure. Input, count and gradient words
use little-endian wire order. The caller supplies the gradient
state and clean count, junk and carry wires. The count, junk and
carry return to zero. The gradient state is available for reuse.
Classical measurement slots can be reused across sequential
components.

The tests compare the pinned outer graph, the QVR binding and
the scaled-add gate and feedback sequence from the shared QVR
reference. Full complex vectors include every input word, gradient
basis shifts and a coherent spectator. The small layouts exhaust
raw measurement histories. Larger layouts check complete sparse
trajectories with all-history cleanup certificates. The adjoint is
also checked against the negative of the original phase increments.

| Input bits | Exponent | Epsilon | Gradient bits | Clean work bits | Total wires | Literal T-family | Literal CX | Literal measurements | Source model Toffoli / And / measurements |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.25 | 0.25 | 3 | 3 | 7 | 8 | 19 | 2 | 1 / 0 / 0 |
| 2 | 0.125 | 0.5 | 3 | 5 | 10 | 12 | 35 | 3 | 1 / 1 / 1 |
| 3 | -0.375 | 0.25 | 4 | 6 | 13 | 24 | 69 | 6 | 4 / 1 / 1 |
| 4 | 0.125 | 0.25 | 4 | 9 | 17 | 24 | 82 | 6 | 2 / 3 / 3 |
| 8 | 0.625 | 0.25 | 4 | 14 | 26 | 44 | 156 | 11 | 4 / 7 / 7 |

Literal counts exclude caller-owned gradient preparation and
unpreparation. The recursive source qubit model succeeds for
the tested two-bit case and reports eight qubits. The source
one-bit graph cannot bind the empty junk during erasure. The
native emitter uses the single self-inverse CX from forward
counting. For the tested three-, four- and eight-bit cases, the
locked source adjoint classical path and recursive qubit estimate
fail on mixed QFxp binding. The source cost model column is
reported independently of these executable and qubit boundaries.

Run `python -m examples.algorithms.register_hamming_weight_gradient_phasing`
for the `circuit.append(op)` example.
