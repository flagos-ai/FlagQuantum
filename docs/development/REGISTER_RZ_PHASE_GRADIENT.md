# Variable Z rotations through phase kickback

The shared emitter ports the locked Qualtran v0.7.0 RzViaPhaseGradient
arithmetic graph. The four Cast nodes reinterpret wire types without
gates. The original ControlledAddOrSubtract then uses unsigned
big-endian words, subtracting on target zero and adding on target
one. Its adjoint swaps the two branches. The angle word can be
narrower than the gradient and is zero-extended as an integer.
The phase scale uses the gradient width. For angle integer a and
an n-bit negative-exponent gradient, the resulting target operation
is `Rz(4πa / 2^n)`. The gradient is caller-owned and unchanged.

The emitter reuses the pinned measured complement/Add/complement
decomposition, with n-1 clean work wires and matching classical slots.
The source graph reader verifies both directions, including cast
aliases, exact control polarity, Add dependencies and feedback.
The numerical checks compare both complete complex branches with
their signed phases. They cover all raw histories through three
gradient bits, all-history And certificates over the tested input
support and full sparse trajectories for larger cases.

| Angle, gradient | Physical wires | Literal T-family | Literal CX | Literal measurements | Source model And / measurements / qubits |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1, 1 | 3 | 0 | 3 | 0 | 0 / 0 / 3 |
| 2, 3 | 8 | 8 | 26 | 2 | 2 / 2 / 8 |
| 4, 5 | 14 | 16 | 54 | 4 | 4 / 4 / 14 |
| 8, 8 | 24 | 28 | 97 | 7 | 7 / 7 / 24 |

The source model aggregates Clifford operations. The CX column is a
literal count. The one-bit source aggregate also omits four
Clifford operations from the decomposed stream.

Run `python -m examples.algorithms.register_phase_gradient_rotations`
to apply both constant Z and variable Rz through `circuit.append(op)`
with a shared gradient, and verify complete gradient and work cleanup.
