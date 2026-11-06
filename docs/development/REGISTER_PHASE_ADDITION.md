# Fixed-point phase-register addition

The shared emitter implements the concrete arithmetic action of the
pinned Qualtran v0.7.0 AddIntoPhaseGrad. Both registers are big-endian
unsigned fractions. For input width m, phase width n and right shift r,
the increment is `floor(x * 2^(n-m-r))`, with signed accumulation modulo
`2^n`. The optional coherent control can activate on either polarity.

The compiler keeps the most significant `min(m, n-r)` input bits when
`r < n`. It adds or subtracts that word from the corresponding phase
prefix, including the higher carry bits. Lower phase bits keep their
original values. When `r >= n`, the phase register is unchanged and no
work is required. The controlled route loads only retained input bits,
then uses the shared Add or Add adjoint. All work returns to zero with
caller-owned measurement slots. The source, control and truncated bits
remain coherent.

The pinned AddIntoPhaseGrad raises DecomposeNotImplementedError.
Its classical action and small Cirq matrices supply numerical
references. Native gate parity is against the original executable
Add/CAdd components and their exact adjoints, using the aligned
wire operands. Resource estimates for the undecomposed source leaf
and measurements from the native IR are tracked separately. The
source leaf formula can even report a negative Toffoli count at one
phase bit, where the uncontrolled native circuit uses one CX.

Tests verify both signs, both control polarities, width extension,
truncation, shifts, wraparound and the inverse. They compare complete
complex vectors with a coherent spectator and certify all And
histories on the tested input support. Small layouts execute every
raw history directly. Larger layouts also compare complete sparse
trajectories, including all eight-bit source words and selected phase
words at carry boundaries.

| Input, phase, shift, mode | Physical wires | Clean work | Literal T-family | Literal CX | Literal measurements | Source model Toffoli / qubits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1, 1, 0, add | 2 | 0 | 0 | 1 | 0 | -1 / 3 |
| 2, 5, 0, add | 8 | 1 | 4 | 9 | 1 | 3 / 7 |
| 8, 8, 2, add | 23 | 7 | 28 | 77 | 7 | 6 / 16 |
| 8, 8, 2, controlled add or subtract | 30 | 13 | 52 | 113 | 13 | 12 / 17 |

The pinned leaf's model includes no measurements. Its qubit counts
come from the signature and modeled callee fallback. The literal
counts include the explicit clean work used by the shared adders.

Run `python -m examples.algorithms.register_phase_addition` to prepare a
gradient, perform controlled shifted subtraction through
`circuit.append(op)`, and observe the expected kickback phase with the
gradient and work restored to zero.
