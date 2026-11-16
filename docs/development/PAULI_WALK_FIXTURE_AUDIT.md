# Pauli walk fixtures

Q-0502 and Q-0503 match their frozen Qualtran v0.7.0 blobs at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The native fixtures
live under tests and reuse the existing PREPARE, SELECT and walk owners.

## Uniform ring

The ordered X/Y/Z pair products, dense Hamiltonian and normalization
match at two, three and four sites. The selection word is big-endian.
The full SELECT primitive stream, physical operands and measurement
feedback agree. The forward and adjoint walk channels are compared
against the full source tensors on coherent inputs, including the
padded selection address at three sites. Every small measurement
history is projected; the longer three-site circuit checks all-zero,
all-one and alternating histories. The existing uniform PREPARE and
walk regressions cover their constituent measurements.

The three-site walk emits 48 T-family instructions, 12 measurements
and four Rz gates across ten wires. The distinct original resource
model declares 12 AND, 12 measurements, 84 Cliffords, four rotations
and a nine-qubit logical peak. The one-site XY product is non-Hermitian.
The wrapper with nonempty control values fails because there is no
declared control port. Both original refusals are covered.

## Randomized oracles

The seeded PREPARE matrix, squared coefficients, Pauli draws and next
RNG draw are identical to the source. The native analytical Hamiltonian
uses the declared term ordering. The executable SELECT instead binds
its address little-endian, as the source does. Uncontrolled, zero-control
and one-control operator matrices were compared. The full forward and
adjoint walks match the locked source, with all measurement histories
projected and clean work at one and two selection bits.

For seed 42 with two target qubits, at a two-bit selection width the
largest elementwise difference between the original projected SELECT
and the reported Hamiltonian is 0.382220521869599. The one-selection-bit
case agrees to numerical precision. These two source results are
compared independently.

The two-selection-bit native walk emits two opaque matrix instructions,
12 T-family instructions and three measurements across six wires.
The original logical width is six. The original QEC query cannot
decompose the matrix-backed PREPARE. The randomized source SELECT is
a sequence of controlled dense Paulis, while native emission uses the
existing unary SELECT tree. The complete operator is verified, while
Q-0503 remains open for the source Hamiltonian and decomposition gaps.

The concentrated native suite passed 98 tests. Strict typing,
formatting, team scope, architecture, IR and public API checks passed.
