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

Complete `for_testing/random_select_and_prepare.py` now uses the native
`compiler/registers/lookup/ordered_pauli_select.py` owner through the existing
`SelectBlockEncoding` and `circuit.append(op)` entry. Random fixtures previously
used the unary SELECT tree. They now emit the original ordered, individually
controlled Pauli rows. Uniform ring and ordinary `SelectPauliLCU` retain their
existing owners.

The first selection wire is the least significant address bit. An optional
outer control precedes selection controls. The emitter owns no clean workspace
or classical slots. I/X/Y/Z rows, omitted addresses, unit-modulus coefficients,
zero/one control polarities and row-reversed adjoints preserve source order.
The negative real scalar's signed zero is canonicalized to the original Pauli
inverse representation before choosing the principal phase. Existing recursive
single-qubit synthesis and controlled-H phase helpers own local matrix and
CCZ/diagonal decomposition. Synthesis factors only local 2x2 matrices.

The exact primitive comparison boundary is the unchanged source's Cirq 1.7.0
SDK decomposition, imported through the existing strict adapter. Tests compare
every operation, operand, parameter, local matrix and metadata at that boundary.
Complete complex matrices and native IR round trips are verified through three
selection bits, with both outer-control polarities. Actual random fixture
append paths use the same emitter. Signed/complex row phases and incomplete
rows have independent source comparisons. Seeded PREPARE matrices, squared
coefficients, Pauli draws and subsequent RNG state remain equal. PREPARE has one
original MatrixGate child and one native opaque matrix leaf in each direction.
Nonreal first-column inputs fail in both implementations.

Existing full forward/adjoint walk checks are reused on coherent inputs at one
and two selection bits, projecting every measurement history and checking the
entire unnormalized complex output including zero work. SELECT introduces no
measurement. The two-bit walk retains one measurement in its existing native
reflection component. Full walk tensor/channel equality is the comparison for
that composition. A raw whole-walk SDK snapshot has a different scalar-carrier
and matrix-leaf representation, and is not asserted to be the same primitive
stream. This distinction also applies to primitive counts below.

### Actual primitive counts and independent models

The following values use seed 42 and two target qubits. Local `any` instructions
are 2x2 source matrix leaves and are not expanded Clifford+T counts. `ccx` remains
an admitted primitive, with no implicit T expansion. `matrix_fixture` is the
opaque PREPARE matrix.

| Operation | Measured native IR primitives | Independent source model |
| --- | --- | --- |
| One-bit SELECT | 2 X, 1 S, 3 local matrices, 3 CX, 1 CZ; no measurement or work | 5 Cliffords; logical peak 3 |
| Two-bit SELECT | 14 X, 8 Tdg, 7 T, 40 CX, 39 local matrices, 1 CCX; no measurement or work | 4 AND, 10 Cliffords, 4 measurements; logical peak 5 |
| One-bit walk | 6 X, 1 S, 3 local matrices, 3 CX, 2 CZ, 2 PREPARE matrices, 1 phase; 4 used wires | Logical peak 4; QEC query refuses matrix PREPARE |
| Two-bit walk | 24 X, 10 Tdg, 9 T, 46 CX, 39 local matrices, 1 CCX, 2 PREPARE matrices, 3 H, 1 S, 2 CZ, 1 measurement, 1 reset, 1 phase; 6 used wires | Logical peak 6; QEC query refuses matrix PREPARE |

The source logical SELECT model follows its composite controlled-via-AND graph.
Its SDK front end follows controlled dense-Pauli synthesis. Those are different
source protocols, so the logical model is retained independently from the
measured SDK-aligned stream. Each actual SELECT primitive count equals the
corresponding unchanged source SDK count at the specified leaf boundary.

### Retained source and dependency boundaries

For seed 42 at two selection bits, the largest difference between the original
projected SELECT and its reported Hamiltonian remains 0.382220521869599. The
one-bit case agrees to numerical precision. The analytical Hamiltonian and
actual operator are compared independently. The source discrepancy is reproduced directly.

Original PREPARE foreign libraries at depths 0/1/2 preserve exact serialized
bytes through the native library reader/writer. Original reconstruction refuses
its unregistered `TestPrepareOracle` class. Original SELECT serialization fails
on its DensePauliString object array with pickle disabled. Both original
failures remain explicit. The specialized one-bit source control callback is
executed in a real builder, and the corresponding native controlled append is
compared to the original SDK stream.

A separate Cirq 1.6.1 probe passes 59 cases and fails five. Two failures expose
the older SDK's generic two-control Y synthesis, where 1.7 supplies CCY. Three
uniform-ring failures come from an existing reference helper using `cirq.CY`,
which 1.6.1 does not expose. Exact primitive equality is certified for the
Cirq 1.7.0 profile used above. Cross-version SDK primitive drift remains explicit.

Whole-file acceptance retains these original source limitations and
version-specific decomposition boundaries. It closes the random SELECT
execution-path gap under the declared reference profile. Source blob verification and the Core API guard pass.
