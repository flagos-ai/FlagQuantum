# Programmable rotation arrays

The shared emitter follows the pinned Qualtran v0.7.0
`ProgrammableRotationGateArrayBase` generator. It concatenates
the MSB-first angle columns, groups them by the load width and
updates the same load word with each XOR delta. The final lookup
erases that word. Each occupied column invokes the caller's
controlled base gate at exponent `2**(-1-bit_power)`. This binding
preserves the base gate's phase convention and lets the caller
select its native implementation.

The interleaver runs at the source's angle-index transition. In
particular, a zero-width middle angle contributes no columns and
may suppress the following interleaver. An all-zero array emits
only the empty final lookup. The source channel is bounded to
valid selection indices.

Tests read the original Cirq generator, expand its QROM leaves
with the source gate reader, and compare gate order, operands,
measurements, feedback and exact controlled matrices for X, Z and
two-target CZ bases. They project every raw measurement history
for the tested layouts, including a coherent address/target
superposition and spectator. The load and QROM work return to zero.
A native controlled-phase instance also executes through
`circuit.append(op)` with the trajectory runtime.

| Rows / addresses / load bits | QROM calls | Native T-family / CX / measurements / wires | Controlled rotations / interleavers | Source QROM And model |
| --- | ---: | --- | --- | ---: |
| 2 / 3 / 2, X | 4 | 8 / 22 / 2 / 6 | 5 / 1 | 2 |
| 3 / 4 / 3, Z | 3 | 20 / 49 / 5 / 7 | 6 / 2 | 5 |
| 2 / 3 / 2, CZ | 4 | 12 / 33 / 3 / 7 | 5 / 1 | 3 |

Literal T-family and CX counts are for the lookup stream. The
controlled rotation and interleaver matrices in the oracle are
unsynthesized. The source model column sums the individual
Qualtran QROM estimates.
