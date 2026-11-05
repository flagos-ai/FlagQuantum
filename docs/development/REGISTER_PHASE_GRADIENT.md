# Phase-gradient unitaries and state preparation

The shared emitter maps a big-endian word to the pinned Qualtran
v0.7.0 sequence of Z powers. For exponent t and width n, a basis word
k acquires phase `exp(2π i t k / 2^n)`. The controlled path applies that
phase only when its positive control is set. It emits the original
CZPow decomposition for each wire. A four-T logical And computes onto
one clean caller-owned work wire, a phase acts on the work wire, and
a measurement with CZ feedback resets that wire. The same quantum
wire and classical slot are reused throughout. Negating t gives the
adjoint.

State preparation applies Hadamards followed by the same gradient
unitary. Its inverse applies the opposite phase then the Hadamards.
The existing FlagQuantum preparation default stays positive, as used
by the THC consumer. Passing exponent -1 matches the pinned source's
state default. Tests compare the source state tensors, the native
output amplitudes and restoration of the all-zero register. The
existing positive-exponent forward and inverse IR streams are checked
instruction for instruction.

The source oracle reads the original Soquet graph and the And leaf's
Cirq decomposition. It compares the Z operands, phase order, angles,
measurement feedback and clean work mapping. The source divides its
epsilon across the phase leaves. FlagQuantum emits the angles directly,
with no rotation synthesis in this emitter. Numerical tests include
the complete complex vectors for both controls, both exponent signs,
a coherent spectator, and all raw measurement histories through four
bits. The eight-bit example verifies three complete forced histories
in each direction.

| Gradient and exponent | Physical wires | Literal T-family | Literal CX | Literal phases | Literal measurements | Source model |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4-bit, +1 | 4 | 0 | 0 | 4 | 0 | 1 T, 2 Clifford, 1 rotation |
| 3-bit controlled, -0.75 | 5 | 12 | 18 | 3 | 3 | 3 And, 3 rotations, 3 measurements |
| 8-bit controlled, +0.625 | 10 | 32 | 48 | 8 | 8 | 8 And, 8 rotations, 8 measurements |

Literal T-family counts cover the emitted And gates. The source
model's classifications and any later rotation synthesis are separate.
The controlled source qubit estimate includes one clean work wire and
matches the physical layout.

Run `python -m examples.algorithms.register_phase_gradient` to apply
and reverse the controlled unitary through `circuit.append(op)`.

The pinned AddIntoPhaseGrad has a classical action and a Toffoli model.
Its original decomposition call raises DecomposeNotImplementedError.
Executable phase-register arithmetic therefore has a separate parity
scope.
