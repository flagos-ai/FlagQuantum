# Periodic Ising Trotter files

This batch covers Q-0445–Q-0448 in Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The four pinned files match
their Git blobs. The original unitary tests returned 11 passes and
two skips. The unchanged 18-cell notebook ran in an independent kernel.

The package initializer documents the periodic transverse-field Hamiltonian
and exports its X and ZZ terms. The native owner provides both terms.
The native notebook reproduces the two three-site fixtures at rotation
angle 0.04, including each graphical signature and direct call graph.
Its independent kernel rejects reference-library imports.

| Term | Locked direct model | Native primitive stream on three sites |
| --- | --- | --- |
| X | Three Rx | Three Rx in register order |
| ZZ | Six CNOT and three Rz | Three ordered CX–Rz–CX blocks, including the last-to-first edge |

The typed through signature uses a QBit for one site and an opaque
word for larger sizes. The direct model retains CNOT and rotation
multiplicities and uses the original live logical width. Native
instructions apply exact rotations. The source synthesis descriptor
retains the angle and precision convention: X uses the default Rx
precision even when a different tolerance is set on the source term;
ZZ forwards its term tolerance. This metadata does not request physical
rotation synthesis.

The tests compare the exact forward and adjoint instruction streams
on permuted wires and full complex matrices on contiguous wires. They
include both notebook fixtures, two-site ZZ with both directed edges,
and representative generic, T-like and Clifford angles. There are no
measurements or work wires. The three-site ZZ fixture emits six CX
and three Rz; the model reports three rotations and a three-qubit peak.

The locked one-site ZZ decomposition passes the same live operand as
control and target and fails in the source builder. The native binder
rejects it at construction. Two distinct sites remain executable and
keep both source directed edges.
