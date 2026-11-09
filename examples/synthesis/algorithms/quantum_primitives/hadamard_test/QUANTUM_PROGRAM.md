# Hadamard-test ownership and readout

Complete four-bit-target Hadamard-test quantum program. Raw five-qubit joint states and ancilla readout checked against original and independent Fourier Circuit. Complete Notebook workflow remains pending.

Native psi is an explicit Output. Read only expectation_value using Program.probabilities to reproduce source drop readout. The entangled target remains in the charged five-qubit state: it is not reset, clean workspace, freed or traced into a pure statevector. The byte-identical tutorial reuses the algorithm program. Qmod within-H/apply-control-QFT is expressed as H/control(qft)/H in the root native qfunc; no source importer or source file is used in user examples.

The test compares absolute complex128 amplitudes without phase alignment,
the source's full joint state, logical bit order and ancilla marginals. The
target has nonzero amplitudes outside zero and must not be advertised as clean.
Compiler-inserted workspace is zero for the scalar control; all five qubits
remain charged to qubit and memory budgets. The reusable Fourier-control tests
also cover asymmetric coherent states, every small column, comparison workspace
cleanup, overlap, Const, initialization and budget refusals.

[README](README.md) contains the executable native and independent Circuit
commands. Original-source execution stays in test fixtures. Release
authorization remains false.
