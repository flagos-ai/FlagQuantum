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

## Complete source constructor binding

Both frozen identifiers now resolve through
`compiler/formats/legacy_ising_factories.py`. Their common typed composition
owner is `registers/primitives/ising_component.py`: one MSB-first system word,
explicit Split, ordered per-site calls, and Join. Rotation and CNOT definitions
come from the existing primitive factories. No angle kernel, work allocator,
backend choice or second resource estimator is introduced. The existing
Algorithms operations and their stricter numerical validators are unchanged.

The constructor retains source `angle` and `eps` in the owned library wire.
For X, each Rx still receives its source default precision, not the root eps.
For ZZ, Rz receives the root eps. The shallow named constructor and stored
graphs at depths one and two have identical ordered calls and named operands.
The two-site ring still has two directed edges. Symbolic angles lower through
the existing parameter expressions and bind to ordinary Circuit parameters.

`test_ising_source_constructors.py` verifies both original examples and doc
specifications, all three wire depths, complete complex operators, exact
forward/adjoint streams, permuted wire binding, a complex spectator, unused
clean storage, source precision, and construction with optional SDK imports
blocked. Zero/symbolic site counts remain logical declarations; one-site ZZ
preserves its source failure and cannot execute as identity. With no
measurement, the only history is empty and no work is allocated or leaked.

The concentrated constructor/Ising/Trotter/rotation/fixed-gate scope passes
240 cases in each Cirq lane. Local integration passes 145 cases. The unchanged
complete original tests pass 11 cases with their two source-owned count skips.
The frozen implementation and test blobs are respectively
`c9f9b14ce57038fc5e88ffff1d54e4f66a5376fa` and
`542719fdd70750308bcc73dfd537a51b661dacd2`; native reference tests also require
Qualtran 0.7.0 and a real Kickmix 0.1 Simulator.

| Three-site term at angle 0.04 | Executed named primitive counts | Original model |
| --- | --- | --- |
| X | 3 Rx; no workspace or measurement | 3 rotations, peak 3 |
| ZZ | 6 CX, 3 Rz; no workspace or measurement | 6 Clifford, 3 rotations, peak 3 |

Continuous rotations are not a synthesized T inventory. Logical counts and
unused caller storage are separate from executed gates. Original unsupported
dimensions and nonfinite/native parameter admission remain outside physical
execution. Earlier full-file acceptance is reused, not counted twice; this
delivery closes its missing named-constructor path under the still-open
Q-0876 catalogue. The [303-key audit](CONSTRUCTOR_CATALOGUE_AUDIT.md) separates
registration from acceptance and retains unresolved catalogue evidence.
