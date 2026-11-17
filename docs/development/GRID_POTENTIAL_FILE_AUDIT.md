# Grid potential file audit

This audit covers Q-0429 and Q-0430 in Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`. The unchanged original test file returned ten passes and three skips. Q-0430 closes with the original cases and the native comparisons. Q-0429 retains the execution gaps below.

The pair takes two opaque through registers, `system_i` and `system_j`, with three equal-width axis words each. Total potential takes one opaque `system` register shaped `(electrons, 3)`, using the same coordinate layout as grid kinetic energy. Tests check source port shapes, order, and row-major indices. The upper-triangular traversal reuses each electron's returned wires in the next incident pair and returns the final wires in their original positions.

At the pair level, the checks follow all three coordinate operands into the forward adders, the sum of squares and the adjoint adders. They also follow the cast and QROM selection, the four coefficient banks in order into the polynomial, the squared-distance and polynomial words into Newton, and its estimate through QVR into release. Every source allocation and release is accounted for. The resource model uses the source's greedy scheduling order and shorter coefficient-prefix table. Its two costing modes are compared per child and in aggregate.

The native dynamic path accepts `circuit.append(op)`, binds big-endian coordinate words and reclaims caller-provided workspace after each pair. It computes signed axis differences and clears the squared distance after applying the phase. Native regression covers all one- and two-bit squared-distance inputs and its measurement histories, and full complex pair-phase states in both directions with feedback, including the cubic option. The one-electron operator is the identity.

| Quantity | Native emitted gates | Locked estimate |
| --- | --- | --- |
| Two-electron baseline fixture | 3,104 T-family instructions, 776 measurements, 177 workspace wires | Peak 81 logical qubits |
| Two-electron cubic fixture | 4,136 T-family instructions, 1,034 measurements, 212 workspace wires | Model peak 81 logical qubits |

The original pair feeds the second axis straight into the out-of-place adder. At equal unit inputs that leaf produces two, while the native difference is zero. The locked comments describe a sign adjustment but the decomposition does not implement it. The locked polynomial and Newton specify costs without primitive decompositions, QVR leaves the angle unspecified, and the original `Free` calls provide no matching native cleanup stream. Arbitrary user-supplied pair coefficient data remains outside the native physical binding.

The shared chemistry drawer preserves pair `U_{label}(dt)_ij` and total
`U_{label}(dt)` labels, with default, custom `Coulomb`, and empty labels.
The whole layouts include ordered `system_i`, `system_j`, and shaped `system`
axis TextBoxes. Source geometry comparisons at three grid sizes, followed by
JSON round trips and rendering, cover these presentation bindings. Existing
physical execution and resource evidence above is reused unchanged.

The grid package initializer Q-0424 exports kinetic, pair potential and total
potential. The source identities and all three imports are checked against
the defining source modules. Their native owners are GridKineticEnergy,
GridPairPotential and GridPotentialEnergy.

GridPairPotential binds the two three-axis ports through the existing
two-electron total-potential core. The coordinate words are unsigned and
big-endian. The caller supplies the phase angle and clean workspace.
Independent numerical tests compare every complex amplitude for all
64 one-bit coordinate inputs, including coincident points, for the C1 seed
and cubic option in both phase directions. The standalone C1 instruction
stream equals the total-potential stream for the same two points. The
one-update fixture emits 3,104 T/T† and 776 measurements, borrowing
177 work wires and 13 feedback slots. The output support certifies work
cleanup.

Q-0424 closes the package's three export mappings. Q-0429 retains its
source addition, angle, arbitrary coefficient and release boundaries.
