# Diagonal-gate tutorial audit

Q-0233 is the complete 20-cell Z, S and CZ tutorial from Qualtran v0.7.0
commit `bcb17bb25360ed68ce3bc084645db3e929799389`. Its Git blob
`59ab7dd205aa110d7c35178398b93d02be34cca1` and its four direct source/test
blobs for Z-basis and S match the frozen inventory.

## Direct dependencies and source coverage

| Source topic | Native owner and evidence |
|---|---|
| ZGate instance and graphical signature | Shared Pauli lowering emits atomic Z. The one-bit THRU q signature and complete complex operator match the source. It changes phase on basis one and fixes the basis label. |
| SGate instance, graphical signature and square identity | The native S and SDG opcodes implement diag(1, i) and diag(1, −i). The whole complex matrices match the source, and the native checks verify S²=Z and its exact inverse. |
| CZ instance and graphical signature | Shared controlled Pauli lowering emits atomic CZ in original q1/q2 order, with both ports THRU. Full complex operator and spectator amplitudes retain a minus sign only for the joint one input. |
| Three original visualization sections | Native signature tables preserve all original ports and instance groupings. The native teaching asset uses the existing `circuit.append(op)` path and prints source logical categories separately from emitted primitives. |

The existing native circuit API adds a coherent control to each gate.
Tests compare its entire complex operator to a Cirq controlled
reference with physical axis order accounted for. Native controlled
S and S adjoint use exact cphase angles +π/2 and −π/2. This path has no
scratch wires. The distinct measured control-predicate implementation
is covered by the existing Pauli tests, which check original
dependencies, corrections and zero work on every small branch.
The new tutorial does not replace either control path.

| Base execution | Actual emitted IR | Original source model |
|---|---|---|
| Z | one Z | one Clifford |
| S | one S | one Clifford |
| CZ | one CZ | one Clifford |
| S adjoint | one SDG | one Clifford |

Each base example has zero measurements and zero scratch. The controlled
matrix checks establish exact phase and do not assign physical T costs
to the controlled phase opcodes.

The original 20-cell notebook and native 24-cell notebook execute in
independent kernels. The native kernel rejects reference imports. The
native version preserves the original three signatures and adds two
executable exact-phase sections. The unchanged Z-basis and S companion
suites, with their other notebook tests excluded, report 40 passed,
18 skipped and 2 expected failures. Those outcomes include atomic and
undecomposed cases and two existing integer state/effect count modes.
The companion source files retain their separate closure status.

Q-0233 closes as a complete teaching asset.
