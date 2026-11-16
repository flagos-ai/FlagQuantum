# Cirq testing helper source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The source helper and test blobs
are `203207f7591c99f0e66dfce9f90980fc996f37ac` and
`cc35f145ea9be79e1108ae0e4c76330f544a72f3`.

The optional Ecosystem helper binds a Cirq gate to an owned native signature.
It reuses the same register naming and concrete quantum shape validation as
the register exporter. The bound operation includes left, right and through
ports in signature order. The full Cirq decomposition reserves those ports
with an isolated manager and appends sorted working wires to the simulator
order. Each fixture creates a fresh default manager. The SDK objects stay
inside Ecosystem.

The basis helper executes Cirq in complex128 in the supplied wire order and
matches the source Dirac notation and failure behavior. An extra fixture
checks a reversed wire order and detects a changed output phase.
The legacy assertion checks the declared cost of an SDK object against its
one-step decomposition. It rejects absent annotations, absent
decompositions and mismatches, while allowing a valid empty decomposition.

## Verification

The three original tests pass against the unchanged source. The original
negative-control MultiAnd checks each named port and the exact ordered
decomposition, then compares all complex amplitudes for a coherent input.
The separate work-wire fixture reserves a colliding input name, verifies two
CNOTs at the first decomposition level and compares the SDK's full
decomposition. The roundtrip restores the complete coherent state and its
zero work wire. There are no measurements in this roundtrip.

The optional consistency helper compares legacy Clifford, T and rotation
estimates. The fixture's measured decomposition stream and the analytical
legacy costs are independent.
