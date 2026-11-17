# Dense rotation preparation source audit

The reference is Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`. The catalogue notebook, source,
test file, and controlled tutorial blobs are
`ee713c6e8b5f077f72c2aa056fdf91deddf42d47`,
`2466a11c3303e23c066602dce535750e650f83bf`,
`6b7a770579188189c032c55a9c6bf5636314df13`, and
`834025f31ea5b74b47a4113712fd722fd71a03af`.

| Source responsibility | FlagQuantum owner |
| --- | --- |
| Coefficients, target width and direction | `RotationPreparation` binds a power-of-two normalized coefficient vector and a concrete phase width. Uncompute reverses stage order and uses conjugated rotation words. |
| Signature and work | Controls, target and gradient are big-endian through words in source order. The gradient belongs to the caller. The angle register, QROM work, addition work and phase flag are temporary. |
| `RotationTree` | `rotation_tables` owns angle quantization, zero subtrees and signed forward/inverse words. |
| `PRGAViaPhaseGradient` | The compiler lookup owns QROM, gradient addition and adjoint QROM, including cleanup and feedback. |
| Resource declarations | Concrete calls retain ordered prefix address widths and distinct ROM words. Symbolic source factories use two calls to the same full-shape declaration. |
| Symbolic width fallback | The shared compiler formula retains `Max(0, phase_width)` for the angle allocation alongside the live gradient and address. This matches both original factories with unconstrained phase-width symbols. |

The [catalogue](../../examples/notebooks/state_preparation_via_rotation.ipynb)
displays all three original signatures, source-sized concrete calls and both
symbolic resource declarations.
The [controlled tutorial](../../examples/notebooks/state_preparation_via_rotation_tutorial.ipynb)
preserves the seeded three-bit vector and four-bit precision, then uses
`circuit.append(op)` for preparation, one coherent control, and uncompute.
Its runtime checks the inactive and active complex amplitudes, clears the
returned gradient with the inverse gradient circuit, and verifies zero scratch.

## Verification

The native tests load the unchanged numeric grids for exact and approximate
states, adjoints, inactive control, a single coherent control and an entangled
two-wire control from the pinned source test. They compare gate and feedback
operands against source decomposition, evaluate complete complex output with
gradient and zero scratch, and check native resources against source models.
The original three-bit PRGA fixture independently checks every address in
control superposition and phase kickback for four words. Larger fixtures sample
three histories. Existing bounded cases retain exhaustive small-history checks.
The original suite including both notebooks passed 21 cases with six skips and
three expected failures.

| Original fixture direction | Emitted T/T† | Emitted CX | Emitted Rx | Emitted measures | Reserved wires | Source And | Source Clifford | Source measures | Source logical wires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prepare | 56 | 122 | 6 | 14 | 11 | 10 | 44 | 10 | 11 |
| Uncompute | 72 | 158 | 6 | 18 | 11 | 14 | 60 | 14 | 11 |

The model counts the six special-angle Rx calls as Cliffords. The emitted
stream exposes those rotations as Rx. Each of the four gradient additions
contributes a measured carry to the emitted stream. The source model has no
Toffoli for this two-bit addition.

## Typed declarations and call identities

The state-preparation and PRGA signatures share the source's scalar
through-port rules. One-bit ports use the qubit type, zero-width ports
are omitted, and larger or unresolved widths retain opaque words.

The direct state-preparation graph retains both signed Rx angles and
their precision, two X calls, and each PRGA's exact width, controls
and ROM contents. With unknown coefficients or gradient width, the
amplitude and phase calls share one shape-only PRGA identity with
multiplicity two. Each PRGA declares one QROM identity with multiplicity
two and one fixed-point phase addition. The QROM load and erase also
share the existing table definition in executable binding and the
concrete resource model. Concrete, inverse, shape-only, symbolic-phase
and singleton declarations were compared to the locked source.

## Remaining boundaries

Q-0703 retains executable symbolic binding, the original missing singleton
target port, and the abstract source `AddIntoPhaseGrad` leaf without a gate
decomposition. The native PRGA translation uses the accepted shared add
lowering at that boundary. Original symbolic decomposition and serialization
skips and expected failures remain recorded against the unchanged source.
