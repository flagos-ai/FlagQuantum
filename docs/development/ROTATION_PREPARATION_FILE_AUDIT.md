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

## Complete source definition map

The immutable source contains three classes and one converter; all its members
have an existing native owner. This batch adds execution evidence, rather than
another state-preparation implementation.

| Complete source definition | Native implementation or acceptance |
| --- | --- |
| `_to_tuple_or_has_length` and constructor validation | Concrete tuples in `RotationPreparation`; unresolved shape declarations in `symbolic_rotation_resource_model`. Source normalization, power-of-two size and precision greater than one are preserved. |
| `state_bitsize`, `n_coeff`, `signature` | `source_rotation_preparation_signature` and catalogue typed ports, including omitted zero-width control ports. |
| `rotation_tree`, `prga_prepare_amplitude`, `prga_prepare_phases` | `rotation_tables` and exact prefix-width `PhaseGradientLookupCall` declarations. |
| Composite body, amplitude stages, phase stages | `RotationPreparation.emit`, phase-flag ownership and ordered compiler lookups. |
| State-preparation call graph | `source_rotation_preparation_calls` and independent resource formulas. |
| All three factories and DOC example list | Existing catalogue and notebooks; both symbolic factories now have actual resolved external execution evidence. |
| PRGA signature, QROM, addition, body, call graph, representation | Compiler `phase_gradient_lookup`; representation is presented through catalogue call labels rather than a foreign class name in the native API. |
| RotationTree constructor, probability sums, `_p0`, `_angle_0`, amplitude/phase quantization, ROM getter | `rotation_tables`; original numeric grids, zero subtrees and quantized offset tests are reused. |

## Resolved source objects and owned execution

`test_rotation_source_binding.py` resolves the constructor parameters of both
original symbolic factories with `attrs.evolve`. The phase-only factory retains
its original eight coefficients; the shape factory resolves to the original
controlled four-coefficient test vector. The twelve combinations cover zero,
one and two controls, and both values of the source `uncompute` flag. Unresolved
objects still reject execution. They are never silently converted to integers.

The actual external object is captured in both directions. Only its native
`CircuitIR` snapshots pass through JSON and are rebound through the existing
`BlackBoxPrepare` emitter. Consumers continue to use `circuit.append(op)`.
Every logical basis column, including arbitrary gradient words, agrees with the
standalone native operation in full complex amplitude and leaves every private
wire zero. The captured actual adjoint restores every column. Complete operator
unitarity and negative-exponent gradient factorization are checked separately.
The shared erasure certificate proves all raw measurement histories represent
the same channel; complete-column numerical cleanup discharges the separate
final-zero obligation. No measured gate stream is reversed to create an inverse.

`uncompute=True` is a source-defined quantized recipe. It is compared to the
matching source object; the actual generic adjoint is captured separately. The
two notions are not assumed identical at every rounding boundary.

For the original eight-coefficient factory at two phase bits, the captured and
standalone native instruction histograms match exactly:

| Controls | Source recipe | Instructions | Explicit T/T-dagger | CX | Rx | Measures | Captured wires | Native wires | Source And / Clifford / measures / peak |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | Prepare | 304 | 56 | 122 | 6 | 14 | 11 | 11 | 10 / 44 / 10 / 11 |
| 0 | Uncompute | 392 | 72 | 158 | 6 | 18 | 11 | 11 | 14 / 60 / 14 / 11 |
| 1 | Prepare | 440 | 88 | 170 | 6 | 22 | 13 | 13 | 18 / 52 / 18 / 13 |
| 1 | Uncompute | 528 | 104 | 206 | 6 | 26 | 13 | 13 | 22 / 68 / 22 / 13 |
| 2 | Prepare | 576 | 120 | 218 | 6 | 30 | 17 | 15 | 26 / 60 / 26 / 15 |
| 2 | Uncompute | 664 | 136 | 254 | 6 | 34 | 17 | 15 | 30 / 76 / 30 / 15 |

Capture protects complete exported-frame lifetimes and reserves two additional
private names in the two-control case. Those names are clean, and do not add
gates. The source model is an independent declaration: the source abstract
`AddIntoPhaseGrad` supplies no primitive recipe, so capture substitutes the
accepted native measured add. Equal captured/native histograms do not turn that
substitution into an upstream physical gate decomposition. Existing stage tests
retain the gate/operand/feedback comparisons at this declared lowering boundary.
The six Rx leaves are not reported as synthesized Clifford+T circuits.

## Remaining boundaries

Q-0703 no longer lacks executable symbolic binding. It remains open for the
unchanged source singleton missing-target-port failure and the abstract source
`AddIntoPhaseGrad` leaf without a primitive decomposition. Native PRGA uses the
accepted shared addition at that explicit boundary. Original symbolic
decomposition and serialization skips and expected failures remain recorded
against unchanged source. No source defect is repaired in the reference.

## Current batch validation

The twelve complete bound-operator cases passed in 67.64 seconds, including the
persisted histogram assertions. The concentrated rotation, PRGA, phase-addition,
sparse-binding and erasure regression passed 211 cases in 78.54 seconds. Strict
mypy on the new test, Black, Ruff, Algorithms scope, architecture, multilevel IR
and protected API checks passed. A final smoke case after the explicit pair-of-
snapshots typing correction passed. Existing notebook and CPU integration
acceptance is reused: this batch changes only tests and this audit, with no
production behavior or public contract changes. No new parent or child progress
item is counted.
