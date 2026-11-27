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


## Faithful complete-source acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes the historical open
status above for Q-0703. Complete source review covers the coefficient converter,
constructor validation and defaults, all signature/width/count properties,
RotationTree probability accumulation, zero subtrees, amplitude and phase ROM
rounding, all PRGA properties, both stage helpers, composite and call graphs,
three examples, DocSpec and PRGA representation. Existing native compile-time
angle-word construction, typed preparation stages, QROM and phase-add owners
supply this closure; no production algorithm is added by this acceptance entry.

Logical prepare controls, target and the supplied gradient are MSB-first THRU
ports. Only angle words, carry/QROM scratch and phase flag are private clean
work. Exact generic Adjoint reverses explicit child pairs independently of the
source uncompute table choice. The whole operator includes arbitrary logical
gradient columns; gradient-eigenstate kickback and entangled spectators are
additional checks, not replacements for those operator checks.

The original three factories and every concrete PRGA child retain ordered
recursive calls, operand mappings and exact depth-zero/depth-one definitions.
Existing controls zero/one/two, prepare/uncompute and exact inverse evidence is
reused in this batch. Eight-bit words `(8, 51, 153, 231)` retain source calls,
phase, feedback and all-history erasure certificates; complete private cleanup
and full small operators remain separately checked. The eight-bit kickback test
samples two numerical histories in addition to its all-history certificate;
it is not an exhaustive enumeration of a large dense state simulation.

Known source singleton, HasLength/Shaped codec and unresolved physical-width
failures remain explicit. The original symbolic analytical approximation uses
two full-shape PRGA calls in place of concrete prefix-size stages; it is not a
claim of exact symbolic-to-concrete gate-count identity. AddIntoPhaseGrad is an
atomic source leaf. Native measured integer arithmetic below it and the six Rx
leaves remain visible in the primitive inventory, without claiming a missing
upstream primitive recipe or a fully synthesized Clifford+T stream.

The existing original source test and notebook evidence above is reused without
changing source files or dropping their skips/expected failures. The independent
constructor audit and resource receipts remain authoritative for the four exact
forward/inverse rows, distinct from source uncompute inventories. No new API,
reference dependency, checker waiver or progress child is introduced.


This documentation-only closure reruns 82 dense constructor, complete source,
complex execution, symbolic resource and exact-adjoint cases plus 17 independent
Cirq 1.6 constructor cases, all passing. The immediately preceding accepted
batch's 32 local integration cases, strict production checks and real Kickmix
Simulator verification are reused because no Python implementation changed.
All 46 reviewed source/dependency/asset blobs remain identical to the lock.
Algorithms scope, language, links, architecture, IR, protected API baseline and
whitespace pass. The global dependency-policy findings remain separately open.
