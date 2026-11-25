# Programmable rotation array source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `programmable_rotation_gate_array.ipynb` | `a08ae58ab5eb1ddb3ba6e5cd21a95defade9f864` | Q-0678 |
| `programmable_rotation_gate_array.py` | `7c6b33d09150c49574a9e5a4fb218a504254a1b4` | Q-0679 |
| `programmable_rotation_gate_array_test.py` | `f3486cab5190ecb4b482f8ff0af11b68c0149794` | Q-0680 |

Run `jupyter notebook examples/notebooks/programmable_rotation_gate_array.ipynb`.
The native notebook preserves the ten original cells and adds a body
compilation example through `circuit.append(op)`. Both source and native
notebooks ran in independent kernels. The native kernel rendered two SVGs
with Qualtran, Cirq and quimb imports forbidden.

## Exact source fixture

The tutorial uses angles `[1, 2, 3, 4]`, a two-bit angle word, and a
Z-power target. The five logical through wires hold the bounded selection,
angle word and target. One extra wire is borrowed sequentially by the
QROM and the measured controlled rotations. The emitted route checks the
scope of the borrowed wire and its measurement slot. The angle word and
scratch return to zero for each selection and measurement history.

The original generator uses XOR tables `(0,1,1,2)`, `(2,1,3,2)`,
and `(2,0,2,0)` and controlled powers `1/8`, `1/4`, and `1/2`.
The tests verify the ordered Cirq operations and matching Bloq children
without the structural Split/Join nodes. They expand the QROM leaves and
the sourced CZPow compute, phase and measurement-feedback sequence. The
direct path retains a controlled-phase instruction per CZ power.
Both routes exhaust their raw measurement histories on one coherent
superposition of all addresses and both target values.

| Resource view | T/T† or modeled T | CX | Uncontrolled phases | Controlled phases | Measurements | Wires |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Literal direct phase route | 24 | 52 | 0 | 3 | 6 | 6 reserved |
| Literal measured CZ route | 36 | 70 | 3 | 0 | 9 | 6 reserved |
| Source QEC categories | 1 | — | 1 general rotation | — | 9 | 6 logical |

The source model also declares nine And and 32 Clifford operations. Native
phase instructions carry exact angles before synthesis.

## Original test grid and evidence

The unchanged original suite yielded 49 passes and 18 failures. All 18
failed during Cirq's reduced-state calculation of the auxiliary register
with a NumPy eigensolver nonconvergence error. The native audit exercises
both original angle lists, every kappa from 1 to 11, and all three
constructors, including controlled X/Z interleavers with their own target.
It verifies the bound source gate stream, each QROM XOR map and every
local erasure invariant, followed by two sparse coherent trajectories
against independently assembled per-address target matrices. The sparse
comparison includes relative phases between addresses and zero load and
scratch. Small cases from the existing suite project all raw histories.

## Complete source definition map

| Complete original member | FlagQuantum ownership and evidence |
| --- | --- |
| Base constructor and angle/precision properties | Existing unsigned angle rows and load-word binding in `programmable_rotation_array_ir`; original constructor grid and unequal-row rejection are retained. |
| Cached `rotation_gate(exponent)` | Optional gate conversion resolves the original fractional gate, including its scalar phase, into native local matrix instructions; the compiler callback receives native payloads only. |
| Selection, load, rotation-target and complete signatures | Existing bounded selector, opaque load/target words and explicit unrelated-target layout. |
| Abstract interleaver and unrelated-target hooks | Native per-stage instruction callbacks and declared `interleaved_target`, including original custom CNOT/CZ subclass. |
| Complete base decomposition | Existing XOR delta batches, source-sized QROM calls, ordered controlled bit rotations, boundary interleavers and final unload. |
| Concrete constructor and default identities | Existing validation, identity-stage callbacks and original factory/parameter-grid comparisons. |
| Concrete interleaver and empty extra-target property | Native local instructions retain exact target scope; the generic matrix case and original independent-target subclass both execute. |
| Original factory and DOC list | Existing native catalogue/notebook and fixed source blobs. |

## Actual foreign gates and native replacement

`test_programmable_array_source_binding.py` covers one- and two-target-bit
external gates, load words of one and two bits, generic interleavers and the
original independent-target subclass. Seeded external unitaries include explicit
scalar phases, so comparisons cannot discard relative phases between selection
sectors. Only local controlled rotations/interleavers become matrix leaves;
QROM, XOR updates, measurement feedback and cleanup remain ordinary native
instructions. The source algorithm is never contracted into one matrix to emit
its implementation.

Both actual source directions are captured separately and JSON-restored as
owned native IR. The existing named consumer is invoked with `circuit.append`.
The independent native compiler callback uses only copied local matrix payloads.
Every logical basis column, including arbitrary initial load words, is compared
in complete complex amplitude. True source adjoint composition restores the
input. Raw-history identity is certified independently from complete-column
private-work cleanup. With zero initial load, separately assembled per-address
target operators prove angle-load restoration and exact output phases.

A real optional-boundary gap was found for pure Cirq gates that provide only
`_unitary_` and `__pow__`, without a `matrix` attribute or decomposition. The
feedback importer now admits that explicit local unitary protocol using its
existing shape, finite-value and unitarity validation and copied native
payloads. Scalar/controlled instances work at both target widths. Gates with a
provider decomposition keep that decomposition; register-component signatures
and their potentially whole-algorithm unitary protocols are excluded from this
fallback. A regression supplies an interleaver whose unitary hook raises if
called, and verifies its declared H decomposition and both directions execute.
Known primitive gates keep their normal lowering and counts.

For two three-entry angle rows `(1,2,3)` and `(3,0,1)`, with kappa two:

| External local gates | Actual instructions | Explicit T/T-dagger | Measures | Local matrix leaves | Actual wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-bit rotation and interleaver | 57 | 8 | 2 | 5 | 6 |
| Two-bit rotation and interleaver | 57 | 8 | 2 | 5 | 7 |
| One-bit rotation, independent-target CNOT interleaver | 57 | 8 | 2 | 4 | 7 |

The source-capture/native histograms agree. A matrix leaf is not counted as a
finished Clifford+T synthesis. For the explicit one-bit MatrixGate fixture, the
source independently reports 10 And, 24 Clifford, 17 rotations, 10 measurements
and peak six. For the two-bit MatrixGate fixture, source QEC aggregation raises
`DecomposeTypeError` on atomic controlled And, while its width query reports
eight. Those declarations describe another proposed decomposition depth and
are not the actual 57-instruction stream or its seven reserved wires. The
existing original Z-power resource model remains independently accepted.

## Remaining source boundaries

Q-0678 and Q-0680 retain their earlier accepted assets and tests. Q-0666 is now
closed by the subsequent rotations-package audit. Q-0679's actual general
local-gate import and native callback replacement gap is covered for explicit
matrices and pure unitary-only gate protocols at the owning boundary. It remains
open for general Clifford+T synthesis and complete arbitrary-gate source cost
queries. No synthesized T cost, arbitrary-size runtime bound, or fully universal
hardware lowering is claimed from these local matrix leaves.

The unchanged upstream 18 eigensolver/reduced-state failures remain in the prior
source-suite evidence. This batch does not alter reference gates or original
assertions, rerun that unchanged failure without cause, or add child progress
items in place of completing a source file.

## Phase-exact named local gates, 2026-11-26

The complete-file review identified a concrete gap in the matrix-leaf route:
generic single/two-target payloads executed correctly but had no named native
lowering retaining their absolute scalar phase. Existing target-basis Euler/KAK
synthesis intentionally drops that phase. That behavior cannot be reused
unchanged inside a controlled rotation, where the scalar becomes a relative
phase between control sectors.

The new Compiler-owned `optimization/local_unitary_synthesis.py` composes the
existing exact one-qubit emitter, full three-entangler KAK factors, scalar phase
carrier and controlled-Rz owner. It recovers the KAK scalar from only the supplied
4x4 local factors; it does not form or simulate an array operator. The full KAK
form avoids fidelity-based count selection discarding small nonzero interactions.
Controls lift the exact local factors, placing a formerly global scalar on the
control. Local result gates are Rz/Ry/phase/X/CX/CCX. There is no matrix atom,
scratch qubit, classical slot or measurement in this synthesis owner. The old
target-basis synthesis contracts and default behavior are unchanged.

The original array callback interface uses this owner without changing the XOR
traversal or QROM consumer. One/two-target external MatrixGate and pure-unitary
fixtures, both load widths and the independent-target subclass run through both
the matrix and named-gate callbacks. Complete dirty-load operators, exact source
inverse composition, every-history erasure certificates, relative address phases
and private-work zero agree. All prior literal call/operand/feedback evidence is
reused for the unchanged array/QROM layers; independent local synthesis is not
claimed to reproduce Cirq's primitive decomposition or its counts.

The SDK-free runnable example is
`python -m examples.algorithms.local_rotation_array`. It appends an array with
two three-entry angle rows, a phased two-target SWAP base and a phased H-tensor-H
interleaver. Both nonzero scalar phases are significant across selection and
control sectors. The example checks load/work zero for every local trajectory.
Pure local checks compare complete complex operators with/without controls,
scalar/degenerate inputs, reversed targets, SDK imports blocked and interactions
of magnitude 1e-7 and 1e-9. No phase-insensitive fidelity assertion substitutes
for those comparisons.

The example's actual potential inventory is 11 X, 6 H, 4 T, 4 Tdag, 240 CX,
2 S, 2 measures, 2 feedback CZ, 2 resets, 170 phases, 58 Ry, 12 CCX and 18 Rz;
there are zero matrix leaves. Its one classical slot is reused by QROM and
the sole quantum workspace is restored. These named continuous-angle gates are
not synthesized Clifford+T counts. The existing generic MatrixGate source model
and its controlled-And failure remain recorded separately above; no successful
new source estimate is inferred from the new primitive stream.

Q-0679 remains open for the original full acceptance requirements: arbitrary
Clifford+T synthesis/precision, larger local matrix decompositions and complete
arbitrary-gate source cost queries. Shallow source constructor metadata loss and
the documented source/SDK failure profiles are unchanged. This batch closes the
phase-exact named-gate dependency for the complete admitted one/two-target local
matrix family, not the entire parent file.

## Current batch validation

Eight focused cases passed in 2.83 seconds. The expanded source-import,
rotation, sparse/PREPARE/LCU and erasure regression passed 656 cases in 73.79
seconds. Strict mypy on the changed importer and new test, Black, Ruff,
Ecosystem/Algorithms scopes, architecture, multilevel IR and protected API checks
passed. Serial coverage with 56 cases passed in 8.46 seconds and measured 89%
for the complete feedback-import module; unrelated existing guards remain
uncovered, so this is not a 100% claim. Sixty CPU/representation integration
cases passed in 45.13 seconds, and the SDK-free native rotation-array notebook
passed in an independent kernel in 2.26 seconds. Existing unchanged-source
failure evidence is reused.


Batch regression: 296 concentrated cases pass in the Cirq 1.7 lane. The Cirq
1.6 compatible scope passes 244 cases. The other 52 cases belong to the existing
`test_controlled_unitary_synthesis.py` fixture, which pins the SHA256 of Cirq
1.7's controlled decomposition. Running that fixture in 1.6 correctly produced
52 setup errors on its source hash; the intake log is retained and its guard is
unchanged. Those 52 cases pass in their pinned 1.7 lane. This is not a native
operator failure or an optional-dependency skip. Local integration passes 145
cases; strict four files, formatting/lint, Compiler scope, language/links,
architecture, multi-level IR and protected API checks pass. The runnable example
also executes through both local strategies in the resource receipt.
