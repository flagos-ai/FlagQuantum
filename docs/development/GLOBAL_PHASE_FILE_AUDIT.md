# Global phase source family

The reference is Qualtran v0.7.0 at `bcb17bb25360ed68ce3bc084645db3e929799389`.
The unchanged files retain their source attribution.

| Inventory | Frozen Git blob | Disposition |
|---|---|---|
| Q-0236 `global_phase.ipynb` | `949bbece8c4251de0fb4f544a455512cb5a42b45` | Closed by the native `register_global_phase.ipynb`. |
| Q-0237 `global_phase.py` | `08e0273e6e62562dd50f6af0fb2f30b04bb9033f` | Concrete execution, whole-source scalar capture and independent models verified. Ordinary zero-wire static conversion and unbound physical scalar execution remain open. |
| Q-0238 `global_phase_test.py` | `561a9510eb8e3f18507a7558e19d0c80b04ad95f` | Scalar, controlled and resource assertions mapped. Native equivalents for the original external round trips and generic factory checks remain open. |

## Signatures, tensor and execution

The source factory selects exponent 0.5. The logical signature is empty and
the source tensor has rank zero with value `exp(i * pi * exponent)`. The source
declares this scalar atomic. The concrete native lowering borrows one existing
wire in an arbitrary state and returns it with its original logical contents.
It multiplies the whole state by the scalar. The adjoint negates the exponent.
`global_phase_exponent` matches the source principal argument in units of pi for
concrete coefficients and preserves the symbolic argument expression. Native
physical instructions require a finite concrete angle. Source `eps` affects
metadata, while the native exact emission consumes the exponent directly.

The single positive control lowers to a phase on that wire, corresponding to
source `ZPowGate`. A negative control wraps it in two X instructions. Control
bits correspond to the supplied wire tuple in order. Multiple controls use
the shared conjunction and its measured inverse, spending one clean work wire
and one classical slot per extra control. The existing tests compare the locked
source operation stream with reversed physical bindings and a spectator for
both exponents and every tested forward and inverse measurement history. Full
complex vectors verify relative phase and work cleanup independently of counts.

## Original tests

| Original function | Evidence and remaining boundary |
|---|---|
| `test_unitary` | Native `append` produces the full scalar multiple of identity at all ten original seeded angles. The entangled-carrier test also verifies the exact adjoint. |
| `test_controlled` | Both original control polarities and ten angles each agree with the locked source and its Cirq reference matrices. |
| `test_cirq_interop` | The unchanged source round trip passes. The native Cirq importer reports `global_phase_not_represented` for a zero-wire gate. The native carrier consumes an explicit wire binding. |
| `test_pl_interop` | The unchanged source adapter test passes. Its PennyLane matrix at exponent 0.5 is `-i I`, while the source tensor is `+i` and native append gives `+i I`. This records the source adapter's phase convention without altering either source path. |
| `test_t_complexity` | Source T complexity and QEC categories for the scalar are zero. Native model and emitted instructions are compared independently. |
| `test_global_phase` | The original atomic factory runs with skips for decomposition, block counts and decomposition-dependent QDType checks. The native notebook exercises its concrete phase and graphical signature. The generic source factory interface has no full native replacement. |

The unchanged suite recorded eight passes and three mode skips. All 26 native
global-phase tests and 106 integration cases passed. The broader regression
profile recorded 1,138 passes and three accelerator or optional-backend skips.
Numerical tensor contraction, concrete lowering, foreign conversion, and
resource counting retain their existing owners.

## Executed instructions and declared source resources

The following counts are for exponent 0.5. Peak qubits in the model are for
the source operation alone.

| Control pattern | Emitted instructions | Source QEC categories and peak |
|---|---|---|
| None | 2 X, 2 phase on one carrier | Empty categories, peak 0 |
| `1` | 1 phase | 1 Clifford, peak 1 |
| `0` | 2 X, 1 phase | 3 Clifford, peak 1 |
| `01` | 4 X, 3 H, 2 T, 2 Tdg, 6 CX, 1 S, 1 phase, 1 measure, 1 CZ, 1 reset | 1 AND, 2 Clifford, 1 measurement, peak 3 |
| `010` | 8 X, 6 H, 4 T, 4 Tdg, 12 CX, 2 S, 1 phase, 2 measure, 2 CZ, 2 reset | 2 AND, 11 Clifford, 2 measurements, peak 5 |

## Notebook

The 12 original cells cover parameters, the concrete factory, empty graphical
signature, scalar tensor and positive control. The 14 native cells retain those
topics and demonstrate append, the exact inverse, and separate resource reports.
The scalar is approximately `+i`; its positive-control matrix is `diag(1, i)`.
Both notebooks ran in independent kernels. The native kernel rejected imports
of Qualtran, Quimb and Cirq. The rendered zero-port signature was inspected.

```bash
jupyter nbconvert --execute --to notebook --stdout examples/notebooks/register_global_phase.ipynb
```


## Whole-source scalar and controlled capture

The existing named register emitter now requests the snapshot owner's scalar
carrier when the foreign signature has no logical ports. This reuses the same
physical storage convention as block-encoding capture. No new serialized type
or logical port is introduced. A scalar emitter requires one clean work wire,
zero classical slots, and retains the source's independently captured adjoint.

`test_global_phase_source_binding.py` captures zero, positive, negative and
outside-principal-range exponents. The explicit snapshot primitive phases use
the source coefficient's principal argument. Both original directions agree
with the full complex scalar, preserve arbitrary coherent carrier values, and
return an initially clean carrier to zero. The existing SELECT callback accepts
the zero-port child through `circuit.append(op)` with JSON-restored snapshots.
Forward then actual source adjoint restores every complex matrix column.

For a nonidentity scalar the measured inventory is two X and two phase gates;
the identity scalar emits none. The independent source declaration has zero QEC
categories and zero logical peak width, while physical capture owns one carrier.
These are different resource accounts, retained explicitly.

All four two-bit QUInt control values are captured in both actual directions.
After correcting scalar leaf admission, the locked source/Cirq dispatch yields
nine primitives for values zero and two (two X, two phase, three Rz and two CX),
an eight-instruction specialty decomposition for value one, and one cphase for
value three. The previous one-matrix capture for zero and two was an importer
misclassification, not the retained source decomposition. Direct controlled
GlobalPhase operations with one and two controls independently reproduce the
original decomposition, including phase angles and operands. Each complete matrix column matches the source, including the
active row's exact complex phase. These captured paths require no work or
measurement. The independent declared model still uses a conjunction and its
measurement inverse with peak width three. Existing source-style measured
lowering and every-history checks remain the evidence for that separate path.

The original global-phase Cirq operation and source reverse conversion agree.
The ordinary static importer retains `global_phase_not_represented` for its
zero-wire circuit. The explicit anchored feedback importer and named scalar
capture return the same owned primitive body. The source PennyLane adapter at
exponent one-half remains minus i times identity while its tensor and native
scalar are plus i. The exact original assertion is executed in the existing
optional environment. Neither sign convention is silently substituted.

Q-0237 remains open for the ordinary zero-wire SDK contract and unbound physical
capture. Q-0238 retains the already recorded static adapter gap. The newly
verified zero-port owned execution is available independently of these gaps.

The final concentrated run passed all 1453 cases with no skips in the existing
optional environment. Changed-surface strict typing, formatting, lint, Ecosystem
scope, architecture, multilevel IR and public API checks passed. The pure zero-wire
static import first raises the existing minimum-one-qubit ValueError; adding an
identity carrier reaches the separately asserted `global_phase_not_represented`
refusal. The source's original approximate Cirq equality tolerance is retained.

The scalar-admission correction passed the 1589-case concentrated regression,
including the unchanged controlled-H front-end suite and feedback matrix cases.
All four changed Python files passed strict typing and the repository checks.
