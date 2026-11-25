# Phase-gradient source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `phase_gradient.ipynb` | `ffe4116032691edf96d3ede42c32b03e2fbfa3d6` | Q-0670 |
| `phase_gradient.py` | `97885bc9d4c054df375854ea6d9d4d3afecb63e4` | Q-0671 |
| `phase_gradient_test.py` | `af6677a79bdc1bff2a02dd44dd22b618ea417263` | Q-0672 |

## Tutorial and register bindings

Run `jupyter notebook examples/notebooks/phase_gradient.ipynb`. The native
notebook retains the 35 original cells and adds an append example and its
explanation. It compares four concrete fixtures and one symbolic declaration.
All concrete fixtures run through `circuit.append(op)`. The reference notebook
ran separately unchanged, and the native notebook rendered eight SVGs in an
independent kernel with reference imports forbidden.

All words use big-endian fixed-point order. Preparation writes the new gradient
word from zero with the source default exponent `-1`. The existing native
positive default remains available for callers that use it. Addition keeps the
input and gradient as through ports, preserving the input plus any spectator
wires. Measured carry work starts and returns at zero. The scaled fixture
keeps the source's two-bit fractional coefficient format, which saturates the
supplied coefficient 2 to 0.75.

## Original tests and direct dependencies

The locked source suite passed 57 tests and skipped three symbolic
decomposition and dtype checks. The native tests map the original grids:

| Source cases | Native verification |
| --- | --- |
| State widths 6, 7, 8 and exponents ±0.124, ±1 | Source H and Z-power sequence, full complex state against the source tensor, inverse to zero |
| Unitary widths 6, 7, 8, exponents −0.5, 1, 0.1, controlled and uncontrolled | Source CZ/Z expansion and phase operands, precise angles, full forward and inverse amplitudes and clean work. All raw histories through four measurements, three specified histories above that |
| Addition widths 4 and 7, no control and both control values | Source fixed-point permutation and declared cost, with Add/CAdd and their adjoints compared at the primitive layer. Exhaustive small histories and structural certificates plus sparse checks for larger cases |
| All five original scaled-addition fixtures | Source coefficient schedule, slice binding and adjoint addition leaves. Declaration multiplicity compared independently of total Toffoli count. Structural clean-work certificates, complex forward and inverse checks, and full small history projection |
| Symbolic unitary | Source symbolic rotation multiplicities and width formulas, including the controlled variant |

The native resource model preserves a source graph quirk at zero exponent:
the identical Z or CZ keys coalesce to one declared child. The emitted stream
still retains one zero-angle operation per wire. A symbolic controlled unitary
keeps the reference width formula `Max(3, n + 1)`. Its concrete declared cost
includes a clean reusable predicate qubit and reports `n + 2`.

| Concrete fixture | Literal phase | Literal H | Literal T/T† | Literal CX | Literal measurements | Source QEC categories | Reserved / declared wires |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Four-bit unitary | 4 | 0 | 0 | 0 | 0 | 1 T, 2 Clifford, 1 rotation | 4 / 4 |
| Four-bit state | 4 | 4 | 0 | 0 | 0 | 1 T, 6 Clifford, 1 rotation | 4 / 4 |
| Four-bit plain addition | 0 | 9 | 12 | 33 | 3 | 2 Toffoli | 11 / 8 |
| Two-bit scaled addition | 0 | 3 | 4 | 8 | 1 | 0 Toffoli | 5 / 4 |

## Exact precision, port and symbolic declarations

The unitary declaration retains each typed Z or CZ child exponent and the
source tolerance `epsilon / width`. The symbolic controlled leaf has exponent
`exponent / 2**width` and multiplicity `width`. At width four and exponent
zero the controlled declared graph has one child, while the execution visits
all four target wires. The unitary signature drops a zero-width through word.
The state signature retains the explicit zero-width output word.

The arithmetic model includes `p - 2` Toffoli for a plain addition into a
gradient of width `p` and `2*p - 4` for either control polarity. Peak width is
`Max(3, n + p)` or `Max(3, n + p + 1)` respectively for input width `n`.
The scaled symbolic example with a `QFxp(4,2)` coefficient declares three
plain additions and preserves the `QFxp(5,3)` input through port. Its source
model gives `3*p - 6` Toffoli and peak width `Max(3, 5 + p)`. These estimates
describe the abstract source declarations. The measured primitive counts
above come from the previous execution comparison. No emitter changed.

## Complete implementation definition map

| Complete original responsibility | FlagQuantum ownership and acceptance |
| --- | --- |
| PhaseGradientUnitary fields, signature and fixed-point dtype | Existing typed unitary declaration retains exponent, precision, optional scalar control and big-endian fraction word. |
| Unitary decomposition and powers | `phase_gradient_unitary_ir`; inverse negates the exponent and actual source adjoint is captured separately. |
| Unitary diagram, call graph, concrete/symbolic factories and DOC list | Existing native catalogue, exact typed leaf/epsilon declarations and independent concrete/symbolic cost tests. |
| PhaseGradientState fields, RIGHT signature and dtype | Existing state declaration plus explicit `local_junk_layout` at the named consumer. Prepared data is local output, not exposed through-junk. |
| State decomposition, concrete factory and DOC list | `phase_gradient_state_ir` with the source negative default explicitly bound; existing source asset and state-grid acceptance. |
| AddIntoPhaseGrad dtype/signature, register protocol and display | Existing fixed-point add signature, concrete port admission and catalogue labels. Unsupported dynamic register replacement stays unsupported. |
| Addition scaled value, apply, classical callback and tensor action | Existing exact fixed-point adder and numerical complete-map/complex-state tests; both control polarities, signs, shifts and truncation are reused from the accepted phase-addition audit. |
| Addition call graph and adjoint | Independent source model and opposite-sign emission. The original has a value/tensor contract, not a primitive gate decomposition. |
| `_fxp` and `_mul_via_repeated_add` | Exact integer truncation, saturation/word slicing and per-term modular addition in the existing compiler owners; original numerical callbacks provide an independent oracle. |
| Scaled-add fields, from-bitsize factory, signature and fixed-point properties | Existing `scaled_phase_add_terms`, native unsigned format arguments and coefficient quantization, with original five-fixture/source-factory evidence reused. |
| Scaled register protocol, numerical value/apply/callback | Existing exact scaled-add emitter, validation and source numerical grids; new admitted mixed-format cases compare every classical map entry too. |
| Scaled decomposition, call graph, factory and DOC list | Existing per-term compiler emission and independent declared full-width child costs. Actual source directions and model levels remain distinct. |

## Actual resolved objects and directional ownership

`test_phase_gradient_source_binding.py` resolves the actual original symbolic
unitary factory at widths two and three, with and without control, at exponent
-0.375. The source stays unresolved until its constructor fields are explicitly
bound. Both directions pass through owned IR JSON and the normal named append
consumer. Every logical input column agrees with the standalone native unitary,
its analytic diagonal and the true source inverse. Controlled measurement
histories are certified separately from complete-column private-work cleanup.
Source/native gate histograms and independent declared costs agree.

The RIGHT state output is bound with `local_ports=("phase_grad",)` and the
existing `BlackBoxPrepare.local_junk_layout`. It starts at zero, remains live as
a prepared local word, and is consumed by the actual inverse. It is never
flattened into arbitrary logical through-junk or required to be zero immediately
after preparation. Widths two/three and exponents -1, +1 and -0.375 preserve an
arbitrary coherent spectator, match the analytic complete gradient state and
return the local word to zero on inverse. The native implementation can replace
the captured source at the same consumer without changing that ownership.

At exponents +/-1, source capture canonicalizes Z/S/T-family leaves; the native
state emitter retains parameterized phase instructions. Both have one phase
operation per gradient bit plus one H per bit, but their raw opcode families
are different profiles and must not be described as identical instruction
names or fully synthesized costs. The source-owned declarations preserve the
corresponding resource categories independently.

## Mixed fixed-point admission and measured resources

For `x=QFxp(5,3)`, a four-bit gradient and coefficient +/-3.25 represented by
`QFxp(4,2)`, actual source import is admitted. Both directions are compared on
all 512 logical basis inputs, including arbitrary existing gradient values.
The native standalone emitter has the same instruction histogram; the complete
forward operator also equals the immutable source classical callback's exact
permutation. Every raw history has the same full channel and all carry work is
zero. These examples cannot be rejected merely because other mixed formats fail.

| Case | Actual instructions | Explicit T/T-dagger | CX | Measures | Actual wires | Source declaration |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Width-three unitary, no control, -0.375 | 3 | 0 | 0 | 0 | 3 | 3 rotations; peak 3 |
| Same, controlled | 54 | 12 | 18 | 3 | 5 | 3 And, 3 Clifford, 3 rotations, 3 measurements; peak 5 |
| Mixed scaled add, either sign/direction | 129 | 24 | 63 | 6 | 12 | 6 Toffoli, no declared measurement; peak 9 |

The independent models price original declarations. Actual scaled carry
measurements are not added to the source model or relabeled as source Toffoli
instructions. Captured feedback uses six result records; the standalone emitter
reuses its three carry slots. Both histories and latest-record semantics are
verified by their complete channels.

## Remaining source boundaries

Q-0670 and Q-0672 retain prior accepted source assets/tests. Q-0671 no longer
lacks actual resolved symbolic execution, local RIGHT-output binding or the
admitted mixed-format path above. It remains open for the original abstract
addition and invalid fixed-point boundaries. `AddIntoPhaseGrad(1,1)` still
declares minus one Toffoli and raises `DecomposeNotImplementedError` on primitive
decomposition; no source recipe is invented for that declaration.

For `x=QFxp(3,2)`, phase width four, coefficient 0.375 and `QFxp(3,3)` coefficient
format, actual import still raises the original `BloqError` for inconsistent
`QFxp(3,2)` versus `QFxp(3,3)` ports. Other source Soquet dtype failures from the
existing fixtures remain recorded. The source Cirq generator and numerical
contract can be tested separately from those failed graph bindings. Neither
blanket mixed-format rejection nor a claim that all dtype failures were fixed
is supported. No reference code or original assertion is modified, no inherited
failure is silently normalized, and no new child progress item is added.

Zero-width declaration and execution are also separated. The original zero-width
unitary has an empty gradient port (only control remains when enabled), an empty
gate-cost declaration, a `QubitCount` RuntimeError and an actual-import
`KeyError('phase_grad')`. The zero-width state retains a RIGHT zero-bit port,
but its source gate-cost/peak queries and physical binding fail. These exact
original failures are retained separately from positive-width symbolic
resolution. No zero-qubit Core API or implicit physical carrier is introduced.

## Current batch validation

Thirteen initial resolved/owned-state/mixed-format cases passed in 14.39
seconds after adding exact histograms. The final concentrated regression,
including three additional zero-width source-diagnostic cases, passed 233
cases in 17.35 seconds. Strict mypy on the new test, Black, Ruff, Algorithms
scope, architecture, multilevel IR and protected API checks passed. Existing
independent source/native notebook and CPU integration evidence is reused;
this batch changes tests and this audit only. The whole source file remains
open for its documented source defects, with no new parent/child completion
count.

## Complete catalog dependency closure, 2026-11-22

All five original factories retain their signatures and metadata. The concrete
unitary, prepared state and scaled addition now expose executable typed Programs
through `examples/algorithms/phase_gradient_decomposition.py`. Symbolic unitary
widths and `AddIntoPhaseGrad` retain their original decomposition refusals.
The original source and nine source/test/notebook/dependency blobs remain frozen.

The Compiler owners are split by responsibility.

- `primitives/pauli_rotation_component.py` supplies the existing single-bit phase
  wrapper to both controlled rotations and gradient construction.
- `phase/phase_gradient_component.py` owns the ordered Split, phase powers and
  Join, and the Allocate, OnEach-H, gradient preparation graph.
- `phase/scaled_phase_component.py` composes existing coefficient slicing,
  Cast, fixed-point views and shared integer Add with its exact adjoint.
- The example provider owns catalog qualification and drawing callbacks.
  Complete child bodies must match before OnEach or PhaseGradientUnitary labels
  are assigned. Changed H and phase constants reject qualification.

The complete recursive source observer finds five exact graphs and twenty-one
atomic/refusal contexts. All five comparable fresh source drawings agree.
There are sixty-seven additional contextual children, including forty-four
explicitly native-only arithmetic and rebinding helpers below the original
atomic AddIntoPhaseGrad boundary. They carry no original decomposition-equality
claim. The state keeps the original opaque Allocate/OnEach view and returns the
fixed-point logical output. That output remains caller-owned and live.

### Execution and independent resource evidence

The normal entry remains `circuit.append(op)`. The actual catalog Programs
match source unitary/state streams on noncontiguous MSB-first wires and an
entangled spectator. Full complex tensors, output ownership, exact recovery,
Program copying and lowered CircuitIR JSON round trips pass. Positive and
negative scaled coefficients exercise all raw histories and their inverse
histories. Dynamic trajectory append confirms normalized branch execution and
zero clean carry work. The atomic native leaf also retains shifted and empty
updates. Shared Program scheduling can reorder disjoint primitives. Fresh
classical slot identities follow measurement order, so comparisons supply the
explicit measured-wire-to-slot map and retain per-wire order and feedback.

| Original factory | Measured primitives | Separate original QEC model | Physical peak / source width |
| --- | --- | --- | --- |
| PhaseGradientUnitary, width 4 | 4 phase | 2 Clifford, 1 T, 1 rotation, default T-equivalent 12 | 4 / 4 |
| PhaseGradientState, width 4 | 4 H, 4 phase | 6 Clifford, 1 T, 1 rotation, default T-equivalent 12 | 4 / 4 |
| AddScaledValIntoPhaseReg, width 2 | 3 H, 8 CX, 2 T, 2 Tdg, 1 S, 1 measurement, 1 conditional CZ, 1 reset | zero Toffoli and zero modeled T | 5 / 4 |

Preparation allocates its four logical output wires at offset zero. The scaled
fixture allocates one clean carry at offset zero and releases it at offset
eighteen. Its coefficient 2 still saturates to the original 0.75 format. Its
source model omits the concrete measured Add cost. The difference is retained
explicitly, with original abstract estimates and native inventories separate.

Existing complete file, symbolic resolution, controlled gradient, mixed-format,
source-binding and source model evidence is reused. Q-0671's original atomic
addition, mixed-dtype, small-width cost and zero-width boundaries remain open.
Q-0120 remains open for other catalog groups. Pending Proposals 116 through 119
retain their separate authorization requirements. Proposal 115 was already
implemented and accepted before this catalog batch.

Persistent receipts use `ftqc-reference-env/g2-phase-gradient-ui-*-20261122`.
They include recursive inventory, source hashes, resources, focused and
concentrated regressions, strict checks and publication verification.

The final concentrated catalog run passes 827 cases with three existing
unrelated deselections and fourteen existing notebook escape warnings. The
SDK-restored execution/reference run passes all 242 cases without skips or
warnings. Both local integration files pass all 29 cases. Eight changed Python
files pass strict typing, Black and Ruff. Compiler/integration scope,
architecture, multi-level IR, protected API and whitespace checks pass.
A mistyped test filename caused an initial empty collection diagnostic, which
is retained separately from the corrected passing execution run.

## Successful symbolic-exponent resource queries

Concrete-width unitary and state resource helpers now accept the original
symbolic exponent domain. Previously their finite-float guard raised TypeError
for theta even though the unchanged original query succeeded. The common
symbolic-value protocol now bypasses only that numerical check; the existing
phase-key construction and gate-family classifier remain authoritative.
Nonfinite numerical values keep their rejection. No emitter, precision rule,
helper graph or physical workspace requirement changes.

Whole-root QEC categories and successful recursive peak queries match the
unmodified source for prepared states, ordinary unitaries and controlled
unitaries at widths one and four. Unknown expressions, composite expressions,
SymPy zero and SymPy rational constants are compared without coercing them to
floats. The source treats SymPy constants as parameterized rotations, including
mathematical zero. The source symbolic-width declaration also retains rotation
count n for zero or algebraically cancelled exponents; the existing symbolic
width formula was already correct and is unchanged. Original dictionary-key
coalescing at finite width remains distinct from literal emitted gate counts.

An independent comparison against published 63a27499b verifies 112 concrete
unitary/state/direct-Hamming/gradient-Hamming cost trees unchanged, including
all child multiplicities, live widths and categories. Final concentrated
regression passes 685 cases in each Cirq lane; local integration passes 145.
Five changed Python files pass strict typing, format/lint, Compiler scope,
language/links, architecture, Program IR and public API checks. Frozen source
assets and actual Kickmix0.1 Simulator are verified in the affected source
suite. Related resource and symbolic logical-graph receipts use
g2-phase-symbolic-resources-*-20261126.

This closes the source-positive symbolic cost-query gap. Q-0671 retains its
previous complete-file boundaries; no parent is accepted on the strength of
this model extension alone. Source signed classical assertions, nominal dtype
refusals, abstract primitive leaves and invalid small-width model estimates
remain separate from the measured native instruction streams above.
