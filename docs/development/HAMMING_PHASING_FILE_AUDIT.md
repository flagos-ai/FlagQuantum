# Hamming-weight phasing source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `rotations/__init__.py` | `e6c63bdf43262fd6e3771a9eebd5c8e7ad75d4b8` | Q-0666 |
| `hamming_weight_phasing.ipynb` | `3279a78efaa2bd0bbf50eefe43d634612b830d7f` | Q-0667 |
| `hamming_weight_phasing.py` | `2c55d84b5b97f9eedbdcf78b9e4ea13d079161b1` | Q-0668 |
| `hamming_weight_phasing_test.py` | `bad2b4c5b0475e6b3232f2ebc6a3da34cdf21e6f` | Q-0669 |

## Original tutorial fixtures

Open [the native notebook](../../examples/notebooks/hamming_weight_phasing.ipynb)
with `jupyter notebook examples/notebooks/hamming_weight_phasing.ipynb`.
It reproduces both four-bit fixtures, types, gradient precision, and
top-level call identities. Both compile via `circuit.append(op)`.
The source exponent is `π/2` and the physical angle for weight `w` is
`π × exponent × w`. The original code comments about the phases of
`1111` and `0101` do not follow that exponent convention.

The gradient word is a through port. Its caller prepares a
negative-exponent Fourier gradient state to obtain phase kickback.
The compiler owns only the explicit count, junk, and addition carry
layouts. The notebook reports the bodies without the caller's gradient
preparation or unpreparation.

## Original numerical and resource grids

The direct grid covers all 28 source width/exponent combinations.
It compares the measured source stream and angles, full complex outputs,
and work return. For widths up to four, it projects every raw history.
For larger widths, two sparse executions check the complete supplied
complex state including work and junk. The five original large direct
counts and the original 5,000-bit T-complexity comparison are included.

The gradient suite covers all ten original cases and the default
high-precision fixture, in both directions. The source comparison
walks each original QVR and scaled-adder child, lowers the original
Add or its adjoint, and keeps the source offsets and feedback. The
arithmetic certificate checks all raw histories on every data word
and the tested gradient basis values, including wraparound. All
gradient values are used when the gradient has at most four bits.
Larger words use four representative gradient values. Actual sparse
executions check complete complex output vectors on those inputs.
The modular increments give the source Fourier phase and meet the
original epsilon for every possible data weight.

The locked original notebook passed. The bounded original rerun had
22 passes, eight source performance skips and four expected simulator
failures, with the ten gradient simulations deselected. The original
combined suite exceeded 240 seconds during those simulations. The
native suite covers their parameter grid using the independent source
leaf and arithmetic comparisons above.

| Four-bit fixture | Literal T/T† | Literal CX | Literal measurements | Source counts | Width |
| --- | ---: | ---: | ---: | --- | --- |
| Direct | 12 | 50 | 3 | 3 And, 35 Cliffords, 3 rotations, 3 measurements | 10 |
| Gradient | 1108 | 2592 | 277 | 561 Toffoli, 3 And, 35 Cliffords, 3 measurements | 79 reserved, 45 declared logical |

The gradient width is 35 bits. The compiler resource model composes the
original count and count adjoint with the original bounded number of
scaled-add calls. Its 45 logical wires exclude measured addition carries.
The native reserved width includes those carries. The source's
`phase_bitsize - 2` approximation also produces a negative Toffoli
declaration for a one-bit gradient. That count is preserved in the
source model and the literal gate inventory is reported separately.

## Remaining source-file boundaries

Q-0667 and Q-0669 are accepted. Q-0668 remains open for source
one-bit empty-junk uncomputation, mixed QFxp adjoint binding, and
recursive width estimation on the affected cases. The source
defects remain executable reproducers in the earlier focused suites.
Q-0666's complete export inventory is accepted.

## Complete source objects and native replacement

Both complete source classes and the original test file have been read. The
direct class's fields, cached signature, compute/weighted-phase/erase body,
wire symbol, independent call graph, factory, and documentation specification
map to `HammingPhasingExample`, the compiler Hamming-phase owner and the
existing catalogue/notebook. The gradient class additionally delegates its
cached `phase_oracle`, `b_grad` and `gamma_dtype` to the accepted QVR format
owner. Its body retains the same count/QVR/count-adjoint order. Drawing-only
labels are teaching metadata, not new physical instructions.

`test_hamming_phasing_source_binding.py` imports actual immutable source
objects in both directions and round-trips their owned IR. The imported and
native `CircuitRegisterEmitter` implementations run through the same
`BlackBoxPrepare` append consumer. Complete logical matrices agree, including
arbitrary gradient words. Every raw-history channel is certified separately
from the full-column final-zero check, and independently captured adjoints
restore the complete logical input. Primitive histograms agree; earlier
leaf-by-leaf operand/dependency/feedback and original numerical-grid evidence
is reused rather than replaced by histogram equality.

Direct widths two, three and four with exponents ±0.375 also agree with the
analytic complex diagonal. Gradient widths two and three with exponents 1
and -1.1934341 at epsilon 0.2 execute successfully. Exponent 1 has exact
Fourier kickback. The negative case exposes an unchanged-source precision
failure: weight one has complex-state norm error 0.21457825186131635;
weight three has error 0.25160865359628104, both exceeding epsilon 0.2.
Its four-bit-gradient phase increments for weights zero through three are
0, 7, 13 and 4 modulo 16. Tests preserve these complete output vectors and
the tolerance violations rather than loosening the requested precision.
Native/source operator agreement therefore does not prove the promised
approximation bound. The earlier original grid uses different tolerances.

The actual source adapter still fails for one input bit in either class
with its empty-junk `QAny(0)` key. For widths two and three, gradient
exponents ±0.375, 0.5 and 0.1 fail at mixed `QFxp(2,0)`/`QFxp(2,2)`
binding, whereas 1.20345 fails at finalization with the reversed types.
These are case-specific source failures; they do not imply every mixed
format or gradient path is unsupported. No source is patched. Tests retain
the one-bit and ±0.375 errors, and the bounded intake log records the other
cases. The earlier unchanged-source gradient-suite timeout is not repeated.

| Actual source capture, epsilon 0.2 | Instructions | T/T† | CX | Measurements | Physical wires |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct, two data bits, ±0.375 | 31 | 4 | 18 | 1 | 5 |
| Direct, three data bits, ±0.375 | 33 | 4 | 20 | 1 | 6 |
| Gradient, two data bits, exponent 1 | 30 | 4 | 19 | 1 | 6 |
| Gradient, three data bits, exponent 1 | 32 | 4 | 21 | 1 | 7 |
| Gradient, two data bits, exponent -1.1934341 | 151 | 28 | 74 | 7 | 12 |
| Gradient, three data bits, exponent -1.1934341 | 153 | 28 | 76 | 7 | 13 |

These are executable primitive inventories. The source's negative one-bit
gradient Toffoli estimate remains a separate model defect, and neither
physical allocation nor measurements are inferred from that estimate.
Q-0668 remains open for the source failures, this precision counterexample,
and the recursive-width boundary;
this extension does not create another progress item or close the parent.

The final concentrated regression passes 601 cases in 49.33 seconds, including
the precision counterexample assertions. Strict typing, Black and Ruff pass
for the new binding test; Algorithms scope and diff whitespace checks pass.
Existing architecture, multilevel IR and protected API evidence is reused
because this batch changes tests and the audit only. The first regression
exposed the two precision violations above (599 passes, two failures); the
final run records their exact output rather than treating them as tolerance
successes. No production implementation or immutable source was changed.

## Stored dependency graph interface

The complete-file catalog closure retains both source roots. Input `x` is an
MSB-first unsigned THRU word. The shared count tree produces owned RIGHT junk
and count words, and its specialized exact adjoint consumes both. Gradient
storage belongs to the caller. Direct weighted powers follow LSB-first source
order. Compiler owns stored calls and lifetimes, while the example provider
owns fresh source glyphs and literal-definition qualification.

The original four-bit gradient root and its QVR child decompose successfully.
Their scaled-add child fails finalization because an exact physical word key
retains `QFxp(3,3)` after a call, against the declared `QFxp(3,0)` cost output.
Native execution restores the declared storage view with an explicit Cast below
this original failure boundary. The source remains unchanged. That child is
recorded as an original dtype refusal, with no exact decomposition or drawing
claim. Existing exact source-success scaled graphs retain their previous bodies.


## Complete catalog graph evidence

Both four-bit source-success roots retain seven exact recursive graphs and
fresh source drawings, 41 original atomic contexts and the one original
scaled-child dtype refusal described above. The export adds 793 reachable
contextual children. Of these, 746 lie below source refusal boundaries or are
unqualified native helpers. They carry no source decomposition-equality claim.
The count tree and its exact adjoint each retain all 26 original ordered calls.

The stored component is exercised through existing append, typed binding,
Program copy and lowered CircuitIR JSON entry points. Complete immutable source
operators at widths two and three agree in both directions. Noncontiguous
reversed inputs, an entangled spectator, every raw forward/inverse measurement
history and private-work zero are checked. Literal-definition mutation tests
reject swapped CX operands and altered S phases. Larger four-bit factory
evidence uses complete call graphs and primitive inventories, not a full
79-qubit statevector.

The four-bit direct body uses 12 T/T-dagger, 50 CX, three phase instructions
and three measurements on ten physical wires. The gradient body uses 1108
T/T-dagger, 2592 CX and 277 measurements on 79 physical wires. Minimal clean
pool rebinding retains every instruction and allocation event. The independent
source models retain direct 3 And / 35 Clifford / 3 rotations / 3 measurements
and gradient 561 Toffoli / 3 And / 35 Clifford / 3 measurements. Default
T-equivalent estimates are 45 and 2256 respectively. Original direct QubitCount
returns ten. Original recursive gradient QubitCount raises the mixed-QFxp
finalization error; 45 is the existing declared logical formula, not a
successful upstream recursive query.

Thirteen frozen source/test/notebook/dependency blobs remain unchanged.
Qualtran 0.7.0 and native Kickmix 0.1 Simulator were verified. The first resource
receipt caught ValueError rather than the upstream RuntimeError wrapper; the
corrected receipt asserts the precise root and mixed-QFxp diagnostic.
Q-0668 remains open for the previously recorded original boundaries, and
Q-0120 remains open for other catalog groups.


## Concentrated acceptance

Execution/reference regression passes 960 cases, including the shared count
component, Hamming phasing, variable/cost phasing, gradients, controlled rotations
and restored optional SDK lane. Relevant local integration passes 29 cases.
The catalog set passes all 860 selected cases across a retained 97-case prefix
and a 763-case suffix. The initial run stopped on ENOSPC during generated JSON
export. Only that batch's verified generated JSON/text directory was removed.
The suffix runner asserts the unchanged 860-case collection and exact first
remaining node, then reuses the accepted prefix. A separate plugin-load
diagnostic is retained; the incompatible optional platformdirs pytest plugin
is excluded as in the accepted SDK lane. Fourteen existing notebook syntax
warnings remain. Strict mypy over eight files, Black, Ruff, Compiler/integration
scope, architecture, multilevel IR, protected API and whitespace checks pass.

## Symbolic model and logical dependency closure follow-up

The existing direct cost helper now accepts source-valid symbolic exponents
instead of rejecting them at isfinite. It still uses the original unweighted
Z-power declaration, existing gate-family classification and source count plus
exact-adjoint owners. Both original cost profiles are selectable. SymPy zero
and rational constants retain parameterized classification; numerical zero
and rational floats retain their own original classification. Symbols are
not algebraically replaced with cheaper constants to change source estimates.

The symbolic gradient helper composes the same count/erasure gates with the
existing symbolic QVR resource formula. Shared QVR phase/gradient bound math
now has one owner reused by both modeling and signature construction. Whole
source gate expressions match at one, two and four data bits for symbolic,
zero and rational SymPy exponents, concrete or symbolic precision and both
cost profiles. When recursive source QubitCount succeeds at symbolic precision,
the entire peak expression matches including the atomic three-wire fallback.
Original empty-junk and symbolic-gamma recursive failures remain explicit;
declared live storage is not relabeled as a successful source peak query.

The source-positive symbolic-precision HWP constructor now retains its complete
three-call logical root and reuses the existing QVR constructor for the
one-call scaled-add dependency. Previously symbolic precision was refused
before that root could be built, while a symbolic exponent stopped at an
opaque QVR declaration. Exact source producer/operand/return traces now match
both call layers at data widths one, two and four, with concrete, symbolic and
composite precision expressions. Root and child typed signatures agree with
the actual immutable source wire. The unresolved scaled-add child remains a
declaration; physical binding still refuses it. No identity, substitute phase,
backend selection or new Core representation is introduced.

The original numeric-coefficient/symbolic-precision path fails constructing its
fixed-point template. That source-negative profile keeps its native rejection;
the successful symbolic-coefficient path does not justify broadening it.
The source precision, coefficient and static field metadata remain in the
existing input library and are reinterpreted by their existing constructor
owners before physical execution. User operations still enter circuit.append(op).

Final concentrated regression passes 685 cases in each Cirq lane; local
integration passes 145. Strict typing covers five changed Python files; format,
lint, Compiler scope, language/links, architecture, IR and public API checks pass.
All 112 sampled published concrete cost trees remain equal to 63a27499b.
The full vector, measurement-history, private-work, source blob and actual
Kickmix reference checks are included in the affected source regression.
Receipts use g2-phase-symbolic-resources-*-20261126.

This follows the complete file and its count/QVR/scaled-add dependencies. It
closes the identified source-positive resource and logical-call gaps, without
accepting Q-0668 prematurely. The original one-bit physical refusal, mixed
fixed-point child failures, recursive-query boundaries and the recorded
negative-exponent precision violations remain open under its full-file audit.
Native primitive tables and independent source model estimates above remain
separate and unchanged.
