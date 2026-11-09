# Shared register metadata

`core/ir/program/registers.py` and `signatures.py` own standalone metadata
prerequisites 0.1 through 0.6
specified in [proposal 103](API_CHANGE_PROPOSAL_103_REGISTER_COMPONENT_PROGRAMS.md).
It is internal and adds no root export, Program IR opcode, value kind, serialized
schema or executable component profile.

```python
from flagquantum.core.ir.program.registers import Register, RegisterSide, RegisterType

word = Register("words", RegisterType("uint", 8), shape=(2, 3))
assert list(word.all_idxs()) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
assert word.total_qubits() == 48
assert word.adjoint() is word

effect = Register("word", RegisterType("uint", 8), side=RegisterSide.LEFT)
assert effect.adjoint().side is RegisterSide.RIGHT

empty_scalar = Register("empty", RegisterType("opaque", 0))
assert list(empty_scalar.all_idxs()) == [()]
assert empty_scalar.total_bits() == 0
assert list(Register("empty_array", word.dtype, shape=(0, 2)).all_idxs()) == []
```

Existing operations continue through `circuit.append(op)` and existing bound
emitters. Signature construction now feeds Add symbolic resource estimates and
QROM physical layout. General typed register-to-wire binding, component graph
calls and graph traversal remain pending.

## Preserved source behavior

Register methods follow unchanged Qualtran v0.7.0 `_infra/registers.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`:

- an integer shape becomes a one-dimensional tuple; all other supplied shapes
  are consumed into a tuple before nominal dtype validation;
- shape and element width are separate: a symbolic width need not make the
  concrete shape unindexable;
- scalar zero-width ports still have a logical index, whereas zero-length
  array dimensions have none;
- indices use Cartesian-product order; no implicit reshaping or bit reversal
  occurs;
- totals multiply the exact element count by each quantum/classical width;
  symbolic formulas are retained rather than rounded;
- adjoint exchanges LEFT/RIGHT, retaining the dtype and shape, and returns the
  same object for THRU; an unknown side is not normalized into a valid one; and
- the shared symbolic-object predicate retains raw single-checker results,
  short-circuiting and original import/callback failures.

Source dtype names are replaced by a native immutable `RegisterType` descriptor.
It carries encoding identity, exact quantum/classical widths and ordered type
parameters. Equality distinguishes opaque/unsigned encodings, classical/quantum
storage and bounds even when bit counts agree. Shared `register_types.py`
factories construct bit, opaque, unsigned, bounded unsigned, signed,
ones'-complement, Montgomery and fixed-point descriptors using the source
constructor rules. Classical forms retain their storage identity. Finite-field
constructors remain separate work. Numerical domains and bit codecs remain
Simulation-owned.
Unknown encodings and symbolic descriptors do not thereby acquire execution
support. Source Register subclasses with additional constructor fields are also
outside the current acceptance; their evolution behavior needs separate tests.

Core's `value_parameters.py` is the single symbolic predicate used by this
metadata and the existing Simulation consumers. The old private Simulation
module has been removed. Concrete metadata and existing numerical codecs still
run without optional Sympy; a broken installed dependency is not hidden. No
mandatory dependency or provider type is added.

## Verification and limits

The prerequisite 0.1 acceptance ran 250 new cases in
`test_register_metadata.py`. It compares the source Register
methods over opaque, unsigned, signed, bounded, bit and classical type metadata,
all three directions, concrete/symbolic/empty shapes, failure order, raw checker
identity and native immutability. Add and XOR QROM signature scenarios run through
both unchanged source-backed contract metadata and native ordered register tuples.
They replace metadata consumers, not an executable graph implementation.

The prerequisite 0.1 concentrated suite had 1,326 passing tests, no skips and no warnings. It
includes the accepted Add primitive stream, controlled add/subtract full complex
states, control/Power/OnEach measured branches, QROM complex branch/work cleanup,
the independently reported logical resource models, and current Program IR
conformance. These are unchanged consumer regressions; they do not certify a new
metadata-to-gate lowering path. The moved symbolic predicate's function body is
unchanged from the previously accepted implementation. Seven reference blobs are
verified against the immutable source inventory.

The existing public API baseline, Program contract 0.5 and architecture checks
pass unchanged, as do strict typing, format and scope checks on changed surfaces.
The refreshed scope audit records standalone metadata while still refusing a
component call and undeclared quantum register value kind. Prerequisite 0.1 left Signature open. The complete register module, graph fake,
component builder and numerical graph interpreter remain pending after the
Signature slice below. G2 remains open.


## Signature and shared width arithmetic

Prerequisite 0.2 implements the ordered source Signature methods with native
Register records. Its constructor consumes the complete iterable before checking
names, first on the input side and then on the output side. The same name may
appear once on each side. Grouping, positional and slice access, hashing,
adjoint, cached through-only detection and lookup retain source order.

Quantum, classical and combined width queries each take the maximum of their
own input and output totals. For a three-qubit input and seven-classical-bit
output, `n_qubits()` is 3, `n_cbits()` is 7 and `n_bits()` is 7. Adding the first
two peaks would give a different quantity.

The width builder retains the source truth-value filter and one-bit selection
before opaque-width validation. The dtype builder filters on quantum width,
which omits a pure classical port. Explicit construction retains classical
ports and zero-width registers. Foreign-object equality and constructor errors
follow the source behavior, including their failures.

`core/value_math.py` owns source-ordered product and sum folds and shared
minimum/maximum selection. Register totals and Compiler extrema use this same
implementation. Empty argument packs and empty iterables have distinct source
diagnostics. Singleton results retain object identity. Symbolic integer
subclasses are classified through the shared predicate. Exact built-in numeric
inputs keep the existing optional-dependency fast path. Compiler rounding and
logarithm helpers keep their existing ownership and acceptance scope.

The symbolic Add resource fallback obtains its port width from Signature before
applying the original minimum-three-qubit formula. Concrete Add resource counts
retain their separate source model. `QromTable.signature` preserves opaque
control/target types, bounded unsigned selection types, target shapes and source
zero/symbolic filtering. Its existing `register_layout` and `bind_packed` consume
the signature before the established physical partition and emitter.

Run the existing user workflows from this checkout.

```sh
python -m examples.algorithms.resource_model_comparison
python -m examples.algorithms.register_qrom_table
python -m examples.algorithms.register_qrom_table --symbolic
```

Three-bit Add emits 8 T/Tdag primitives, two measurements and eight declared
wires. Its source model reports two And operations, 8 T equivalents and peak
width eight. The symbolic source formula retains `Max(3, 2*n)`. The concrete
QROM example checks all final complex amplitudes and clean work across its
sampled measurement histories. Source model categories and emitted instructions
remain separate report fields.

There are 105 new Signature cases and 73 shared-math cases, compared directly
with unchanged pinned source. Real-consumer replacement tests change Add width
queries and QROM partition metadata through the shared boundary. Full source
dtype constructors, general typed graph connections, explicit function results
and numerical graph traversal remain outside this slice.


The prerequisite 0.2 concentrated suite passes 1,683 cases without skips or
warnings. It includes original Add/QROM primitive comparisons, complete small
measurement-branch amplitudes and clean work, controlled/Power/OnEach numerical
and gate regressions, resource reports and current Program IR conformance.
Nine pinned source blobs match the immutable inventory. Strict typing of seven
production/example files, Black, Ruff, scope, architecture, Program contract and
public API baseline checks pass.

The source mixed-singleton QROM failure remains reproducible in `_load_nth_data`.
Its diagnostic follows the installed NumPy array-to-integer error. The reference
regression compares that native NumPy error and the original source failure
frame instead of assuming an older dependency's message. This adjusts the test
oracle for the environment while preserving the source failure boundary.

## Shared type constructors

Prerequisite 0.3 adds four internal factories in
`core/ir/program/register_types.py`. They return the existing `RegisterType`
without adding an upstream class hierarchy. The width convenience builder in
Signature uses the bit/opaque factories after its existing truth filter and
one-bit selection.

Opaque construction preserves symbolic-first Python integer validation.
Unsigned construction stores the declared width directly. Bounded construction
computes an omitted bound before its variadic symbolic guard and compares the
explicit bound with capacity only when both parameters are concrete. Explicit
`None` remains distinct from an omitted bound. The tests preserve accepted
negative/zero/bool and symbolic inputs and the type of stored scalar values.

The source-backed constructor stand-in passes 124 scenarios before native
implementation. Both implementations pass the same scenarios, yielding 248
differential tests. With Signature, Register and internal multi-level IR
regressions, the Core slice passes 634 tests without warnings or skips. This
constructor comparison reads declared source width directly and excludes the
separate lazy numerical codec construction.

QROM now uses the same opaque control/target and bounded selection constructors.
Its existing axis-fit diagnostic runs before selection construction. Two actual
consumers accept either source-backed or native factories without consumer
changes. The QROM replacement test compares its complete packed instruction
tuple with the existing implementation. Four native constructor scenarios also
run with optional Sympy imports blocked.

The batch passes 2,128 concentrated tests across 45 files in 54.79 seconds, with
no warnings or skips. The 256 constructor/consumer tests include 124 scenarios
per constructor backend plus eight consumer/optional-dependency cases. The
regression includes source gate streams, Add/QROM complex amplitudes, measured
branches, zero workspace, controlled composition, bounded numerical codecs and
separately reported logical resource costs.

Concrete QROM, symbolic QROM and resource-model comparison examples execute.
Concrete QROM reports two logical AND components, two measurements and five peak
logical qubits, while the emitted stream has eight T/Tdag primitives. Add reports
two logical AND components and eight peak logical qubits, with eight emitted
T/Tdag primitives. Conversion assumptions stay explicit in the model report.
Nine previously recorded pinned-source blobs rehash unchanged. Strict Python
3.12 mypy passes for the three changed production modules. Ruff, Black,
architecture, public API baseline and internal multi-level IR checks pass.

Logs are retained in the existing reference environment as
`g2-type-constructor-contract.log`, `g2-type-constructors-focused.log`,
`g2-type-constructors-core.log`, `g2-type-constructors-consumers.log`,
`g2-type-constructors-regression.log` and `g2-type-constructors-examples.log`.
Other encoding families, graph-port compatibility, typed binding, explicit
component returns and graph execution remain pending G2 work.

The default smoke/unit tier with four workers stops at the previously reproduced
`test_native_fused_rotation_layer_only_promotes_terminal_regions` assertion.
This run reports 13,495 passed, 51 skipped, 398 warnings and one failure in
87.89 seconds. The isolated original-baseline reproduction is retained in
`g2-register-organization-baseline-failure.log`. The default run is recorded in
`g2-type-constructors-default.log`. The existing Python 3.10 unused fallback-import
suppression diagnostic in `simulation/numerics/conformance.py` remains open.

## Signed and classical integer constructors

Prerequisite 0.4 extends the existing factory module with signed two's-complement
and ones'-complement quantum words and classical bit, unsigned, signed,
ones'-complement and bounded unsigned words. Signed constructors keep the source
absence of eager width checks. Ones'-complement construction rejects only a
Python integer one, including True. NumPy and symbolic integer one retain their
source constructor outcome.

Classical integer factories retain `("storage", "classical")` in the existing
ordered type parameters. Bounded classical words append their bound parameter.
This preserves the distinction between zero-width classical and quantum types.
The existing classical bit identity remains `RegisterType("bit", 0, 1)`.
Signature explicitly retains zero-width scalar ports and continues its source
classical-only filtering in `build_from_dtypes`.

The original ones'-complement width-guard body moves unchanged to Core. Both
Simulation's numerical codec and signed truncation use that authoritative
function at their original validation sites. The shared bounded resolver keeps
quantum `iteration length` and classical `bound` diagnostics distinct and retains
omitted-bound exponentiation before symbolic classification.

The source-backed contract adds 182 scenarios before implementation. Native
factories pass the same scenarios. Codec, guard identity and optional-Sympy cases
bring the constructor suite to 642 tests. With Register and Signature regression,
997 cases pass without warnings or skips. The shared-guard relocation separately
passes 1,678 cases with 64 existing paired source/native NumPy warnings. Its body
AST is identical to the pre-move implementation.

The symbolic Add resource consumer now also uses the shared unsigned factory.
Both source-backed and native replacement implementations produce the same
formula and signature widths. The final constructor suite contains 644 tests.
The concentrated batch passes 4,298 tests across 51 files in 59.92 seconds,
with 67 existing numerical warnings and no skips. Coverage includes source gate
streams, complete complex amplitudes, measured branches, zero workspace and
separately reported resource models. Nine pinned reference blobs remain unchanged.
Strict Python 3.12 typing, Ruff, Black, team scope, architecture, public API
baseline and internal multi-level IR checks pass.

Four existing examples execute. Signed conversion emits four T/Tdag primitives,
one measurement and one work qubit. Its abstract source model reports one
Toffoli, zero explicit measurements and four T equivalents. The source root
conversion has no primitive decomposition, so whole-root gate parity remains
unavailable. Signed truncation emits three CX with no T or measurements and
clears two redundant bits. Its source model counts one logical Clifford
component. Duplication emits four T, one measurement, five reserved work qubits
and eight declared wires. Its source model reports four T equivalents, one
measurement and seven peak qubits. QROM emits eight T/Tdag with one work qubit,
while its model reports two AND, ten Clifford, two measurements and five peak
qubits. Add emits eight T/Tdag, while its model reports two AND, eleven Clifford,
two measurements and eight peak qubits. These counting bases remain separate.

Final logs are `g2-integer-type-regression.log`,
`g2-integer-type-add-consumer.log`, `g2-integer-type-examples.log` and
`g2-integer-type-default.log` in the reference environment. The default tier
reports 13,445 passed, 51 skipped, 398 warnings and the same independently
reproduced rotation-layer baseline failure in 74.01 seconds. The Python 3.10
fallback-import unused-ignore diagnostic remains open. Other dtype constructors,
port compatibility, typed binding and component calls/returns/traversal remain
pending G2 work.

## Montgomery integer constructors

Prerequisite 0.5 adds quantum and classical Montgomery factories in the existing
Core constructor module. Both retain declared width and optional modulus without
arithmetic, coercion or numerical validation. Encoding is `montgomery`. A
non-None modulus is retained as a named parameter. Absent and explicit None
moduli have the same identity. Classical storage retains its first marker,
including zero-width words.

The inherited source symbolic classifier examines width and ignores modulus.
The existing RegisterType classifier now preserves that rule for Montgomery
encodings. Other encodings retain their parameter-aware classification. Tests
include a concrete width and symbolic modulus, whose numerical conversion fails
at the original Simulation boundary. Constructor operation without optional
Sympy is covered separately from symbolic classification, which uses Sympy.

Before implementation, 326 source-backed constructor and Signature scenarios
pass. Both backends then run the same cases. Additional native classifier,
optional-dependency and numerical replacement cases bring this module to 704
tests. With existing Register, Signature and constructor tests, 1,703 cases pass
without warnings or skips. The replacement cases exhaust small stored-residue
conversion/product/inverse domains, including a composite modulus, using the
existing Simulation functions. Metadata construction emits no gates and owns no
work qubits. Montgomery gate-algorithm verification remains separate.

The final Montgomery batch passes 2,660 tests across 20 files in 150.52 seconds,
with 192 existing paired source/native NumPy overflow warnings and no skips.
It reruns the source primitive streams, exact adjoints, complete complex
amplitudes, measured branches, workspace cleanup and logical resource checks
for the existing modular arithmetic paths. The three executed examples are
`classical_montgomery_values`, `register_qrom_table` and
`resource_model_comparison`. The Montgomery example now consumes the new
quantum/classical metadata widths in existing numerical functions. Its append
path emits 128 T/Tdag, 32 measurements and 21 declared wires. The source model
reports 128 T equivalents, 32 measurements and 21 logical wires separately.
The example constructs forward/adjoint IR. Execution evidence comes from the
regression suite. Source composite-modulus Fermat discrepancies remain open.

Nine locked source blobs remain unchanged. Strict Python 3.12 typing, Ruff,
Black, Integration/Simulation scope, architecture, public API baseline and
multi-level IR checks pass. Logs are retained as
`g2-montgomery-type-native.log`, `g2-montgomery-type-regression.log`,
`g2-montgomery-type-default.log` and `g2-montgomery-type-examples.log`.
The default tier reports 14,149 passed, 51 skipped, 398 warnings and the same
previously reproduced rotation-layer failure in 50.21 seconds. The existing
Python 3.10 unused-ignore limitation remains open. Fixed-point/finite-field
constructors, graph compatibility and typed component execution remain pending.

## Fixed-point constructors and shared shape validation

Prerequisite 0.6 adds quantum and classical fixed-point metadata constructors.
They retain the original declared width, fractional width and signed value,
including scalar types. Ordered parameters are fractional_width then signed.
Classical words retain their first storage marker. Shape validation follows the
source eager decision tree and protocol evaluation order. Native diagnostics
replace only the QFxp name with fixed-point register. The source num_qubits
wording is retained for both quantum and classical metadata construction.

The shared `validate_fixed_point_parameters` guard belongs in Core. Its default
context preserves _Fxp codec diagnostics. Register constructors select their
source-specific context. Fixed-point RegisterType symbolic classification checks
width and fractional width and ignores signed/storage, including custom signed
objects and symbolic signed flags. The Montgomery rule remains width-only.

The source-backed prerequisite passes 430 cases before implementation. The same
constructor, Signature and protocol scenarios run with native constructors.
Additional classifier, complete small integer-domain/array codec replacement
and optional-Sympy cases bring this module to 910 tests. With existing metadata
and type constructor regression, 2,613 cases pass without warnings or skips.
Existing numerical codecs accept the same source-backed or native descriptors.
Metadata constructors emit no gates and allocate no work qubits. Real-to-word
quantization and finite-field constructors remain pending work.

Simulation's original shape-validation entry now delegates to the shared Core
guard while preserving its signature and all existing validation sites. The
real array/domain/scale consumers pass with either the source-backed or native
guard. Their focused regression passes 1,334 cases without warnings or skips.

The full batch passes 4,157 concentrated tests across 26 files in 50.19 seconds,
with three existing paired numerical warnings and no skips. It includes phase
and fixed-point source gate streams, exact adjoints, complete complex amplitudes,
measured branches, zero workspace and separate resource accounting. Four
existing examples execute. The fixed-point append example consumes the new
metadata widths and restores every complex amplitude and its spectator after
the inverse. It emits three phase instructions, zero explicit T, zero
measurements and zero work. The source model counts two Clifford and one T.
Montgomery, QROM and Add examples retain their separate emitted/model reports.

Nine pinned source blobs remain unchanged. Strict Python 3.12 typing, Ruff,
Black, Integration/Simulation scope, architecture, public API baseline and
multi-level IR checks pass. Logs are `g2-fixed-point-type-contract.log`,
`g2-fixed-point-type-native.log`, `g2-fixed-point-type-consumer.log`,
`g2-fixed-point-type-regression.log`, `g2-fixed-point-type-examples.log` and
`g2-fixed-point-type-default.log` in the reference environment. The default tier
reports 15,061 passed, 51 skipped, 398 warnings and the independently reproduced
rotation-layer baseline failure in 86.65 seconds. The Python 3.10 unused-ignore
limitation remains open.

## Shared scalar width arithmetic

Prerequisite 0.7 moves source ceil, floor, log2 and bit_length semantics to the
existing Core value_math owner. Concrete rounding uses the source NumPy scalar
operation followed by int conversion. Symbolic cases retain the source Sympy
expression classes. Bit length retains the zero interval and ordered
floor/log2/ceil chain. The former Compiler helpers classified NumPy scalars as
symbolic and returned Sympy integers. The new implementation preserves the
source Python int result and integer-subclass symbolic protocols.

NumPy already belongs to the required dependency set. This helper uses scalar
mathematics and exposes int or symbolic expressions across the metadata boundary.
Concrete native operation without optional Sympy remains supported. The source
stand-in passes 109 cases before implementation. The native differential and
optional-dependency slice passes 223 cases without warnings or skips. Outcome
comparisons include scalar types, NaN, infinities, large integers, exact
exceptions, captured warnings and symbolic protocol evaluation order. The
source audit is retained as g2-width-math-audit.log.
