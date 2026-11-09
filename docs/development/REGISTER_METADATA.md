# Shared register metadata

`core/ir/program/registers.py` and `signatures.py` own standalone metadata
prerequisites 0.1 and 0.2
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
storage and bounds even when bit counts agree. It is a descriptor of an already
validated type, not a port of all upstream dtype constructors. Numerical domains,
bit codecs and source dtype-constructor validation are not implemented here.
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
