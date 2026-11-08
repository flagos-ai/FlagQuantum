# Shared register metadata

`core/ir/program/registers.py` owns the standalone metadata prerequisite 0.1
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

This is metadata, not a new user circuit construction API. Existing operations
continue through `circuit.append(op)` and existing bound emitters. Signature
construction, register-to-wire binding, component graph calls and graph traversal
remain separate pending work.

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

`test_register_metadata.py` runs 250 new cases. It compares the source Register
methods over opaque, unsigned, signed, bounded, bit and classical type metadata,
all three directions, concrete/symbolic/empty shapes, failure order, raw checker
identity and native immutability. Add and XOR QROM signature scenarios run through
both unchanged source-backed contract metadata and native ordered register tuples.
They replace metadata consumers, not an executable graph implementation.

The concentrated suite has 1,326 passing tests, no skips and no warnings. It
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
component call and undeclared quantum register value kind. Neither source
Signature nor the complete register module, graph fake, component builder or
numerical graph interpreter is accepted by this prerequisite. G2 remains open.
