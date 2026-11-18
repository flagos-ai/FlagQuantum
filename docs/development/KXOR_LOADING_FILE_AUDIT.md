# Complete k-XOR instance loading source mapping

The frozen Qualtran v0.7.0 revision is
`bcb17bb25360ed68ce3bc084645db3e929799389`. The complete
`optimization/k_xor_sat/load_kxor_instance.py` maps to the Algorithms owner
`flagquantum.algorithms.kxor_loading`. Its three oracle declarations, two
example factories, typed properties, concrete lookup body, symbolic call graphs
and specialized-control route selection are mapped together. The Compiler
comparator dependency has its own independent commit.

## Interfaces and ownership

`LoadConstraintScopes` preserves a bounded unsigned THRU index `j` and a RIGHT
opaque packed scope `U`. Concrete QROM targets are k scalar index words in the
original column order. The symbolic QROM uses the original single packed target.
Scope words and their constituent indices are big endian. The caller supplies a
clean output, clean shared QROM workspace and classical slots through `make_on`.
The result enters the normal `circuit.append(op)` workflow. Repeating the XOR
lookup clears its loaded scope on valid indices. The same fixture includes a
coherent spectator outside the operation's owned wires.

`LoadUniqueScopeIndex` retains both original THRU ports, even though the source
prose describes a RIGHT output. Its model invokes the unspecified equality
constant once per unique scope. Shared Compiler equality costs expand that
placeholder using the original symbolic-constant rule.

`PRGAUniqueConstraintRHS` retains its optional opaque control, bounded index,
qubit target, unsigned fixed-point angle type, raw rounded angle values and
sorted signs. It declares two Hadamards, one phase-gradient Rz, one constant
comparison, one controlled S and two calls to the self-adjoint amplitude QROM.
The existing native lookup, controlled-add/subtract, constant comparison and
controlled-S owners provide an independent QEC-category query. The source's
single-positive control specialization, already-controlled conjunction route and
generic fallback selection are compared directly. The generic fallback is the
original `ControlledViaAnd` wrapper.

## Physical and declared evidence

The complete original example emits 86 primitives, with three measurement
operations, two clean work wires and two reusable result slots. Its source
operands, dependencies and feedback match the native stream. A four-variable
three-scope fixture emits 27 primitives, including four T-family gates and one
measurement, with one work wire and one slot. Its complete complex state matches
an independently packed scope action in both forward measurement branches.
All four joint histories of loading twice match the original input with the
correct unnormalized branch factors. Every work and loaded-output amplitude is
checked, including absolute phase and spectator coherence. QROM replacement is
tested through the same loader consumer.

The original large-example QEC declaration and the symbolic packed-target
formula match every native category. Index and RHS oracle QEC categories match
for symbolic and concrete instances, both controlled variants and all seven
gate categories. These estimates remain separate from the emitted 86/27
primitive inventories. Source declaration-only roots carry their own execution
boundary. The source omits a phase-gradient port from the RHS root signature
while requiring one in its Rz child.

## Retained acceptance gaps

The untouched non-notebook source suite reports six passes, three strict source
skips and three expected failures. The skips concern symbolic decomposition,
count-only declarations and corresponding dtype checks. The expected failures
retain the concrete count-versus-decomposition discrepancy and both original
KXorInstance serialization limitations. The original notebook remains separately
tracked under Q-0616. The source has no composite decomposition for the unique
index or RHS root. Their primitive streams, full states and measurement histories
remain unavailable. Native control route inspection declares the same wrapper
choice, and executable root-level control-adder wiring remains open.

Q-0617 remains active for that concrete control-adder admission. Q-0618 remains
active for the original count/serialization discrepancies and notebook evidence.
The complete source mapping adds no new sub-checklist identifiers.

## Existing user workflow

```python
from flagquantum import Circuit
from flagquantum.algorithms.kxor_loading import example_scope_loader

loader = example_scope_loader()
circuit = Circuit(17)
circuit.append(loader.make_on(
    (0, 1, 2), tuple(range(3, 15)),
    workspace=(15, 16), classical_bits=(90, 91),
))
assert len(circuit.to_ir().instructions) == 86
```

## Validation

The concentrated batch passed 481 tests. Four original notebook/IPython warnings
remain visible. Black, Ruff, strict mypy across all four changed Python files,
Algorithms and Compiler ownership checks, architecture, multi-level IR and the
protected API snapshot passed. The original standalone suite retains its
six passes, three skips and three xfails as recorded above.
