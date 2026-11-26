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
remain unavailable. The later typed-context closure below supplies actual generic and specialized
control-root wiring around the unchanged positive declarations. It does not
create physical bodies for those source leaves.

Q-0617 remains active for original bodyless controlled/oracle execution and
instance serialization boundaries. Q-0618 was subsequently accepted by the
complete original-test disposition; its original failures remain documented in
KIKUCHI_KXOR_SOURCE_TEST_ACCEPTANCE.md, not silently fixed.
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

## Exact adjoint model dependency

The block-encoding closure exercises inverse RHS/matrix entry costs. The RHS
model now routes the inverse phase-gradient rotation to the existing exact
controlled-add/subtract polarity and reverses the controlled S declaration.
All seven categories match untouched concrete/symbolic source queries, including
the specialized controlled RHS. Inverse RHS and uncontrolled matrix costs have
two fewer Cliffords than their forward models. Forward callers retain their
existing results. The generic controlled-adjoint matrix source query reaches an
atomic controlled And and fails. Native model admission reports that boundary
explicitly. This dependency adds resource evidence and carries no new physical
primitive, amplitude or restoration claim. Concentrated regression passed 618
tests. Strict typing, formatting, lint, scope, architecture, IR and API checks
passed for the three changed Python files.


## Complete typed loading and control context (2026-11-27)

The Algorithms owner `kxor_loading_component.py` now composes the actual complete
scope root from existing Compiler Allocate, QROM and Partition owners. The
selection index is THRU and MSB-first. Each source scalar scope index is allocated
in column order, loaded by the single shared QROM call and packed by the exact
inverse Partition. The packed scope is a RIGHT output: its live storage transfers
to the caller. Physical binding requires explicit allow_workspace_outputs; the
scope is not falsely zeroed or released as private scratch. Private QROM
predicates are still erased and released normally.

The Ecosystem owner `_kxor_loading_component_library.py` binds an explicit native
instance context; it does not serialize an upstream KXorInstance or register a
fabricated shallow constructor. No foreign object reaches Compiler. Concrete
ordinary scope roots execute. Symbolic scope, unique-index and RHS roots retain
the source bodyless declarations. The existing flat make_on/append workflow is
unchanged, and the new typed root is compared against its previously validated
source stream rather than reimplementing the lookup.

Single-positive controls preserve the actual positive source declaration.
Negative and multiple external controls reuse ControlledViaAnd and the shared
predicate owner. An already-controlled RHS uses the original two-control
conjunction order, including zero activation, or its generic outer reduction.
The shared specialized component owner consumes a declared RHS body explicitly;
whole physical binding refuses that missing body. Roots, named operands and
control/signature layouts are compared with original decompositions across all
three oracle classes. No controlled scope XOR recipe or RHS phase-gradient port
is invented: original positive scope control raises DecomposeTypeError for
non-THRU registers, while index and RHS have no decomposition at all.

### Reused source evidence and new conformance

Complete original source/test/notebook and seven-category model evidence from
6ff0f8a85, bdcd02cc4 and 54949021e is reused. Eight source/direct-control blobs
match the locked CSV manifest. The complete new typed small and original large
roots have the same 27 and 86 primitive instructions, operands, parameters and
measurement feedback as the accepted lookup consumer. The small rectangular
source tensor gives the full complex reference on valid indices; every raw
history agrees and only private work is zero. SDK-blocked ordinary append and
native trajectory execution also preserve the transferred scope state.

| Fixture | Actual instructions | Measurements | Transferred output bits | Peak private bits | Minimum physical wires | Source QEC estimate | Source peak qubits |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| Three scopes, four variables | 27 | 1 | 4 | 1 | 7 | And=1, Clifford=9, measurement=1 | 7 |
| Complete original example | 86 | 3 | 12 | 2 | 17 | And=3, Clifford=32, measurement=3 | 17 |

Peak private counts exclude the still-live RIGHT output; root allocation peaks
including it are five and fourteen bits. Minimum-wire counts use a caller pool
trimmed to actual leases, not arbitrary unused supplied storage. The QEC And
estimates remain separate from their physical T/CX/measurement recipes. Complete
controlled-root primitive inventories and complex maps remain unavailable,
not zero, because their original positive leaves have no executable bodies.

Concentrated regression passes 31 cases without skips per actual Cirq 1.7.0 /
Qiskit 2.5.0 and Cirq 1.6.1 / Qiskit 2.0.0 profile; real Qualtran 0.7.0 and
Kickmix 0.1 Simulator are asserted. Local integration passes 145 cases. Three
Python files pass strict typing, Black/Ruff, Algorithms/Ecosystem scopes,
English/local links, architecture, IR and protected API checks. An initial test
fixture restored all sys.modules and caused duplicate PyTorch registration; the
final fixture only masks the three SDK entries. No production workaround,
source modification, additional dependency or source-defect waiver is used.
Q-0617 stays open for the retained original physical/serialization boundaries.
