# Black-box SELECT and SELECT interface source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `black_box_select.py` | `8cb8d402b17f17952aa3f06b3fca8ed4cc048239` | Q-0588 |
| `black_box_select_test.py` | `6bd3dd68d3adef743a4ae6990180951c40d928d0` | Q-0589 |
| `select_base.py` | `dc72bbc1a41b7cf499b1273f1269984289c48672` | Q-0590 |

## Packed binding

`packed_select_layout` returns the two opaque through ports and the
existing AutoPartition binder. Each word takes its declared register
group in order. Array ports occupy their row-major flattened span.
Symbolic dimensions are kept in the declaration. Physical binding
requires concrete widths and quantum through ports.

The locked 2×2 Hubbard example calls one AutoPartition with the original
eight one-bit selection registers and eight-bit target. The
[native example](../../examples/algorithms/black_box_select_catalog.py)
binds the corresponding two eight-bit packed words and appends the
existing SELECT operation. Its gate stream is identical to the native
named-port entry. Run `python -m examples.algorithms.black_box_select_catalog`
to report 19 wires, 102 T/T†, and 15 measurements. The original toy oracle case exposes the one-bit
selection and five-bit system, with a reversible explicit wire binding.

## Execution and resources

The primitive comparison reads the unchanged Hubbard child with explicit
bindings. The Majorana stages allocate two predicate wires and one
parity carrier. Their different temporary allocation order is reconciled
by a bijection on the three work wires before the final spin-up traversal.
Classical feedback is compared by the ordinal of the preceding measurement
that wrote each slot. The original scalar `-i` is represented by two
phase instructions interleaved with two X instructions on a live anchor,
preserving global phase. The comparison retains each gate's operands,
angles and dependencies on its wires.

A coherent input spans all 256 selection labels with two system basis
vectors per selector. Four normalized raw measurement histories compare
all complex output amplitudes and require source and native work bits
to be zero. Existing SELECT tests cover controlled, adjoint and other
Hubbard sizes through their named interfaces.

| Fixture | Emitted primitives | Declared source model |
| --- | --- | --- |
| Uncontrolled 2×2 Hubbard | 102 T/T†, 177 CX, 15 measurements | 6 CSwap, 15 And, 23 Cliffords, 15 measurements |
| Width | 19 reserved, including 3 clean work wires | 19 peak live wires |

The default model gives 84 T equivalents. Charging seven T per exact CSwap
gives 102. The abstract Clifford total and the measured primitive CX
count describe different inventories.

## Remaining source-file boundaries

Q-0589 is accepted. The original suite had four passes, one count skip
because the wrapper's graph comes from decomposition, and one expected
serialization failure. The native audit reads the actual partition
and compares its executable child.

Q-0588 remains open for controlled packing and arbitrary foreign SELECT
objects. The locked wrapper exposes child control metadata but omits
control from its packed signature. A controlled Hubbard child fails with
`Bloq registers do not match given partitions`. The native packed
binder diagnoses this before binding. The existing named SELECT path
continues to own controlled execution.

Q-0590 retains typed register roles in the concrete owners. The toy and
Hubbard cases verify their original control, selection, and target order.
The general foreign SelectOracle subclass protocol remains open.

Strict typing passes on the new layout, fixture and file audit. The
shared reference reader retains its baseline dynamic Cirq and metadata
typing diagnostics.
