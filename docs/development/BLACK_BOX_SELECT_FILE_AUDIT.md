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

Q-0588 remains open for the locked controlled-packing defect. The locked wrapper exposes child control metadata but omits
control from its packed signature. A controlled Hubbard child fails with
`Bloq registers do not match given partitions`. The native packed
binder diagnoses this before binding. The existing named SELECT path
continues to own controlled execution.

Q-0590 retains typed register roles in the concrete owners. The toy and
Hubbard cases verify their original control, selection, and target order.
The general foreign `SelectOracle` subclass protocol and resolved symbolic
metadata are now consumed by the grouped-object binding below.

## Complete interface and packed external-object binding

The complete `SelectOracle` base, `BlackBoxSelect`, both documentation records,
the three original interface properties, the cached source signature, the packed
properties, size calculations, string value, AutoPartition body, example factory
and original tests were read. The source files match their frozen v0.7.0 blobs.
The existing Hubbard factory/decomposition and AutoPartition evidence above is
reused rather than recounted.

The native `SelectOracle` declaration preserves control, selection and target
descriptor order. `PackedSelectDeclaration` keeps the packed source metadata and
control descriptors, uses `packed_select_layout` for the opaque packed signature,
and requires FlagQuantum-owned concrete quantum through descriptors. The existing
layout owner validates direction, classical content and unique port names. Arrays
and named groups flatten in source order. Physical bits within each packed word
remain big-endian, with address fields ahead of the system word. `bind` hands the
captured emitter to the existing `SelectBlockEncoding` and identity PREPARE
consumer. Caller-owned clean work and feedback slots remain with that consumer.
The external SDK object is converted to circuit snapshots at the Ecosystem
boundary. Native declarations and their register types are separate objects,
including when symbolic dimensions are resolved. No new package export or public
API was added.

`test_select_oracle_binding.py` resolves a symbolic-width external subclass and
replaces its actual source `BlackBoxSelect` object through the same native
`circuit.append(op)` consumer. Width cases use one/two address bits and one/two
system bits. Both source directions match exact instruction dependencies,
operands, JSON IR, complete complex matrices, and every basis transition. There
are no measurements or workspace wires. Literal streams contain 2 or 3 CX,
according to address width. The independent source model reports the same live
logical width and its Clifford categories remain distinct from the emitted CX
inventory. The abstract base member set, documentation import, grouped signature,
unresolved-shape rejection, and original control/selection/target order are
checked. The original controlled Hubbard wrapper still advertises its control
separately while omitting it from its packed signature, so physical controlled
packing remains refused under Q-0588.

The focused dependency run passes all eight interface, wrapper and binding tests.
The concentrated regression and separate strict type, scope, format,
architecture, multi-level IR and API checks are recorded in the batch handoff.

Strict typing passes on the new layout, fixture and file audit. The
shared reference reader retains its baseline dynamic Cirq and metadata
typing diagnostics.

## Native declaration boundary review

The grouped binding tests construct an actual native `SelectOracle` subclass,
independently of the foreign source subclass. The packed binder reuses the
existing layout validator and builds the concrete layouts in one traversal.
An additional source subclass uses genuine two-element array ports and an S
phase so its adjoint differs from its forward operation. Both imported
directions pass through the native declaration, packed consumer, append entry,
and JSON roundtrip and match the source wrapper's complete complex matrices.
RIGHT-only and classical target descriptors fail at binding, as do unresolved
widths. Control metadata remains preserved and controlled packing remains
refused under the original source defect.

## Faithful whole-file disposition, November 28

The earlier open-parent statement used the previous source-defect rule. Under
the approved [G2 closure rule](G2_FAITHFUL_MIGRATION_CLOSURE_PROPOSAL.md),
Q-0588 can close after whole-file and direct-dependency verification. All source
properties, grouped signatures, width sums, AutoPartition composition, example
and documentation have native owners. The abstract oracle protocol, generic
array packing and symbolic-to-concrete handoff retain their existing evidence.

The controlled wrapper's omitted control port is a reproduced upstream defect,
not a missing valid packing algorithm. Keep its actual ValueError and native
early refusal as a packing-correction follow-up; named controlled SELECT remains
separately executable. No controlled packed execution is claimed.

A fresh boundary intake verifies the successful empty-system child. Existing
packing already supports it: the two signed empty rows produce Z on the one
selection wire, with two X instructions and one Z, no work and no measurement.
The complete source operator, both native directions, ordinary append and JSON
roundtrip are now checked in test_select_oracle_binding.py. The source's live
logical width and source gate model remain separate from this literal inventory.
The source controlled empty-system child still fails the same partition check.
No new execution implementation or protected behavior was necessary.

Final batch evidence: 213 concentrated cases and 213 independent Cirq 1.6
cases pass, including the empty-system packed consumer. The two unchanged
original files report 14 passes, one original skip and two original expected
failures; their notebook case is deselected here and reuses the existing complete
notebook receipt above. The unchanged integration boundary reuses this head's
248 passing cases. Seven Python owners/tests pass strict typing, Black and Ruff;
architecture, IR, API, required/Cirq contracts, scope, English and links pass.
The changed test's actual missing Cirq, NumPy, Qualtran and Kickmix imports are
checked in both optional and mandatory modes (eight cases). Optional runs skip
explicitly; mandatory runs preserve the original missing-dependency exception.
Two preexisting unguarded imports are repaired, reducing dependency findings
from 570 to 568, with none on the changed test. Source fixtures retain frozen
blobs and actual Qualtran 0.7.0 / Kickmix 0.1 Simulator assertions. Receipts use
prefix g2-select-source-closure; implementation owners remain unchanged.
