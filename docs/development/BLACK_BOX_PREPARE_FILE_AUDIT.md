# BlackBoxPrepare source-file audit

The reference is unchanged Qualtran v0.7.0 at commit
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| File | Frozen Git blob | Ledger item |
| --- | --- | --- |
| `state_preparation/__init__.py` | `024a57c221ae73f95645dbaa4247f423b037a932` | Q-0690 |
| `black_box_prepare.py` | `42aacba78eb58628d93eee0db860754827d2f42e` | Q-0691 |
| `black_box_prepare_test.py` | `8dc0acab59408715563bfd651249bc516162fea5` | Q-0692 |

Run `python -m examples.algorithms.hubbard_lcu_catalog` for the original
packed 2×2 Hubbard PREPARE. The example uses `circuit.append(op)` to
append the packed preparation alone, with the 10 through wires and two
caller-supplied clean scratch wires.

## Original source and edge cases

The original fixture wraps `PrepareHubbard(2, 2, 1, 4)`. The selection
port packs `U, V, p_x, p_y, alpha, q_x, q_y, beta` in that order. Junk
packs the two-bit `temp` word. The source has one AutoPartition child.
The native wrapper preserves the packed order and normalization across
both directions. It emits the same gate stream as the named binding.
The named Hubbard test checks the exact source parameters, underlying
gate sequence, operands and measurement feedback.

The packed source was simulated with its original operations. The
undecomposable ModAddK leaf uses its source classical permutation as an
explicit matrix. All eight native measurement histories match the full
complex source state and restore both scratch wires. The inverse
restores the all-zero input on those histories.

The original dummy oracle packs its named one-bit selection and
five-bit junk. The one-bit public signature uses bit encoding, while
its group descriptor stays opaque. The tests check packed port names,
order, bind-time operand routing, and absent normalization for an
emitter without coefficient data. Zero-width descriptors are retained
and omitted from the public signature. A zero-width selection returns
the other bound ports without invoking the child. A nonempty selection
and zero junk binds only the selection child. The original suite
reported four passes, one decomposition-only count skip, and one
expected failure for serialization.

| 2×2 Hubbard body | T/T† | CX | Measurements | Declared resource categories | Wires |
| --- | ---: | ---: | ---: | --- | --- |
| Native packed, either direction | 47 | 66 | 3 | — | 12 reserved |
| Source model | — | — | 2 | 5 CSwap, 2 And, 11 Clifford, 4 rotations | 12 logical |

## Typed and symbolic packing

`PackedPrepareDeclaration` reads the ordered register descriptors from
the native PREPARE contract. Its widths sum each descriptor's total
bits, including array shape. It preserves the source's opaque group
descriptors, zero-width filtering and one-bit top-level encoding.
A shaped selection of width `n` twice plus a one-bit register declares
`2*n + 1` selection bits. Three width-`n` junk elements declare `3*n`
junk bits. The coefficient norm passes through unchanged in metadata,
including when it is symbolic. A missing norm raises the source exception
on access and is represented by `None` on a bound emitter.

The named Hubbard consumer derives its concrete layout from the same
declaration. The shaped fixture verifies ordered binding on reversed
physical wires through `circuit.append(op)`. Existing Hubbard
and alias tests cover emitted gates, measurement histories, complex
amplitudes, scratch restoration and source resource estimates.

## Package and remaining boundaries

The four exports in Q-0690 point to the following native owners.
Their individual source-file acceptance remains tracked under their
own parent items.

| Source export | Native owner |
| --- | --- |
| `PrepareUniformSuperposition` | `compiler/registers/preparation/uniform_preparation.py` |
| `StatePreparationAliasSampling` | `algorithms/primitives/alias_preparation.py`, dense table |
| `SparseStatePreparationAliasSampling` | `algorithms/primitives/alias_preparation.py`, sparse table |
| `StatePreparationViaRotations` | `algorithms/primitives/rotation_preparation.py`, rotation program |

Q-0690 and Q-0692 are covered. Q-0691 retains the original object-serialization
failure recorded below. External import, owned execution-artifact round trips
and explicitly resolved symbolic bindings are now exercised. An unresolved
symbolic width remains declaration-only rather than silently coerced.

## Complete definition and dependency map

| Source member | Native owner and acceptance |
| --- | --- |
| `selection_registers`, `junk_registers` | `PackedPrepareDeclaration` preserves opaque packed group descriptors; existing tests cover one-bit top-level encoding and zero-width omission. |
| `selection_bitsize`, `junk_bitsize` | Ordered sums of descriptor total bits, including array shapes and symbolic dimensions; concrete layouts are resolved before binding. |
| `l1_norm_of_coeffs` | Original normalization expression passes through; absent source normalization retains its exception and a bound emitter records unavailable normalization. |
| `signature` | Packed selection followed by junk, distinct from the child's named order; the full frozen Hubbard factory checks 8 selection and 2 logical through-junk bits. |
| `build_composite_bloq` | Existing AutoPartition mapping and zero-selection early return; the actual external packed source is captured through `import_register_emitter`, not replaced with a dense matrix. |
| `__str__` | `PackedPrepareDeclaration` and the original return `Prep`. |
| `_black_box_prepare`, `_BLACK_BOX_PREPARE_DOC` | Actual original Hubbard object, its documented example and child partition mapping are checked directly. |

### Actual external object replacement

`test_original_packed_object_roundtrip_replaces_native_hubbard` captures both
directions of the original packed object, serializes only owned CircuitIR, and
rebinds the restored emitter through the existing BlackBoxPrepare callback.
The user still appends its operation to a Circuit. No SDK object survives in
the restored execution snapshots. Every one of the eight measurement histories
is compared with the existing native Hubbard recipe on a coherent arbitrary
state over all ten logical bits, including arbitrary logical junk. A separate
all-history certificate proves private workspace restoration.

| Execution route | Wires / private work | Instructions | T/T-dagger | CX | Measurements |
| --- | --- | ---: | ---: | ---: | ---: |
| Actual external packed object, either direction | 14 / 4 | 170 | 47 | 66 | 3 |
| Existing native packed Hubbard | 12 / 2 | Existing recipe | 47 | 66 | 3 |
| Independent unchanged source model | 12 logical peak | Not a primitive count | 5 CSwap + 2 And, not expanded here | 11 Clifford; 4 rotations | 2 |

Distinct private allocator reservations are not claimed to be identical
physical widths. No unused source allocation is silently removed to match the
native count. Full output comparisons include zero amplitude outside clean work.

The accepted PrepareOracle symbolic fixture now also wraps the resolved
external child in the actual source BlackBoxPrepare/AutoPartition chain.
Widths 1, 2 and 3 in both directions pass full complex operator comparisons,
IR JSON restoration, instruction equality and normalization checks. Symbolic
width and angle are resolved explicitly through immutable constructor values;
the unbound external packed object is rejected before execution. The existing
shaped-symbolic descriptor and zero-selection tests remain part of this closure.

### Remaining original serialization failure

The unchanged `_black_box_prepare` serialization check still reports FAIL:
NumPy raises ValueError for an inhomogeneous array shape `(2, 2)`. This is an
original object-serialization failure, separate from the successful native
execution-artifact round trip. It is neither patched in the reference nor
reported as parity. Under the frozen parent closure criteria Q-0691 remains
open for that exact boundary; unresolved symbolic execution is no longer
misreported as a missing concrete implementation.

The focused source/dependency suite passes 25 cases (3.75 s). The concentrated
packed PREPARE, named/directed LCU and register-import regression passes 107
cases (43.39 s); the final serialization assertion is included in the focused
run. Both changed test files pass Black and Ruff and their Algorithms scope.
No production behavior changes in this batch: the previously accepted 169-file
strict result, architecture/IR/API checks and 60 CPU/representation cases are
reused from the immediately preceding scalar dependency batch.


## Faithful full-file acceptance (2026-11-28)

The approved faithful-migration closure rule supersedes the historical open
status above. Q-0691 covers the complete locked source file and its direct
AutoPartition/PrepareOracle dependencies. The original object-serialization
failure remains an upstream defect, separately from native IR restoration.

One executable path needed a production fix: a selection word with physical
width `sympy.Integer(3)` passes the unchanged source SDK decomposition, although
its tensor protocol rejects symbolic Partition. Native packed binding now uses
the existing Core executable-width projection under approved Proposal 130,
retaining the original dtype metadata. This is raw physical storage admission;
it does not resolve a free width or interpret symbolic numeric metadata.
Array dimensions still pass through the existing concrete-shape contract.

`test_packed_literal_storage.py` checks Python and exact SymPy integer widths,
concrete shaped words, selection/junk ordering, reversed physical wire maps,
both directions and a coherent spectator through `circuit.append(op)`. The
explicitly specified identity child is compared with the unchanged source SDK
operator, including full complex output and IR restoration. Its measured
primitive, measurement and private-work counts are all zero; this fixture is
not a resource model for Hubbard PREPARE. Free widths and symbolic array
extents remain rejected. Existing Hubbard full-history and resource evidence
above remains applicable and is rerun in the batch regression.

Unknown child semantics still require an explicit owned body and inverse.
No child role or adjoint is inferred by name, and logical junk is not disposed
of as private workspace. The zero-selection source early return is retained.


Batch verification: 108 concentrated source/execution cases, 30 independent
Cirq 1.6 cases and 32 local integration cases pass. The two Python files pass
strict typing, Black and Ruff; Algorithms scope, architecture, IR, protected
API baseline, required checks, language, links and whitespace pass. All 42
frozen source/dependency blobs are unchanged. Fourteen actual dependency-absence
cases distinguish optional skips from mandatory propagation of the original
error. The dependency-policy scan still has 570 existing findings in 362 files;
none belongs to the new test consumer. That workflow remains open.
