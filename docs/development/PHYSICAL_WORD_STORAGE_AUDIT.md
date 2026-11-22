# Physical word storage

## Complete source scope and ownership

This batch closes K-0168 through K-0177, the complete locked
`mem/monotonic_arena.cc`, `.h`, `.test.cc`, `mem/stride_ptr.h`, `.test.cc`,
`mem/stride_span.h`, `.test.cc`, and `mem/util.cc`, `.h`, `.test.cc` files.
The source lock is `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
The quoted include closure contains eleven frozen Git blobs, including
`build/iota.h`. The fixture generator verifies every blob against the original
inventory before compiling unchanged implementation files.

Compiler `artifacts._word_storage` owns aligned bytes, physical integer pointers,
strided spans, repeated byte copies and unsigned-32 maxima.
`artifacts._word_arena` owns append-only chunks, growth, tail reservations,
rewinding, copying and transfer. Existing `formats.kmb._Columns` uses these
arenas for its ten operand columns. Format readers, mutable circuit builders,
serialization and normal `circuit.append(op)` consumers retain their existing
entry points. Simulation continues to own all numerical execution.
Immutable `OperandArray` remains the authoritative typed operand view.

## Interface mapping

| Original operation | Native representation |
| --- | --- |
| Aligned typed allocation and freeing | `AlignedBytes`, retained padded bytearray, Python lifetime |
| `stride_ptr` dereference and signed indexing | `WordPointer[0]`, `WordPointer[index]` |
| Addition, subtraction, compound displacement | Pointer arithmetic in element units |
| Prefix and postfix increment | `advance`, `advance_after` |
| Pointer equality | Raw address equality, independent of stride and interpretation |
| Span pointer, count, size, empty, first and last | `pointer`, `count`, `len`, Boolean length, indexing |
| Span checked access | `at`, with logical bounds |
| Span unchecked access | Indexing, with allocation bounds |
| Span iterator and end sentinel | Counted Python iterator, including zero stride |
| Span casts and iota indexing | `cast_words`, `indexed_pointer` |
| Reverse, skips, keeps and subspan | Same transforms with source clamping and empty-pointer rule |
| Span stream output | `diagnostic_text`, native class name and actual address |
| Arena size, iteration, chunk callback | `len`, iteration, `iter_spans` |
| Reservation and tail return | `grab_writeable`, `rewind_tail` |
| Single, repeated, span and arena appends | `append`, `append_repeat`, `extend` |
| Reversed span and arena append | `append_reversed`, preserving source chunk traversal |
| Contiguous raw copy | `copy_into`, independent of destination view stride |
| Deleted copy and move ownership | Copy refusal, `move`, `move_from`, self-move preservation |
| Clear with largest allocation retained | `clear` |

Growth uses the original greater-than-or-equal spill boundary, effective pointer
alignment, doubled maximum of previous capacity, requested bytes and four
alignment units. Frozen chunk sizes, used bytes, capacities, retained current
allocation, alignment and reversal order are compared directly with the source.

## Evidence and replacement

All nineteen unchanged original GoogleTests pass. The generated source corpus
covers signed strides -2 through 2, zero/one/four element spans, clamping beyond
the range, signed displacement, prefix/postfix mutation, alias equality, casts,
iota, checked refusals, byte repetition, maxima and overflow allocation.
Arena probes cover one-, two-, four- and eight-byte words, multiple alignments,
chunk spilling, copying, reverse copying, reserve/rewind/reuse, move and clear.
Regeneration must reproduce `tests/unit/data/memory_support.json` byte for byte.

Native tests additionally exercise signed and unsigned integer storage at
1/2/4/8/16 bytes, allocation failure, invalid extension preservation, self-append,
copy refusal, retained pointer lifetime and bounded dereference. Both new modules
have 100% measured statement and branch coverage in the concentrated regression.
This coverage describes those two modules only.

The actual builder factory is replaced between `WordArena` and its previous
`list` implementation without changing consumers. Both implementations pass
large spill/copy/reverse/clear and KMB round trips. The sixteen frozen builder
consumer cases are reused under both storage implementations. Each checks
ordered raw operations, KMX round trips, the full complex five-qubit state,
classical feedback, zero workspace, all 32 exact basis phases and per-lane actual
counters. Existing broadcast measurement, QROAM erasure and coherent/trajectory
integration paths run in the same batch. The batch passes 707 concentrated and
121 integration checks, with no skipped cases, and strict typing on all five
changed Python files.

## Resource accounting

| Quantity | Observation |
| --- | --- |
| Storage operations emitted primitives | 0 |
| Storage operations measurements and quantum workspace | 0 |
| Builder replacement ordered operations | Exact equality with the frozen source cases |
| Consumer actual per-lane primitive counters | Exact source equality, including conditional activation |
| Analytical resource model introduced by this batch | None |

Physical storage allocation is host memory. It contributes no quantum gates or
logical workspace. Existing analytical estimates retain their separate owner.

## Native adaptations and remaining boundaries

Python allocations are zero-initialized. Source uninitialized storage is never
read to manufacture an observation. Native pointers retain allocation owners
through arena clear and move. Source pointers can dangle after owner release.
Native dereferences enforce allocation bounds. Arbitrary invalid pointer use,
undefined overlap, uninitialized values and C++ object lifetime violations are
outside the compared domain.

Allocations occur before ownership mutation, preserving the previous arena on
allocation failure. The original helper can mutate before allocation fails.
The native `MemoryError`, `IndexError` and `ValueError` categories follow the
Python internal ownership boundary. Source alignment requirements are validated
at runtime. Const templates map to Python ownership conventions. There is no
claim of arbitrary C++ object construction/destruction or floating-point typed
pointer coverage. Actual format columns are unsigned 32-bit words. Additional
native signed/128-bit boundary checks are distinguished from original typed
arena observations.

Raw word storage follows host byte order. Existing KMB encoders continue to own
canonical wire byte order. Raw addresses differ between processes. Diagnostics
compare their numeric address and contents within the corresponding process.

A local four-sample construction/freeze/KMB measurement for 1000 CX records gave
median 2.874 ms for the previous lists and 7.853 ms for aligned arenas, about
2.73 times the construction/serialization time. Both emitted 1000 records and
9288 bytes. The retained receipt is
`ftqc-reference-env/g2-memory-support-local-storage-timing-20261122.json` outside
the checkout. This measured Python storage overhead remains an optimization
gap. It establishes no execution-kernel performance claim.

## Contributor path

Run `tests/unit/test_physical_word_storage.py` with the established locked test
environment. For storage changes, include existing KMX/KMB, mutable builder,
physical counter, broadcast measurement and QROAM erasure tests. Regenerate the
fixture with `tests/unit/fixtures/generate_memory_support.py` using the original
source directory and immutable CSV inventory. Run strict typing, format, scope,
architecture, IR and public API checks before publishing.
