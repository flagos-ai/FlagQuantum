# Complete Python integer-table conversion utility

## Scope and ownership

K-0183 covers all 40 lines of circuit_builder_util.pybind.cc at immutable
Kickmix revision `9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. The complete
utility, its two lookup callers and their recursively quoted source dependency
closure are verified against 43 inventory Git blobs.

The utility has one definition, read_python_int_list_into_table_data. Native
Compiler lookup_binding already owns Python integer conversion, table shape
admission and the immutable logical row representation. The utility is mapped
into that owner. Existing recursive_lookup and measured_unlookup own physical
operations and component calls. Existing lowering and Simulation own execution.
There is no additional scalar type, table hierarchy or numerical executor.

Source entries and source target words are LSB-first. Native address/target
ports are MSB-first. The existing caller reverses physical word order once at
its boundary. The shared split_integer encoding retains native row order, so
no extra reversal or padded byte becomes a logical target bit. The caller owns
target/workspace. Conversion creates no physical wires or measurements.

## Sequential conversion repair

The previous native implementation converted every entry before checking any
range. A first entry -1 followed by a custom integer conversion that raises
therefore surfaced the later exception. The unchanged source rejects table[0]
immediately and never evaluates the later entry.

The native owner now indexes, converts, validates and encodes each entry before
reading the next one. Source ValueError diagnostics retain the original entry
index, converted integer and target width. Length is read once for iteration;
shape validation uses the resulting immutable rows. Zero-width targets still
convert and validate every entry, preserving arbitrary-length all-zero tables.
Invalid construction returns no partially mutated Program. Existing native
integer normalization and shared big-endian integer encoding are retained.

## Evidence and resources

42 unchanged-source forward/joined caller circuits cover target widths
0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128 and 129.
One-address-bit cases additionally cover widths 9, 17, 65 and 129. Every
complete marked gate stream, operand, feedback slot and static histogram is
compared literally. Actual three-lane executions compare every physical output,
exact lane phase and actual counters. All work wires are zero. Joined callers
also return every target wire to zero. Wide runs use the existing packed
numerical owner. No exponential dense-state or exhaustive wide-history claim
is inferred from those runs.

Five source conversion observations record callback order and exact diagnostics.
Existing accepted 1030 forward/erase/joined marked-stream receipts, normal
circuit.append(op) caller, complex address/spectator states and exhaustive
small reset/HMR branch checks are rerun. Their established workspace ownership
and zero-support assertions are reused.

Conversion itself adds zero primitives, measurements or quantum workspace.
Caller static histograms and actually activated three-lane counters are separate
fixture fields. The source utility has no analytical resource model.

## Original subclass boundary and follow-up closure

### Follow-up contract before implementation

The next complete utility closure retains an int subclass before invoking its
dynamic bit_length, negativity comparison and to_bytes methods in source order.
Other values retain the original int conversion. Encoding consumes exactly the
requested low little-endian bytes and low target-width bits. Extra bytes and
high padding bits do not enter the table. Zero-width rows still perform all
three callbacks. Source-defined callback exceptions propagate in order.

Fresh unchanged-source observations cover encoded zero/one, extra bytes,
padding, underreported width, negative comparison override, zero-width encoding
and callback failures. Existing full ordinary-value, phase, work-zero and
small complex-history evidence will be rerun. The source reads short returned
buffers without checking their size. Native code must reject those buffers
before reading them. Such source-undefined memory access is not a numerical
reference observation. No source modification or new execution owner is used.

The source keeps Python int subclasses and dynamically calls their bit_length
and to_bytes methods. Before this follow-up, native lookup_binding normalized
entries with Python int, removing those subclass overrides. An unchanged-source subclass
with value 1 and an overridden bit_length returning 100 is rejected for a
one-bit target. The earlier native owner accepted its normalized integer value
1. The original fixture preserved both outcomes without claiming equivalence.

The initial sequential failure-order repair left K-0183 open for this
subclass admission/encoding boundary. Source lookup work shortages, partial
source allocator mutation on failure and dense address portability retain
their separate K-0232 dispositions. All source references are unchanged.

Local acceptance passes 49 focused utility scenarios, 3114 concentrated checks
and 121 integration checks without skips. All 33 unchanged original lookup
caller tests pass. Strict typing passes the three changed Python files. Black,
Ruff, Compiler scope, architecture, multilevel IR and protected API checks pass.
The dependency finding set remains the identical 720-entry baseline. Fixture
regeneration is byte-exact. This accepts the sequential conversion repair and
its stated ordinary-value scope while leaving K-0183 open at that time.

### Complete dynamic conversion delivery

The follow-up preserves integer subclass identity through bit_length,
negativity comparison and to_bytes. It also preserves integer-protocol results
returned by bit_length. Allocation counts and bit-length casts share one
Compiler-owned unsigned_size conversion. This relocates the previous
allocation-only helper without duplicating its conversion rules. Existing
builder protocol tests run against this replacement.

The byte decoder uses the bytes buffer directly, independent of Python subclass
indexing or reported length, and consumes exactly the low requested bytes and
target bits. Full forward and joined erasure streams preserve source callbacks,
operands, metadata and feedback. All 16 conversion observations retain original
errors or their complete successful streams. Successful cases also execute both
forward and joined streams through three actual lanes with exact phases and
separate static/actual counters. Eight one-bit encoding cases enumerate every
reset/HMR history on complex spectator inputs, checking exact signed
unnormalized amplitudes and zero target/work support. The existing wider
ordinary-table corpus and small arbitrary-address branch evidence are reused.

The earlier subclass mismatch is closed. Native malformed-result diagnostics
remain native. An encoder must return a bytes buffer with enough storage for
the requested width. The original unchecked short-buffer read is excluded
from numerical evidence and explicitly rejected. Source wrapper work shortages,
post-failure allocation mutation and dense-address portability retain their
separate K-0232/K-0180 dispositions.

The complete utility parent K-0183 is accepted. All 43 frozen source blobs and
byte-exact fixture regeneration pass. The final corpus has 42 wide ordinary
callers and 16 callback observations. Local checks pass 69 focused table cases,
4541 concentrated cases and 121 integration cases without skips. Strict typing
passes nine source files. After the Python 3.10-compatible NotRequired import
correction, 95 focused table/builder checks and strict typing pass again. Black,
Ruff, Compiler scope, architecture, multilevel IR and API checks pass, with the
identical 720-entry dependency finding baseline. The conversion adds no primitive,
measurement or quantum workspace. Static source counts and executed lane counts
remain separate, and no analytical resource model is claimed. Acceptance evidence
is recorded in g2-table-subclass-acceptance-20261123.json beside the worktree.
