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

## Explicit remaining subclass boundary

The source keeps Python int subclasses and dynamically calls their bit_length
and to_bytes methods. Native lookup_binding normalizes entries with Python int,
which removes those subclass overrides. An actual unchanged-source subclass
with value 1 and an overridden bit_length returning 100 is rejected for a
one-bit target. The native owner accepts its normalized integer value 1. The
fixture preserves both outcomes and does not claim their equivalence.

K-0183 remains open for this existing subclass admission/encoding boundary.
The sequential failure-order repair and complete ordinary integer/caller
coverage do not close that parent. Source lookup work shortages, partial
source allocator mutation on failure and dense address portability retain
their separate K-0232 dispositions. All source references are unchanged.

Local acceptance passes 49 focused utility scenarios, 3114 concentrated checks
and 121 integration checks without skips. All 33 unchanged original lookup
caller tests pass. Strict typing passes the three changed Python files. Black,
Ruff, Compiler scope, architecture, multilevel IR and protected API checks pass.
The dependency finding set remains the identical 720-entry baseline. Fixture
regeneration is byte-exact. This accepts the sequential conversion repair and
its stated ordinary-value scope while leaving K-0183 open.
