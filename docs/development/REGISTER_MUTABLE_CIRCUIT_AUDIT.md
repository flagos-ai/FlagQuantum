# Mutable raw circuit construction

The locked Kickmix source files are `mutable_circuit.cc` (K-0023,
`5eb19d23b346a670cddc745332e22ea7af81c973`), `mutable_circuit.h` (K-0024,
`de737445df77213457cbba980c7ecb885c65edbb`) and `mutable_circuit.test.cc`
(K-0026, `1e8cd4f636c63a8d1a94de2ea035f98aa5384c4e`).
The native `KmxCircuitBuilder` shares physical columns and parsing with
the KMB codec and freezes to the existing immutable `KmxCircuit`.

The builder accepts one line, multiline text, or a binary stream from
its current position. Stream and line forms accept literal hashes
inside quoted metadata. The string form follows the source's comment
rules. A syntactically invalid ordinary record leaves earlier complete
statements intact. Debug operands expand in order, including register
members and a shared inline guard. For a debug line that fails on a
later operand, earlier recognized operands remain, matching the source.

Append takes operation records from a frozen value or another builder.
Register replacement is explicit and members retain their physical IDs.
Reversal preserves every operand and angle and reverses complete tags
including empty and non-ASCII payloads. It does not take a quantum
adjoint. Copy owns independent mutable columns and registers, and clear
releases them. Existing frozen snapshots retain their contents. C++
arenas, pointer tagging and release mechanics map to owned lists of
physical IDs. The builder rejects malformed column data at the freeze
boundary, and typed tags replace mutable tag offsets. The KMB decoder
continues to validate offsets when reading raw bytes.

| Locked original test | Native evidence |
| --- | --- |
| `parse_each_operation` | Ordered statement matrix, expanded debug operands and empty or nonempty tags |
| `fail_parsing` | All 44 literal syntax refusals and the committed prefix behavior |
| `boundary_values` | Six refused IDs and both admitted sparse width limits |
| `parse_later_fail_validation` | Twelve duplicate-wire cases parse and fail at freeze |
| `fail_validation_size_mismatch` | Underflow for partial one-, two-, three-wire, classical and tag columns; offsets remain covered by the original malformed KMB fixtures |
| `append_from_kmx_file` | Original file text, caller-owned stream and correct position |
| `bulk_parse` | All 37 ordered records, including guards and register slots |
| `append` | Frozen and mutable inputs, reversal, tag content, copy isolation and snapshot lifetime |

All eight unchanged C++ tests passed. The mixed append probe matches
all 14 stored records against the source. An executable forward/reverse
sample appends through `circuit.append(op)` and checks every complex
amplitude with its spectator work wire at zero. Its physical stream
contains two X, two CX and two Z_POW records. The quick CCZ-state model
counts one from the two rotation records, separately from the emitted
gates and any downstream synthesis.

The locked `testdata/big_circuit.kmx` asset (K-0329, blob
`ca59bc2cc04a685c5cd3e431eb3381d72a2d3d2f`) is 3,379,783 bytes with
SHA-256 `c4686af0cf445c413c69c8b27fa8b8788824e6ccdf6ed093d398a62a51e32b5f`.
The builder reads and freezes its 263,405 records to the locked canonical
KMX and the KMB encoder reproduces the locked binary asset byte for byte.
The unchanged asset stays in the reference checkout. The six native and locked source performance measurements are recorded in
[the format benchmark audit](REGISTER_CIRCUIT_FORMAT_BENCHMARKS.md).
