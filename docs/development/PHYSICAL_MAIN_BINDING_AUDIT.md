# Complete physical Python main registration mapping

## Scope and ownership before implementation

This group covers K-0243, K-0244 and K-0245 at locked Kickmix
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`. Its main registration source has
204 lines, its header is empty, and its original test file has 284 lines.
All 63 source/dependency blobs, including the complete circuit file-binding
implementation, match the immutable inventory. All five unchanged main tests
pass in the locked extension. The extension version, Simulator availability and
binary SHA256 are verified before generating observations.

Registration is distributed to existing native owners. Ecosystem owns Python
paths, streams and external conversion. Compiler owns immutable physical records,
format parsing, serialization, builder transformations and static instruction
statistics. Simulation owns persistent numerical state and actual counters.
Core owns mathematical field values. Private native representations keep these
boundaries. Existing `circuit.append(op)` construction and lowering supply the
execution path.

Caller-supplied streams retain position and lifetime ownership. Reading starts
at the current cursor and consumes the remaining payload. Writing starts at the
current cursor, leaves the stream open and passes text only to a TextIOBase for
KMX. Binary serialization requires a binary writer. Path inputs use the existing
Python filesystem protocol, including bytes paths. Format validation precedes
I/O. No quantum state or workspace transfers during file conversion.

## Complete registration map

| Source declaration or registration | Native owner |
| --- | --- |
| Module version, docstring, `PYBIND11_MODULE` | FlagQuantum package identity and owning module documentation. Foreign module registration is retired. Reference version remains oracle metadata. |
| Circuit class and methods | Compiler `formats.kmx_circuit.KmxCircuit`, existing KMX/KMB parsers, immutable register values, circuit transforms, drawer renderers and resource inspection. |
| CircuitBuilder and builder marks | Compiler `formats.kmx_builder.KmxCircuitBuilder`, existing register/component construction, `ComponentCall` and KMX block marks. |
| array, q, b, r, xb, xbool classes and methods | Existing `OperandArray`, `OperandIdentity`/`OperandKind`, operand views, tagged scalar values and `primitives.operand_binding`. No foreign values enter execution. |
| GF2 field class and methods | Core `_binary_field.BinaryField`, Ecosystem `binary_fields` conversion and existing Compiler field generators. |
| UnaryIterationCursor class and methods | Compiler `_unary_iteration_session.UnaryIterationPass`, accepted physical cursor and shared clean-work leases. |
| OperationStatistics class and methods | Compiler `_physical_operation_statistics.PhysicalOperationStatistics` and Simulation actual counter snapshots. |
| Simulator class and methods | Ecosystem `_physical_basis_session.PhysicalBasisSession` around Simulation `_basis_trajectories.BasisTrajectories`. |
| Six `PYBIND_TYPE_PTR` globals | Existing typed operand and array dispatch replaces extension-global raw type pointers. |
| Empty main header | No declarations or executable behavior to port. |
| Main tests | Existing consumers plus source-generated native file, statistics and numerical composition tests. |

The registration file contains no numerical implementation. Existing dependent
source implementations retain their independent ledger entries and documented
boundaries. Closing registration does not close comparator source crashes,
lookup source work shortfalls, generic builder wrappers or SIMD performance.

## Remaining file-boundary implementation

`ecosystem._physical_circuit_files` composes existing Compiler parsers and
serializers for the original `from_file` and `to_file` behaviors. Private
functions accept Python paths and file-like objects and return native circuit
values. No stable export or schema is changed.

Text-returning input streams use the existing string constructor grammar.
Byte-returning streams and paths use the existing file grammar, preserving the
source distinction for hashes inside quoted fields. Automatic KMX/KMB detection
uses the accepted binary magic decoder. An explicit KMB request rejects text
streams. An explicit KMX request validates text decoding before detection and
rejects KMB bytes. Empty KMX output is zero bytes. Nonempty canonical KMX ends in
one newline. Exact KMB bytes are emitted by the existing encoder.

Python-owned memory replaces temporary C FILE buffers. Path-opening failures
retain ValueError with the source reading/writing diagnostic category. Existing
Compiler parse diagnostics and native class representations are retained. Raw
C pointers, extension class initialization ordering and foreign repr strings
have no runtime parity claim. Custom stream exceptions propagate. Like the
source, custom writer return counts are ignored.

## Evidence and resource accounting

The byte-exact fixture contains five full source circuits and six stream-grammar
probes. It retains complete canonical records, literal KMX/KMB payloads, grouped
physical opcode counts and exact rotation hierarchy projections. The corpus
covers empty circuits, the original main example, measured feedback, conditional
Toffoli/swap operations, arbitrary exact rotations, debug records and metadata.
Native scenarios exercise pathlib, str and bytes paths, opened text/binary files,
StringIO and BytesIO, explicit and detected formats, wrong formats, invalid paths
and failed format validation before output creation.

Twelve existing full-complex, all-reset/all-HMR branch consumers execute with
KMB file round trips inserted at the physical lowering boundary. Their original
assertions and work-zero checks are reused. The normal unary traversal example
runs with the same boundary substitution and its existing `circuit.append(op)`
call. The original main numerical example also executes on the native persistent
session and reads its target before and after X/CX.

File adaptation adds zero primitives, measurements or quantum workspace. Actual
static instruction statistics remain identical to the frozen source. Actual
executed lane counters retain their separate Simulation owner. Source analytical
resource estimates remain separately owned and are not inferred from file
conversion. This boundary has analytical model None.

The five original source tests remain distinct from the native scenarios.
Original source files are unchanged. Batch regression, strict typing, architecture
and dependency receipts are retained under
`ftqc-reference-env/g2-main-binding-*20261123.*` beside the worktree.

Local acceptance passes 1458 concentrated tests and 121 integration tests.
Strict typing passes all three Python files. The production file boundary has
100 percent statement and branch coverage. Black, Ruff, Ecosystem scope,
architecture, multilevel IR and protected API checks pass. The dependency
finding set remains the identical 720-entry baseline.
