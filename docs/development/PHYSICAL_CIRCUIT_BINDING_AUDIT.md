# Complete physical circuit Python bindings

## Frozen source and ownership

K-0178, K-0179 and K-0184 cover the complete circuit.pybind.cc, its header,
and circuit_test.py from Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`.
All 37 root and recursively quoted dependency blobs match the immutable
inventory. The five unchanged original tests pass in the original extension.
The fixture generator verifies its version, Simulator availability and binary
SHA256 before executing source observations.

Compiler owns the immutable KmxCircuit value, raw ordered records, register
membership, structural transformations and static statistics. Drawer owns text,
HTML and flame rendering. Ecosystem owns filesystem and stream conversion.
Simulation owns numerical execution and actual counters. Binding registration
and the foreign Python class name are retired into these native owners.

## Complete declaration map

| Source declaration | Native owner and disposition |
| --- | --- |
| Constructor and str | read_kmx_circuit and KmxCircuit.to_kmx |
| repr | Native dataclass representation retains the native type and fields |
| register_data | KmxCircuit.register_data projects owned OperandIdentity keys and OperandArray values |
| num_qubits, num_bits, num_registers and len | qubits, bits, len(registers) and len(circuit) |
| Equality and inequality | Existing frozen dataclass structural equality |
| Addition | KmxCircuit addition, shared by concatenate_kmx |
| Multiplication and reflected multiplication | KmxCircuit signed-int64 repetition with negative counts clipped to zero |
| text_diagram | Drawer render_circuit_text |
| html_diagram | Drawer render_circuit_html |
| flame_chart_svg | Drawer render_reversible_phase_analysis_svg with source default reference width 100 |
| operation_statistics | PhysicalOperationStatistics.from_circuit |
| reaction_depth | reaction_depth_for_operations with existing physical record projection |
| from_file and to_file | Ecosystem read_physical_circuit and write_physical_circuit |
| Header registration declaration | Native Python module ownership replaces foreign registration ordering |

Register projections are independent dictionaries of immutable operand arrays.
Unnamed registers use typed register identities. Empty intermediate registers,
ordered mixed quantum/classical members, and named empty registers survive.
Wire zero is the highest-order logical bit in the existing numerical consumers;
physical operand order and identifiers are preserved by these transformations.
No physical wires, scratch ownership or classical feedback storage is created.

Addition uses the left register table when present and otherwise uses the right
one. Repetition retains a single register table even for zero and negative
counts. Both operations retain immutable input ownership. Object repetition
supports Python's integer index protocol and rejects values outside signed
int64 before allocation. The existing repeat_kmx function retains its unsigned
uint64 admission, including the maximum count on an empty circuit. Both routes
share one bounded record repetition implementation. Existing record allocation
limits apply before materializing output.

## Evidence

Seven original values, all 49 ordered additions and 63 signed repetitions are
captured by the unchanged extension. Reflected repetition is also checked.
Each observation includes canonical full KMX, every register key/member,
inferred widths, equality, full text diagram and reaction depth. Each also
executes three lanes with identical seeded input masks, recording every output
quantum/classical bit, exact phase, static opcode counts and actual counters.

The native tests replay all observations, copy ownership, dictionary mutation,
integer-index objects, refusal paths and record-limit failures. Existing full
file, graphical, flame, statistics and register suites cover their respective
owners. A value-composition replacement is inserted into the normal
circuit.append(op) unary example and every accepted small reset/HMR branch
consumer. Those consumers verify full complex amplitudes and clean work.

These operations add zero primitives, measurements or quantum workspace.
Concatenation and positive repetition retain every literal record and feedback
operand in order. Static records and actually activated lane counters remain
separate. This binding boundary has analytical model None.

Native parser diagnostics and native representations are retained. The source
addition docstring incorrectly repeats Z q2 in one displayed result, while the
executed operation produces it once. The fixture follows the actual operation.
The accepted graphical owner retains its documented well-formed XML correction
and empty-analysis coordinate handling. Those source defects remain explicit
in their owner audits. No source throughput or foreign ABI claim is made.

Acceptance receipts are g2-circuit-binding-*20261123 files in the persistent
reference environment. The numerical factory precision intake is a separate
open dependency and does not contribute circuit-binding acceptance.

Local acceptance passes 133 focused scenarios, 3007 concentrated checks and
121 integration checks without skips. Strict mypy passes all three Python
files. Black, Ruff, Compiler scope, architecture, multilevel IR and protected
API checks pass. Dependency findings retain the identical 720-entry baseline.
The complete original fixture regenerates byte-exactly.
