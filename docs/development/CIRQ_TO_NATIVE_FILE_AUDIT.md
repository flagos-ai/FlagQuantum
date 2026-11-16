# Cirq-to-native conversion source-family audit

This work follows Qualtran v0.7.0 at
`bcb17bb25360ed68ce3bc084645db3e929799389`.
Q-0717 and Q-0718 track `cirq_interop/_cirq_to_bloq.py` and its original tests.

| Source responsibility | Native evidence and remaining boundary |
| --- | --- |
| Gate recognition and resource estimates | `resource_keys.py` covers exact gates, controlled binary clauses, parameterized powers and the admitted inverse leaves. Ecosystem owns opaque SDK values. |
| One-step decomposition and qubit manager | `_decompose.py` and `_qubit_manager.py` preserve protocol order and distinguish no decomposition from an empty decomposition. Native register import uses declared clean lifetimes for admitted operands and managed decomposition work. |
| `cirq_optree_to_cbloq` | `import_cirq_register_optree` handles default through ports and explicit concrete signatures with shaped bit-axis quregs. It reuses the native builder for split/join, first-use allocations, clean releases and sided output binding. |
| Gate wrappers, circuit conversion and `decompose_from_cirq_style_method` | `import_cirq_register_gate` expands one untagged SDK decomposition. Its default signature retains the original through operands, including idle wires. An explicit native signature can consume a left word after binary reset and defaults to operand-order placement. Optree conversion recognizes gate-owned, typed native definitions and gives caller-supplied definitions precedence. |
| `_QReg` and register helpers | Native port bit order, shape and ownership are checked. An explicit native function definition preserves per-gate word grouping. The importer splits only overlapping words and joins in operand order. The locked optree conversion materializes the SDK borrowed-wire witness as clean Allocate/Free. |
| `_my_tensors_from_gate` and wire symbols | `_gate_metadata.py` reads Cirq's unitary and diagram protocols and returns owned bit-axis tensors and native glyph data. Shared Simulation projection preserves output-then-input order and zero-projects missing sides. Nonunitary reset is refused at the tensor boundary. |
| Global phase and unsupported dimensions | The typed import lowers concrete global phase through the existing scalar primitive on a live wire or an explicitly owned clean carrier. Symbolic phases, wider qudits and symbolic signatures require separate admission. |

All ten frozen original tests execute unchanged. The native fixture compares
shaped ports, two-sided remapping, a right-only output, allocation and release
against the source optree contract. Physical binding applies one H, three
CNOTs and one T in the same dependency order and preserves all complex
amplitudes after append. The owned work wire returns to zero. The separate
locked model reports four Clifford and one T. Independent gates can be
scheduled by the native binder.

Separate fixtures preserve the source's phase with and without existing
wires, and compare the frozen left-only CNOT/reset wrapper with its through
counterpart. Both produce the source density matrix. The sided wrapper
consumes both input wires under explicit ownership and releases both after
reset. A managed clean-work decomposition preserves coherent amplitudes
and releases its allocated wire. The plain physical Cirq importer retains
its established entry point.

The metadata fixtures match scalar global phase, primitive and grouped
word tensors, the source LEFT/RIGHT SWAP transfer, and occupied control
and directional glyphs, including the frozen multibit-shaped label slice.
No physical gates are emitted by tensor inspection or glyph selection.

The original 28-wire grouping fixture now has a native counterpart.
The default signature exposes all 28 lines. Explicit shaped ports retain
24 lines and use two clean two-bit allocations and two releases. Six
three-bit words split, then rejoin in the declared output order. The
literal six-wire seeded source circuit, including iSWAP and its scalar
phase, agrees with the full 64-by-64 complex unitary obtained through
normal append.

The default 28-wire import discovers typed `FunctionRecord` definitions
exposed by the SDK gates. The shaped import uses the explicit definition
mapping. The importer reserves names before lowering primitives and verifies
the native type before inserting a definition into the Program. Gates with
no definition use the existing raw Cirq path. Opaque foreign declarations
stay at the Ecosystem boundary.

Q-0717 covers the locked conversion responsibilities through the native
Ecosystem entry points. Q-0718 covers the unchanged original tests and the
native counterparts above. Symbolic phases, nonbinary operands, and
foreign SDK ownership beyond the locked clean allocation contract remain
outside the admitted path.
