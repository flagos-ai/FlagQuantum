# Checkerboard block encoding

The native program exposes the source's packed fields directly as
`data,s1,s0,j1,j0`. `s1` is the high two-bit selector field; `s0` is its low bit.
`j1` is the high position bit and `j0` its low two-bit field. These five outputs
preserve exactly the original `data,s,j` physical readout order.

The user program uses checked SWAP/CX and an RX rotation followed by a coherent
controlled angle difference. For selector 0 the data receives `RX(angle0)`; for
selector 1 it receives `RX(angle0) RX(angle1-angle0) = RX(angle1)`. `within`/`apply`
reverses the selector Hadamards. Original angles remain `2*acos(-0.69)` and
`2*acos(-0.34)`; classical bindings can supply a different pair without changing
the quantum program. No copied Qmod text, importer wrapper or hand-built matrix
is used in the native entrypoint.

The independent Circuit builds both controlled-RX matrix blocks directly and
checks full complex amplitudes. Tests cover all 128 packed initial words,
including nonzero data, coherent relative phases, two nonzero angle bindings,
absolute phase, bit order, zero hidden work and budget refusal.

Current native costs: 7 physical qubits, 0 workspace, 16 operations, depth 4.
Original source: 7 physical qubits, 22 operations, depth 11. These are measurements
for this concrete source; no general optimality claim follows. The complete
Notebook workflow remains a separate migration item.
