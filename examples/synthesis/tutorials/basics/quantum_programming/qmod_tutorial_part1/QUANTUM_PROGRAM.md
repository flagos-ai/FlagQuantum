# Four-bit phase-oracle interference quantum program

The fixed four-bit source program applies Hadamards, flips the phase at `1111`
and on states with `x[3]=1`, then applies Hadamards again. The native polynomial
phase `pi*(x[0]*x[1]*x[2]*x[3]+x[3])` retains both flips, which cancel at `1111`.
This example has no classical parameters; validation does not invent bindings.

Both objectives match the original and an independent sixteen-entry diagonal
Circuit in every raw complex128 amplitude, including absolute phase and bit order.
The original's hidden auxiliary bit is checked to return to zero.
Native resources are 4 qubits, no workspace, 33 static instructions and depth 27,
with maximum instruction width 2. The source uses 5 qubits, one clean workspace
bit, 9 instructions and depth 5, with maximum instruction width 3. This is a
workspace/instruction tradeoff, not an efficiency improvement or hardware-cost
comparison. Frozen source resources remain unchanged.

Run the commands in [README](README.md) for the native program and Circuit reference.
Only the retained quantum program is accepted; the complete Notebook remains pending.
