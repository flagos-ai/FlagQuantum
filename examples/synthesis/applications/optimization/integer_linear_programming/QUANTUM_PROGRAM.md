# Integer linear optimization quantum program

Complete retained eight-bit, three-layer integer/slack quantum program. Two nonzero parameter vectors, both objectives and independent 256-state cost table/Circuit parity rotations. Full Notebook optimizer workflow remains pending.

Source v.x[0..2] become explicit native QNum ports x0,x1,x2 of two bits each; the two source scalar slack fields become s0,s1. This explicit storage is not general native qstruct or numeric-array support. Retain every source term, offset, gamma-first/beta-last order and RX(beta) mixers. Exact commuting sum-X evolution implements numeric-word mixers with duration beta/2. No source importer in user examples.

Native resources are 8 qubits, zero workspace, 120 static instructions and depth
35. The separately frozen source uses 8 qubits, 130 instructions and depth 34.
Maximum instruction width is two in both. These counts describe current IR;
local evolution fusion and storage scheduling can change them. They are not
hardware basis-gate estimates or an optimality claim.

Run the module commands in [README](README.md). All raw complex128 amplitudes
are compared without global-phase alignment after logical field-order mapping.
The two distinct slack fields and the constant phase at the all-zero state are
explicitly checked. Notebook training and classical optimization remain pending.
