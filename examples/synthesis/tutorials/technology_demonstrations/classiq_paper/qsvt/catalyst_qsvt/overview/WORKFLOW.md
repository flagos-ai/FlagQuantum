# Source mapping and explicit reconstruction limits

| Source code cell (all commented) | Native counterpart |
| --- | --- |
| 2: reflection, block encoding, projectors, QSVT and device declaration | Composable native qfunc, checked arithmetic, exact phase polynomial, adjoint and ten explicit logical wires |
| 4: six data bits, degree three, four phases, CX/RZ/RY/RX decomposition, qjit and MLIR count | Same quantum workload; an explicitly different local exact CX/U3 pipeline with actual instruction counts, width, depth and monotonic durations |

The entire source is commented. Its device declaration supplies size plus three
bits: nine for size six. Its auxiliary index is also size plus three, so index
nine addresses a tenth bit.
The native program explicitly allocates ten logical wires. It reconstructs the
stated circuit with that correction; it does not claim unchanged foreign SDK
execution or certification of qjit/MLIR behavior.

Both block controls select one, unlike the neighboring Qiskit and Classiq
references. The reflection is negative only on data zero. Projector RZ phases
are radians. Adder treats `[data[0], ..., data[5], deletion]` as a big-endian
seven-bit word. Native QNum indexing is LSB-first, so the arithmetic view lists
`[deletion, data[5], ..., data[0]]` explicitly. It adds two only in the selected
branch, then subtracts one unconditionally. Output named wire order is MSB-first
[block,data,aux]. The independent Circuit uses select wire two, deletion one,
projection zero, data [8,7,6,5,4,3], and auxiliary nine. Modular permutations are
verified on all 128 columns; the reflection on all 64 columns.

Native and local-target complete complex statevectors are checked on zero,
nonzero boundary and complex coherent inputs, with a second nonzero phase
binding. Absolute phase and compiler work returning to zero are required.
Logical block and auxiliary bits remain outputs. This is not exhaustive proof
of all 1024 full-program columns. The independent reference's largest dense
complex128 matrix is 256 by 256 (one MiB), for correctness inspection only.
Native synthesis uses no dense matrices.

The local pipeline targets CX/U3. It deliberately does not report source-basis
parity, Catalyst optimization, qjit timing or MLIR CNOT counts. Caller-owned
exact RZ conversion preserves absolute phase. Width, work, instruction, depth,
execution-memory and conversion-growth refusals are tested. The explicit local
budgets are eleven physical bits, one work bit, 100,000 native operations,
200,000 added target operations and 64 MiB of execution memory. The last value
is not a process-RSS bound. The original has no width or work budget.

This workflow is single_device_fast_path. Actual costs and local durations are
not hardware, optimality, speedup or release evidence. Source Notebook outputs
stay empty; actual executed artifacts are stored outside the examples tree.
