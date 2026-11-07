# Unitary block encoding

G2-BE01 ports the locked Unitary block-encoding wrapper to native algorithm
composition. The child has concrete ordered THRU ports described by existing
RegisterLayout. Packed system wires bind those ports through partition_register;
ancilla and resource ports are unchanged bystanders. The callback receives its
named system ports, its own explicit clean workspace, classical result slots
and direction. It owns the child adjoint and work cleanup; the wrapper never
inverts measured primitive IR. No simulation/backend selection occurs here.

The wrapper retains alpha, epsilon, ancilla/resource widths and an identity
signal preparation. Costs are explicit child ComponentCost inputs composed with
all ancilla/resource bystanders, independently from emitted instruction counts.
Only concrete register bindings are admitted in this slice; symbolic layouts,
non-THRU children and a zero-wire scalar primitive are not certified.

The implementation lives in algorithms/primitives/unitary_block_encoding.py;
compiler-owned partitioning, arithmetic and ComponentCost remain authoritative.
The identity signal emitter is shared with LCUBlockEncoding through the existing
BlackBoxPrepare module instead of duplicated. Its normalization remains unset:
the locked identity signal's l1_norm_of_coeffs property raises NotImplementedError,
so no invented coefficient normalization is reported. Source alpha and epsilon
are retained as metadata rather than applied as an extra gate or matrix scaling.

Twenty-four original source configurations cover T, CNOT and two-/three-bit Add,
forward and original composite adjoint, and three auxiliary layouts. Two further
source flows cover noncontiguous physical system/ancilla/resource/work binding.
Primitive dependencies, measurement feedback and operands match the untouched
source. Both QEC profiles match all seven categories and logical peak for each
of the24 configurations. Source blob bbcff63d78cbd58979bee3670597a778fdd4bdd2
matches both the pinned source tree and installed reference environment.

Eight independent full complex-state scenarios enumerate every measurement
history, including arbitrary coherent and entangled ancilla/resource values,
and prove child work returns to zero. Packed register binding changes no gates.
Invalid dimensions, layout names, overlapping qubits, duplicate result slots,
out-of-range circuits and child access to auxiliary wires fail explicitly.
A dimension-only1024-bit model query forbids child execution and returns4092 T
equivalents and3076 logical peak qubits, with no emitted IR report.

Run `python -m examples.algorithms.unitary_block_encoding` for the normal
`circuit.append(encoding.make_on(...))` path. It executes Add and its exact
adjoint with coherent auxiliary bystanders. For the forward block:

| Quantity | Declared source model | Emitted IR |
|---|---:|---:|
| T equivalents / T instructions | 4 | 4 |
| Measurements | 1 | 1 |
| Logical peak / reserved qubits | 7 | 7 |

No physical-resource or execution-memory measurement is inferred from these
counts. Models consume explicitly supplied directional child costs.
There are52 new tests and193 concentrated passing tests without skips, including
existing BlackBoxPrepare, SELECT, LCU and repeated-control source-boundary tests.
Strict mypy covers the new wrapper/example and the two existing preparation
consumers. Black, Ruff, algorithm ownership and architecture checks pass.
This is a concrete THRU wrapper slice, not whole-source-file acceptance or G2
completion. Symbolic bindings, shaped source signatures and explicit controlled
wrapper specialization remain separate items.
