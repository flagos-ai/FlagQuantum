# Workflow contract

All source code cells were reviewed, including their original comments. The
original has no active code, sampling experiment or figure. The native workflow
makes the intended compilation example executable without installing its SDK.

Preserve the original loop indentation: X(x[0]) occurs inside all five MCX iterations. This is a reversible permutation, but differs from modular increment on 32 of 64 positions. Do not silently replace it with arithmetic increment.

The source requests CX/RZ/RY/RX then counts CNOT text in a jitted MLIR string. We count actual executable CX instructions instead. No qjit, Lightning runtime or MLIR-text timing equivalence is claimed.

The source width is six position bits plus coin. Native synthesis permits at most
16 physical qubits and 9 work qubits; actual width is measured, not assumed.
The bounded target conversion permits at most 20,000 added operations and refuses
smaller budgets. Reported depth and gate counts use the actual executable IR.
The target basis is CX/RZ/RY/RX. Exact logical phase uses one final RZ on an existing clean zero work qubit. Its cost is included; the transform is an isometry with zero input/output work, not an identity for dirty work.

Every one of 128 logical basis columns is checked against a 256 KiB independent
permutation Circuit, for native and converted IR. Complex coherent inputs also
check absolute phase, bit order, coin sign and workspace zero. The Catalyst
rotation transform additionally requires a clean final work qubit and explicitly
refuses a program without one. No dropped phase is excused by alignment.

Local execution uses complex128 CPU and `single_device_fast_path`. Durations are
observed local synthesis/lowering/workflow durations, not cross-framework
performance data. No hardware or speedup claim is made. The independent dense
matrix is a correctness oracle only. Native lowering uses named gates.
