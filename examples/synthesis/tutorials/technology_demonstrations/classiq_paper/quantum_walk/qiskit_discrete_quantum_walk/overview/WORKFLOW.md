# Workflow contract

All source code cells were reviewed, including their original comments. The
original has no active code, sampling experiment or figure. The native workflow
makes the intended compilation example executable without installing its SDK.

The source shift is modular increment. Coin=1 applies increment; coin=0 applies its inverse. The logical value is six-bit unsigned, with source array index zero as the low bit.

The source requests U/CX with Qiskit optimization level 3. We replace that private optimization pipeline with a bounded FlagQuantum decomposition; no optimization-level or timing equivalence is claimed.

The source width is six position bits plus coin. Native synthesis permits at most
16 physical qubits and 9 work qubits; actual width is measured, not assumed.
The bounded target conversion permits at most 20,000 added operations and refuses
smaller budgets. Reported depth and gate counts use the actual executable IR.
The target basis is CX/U3. U3 is the three-angle U gate. All explicit conversion rules preserve absolute phase.

Every one of 128 logical basis columns is checked against a 256 KiB independent
permutation Circuit, for native and converted IR. Complex coherent inputs also
check absolute phase, bit order, coin sign and workspace zero. The Catalyst
rotation transform additionally requires a clean final work qubit and explicitly
refuses a program without one. No dropped phase is excused by alignment.

Local execution uses complex128 CPU and `single_device_fast_path`. Durations are
observed local synthesis/lowering/workflow durations, not cross-framework
performance data. No hardware or speedup claim is made. The independent dense
matrix is a correctness oracle only. Native lowering uses named gates.
