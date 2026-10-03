# Comparison Results

This directory contains cross-framework or backend comparison results. These
files can compare runtime, device, interface, dtype, gradient method, and other
benchmark settings, but they are not release-grade scalability evidence.

Comparison payloads must keep `scalability_claim_allowed=false` and include a
non-release blocker such as `comparison_result_not_release_scalability_evidence`.
If a comparison used multiple ranks or devices without proving one logical
workload capacity expansion, it is labeled as rank-local or non-release rather
than scalability.

[`SIMULATOR_COMPARISON_CPU_ARM64_20260923.md`](SIMULATOR_COMPARISON_CPU_ARM64_20260923.md)
is the generated human-readable view of the adjacent raw FlagQuantum, Qiskit
Aer, Cirq, and PennyLane artifacts. Its machine-readable companion is
[`simulator_comparison_cpu_arm64_20260923.json`](simulator_comparison_cpu_arm64_20260923.json).
Contract tests regenerate both files and fail if either view drifts from the
raw measurements.

[`SIMULATOR_ADAPTIVE_DENSE_FUSION_WIDTH_CPU_ARM64_20260924.md`](SIMULATOR_ADAPTIVE_DENSE_FUSION_WIDTH_CPU_ARM64_20260924.md)
records the native rollback A/B and refreshed cross-framework evidence for the
measured complex128 dense-fusion width policy.

[`SIMULATOR_RANDOM_CLIFFORD_CPU_ARM64_20260924.md`](SIMULATOR_RANDOM_CLIFFORD_CPU_ARM64_20260924.md)
records the exact-statevector Random Clifford comparison, product-state
Clifford-layer rollback A/B, public example, and reproduction command.

[`SIMULATOR_TRUNCATED_QFT_CPU_ARM64_20260924.md`](SIMULATOR_TRUNCATED_QFT_CPU_ARM64_20260924.md)
records the exact Truncated QFT comparison and the deferred product-state SWAP
materialization rollback A/B.

[`BATCHED_STATEVECTOR_PENNYLANE_NATIVE_BATCH_CPU_ARM64_20261002_SCORECARD.md`](BATCHED_STATEVECTOR_PENNYLANE_NATIVE_BATCH_CPU_ARM64_20261002_SCORECARD.md)
compares FlagQuantum's native parameter batch with both PennyLane Lightning's
public broadcast expansion and the repeated single-item FlagQuantum bridge. It
records exact times, speedups, correctness, peak RSS, baseline-adjusted execution
growth, example code, reproduction steps, and the remaining memory gap.

[`BATCHED_STATEVECTOR_TERMINAL_FUSED_ROTATION_CPU_ARM64_20261002_SCORECARD.md`](BATCHED_STATEVECTOR_TERMINAL_FUSED_ROTATION_CPU_ARM64_20261002_SCORECARD.md)
records the exact rollback A/B for routing terminal same-wire fused rotations
through the native CPU layer kernel. It includes the rejected broad-routing
experiment, same-run PennyLane Lightning native-batch comparison, exact times,
memory, correctness, example code, limitations, and reproduction steps.

[`BATCHED_STATEVECTOR_ROTATION_CLIFFORD_FUSION_CPU_ARM64_20261003_SCORECARD.md`](BATCHED_STATEVECTOR_ROTATION_CLIFFORD_FUSION_CPU_ARM64_20261003_SCORECARD.md)
records the exact rollback A/B for fusing local rotation tiles with their next
disjoint CX matching. It includes absolute time, the execution-RSS reduction,
same-run PennyLane Lightning comparison, public example, boundaries, raw JSON,
and an exact reproduction command.

[`BATCHED_STATEVECTOR_PRODUCT_STATE_INITIALIZATION_CPU_ARM64_20261003_SCORECARD.md`](BATCHED_STATEVECTOR_PRODUCT_STATE_INITIALIZATION_CPU_ARM64_20261003_SCORECARD.md)
records the direct-from-zero first-layer kernel and final-slice assembly. It
includes the copy-based rollback A/B, absolute time and RSS, a prominent
PennyLane Lightning comparison, public example, boundaries, raw JSON, and an
exact reproduction command.

[`BATCHED_STATEVECTOR_STATIC_PRODUCT_INITIALIZATION_CPU_ARM64_20261003_SCORECARD.md`](BATCHED_STATEVECTOR_STATIC_PRODUCT_INITIALIZATION_CPU_ARM64_20261003_SCORECARD.md)
records the specialized direct initializer for complete static Clifford product
layers. It includes the exact rollback A/B, absolute time and RSS, a prominent
PennyLane Lightning comparison, public example, safety boundaries, raw JSON,
and an exact reproduction command.

The differentiable simulator corpus adds matched exact expectation-value and
full-gradient measurements for FlagQuantum native PyTorch autograd and PennyLane
default.qubit backprop. Its generated Markdown report contains the workload meaning,
public FlagQuantum example, measured times and memory, and reproduction command.

[`ADJOINT_DIFFERENTIABLE_SIMULATOR_CORPUS_CPU_ARM64_20260928.md`](ADJOINT_DIFFERENTIABLE_SIMULATOR_CORPUS_CPU_ARM64_20260928.md)
adds a separate method-matched comparison between FlagQuantum's reversible
statevector adjoint and PennyLane Lightning's adjoint. It also records a
same-process rollback A/B for the native CPU direct-layout backward optimization.
The report includes the initial Z/ZZ Hamiltonian API, full-gradient correctness,
measured forward/backward/total times and speedups, current performance gap,
limitations, and exact reproduction command.

[`NATIVE_CPU_ADJOINT_THREAD_SCALING_CPU_ARM64_20260928.md`](NATIVE_CPU_ADJOINT_THREAD_SCALING_CPU_ARM64_20260928.md)
records method-matched 1/2/4/8-thread FlagQuantum and PennyLane Lightning
adjoint measurements after enabling Torch OpenMP in the package-local native
CPU extension. It includes absolute forward/backward/total time, FlagQuantum
speedup, correctness, a small-workload guardrail, and exact reproduction steps.

[`NATIVE_CPU_ADJOINT_EULER_TRIPLES_CPU_ARM64_20260929.md`](NATIVE_CPU_ADJOINT_EULER_TRIPLES_CPU_ARM64_20260929.md)
records an exact rollback A/B for branchless amplitude-pair traversal and the
same-wire RZ/RY/RX Euler-triple reverse kernel. It reports absolute backward and
total times, speedups, correctness, workload meaning, and reproduction steps.

[`NATIVE_CPU_ADJOINT_EULER_POST_REDUCTION_CPU_ARM64_20261002.md`](NATIVE_CPU_ADJOINT_EULER_POST_REDUCTION_CPU_ARM64_20261002.md)
records a same-binary rollback A/B for reducing raw Euler bilinears before the
coordinate transform. It reports the wide-state VQE gain, the no-Euler QAOA
control, complete PennyLane Lightning comparison, public example, and exact
reproduction command.

[`NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE_CPU_ARM64_20260929.md`](NATIVE_CPU_ADJOINT_TERMINAL_NO_RESTORE_CPU_ARM64_20260929.md)
records the 22-qubit VQE rollback A/B for terminal Euler-layer no-restore,
CX-to-rotation boundary fusion, flat SIMD pair traversal, and reused Pauli
bilinears. It includes exact backward and value-plus-gradient times,
correctness, example code, limitations, and reproduction steps.

[`NATIVE_CPU_ADJOINT_COMPACT_CX_CPU_ARM64_20260930.md`](NATIVE_CPU_ADJOINT_COMPACT_CX_CPU_ARM64_20260930.md)
records the opt-in 22-qubit compact CX mapping comparison against the full
permutation-index rollback and PennyLane Lightning adjoint. It includes exact
timings, isolated peak RSS, metadata size, correctness, example code, the
time-memory tradeoff, and an exact reproduction command.

[`NATIVE_CPU_ADJOINT_HOST_MEMORY_BUDGET_CPU_ARM64_20260930.md`](NATIVE_CPU_ADJOINT_HOST_MEMORY_BUDGET_CPU_ARM64_20260930.md)
records the 24-qubit complex128 CPU adjoint checkpoint cliff, the host-aware
default's absolute forward/backward/total time and peak RSS, and a censored
static-512-MiB rollback. It includes the cgroup safety boundary, public example,
limitations, and exact reproduction commands.

[`NATIVE_CPU_ADJOINT_BUDGETED_BLOCK_CHECKPOINTS_CPU_ARM64_20260930.md`](NATIVE_CPU_ADJOINT_BUDGETED_BLOCK_CHECKPOINTS_CPU_ARM64_20260930.md)
measures the bounded-memory fallback between reversible adjoint and full
rematerialization. It records the exact checkpoint plan, replayed-gate count,
absolute 24-qubit time, rollback speedup, correctness coverage, and reproduction
commands for the 1 GiB and 512 MiB policies.

[`NATIVE_CPU_ADJOINT_LOW_MEMORY_CX_CPU_ARM64_20260930.md`](NATIVE_CPU_ADJOINT_LOW_MEMORY_CX_CPU_ARM64_20260930.md)
records the 24-qubit memory-tiered CPU CX adjoint path. It compares the native
zero-state-scratch in-place kernel with budgeted block checkpoints and the
faster two-state-scratch fused gather, including absolute time, peak RSS,
correctness checks, public example, and exact reproduction commands.

[`NATIVE_CPU_ADJOINT_CX_CYCLES_CPU_ARM64_20260930.md`](NATIVE_CPU_ADJOINT_CX_CYCLES_CPU_ARM64_20260930.md)
records the compact in-place permutation-cycle kernel and its performance and
memory gap to both per-CNOT pairs and fused dual-state gather.

[`NATIVE_CPU_ADJOINT_MEMORY_TIERS_CPU_ARM64_20260930.md`](NATIVE_CPU_ADJOINT_MEMORY_TIERS_CPU_ARM64_20260930.md)
records checkpoint-policy v5 selecting fused gather, compact cycles, or
zero-auxiliary CNOT pairs from explicit modeled memory requirements. It includes
absolute timing, peak RSS, correctness, usage, and reproduction commands.

[`NATIVE_CPU_SHARED_RZZ_CPU_ARM64_20260929.md`](NATIVE_CPU_SHARED_RZZ_CPU_ARM64_20260929.md)
records the 22-qubit QAOA rollback A/B for exact-view parameter deduplication,
shared-RZZ phase lookup, and terminal RX-to-RZZ reverse fusion. It includes
absolute forward/backward/total times, correctness, example code, limitations,
and an exact reproduction command.
