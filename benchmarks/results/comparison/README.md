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
