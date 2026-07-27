# Distributed SV evidence completion

All final comparisons must use 31q, 8 layers, 248 parameters, complex64,
full value + reversible-adjoint gradient, warmup 2 and 5 measured samples.

## Existing payloads to audit

- FlagQuantum 16-GPU two-node: `benchmarks/results/comparison/flagquantum_31q_d8_16xa800_2node_cached_nccl_ch16_all_checkpoint_warm3.json`
- PennyLane 16-GPU two-node: `benchmarks/results/comparison/pennylane_lightning_gpu_31q_d8_16xa800_mpi_final_warm2_rep5.json`
- FlagQuantum 8-GPU Ring final: `generality/flagquantum_31q_d8_ring_8gpu_correct_runner_ch16.json`

The audit must reject payloads with mismatched runner, topology, gradient
protocol, or NCCL configuration. Keep censored/timeout payloads as lower-bound
evidence, never as completed measurements.

## Required final table

For each backend and GPU count record median, raw samples, bootstrap 95% CI,
peak memory per rank, communication bytes, communication kernel count, and
correctness status. A result is publishable only when the payload's effective
configuration matches the table entry.
