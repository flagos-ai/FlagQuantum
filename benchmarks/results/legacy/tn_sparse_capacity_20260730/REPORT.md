# FlagQuantum Sparse-Output TN Capacity Report

Date: 2026-07-30

Hardware: single node, 8 × NVIDIA A800-SXM4-80GB

Backend: PyTorch 2.13.0 + CUDA 13.0 + NCCL
Workload: nearest-neighbour brickwork circuit, depth 4, four shared amplitudes,
complex64

## Outcome

The capacity probe reached 53 qubits without materializing a full statevector.
All eight ranks owned one slice task, and the final NCCL reduction contained
only four complex64 amplitudes (32 bytes).

This is a measured sparse-output capacity result. It is **not** a successful
multi-GPU scaling result.

## Measured scaling at 40 qubits

| A800 GPUs | Latency (s) | Speedup | Parallel efficiency | Peak/rank |
|---:|---:|---:|---:|---:|
| 1 | 44.332 | 1.000× | 100.0% | 8.362 MiB |
| 2 | 51.460 | 0.861× | 43.1% | 8.362 MiB |
| 4 | 51.874 | 0.855× | 21.4% | 8.362 MiB |
| 8 | 50.823 | 0.872× | 10.9% | 8.362 MiB |

The four amplitudes were bitwise consistent across the 1/2/4/8-GPU runs.

## Measured capacity on eight A800 GPUs

| Qubits | Latency (s) | Peak/rank | Slice tasks | Reduction |
|---:|---:|---:|---:|---:|
| 40 | 50.823 | 8.362 MiB | 8 | 32 B |
| 45 | 73.398 | 8.391 MiB | 8 | 32 B |
| 50 | 98.606 | 8.421 MiB | 8 | 32 B |
| 53 | 116.514 | 8.438 MiB | 8 | 32 B |

## Interpretation

- The default tensor-network initial state no longer allocates a dense
  `2**n` statevector. A 53-qubit construction therefore remains small.
- Sparse output is preserved end to end: no full state is materialized and
  only 32 bytes are reduced.
- Capacity is currently limited by contraction scheduling and repeated
  per-slice work, not A800 memory.
- The present slice selection does not reduce the dominant work per rank.
  Increasing GPU count therefore adds process and collective overhead without
  producing speedup.
- The current implementation passes the 53-qubit sparse-output capacity gate,
  but fails the distributed strong-scaling gate.

## Evidence and limitations

- Every point is a completed real NCCL torchrun measurement.
- Raw rank-level latency, peak allocated memory, task ownership, values,
  environment, and git SHA are stored under `raw/`.
- This first capacity sweep uses one measured iteration per point because the
  current executor takes 44–117 seconds per iteration. It should not be used
  for low-noise performance claims.
- The topology is deliberately low width. These results do not imply that a
  53-qubit deep two-dimensional random circuit is tractable.
- A cotengra path-quality and execution comparison remains required before an
  external performance claim.

## Required next optimization

Replace label-order slicing with cost-aware cut selection and compile the
shared contraction once. Rank execution must reuse a common contraction DAG
and eliminate zero/irrelevant slice work. The next scaling gate should require:

1. all ranks own non-empty tasks;
2. per-rank estimated FLOPs decrease with world size;
3. 8-GPU speedup greater than 4× on a contraction-dominated workload;
4. numerical agreement with the one-GPU result;
5. no full-state materialization.
