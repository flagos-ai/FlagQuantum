# FlagQuantum MPS–TN Crossover Report

Date: 2026-07-30

Hardware: one NVIDIA A800-SXM4-80GB
Output: four selected amplitudes, complex64, no full-state materialization

## Why this campaign exists

Qubit count and sparse output are not sufficient reasons to choose a general
tensor network. One-dimensional shallow circuits are natural MPS workloads.
General TN should be selected when the interaction graph has low contraction
width but is unfavorable for a linear MPS ordering.

## One-dimensional chain

For a 40-qubit depth-4 nearest-neighbour chain:

| Backend | Cold start | Steady state | Peak memory |
|---|---:|---:|---:|
| MPS | 2.740 s | 40.85 ms | 14.35 MiB |
| General TN | 4.292 s | 31.44 ms | 8.26 MiB |

The MPS maximum bond dimension was 16. MPS wins the one-shot latency, while the
cached TN contraction is approximately 1.30× faster in this repeated-execution
measurement. The selector therefore treats the chain as an MPS-native workload
by default, while future repetition-aware calibration may choose a cached TN
program for repeated parameter execution.

## Nonlocal binary tree

| Qubits | MPS bond | MPS steady | TN steady | MPS/TN speed ratio | MPS memory | TN memory |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 256 | 0.494 s | 10.61 ms | 46.6× | 20.42 MiB | 8.16 MiB |
| 20 | 1024 | 1.164 s | 12.98 ms | 89.7× | 96.80 MiB | 8.17 MiB |
| 24 | 4096 | 7.833 s | 14.35 ms | 545.8× | 1238.83 MiB | 8.18 MiB |

At 24 qubits the general TN cold start is also 4.05× faster and uses 151.5×
less peak allocated memory. The interaction graph has treewidth one, while the
fixed linear MPS ordering creates rapidly growing bonds.

Maximum absolute MPS–TN amplitude disagreement across the measured cases is
`1.08e-6`.

## Selector consequence

- Shallow nearest-neighbour chain with bounded cut crossings: select MPS.
- Nonlocal binary tree with low graph width: reject the linear MPS fast path and
  select general TN.
- Dense statevector fitting in memory remains the lower-risk default for small
  workloads.
- Repeated execution count must eventually become a calibrated selector input:
  cold-start and steady-state winners can differ.

## Claim boundary

This campaign demonstrates a real MPS–TN crossover. It does not prove that TN
wins on arbitrary nonlocal circuits: high treewidth remains exponentially hard
for general TN as well.
