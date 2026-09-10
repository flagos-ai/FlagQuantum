# JAX JIT Performance Case Design

## Questions

For the same FlagQuantum statevector loss-and-gradient workload, retaining the
PyTorch training interface:

1. What compilation cost does the first JAX JIT call incur?
2. How much faster is JAX per step after compilation than PyTorch Native?
3. How does initial compilation cost grow with qubit count?
4. How many repeated calls are needed before cumulative JAX time beats PyTorch?

## Frozen Experiment Protocol

- Semantics: `single_device_fast_path` on one A800, with no distributed scalability claim.
- Method: exact statevector, complex64.
- Circuit: RX/RY/RZ, linear CX, and first-to-last RXX in every layer.
- Metric: end-to-end loss plus backward gradient time.
- Cold start: a fresh Python subprocess per qubit size, with persistent compilation
  caches disabled.
- Steady state: repeat after the first JAX call; report medians across all
  cold-process samples.
- Correctness: maximum absolute JAX/PyTorch loss and gradient errors at most `1e-4`.
- Default sweep: 4/6/8/10/12 qubits, 2 layers, 3 cold starts, 10 steady samples each.

## Figure Plan

| Figure | Question | Metric | Intended interpretation |
| --- | --- | --- | --- |
| 01 Cold Start | How expensive is initial compilation? | PyTorch first versus JAX build+first | Larger inputs increase initial JAX waits. |
| 02 Steady State | Is it worthwhile after compilation? | Steady loss+gradient latency | Small, frequently repeated calls can benefit from JIT fusion. |
| 03 Speedup | How large is the steady-state gain? | PyTorch/JAX ratio | Report development results for the frozen workload only. |
| 04 Break-even | How many calls amortize compilation? | Cumulative-time crossover | Planners should consider expected reuse when selecting backends. |

Presentations must include CPU/dependency versions, whether initial compilation is
included, warm/cold definitions, repetitions, correctness errors, and
`single_device_fast_path`. These results are neither multi-device scalability
evidence nor universal backend conclusions across all sizes.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/jax_jit_crossover.py \
  --wires 4,6,8,10,12 --layers 2 \
  --cold-repetitions 3 --steady-repetitions 10 \
  --device cuda:0 \
  --output benchmarks/results/local/jax_jit_crossover_a800.json
```

Plotting and publication-specific analysis are maintained outside the source
repository; the JSON result is the reproducible interface.

## A800 Measurements

Environment: one NVIDIA A800-SXM4-80GB, PyTorch 2.13.0+cu130, JAX 0.10.2.
Each size uses 3 fresh processes for cold starts and 10 steady-state samples per process.

| Qubits | JAX cold start | PyTorch steady | JAX steady | JAX steady speedup | Break-even calls |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 5.015 s | 26.589 ms | 0.958 ms | 27.75× | 190 |
| 8 | 8.761 s | 43.124 ms | 2.195 ms | 19.65× | 210 |
| 12 | 12.662 s | 60.018 ms | 3.356 ms | 17.88× | 221 |
| 16 | 14.480 s | 75.256 ms | 10.498 ms | 7.17× | 221 |
| 20 | 17.100 s | 132.453 ms | 127.120 ms | 1.04× | 3147 |

All sizes pass the frozen `1e-4` loss/gradient threshold. JAX JIT provides
substantial steady-state gains for repeated small simulations, with compilation
cost increasing by size. At 20 qubits the steady advantage nearly disappears,
requiring approximately 3147 same-shape calls to amortize cold start.
