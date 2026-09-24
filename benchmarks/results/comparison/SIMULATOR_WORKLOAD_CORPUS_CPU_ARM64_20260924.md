# Cross-framework simulator workload corpus (Apple arm64 CPU)

This report summarizes the raw artifact
[`simulator_workload_corpus_cpu_arm64_20260924.json`](simulator_workload_corpus_cpu_arm64_20260924.json).
It was produced on 2026-09-24 with Python 3.12.14, PyTorch 2.13.0,
FlagQuantum 0.2.0, Qiskit 2.5.2, Qiskit Aer 0.17.2, Cirq Core 1.7.0,
PennyLane 0.45.1, and PennyLane Lightning 0.45.0.

All engines used one CPU thread and complex128 exact statevectors. Each median
is based on nine retained end-to-end calls after one warmup. Conversion,
compilation or device preparation, execution, and result retrieval are included.
Engine order rotates between iterations. All 80 measured engine cases passed
statevector parity at absolute tolerance `1e-10`. Of the 80 timing groups, 77
met the declared 20% relative median absolute deviation stability threshold.
The three groups above the limit remain visible in the raw artifact instead of
being silently discarded: 14-qubit Random Clifford on FlagQuantum (31.2% RMAD),
and 10-qubit local brickwork on Cirq (20.8%) and PennyLane (39.8%). Ratios for
those groups should be treated as exploratory host-noise-limited evidence.

| Workload | Qubits | Gates | Depth | FlagQuantum (s) | Aer (s) | Cirq (s) | PennyLane (s) | Aer / FQ | Cirq / FQ | PennyLane / FQ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hardware-efficient | 10 | 82 | 26 | 0.000999 | 0.024817 | 0.003965 | 0.002063 | 24.83× | 3.97× | 2.06× |
| Hardware-efficient | 14 | 114 | 34 | 0.001760 | 0.025235 | 0.006532 | 0.003500 | 14.34× | 3.71× | 1.99× |
| Hardware-efficient | 18 | 146 | 42 | 0.016362 | 0.057187 | 0.056968 | 0.021387 | 3.50× | 3.48× | 1.31× |
| Hardware-efficient | 22 | 178 | 50 | 0.401835 | 0.718561 | 1.043863 | 0.476849 | 1.79× | 2.60× | 1.19× |
| Truncated QFT | 10 | 135 | 71 | 0.000714 | 0.024871 | 0.004211 | 0.003298 | 34.82× | 5.89× | 4.62× |
| Truncated QFT | 14 | 201 | 103 | 0.003037 | 0.030667 | 0.008379 | 0.007009 | 10.10× | 2.76× | 2.31× |
| Truncated QFT | 18 | 267 | 135 | 0.008044 | 0.075421 | 0.015606 | 0.050151 | 9.38× | 1.94× | 6.23× |
| Truncated QFT | 22 | 333 | 167 | 0.151352 | 0.940726 | 0.154574 | 1.040753 | 6.22× | 1.02× | 6.88× |
| Random Clifford | 10 | 60 | 8 | 0.000955 | 0.024117 | 0.002348 | 0.001865 | 25.24× | 2.46× | 1.95× |
| Random Clifford | 14 | 84 | 8 | 0.003643 | 0.028257 | 0.004249 | 0.002884 | 7.76× | 1.17× | 0.79× |
| Random Clifford | 18 | 108 | 8 | 0.036785 | 0.056273 | 0.015780 | 0.013661 | 1.53× | 0.43× | 0.37× |
| Random Clifford | 22 | 132 | 8 | 0.258565 | 0.654365 | 0.261208 | 0.298529 | 2.53× | 1.01× | 1.15× |
| Local brickwork | 10 | 98 | 12 | 0.001087 | 0.031712 | 0.005074 | 0.004219 | 29.16× | 4.67× | 3.88× |
| Local brickwork | 14 | 138 | 12 | 0.002814 | 0.029838 | 0.008650 | 0.004785 | 10.61× | 3.07× | 1.70× |
| Local brickwork | 18 | 178 | 12 | 0.032028 | 0.060526 | 0.061983 | 0.031604 | 1.89× | 1.94× | 0.99× |
| Local brickwork | 22 | 218 | 12 | 0.672586 | 0.717676 | 1.221049 | 0.717974 | 1.07× | 1.82× | 1.07× |
| Dense nonlocal | 10 | 65 | 19 | 0.001115 | 0.029552 | 0.003509 | 0.003722 | 26.50× | 3.15× | 3.34× |
| Dense nonlocal | 14 | 119 | 27 | 0.004203 | 0.029009 | 0.007501 | 0.004954 | 6.90× | 1.78× | 1.18× |
| Dense nonlocal | 18 | 189 | 35 | 0.043253 | 0.104685 | 0.043925 | 0.016151 | 2.42× | 1.02× | 0.37× |
| Dense nonlocal | 22 | 275 | 43 | 1.084791 | 1.951721 | 0.967858 | 0.413626 | 1.80× | 0.89× | 0.38× |

A ratio above one means FlagQuantum was faster; below one means the external
engine was faster. The table shows why backend selection cannot be based on
qubit count alone: circuit depth, connectivity, and gate family materially
change the crossover. These numbers are reproducible local comparison evidence,
not a universal framework ranking or release/scalability evidence.

The truncated-QFT native path now combines three exact optimizations. First, it
recognizes the portable `RZ-RZ-CX-RZ-CX` controlled-phase decomposition and
applies its equivalent two-qubit diagonal in one statevector pass. A dedicated
22-qubit rollback A/B (`threads=1`, two warmups, five retained calls) measured
0.491851 seconds with that fusion and 1.541631 seconds with
`FQ_CPU_CONTROLLED_PHASE_DECOMPOSITION_FUSION=0`, a 3.13× speedup. The compiled
plan reduced statevector applications from 252 to 93 while preserving the
original IR and exact controlled-phase global phase.

Second, eligible CPU circuits now execute exact, initially independent
statevector components and merge them only when a gate connects the components.
A conservative static cost model enables this path only when its estimated work
is at most 30% of dense execution; custom initial states, batches, smaller
circuits, and entanglement-dense workloads retain the established dense path.
A dedicated 22-qubit rollback A/B measured 0.190328 seconds with product
components and 0.528505 seconds with `FQ_CPU_PRODUCT_STATE_EXECUTION=0`, a
2.78× speedup over the already fused dense path. Compared with the original
no-fusion rollback median, the two layers together are 8.10× faster.

Third, consecutive static controlled-phase decompositions that may share wires
are compiled into one weighted phase graph per segment. The graph preserves the
exact global phase while replacing several component-state applications with
one broadcast multiply. A separate native-only rollback A/B using two warmups,
nine retained samples, and five calls per sample measured 0.007589 seconds
versus 0.012042 at 18 qubits (1.59×) and 0.135640 seconds versus 0.198294 at
22 qubits (1.46×). The 18-qubit compiled product-state plan fell from 75 state
applications to 44. The corpus now records 0.151352 seconds for FlagQuantum
versus the unchanged 0.154574-second Cirq measurement at 22 qubits. External
measurements were not rerun; only FlagQuantum was remeasured with the original
corpus methodology.

Reproduce the artifact from the repository root:

```bash
pip install -e '.[qiskit,cirq,pennylane]'
MPLCONFIGDIR=/tmp/fq-mpl-cache \
flagquantum-benchmark run simulator_workload_corpus \
  --n-wires 10 14 18 22 --threads 1 --warmup 1 --iterations 9 \
  --calls-per-sample 1 \
  --json-output benchmarks/results/comparison/simulator_workload_corpus_cpu_arm64_20260924.json
```
