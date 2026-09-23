# FlagQuantum and Qiskit Aer CPU comparison

This measured comparison uses the same FlagQuantum IR and exact complex128
statevector semantics for both engines. It was recorded on an Apple arm64 CPU
with one Torch/Aer worker thread, PyTorch 2.13.0, Qiskit 2.5.2, and Qiskit Aer
0.17.2. Each steady-state sample contains ten executions and each engine has
nine samples after three warmups. The original run alternated engine order. The
22- and 24-qubit FlagQuantum samples were later refreshed independently while
their original Aer samples were retained, as recorded in the JSON methodology.

| Qubits | Gates | FlagQuantum median | Qiskit Aer median | Aer / FlagQuantum | First-result ratio | Maximum error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 82 | 0.577 ms | 0.668 ms | 1.16x | 0.89x | 6.28e-16 |
| 14 | 114 | 1.307 ms | 2.481 ms | 1.90x | 11.39x | 7.07e-16 |
| 18 | 146 | 13.125 ms | 26.872 ms | 2.05x | 2.32x | 4.74e-16 |
| 22 | 178 | 300.297 ms | 534.415 ms | 1.78x | 1.43x | 1.68e-16 |
| 24 | 194 | 1164.636 ms | 2280.374 ms | 1.96x | 1.58x | 4.90e-16 |

A ratio above one means FlagQuantum was faster. Qiskit Aer returned the first
result sooner only at 10 qubits after conversion and transpilation were
included. At 24 qubits, the large-state dense-fusion update increased the
steady-state advantage from the original 1.06x result to 1.96x. The 22- and
24-qubit FlagQuantum rows were remeasured at source revision
`4d976abe5c415414846c857616ffce6cbadeeefc`; their Qiskit Aer measurements are
the unchanged original samples. All steady-state measurements passed the
declared relative median absolute deviation stability limit.

The updated large-state rows combine the benchmark runner from revision
`aaa3a70c698dc629c37baa2719607444aa3b6381` with FlagQuantum revision
`4d976abe5c415414846c857616ffce6cbadeeefc`. With both changes in one checkout,
reproduce the measurement with:

```bash
pip install -e '.[qiskit]'
flagquantum-benchmark run simulator_compare \
  --n-wires 10 14 18 22 24 --layers 2 --threads 1 \
  --warmup 3 --iterations 9 --setup-iterations 3 --calls-per-sample 10 \
  --json-output \
  benchmarks/results/comparison/flagquantum_qiskit_aer_cpu_arm64_20260923.json
```

The adjacent JSON file contains every raw sample, conversion and compilation
timing, environment metadata, correctness verdict, and claim ceiling. Its
methodology metadata identifies the independently updated FlagQuantum
large-state rows and reused Aer samples. This is a local comparison result, not
scalability or release evidence.
