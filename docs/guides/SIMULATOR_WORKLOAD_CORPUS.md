# Simulator Workload Corpus

One circuit family cannot establish that a simulator is generally faster. The
workload corpus measures several deterministic FlagQuantum IR structures across
the native runtime, Qiskit Aer, Cirq Simulator, and PennyLane Lightning.

## Run the corpus

Install the optional simulators and run the standard CPU matrix:

```bash
pip install -e '.[qiskit,cirq,pennylane]'
flagquantum-benchmark run simulator_workload_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 10 14 18 22 \
  --engines flagquantum_native qiskit_aer cirq_simulator \
    pennylane_lightning_qubit \
  --threads 1 --warmup 1 --iterations 5 --calls-per-sample 1 \
  --json-output workload-corpus.json
```

Use `--workloads`, `--n-wires`, or `--engines` to run a smaller matrix. Missing
optional dependencies, unsupported conversions, and statevector mismatches fail
closed instead of silently removing an engine.

## What is measured

Every case contains:

- the deterministic FlagQuantum IR content hash;
- gate count, logical depth, opcode histogram, parameter count, and gate arity;
- two-qubit wire-span and nearest-neighbor features;
- raw end-to-end timing samples and their median for each engine;
- relative median absolute deviation with a 20% local stability threshold;
- exact-statevector parity against FlagQuantum native;
- each external engine's median time divided by FlagQuantum's median time.

The timed unit is one user-facing run call. Conversion, compilation or device
preparation, execution, and result retrieval are included. An
`engine_over_flagquantum_median` value above one means FlagQuantum was faster;
a value below one means the external engine was faster.

`--threads` sets PyTorch and the standard OpenMP/BLAS environment limits. The
Qiskit path additionally uses the public `QiskitAerBackend` extension so Aer's
`max_parallel_threads` is set explicitly rather than inferred from the host.

The corpus is comparison evidence, not a universal ranking or a scalability
claim. Results apply only to the recorded versions, host, thread count, dtype,
workloads, and widths. In particular, low-qubit results are dominated by fixed
overheads and must not be extrapolated to memory-pressure regimes.

## Workload families

| Family | Structural purpose |
| --- | --- |
| `hardware_efficient_statevector` | Parameterized rotations, linear CX, RZZ, and SWAP |
| `truncated_qft_statevector` | Long dependency chain with controlled phases decomposed into the portable gate subset |
| `random_clifford_statevector` | Shallow deterministic Clifford layers with shuffled two-qubit pairs |
| `local_brickwork_statevector` | Fixed-depth parameterized nearest-neighbor circuit |
| `dense_nonlocal_statevector` | All-to-all CZ connectivity between two single-qubit layers |

The feature schema is deliberately backend-neutral. A later evidence-based
advisor can match a user's circuit to measured cases without claiming that this
finite corpus enumerates every possible circuit.
