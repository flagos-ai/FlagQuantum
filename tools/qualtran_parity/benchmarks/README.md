# Dynamic SELECT CPU measurements

From the repository root, use a FlagQuantum environment with Qualtran 0.7.0,
Cirq Core 1.4.0, NumPy 1.26.4 and PyTorch 2.13.0.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
python tools/qualtran_parity/benchmarks/compare_dynamic_select.py --shots 32
python tools/qualtran_parity/benchmarks/same_circuit_speed.py
```

The JSONL files record local ARM64 CPU timings at FlagQuantum commit
`1fc7668b`, using complex128 and one Torch thread. The first benchmark
uses the two native decompositions with the same H/S input and records
medians over five runs after one warmup. The second uses the FlagQuantum
decomposition in both executors with the same RY/S input and records
medians over three runs after one warmup. The benchmark-only Cirq converter
preserves instruction order and checks complex amplitudes against the
coherent reference at three seeds. Both scripts check sampled workspace reset.

At 16 entries and 32 shots, native decomposition took 4.31 ms in
FlagQuantum and 11.19 ms in Qualtran. Sampling their respective circuits
took 145.88 ms with FlagQuantum trajectories and 589.43 ms in Cirq.
On the shared circuit, trajectory times were 118.47 ms and 162.38 ms,
respectively. FlagQuantum batched sampling took 18.56 ms. The shared
circuit comparison uses a different input, so the two ratios are not
a quantitative attribution of the decomposition effect. These are CPU
simulator timings for a small workload, not hardware estimates.
