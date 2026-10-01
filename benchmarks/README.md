# FlagQuantum Benchmarks

FlagQuantum provides one discoverable command for maintained benchmarks:

```bash
pip install -e .
flagquantum-benchmark list
```

Inspect a benchmark before running it:

```bash
flagquantum-benchmark info statevector_local
```

Every maintained runner accepts its existing benchmark arguments after
`run`. A CPU quick start that does not require accelerator setup is:

```bash
flagquantum-benchmark run environment_probe \
  --json-output benchmarks/results/local/environment.json

flagquantum-benchmark run statevector_local \
  --device cpu --n-wires 12 --batch-size 4 --layers 2 \
  --warmup 1 --iterations 3 \
  --json-output benchmarks/results/local/statevector_cpu.json

python benchmarks/dynamic_trajectory.py \
  --shots 100 1000 --mid-circuit-measurements 1 2 4 \
  --json-output benchmarks/results/smoke/dynamic-trajectory.json
```

The flagship MPS training runner executes on the JAX kernel backend, so it
needs the `jax` extra and exits with `JAX backend unavailable` without it:

```bash
pip install -e '.[jax]'
python benchmarks/flagship_mps_training.py \
  --cases dimer:20 --steps 1 --iters 1 --warmup 0 \
  --json-output benchmarks/results/local/mps_smoke.json
```

### Interoperable simulator comparison

Compare equivalent exact-statevector execution from the same FlagQuantum IR:

The executable implementations are
[`simulator_compare.py`](../flagquantum/benchmarking/simulator_compare.py) for
FlagQuantum/Qiskit Aer and
[`external_simulator_compare.py`](../flagquantum/benchmarking/external_simulator_compare.py)
for Cirq/PennyLane Lightning. They are installed through the
`flagquantum-benchmark` CLI; the commands below execute those source files and
write the complete raw samples, not a manually assembled summary.

```bash
pip install -e '.[qiskit]'
flagquantum-benchmark run simulator_compare \
  --n-wires 10 14 18 22 24 --layers 2 --threads 1 \
  --warmup 3 --iterations 9 --setup-iterations 3 --calls-per-sample 5 \
  --json-output benchmarks/results/comparison/simulators.json

pip install -e '.[cirq,pennylane]'
# Exact versions used by the checked-in Apple arm64 result:
pip install 'cirq-core==1.7.0' 'pennylane==0.45.1' \
  'pennylane-lightning==0.45.0'
flagquantum-benchmark run simulator_compare_cirq \
  --n-wires 10 14 18 22 24 --layers 2 --threads 1 \
  --warmup 3 --iterations 9 --setup-iterations 3 --calls-per-sample 10 \
  --json-output benchmarks/results/comparison/cirq.json
flagquantum-benchmark run simulator_compare_pennylane \
  --n-wires 10 14 18 22 24 --layers 2 --threads 1 \
  --warmup 3 --iterations 9 --setup-iterations 3 --calls-per-sample 10 \
  --json-output benchmarks/results/comparison/pennylane.json
```

Generate a validated JSON and Markdown report from compatible raw artifacts
without rerunning any simulator:

```bash
flagquantum-benchmark run simulator_comparison_report \
  benchmarks/results/comparison/flagquantum_qiskit_aer_cpu_arm64_20260923.json \
  benchmarks/results/comparison/cirq_cpu_arm64_20260923.json \
  benchmarks/results/comparison/pennylane_lightning_cpu_arm64_20260923.json \
  --json-output comparison.json --markdown-output comparison.md
```

The report generator rejects mismatched workload matrices, hosts, devices,
thread counts, sampling settings, tolerances, and reference semantics. Pass an
earlier generated report with `--baseline` to classify timing changes against
`--regression-threshold-percent`. Historical classifications are informational;
they do not become a CI performance gate.

The runner reports conversion, compilation, cold execution, and steady-state
execution separately. Steady-state samples alternate engine order inside one
process. Every case must pass exact-statevector parity before the payload passes;
unsupported conversion or a missing engine fails closed. The ratio is Qiskit Aer
time divided by FlagQuantum native time, so a value above one means FlagQuantum
is faster for that case.

The Cirq and PennyLane runners time only the selected external simulator. They
execute FlagQuantum once per case as an untimed correctness reference, allowing
a report to reuse a compatible measured FlagQuantum result rather than distort
it with a second run. Their payload records that measurement scope explicitly;
it must not be presented as a fresh FlagQuantum measurement.

These are local comparison results, not scalability or release evidence. A
result must retain `scalability_claim_allowed=false` and the comparison evidence
blocker documented in `benchmarks/results/comparison/README.md`.
Steady-state stability uses relative median absolute deviation so a single host
interrupt remains visible in the raw samples without invalidating an otherwise
resolved median.

The standard width matrix intentionally spans three regimes: 10 and 14 qubits
expose fixed and interactive-latency overheads, 18 and 22 qubits expose the
transition toward statevector-kernel cost, and 24 qubits provides a practical
memory-pressure point for a 4 GiB-class host. Results from only the small-width
regime must not be extrapolated into whole-simulator performance claims. Widths
of 26 qubits and above are separate, non-gating capacity probes because the two
engines and their working buffers can exceed 4 GiB even though one complex128
statevector is smaller than that limit.

The single hardware-efficient circuit above is useful for longitudinal
regression tracking, but it is not representative of every simulator workload.
Run the feature-labelled workload corpus to compare several circuit structures
through the same user-facing FlagQuantum execution paths:

```bash
pip install -e '.[qiskit,cirq,pennylane]'
flagquantum-benchmark run simulator_workload_corpus \
  --n-wires 10 14 18 22 --threads 1 --warmup 1 --iterations 5 \
  --calls-per-sample 1 \
  --json-output benchmarks/results/comparison/workload-corpus.json
```

The corpus records deterministic IR hashes and backend-neutral features beside
the raw timing samples. Its six families cover hardware-efficient, truncated
QFT, random Clifford, nearest-neighbor brickwork, dense nonlocal, and
low-entanglement SWAP-routing circuits.
See [the workload corpus guide](../docs/guides/SIMULATOR_WORKLOAD_CORPUS.md) and
the [checked-in Apple arm64 measurement](results/comparison/SIMULATOR_WORKLOAD_CORPUS_CPU_ARM64_20260924.md).
The focused 22-qubit SWAP-routing run is available as a
[raw comparison artifact](results/comparison/simulator_swap_routing_cpu_arm64_20260924.json).
The dense nonlocal CZ-graph optimization has a
[reproducible comparison report](results/comparison/SIMULATOR_DENSE_NONLOCAL_CPU_ARM64_20260924.md)
and a [raw comparison artifact](results/comparison/simulator_dense_nonlocal_cpu_arm64_20260924.json).
The static Random Clifford product-state routing change likewise has a
[reproducible comparison report](results/comparison/SIMULATOR_RANDOM_CLIFFORD_CPU_ARM64_20260924.md)
and a [raw comparison artifact](results/comparison/simulator_random_clifford_cpu_arm64_20260924.json).
The Truncated QFT controlled-phase graph optimization has a
[reproducible comparison report](results/comparison/SIMULATOR_TRUNCATED_QFT_CPU_ARM64_20260924.md);
its refreshed FlagQuantum measurements and unchanged external measurements are
stored in the workload-corpus artifact linked above.
These end-to-end timings include external conversion and backend preparation;
they answer the user-facing workload comparison question, not isolated kernel
throughput.

### Batched exact-statevector throughput

Measure one user task that evaluates the same circuit structure for independent
parameter bindings and returns one exact statevector per binding:

```bash
pip install -e '.[qiskit,cirq,pennylane]'
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
flagquantum-benchmark run batched_statevector_corpus \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 10 14 18 --batch-sizes 1 8 32 \
  --threads 1 --warmup 1 --iterations 5 \
  --json-output benchmarks/results/comparison/batched-statevectors.json \
  --markdown-output benchmarks/results/comparison/batched-statevectors.md
```

To measure the same task's real process-memory high-water mark without one
framework contaminating another, use the isolated RSS runner. It keeps warm
timing/correctness in the normal corpus and launches one fresh process per
engine/workload for memory:

```bash
flagquantum-benchmark run batched_statevector_memory \
  --workloads hardware_efficient_statevector truncated_qft_statevector \
    random_clifford_statevector local_brickwork_statevector \
    dense_nonlocal_statevector \
  --n-wires 18 --batch-sizes 32 \
  --engines flagquantum_native_batch flagquantum_native_layout_retention \
    flagquantum_native_functional_windows flagquantum_native_monolithic_batch \
    flagquantum_native_serial qiskit_aer_bridge cirq_simulator_bridge \
    pennylane_lightning_bridge \
  --threads 1 --warmup 2 --iterations 11 --memory-probes 3 \
  --json-output benchmarks/results/comparison/batched-memory.json \
  --markdown-output benchmarks/results/comparison/BATCHED_MEMORY.md
```

The checked Apple-arm64 result and interpretation are in
[`BATCHED_STATEVECTOR_MEMORY_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_MEMORY_CPU_ARM64_20261001_SCORECARD.md).
The follow-up
[`BATCHED_STATEVECTOR_MEMORY_PREALLOCATION_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_MEMORY_PREALLOCATION_CPU_ARM64_20261001_SCORECARD.md)
measures bounded zero-state allocation and inference-only one-qubit output
preallocation against the exact legacy window path. Its RSS values are medians
of three fresh processes, with every raw observation retained in JSON.
The focused
[`BATCHED_STATEVECTOR_LAYOUT_LIFETIME_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_LAYOUT_LIFETIME_CPU_ARM64_20261001_SCORECARD.md)
then measures early input-layout release against exact legacy retention on the
Random Clifford and local-brickwork allocation hotspots.
The native static-Clifford layer follow-up is documented in
[`BATCHED_STATEVECTOR_NATIVE_CLIFFORD_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_NATIVE_CLIFFORD_CPU_ARM64_20261001_SCORECARD.md).
It compares the optimized path with its explicit rollback and PennyLane
Lightning in the same run, including exact timing, peak RSS, correctness,
applicability boundaries, and the full reproduction command.
The next
[`BATCHED_STATEVECTOR_NATIVE_PARAMETERIZED_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_NATIVE_PARAMETERIZED_CPU_ARM64_20261001_SCORECARD.md)
measures the native batch-specific RX/RY/RZ layer against its exact rollback and
PennyLane Lightning. It records a 1.132x Random Clifford and 1.146x local-
brickwork improvement, while preserving autograd fallback and the public API.
The disjoint mixed-entangler follow-up is documented in
[`BATCHED_STATEVECTOR_NATIVE_CLIFFORD_MATCHING_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_NATIVE_CLIFFORD_MATCHING_CPU_ARM64_20261001_SCORECARD.md).
It fuses each pairwise-disjoint CX/CZ matching into one native CPU traversal,
records 1.311x and 1.199x rollback speedups, and prominently retains the measured
PennyLane Lightning lead and the exact reproduction command.
The focused phase-loop follow-up is documented in
[`BATCHED_STATEVECTOR_CLIFFORD_PHASE_MAP_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_CLIFFORD_PHASE_MAP_CPU_ARM64_20261001_SCORECARD.md).
It encodes CZ signs in the cached CX gather map below 22 qubits and records a
1.112x Random Clifford improvement. Its CZ-free local-brickwork control is
explicitly classified as not applicable, while the compact wide-state fallback
and measured PennyLane Lightning comparison remain visible.
The static-layer arithmetic follow-up is documented in
[`BATCHED_STATEVECTOR_STATIC_CLIFFORD_LAYER_CPU_ARM64_20261001_SCORECARD.md`](results/comparison/BATCHED_STATEVECTOR_STATIC_CLIFFORD_LAYER_CPU_ARM64_20261001_SCORECARD.md).
It collapses exact H/S/Sdg/X/Y/Z layers into one native traversal and records a
1.974x focused rollback speedup on the 18-qubit, batch-32 Random Clifford task.
On this measured task, FlagQuantum is 1.468x faster than the same-run PennyLane
Lightning bridge; the inapplicable local-brickwork control remains explicit.

Gate a fresh run against that maintained batch profile without confusing a
different machine, runtime family, timing scope, or memory API for a pass or a
regression:

```bash
flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/batched_statevector_layout_lifetime_cpu_arm64_20261001.json \
  candidate-batched-memory.json --max-slowdown 1.20 \
  --max-memory-growth 1.10 --minimum-samples 5 \
  --minimum-memory-probes 3 \
  --json-output candidate-batched-memory-gate.json
```

For this schema the gate requires the complete workload/width/batch/hash case
identity, exact-statevector correctness, stable native timing, at least the
requested timing samples, and fresh-process peak-RSS probes. A runtime-profile
or measurement-method mismatch returns `incomparable` rather than a false pass.
The checked Apple-arm64 replay, exact times, RSS ratios, unchanged PennyLane
Lightning context, and reproduction commands are in
[`BATCHED_STATEVECTOR_REGRESSION_GATE_CPU_ARM64_20261001.md`](results/comparison/BATCHED_STATEVECTOR_REGRESSION_GATE_CPU_ARM64_20261001.md).

FlagQuantum executes all bindings through its native parameter-batch path. The
current Qiskit Aer, Cirq, and PennyLane bridges accept one statevector request at
a time, so this runner repeats the public bridge call for each binding. It
therefore measures the complete task as exposed through FlagQuantum today; it
does not claim to measure each external framework's best native batching API.
The payload records total time, time per returned statevector, statevectors per
second, raw samples, correctness, stability, execution strategy, and logical
statevector storage. It does not measure process peak RSS.

See the [Apple arm64 scorecard](results/comparison/BATCHED_STATEVECTOR_CPU_ARM64_20260930_SCORECARD.md)
for the checked-in result, its important 18-qubit batching limitation, exact
times, ratios, and the focused stability rerun.

The follow-up
[wide-batch windowing scorecard](results/comparison/BATCHED_STATEVECTOR_CHUNKING_CPU_ARM64_20261001_SCORECARD.md)
measures the automatic 64 MiB CPU statevector windows against the same native
batch with windowing disabled, FlagQuantum scalar execution, Qiskit Aer, Cirq,
and PennyLane Lightning. It records exact task times, `1.05x`-`1.27x`
monolithic-batch speedups, remaining limitations, and the complete reproduction
command.

The [CPU phase-1 scorecard](results/comparison/CPU_PHASE1_SCORECARD_CPU_ARM64_20260930.md)
summarizes the maintained Apple arm64 corpus: FlagQuantum wins all 20 recorded
workload/width cases against each of Qiskit Aer, Cirq Simulator, and PennyLane
Lightning, with concrete times, speedup ranges, provenance, limitations, and
reproduction commands shown before the regression-gate details.

### Profile-aware CPU regression gate

After producing a candidate artifact on the same measurement profile as a
checked-in baseline, enforce correctness, stability, sample count, case-matrix
coverage, and a bounded median-time regression:

```bash
flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/simulator_workload_corpus_cpu_arm64_20260924.json \
  candidate-workload-corpus.json --max-slowdown 1.20 \
  --minimum-samples 5 --json-output cpu-forward-gate.json

flagquantum-benchmark run cpu_performance_gate \
  benchmarks/results/comparison/adjoint_differentiable_simulator_corpus_cpu_arm64_20260928.json \
  candidate-differentiable-corpus.json --max-slowdown 1.20 \
  --minimum-samples 5 --json-output cpu-adjoint-gate.json
```

Exit status `0` means pass, `1` means a correctness, stability, coverage, or
performance failure, and `2` means the artifacts are not comparable. Platform,
Python and PyTorch version families, device, thread limits, measurement scope,
and calls per sample must match. This prevents timings from an arbitrary CI
runner from approving or rejecting a baseline recorded on different hardware.

Measure whether the silent `fq.train` path avoids per-step CUDA scalar reads:

```bash
python benchmarks/internal/evidence/train_host_sync.py \
  --device cuda:0 --parameters 1024 --steps 2000 --repeats 9 \
  --output benchmarks/results/local/train_host_sync.json
```

Use `--workload quantum_statevector --wires 12 --parameters 24` to validate the
same behavior through a real `fq.Module` statevector training path.

Use `--device cuda` for the local statevector benchmark on a GPU host.

### CPU path decomposition

`statevector_local` reports one aggregate number for a mixed circuit, which
detects a broad regression but cannot attribute it. The CPU fast paths differ
per gate family, so attribute a CPU change with the path-decomposed runner:

```bash
flagquantum-benchmark run statevector_cpu_paths \
  --n-wires 20 --layers 8 --warmup 2 --iterations 5 \
  --json-output benchmarks/results/local/statevector_cpu_paths.json
```

It measures a rotation chain, a diagonal chain, a fused two-wire diagonal chain,
a CX ladder, a mixed chain, and joint marginal probabilities separately, records
the engine's own runtime statistics per case, and exits non-zero when a case
stops matching its reference. Restrict a run with `--cases`, control the
intra-op thread count with `--threads`, and lower `--n-wires`/`--layers` for a
smoke run.

The two-wire diagonal chain repeats one `cz` pair per layer rather than walking
a ladder, because two-qubit regions only fuse when consecutive gates share the
identical wire tuple. It is the case that separates the diagonal kernel from the
dense one where the single-wire kernel cannot reach: a two-wire region is
outside that kernel's domain.

Ratios computed from that payload share one host, one input, and one warmup
policy. Each case reports its candidate time, its reference time, and the paired
`reference/candidate` ratio; stability is gated on whether that ratio's standard
error is small enough to separate it from host noise, and `all_cases_stable` is
`false` when it is not. Read the paired ratio rather than a quotient of two
separately recorded medians, and treat `passed: false` as "this host could not
resolve the ratio", not as a library verdict.

Every ratio in that payload is internal: the candidate is the shipped path and
the reference is another route in this repository or a reduction computed from
the same state. No cross-framework comparison is run, so none of these numbers
is a claim about another library.

The payload also records the CPU kernel switches in force (`execution_flags`),
because a switch decides which kernel a case takes. Two payloads are comparable
only when that block agrees, and a switch that is off by default reads as
`"source": "code_default"` rather than as an absent key:

```bash
# Default state: the pre-existing kernels.
flagquantum-benchmark run statevector_cpu_paths \
  --cases rotation_chain --json-output /tmp/off.json
# The same case with the opt-in elementwise single-wire kernel.
FQ_CPU_SINGLE_WIRE_ELEMENTWISE=1 flagquantum-benchmark run statevector_cpu_paths \
  --cases rotation_chain --json-output /tmp/on.json
```

Run the two arms alternately inside one process rather than dividing two
separately recorded medians; the evidence limits in each payload say why.

Each case also carries `ir_gate_counts`, counted straight off the circuit IR. It
is there because `runtime_statistics` is written by the fusion code it reports
on, so it cannot on its own show that the fusion code counted correctly. The IR
is the input to that code, so the two numbers are independent, and the diagonal
count is taken with the same rule the compiler routes on - a named diagonal gate
carrying no matrix of its own.

Distributed measurements are launched with the usual `torchrun` environment;
the maintained report builders are also exposed by the same command:

```bash
flagquantum-benchmark run statevector_weak_scaling INPUT... \
  --json-output benchmarks/results/smoke/weak-scaling.json
flagquantum-benchmark run statevector_strong_scaling INPUT... \
  --json-output benchmarks/results/smoke/strong-scaling.json
flagquantum-benchmark run statevector_training_scaling INPUT... \
  --json-output benchmarks/results/smoke/training-scaling.json
```

## Directory contract

| Directory | Stability | Purpose |
| --- | --- | --- |
| `flagquantum/benchmarking/` | Supported | Packaged CLI, registry, result contract, and maintained runners. |
| `benchmarks/runners/` | Maintained | Reproducible hardware and workload entry points. |
| `benchmarks/results/` | Evidence | Structured local, comparison, smoke, and certified scalability results. |
| `benchmarks/internal/evidence/` | Internal | Development-evidence generators required by current contracts. |

Only `local/`, `comparison/`, `smoke/`, and `scalability/` are evidence
classes. Historical experiment families, plotting scripts, and ad-hoc analysis
belong in the external evidence archive. Migrate them out in reviewable batches according to the
[repository governance policy](../docs/development/REPOSITORY_GOVERNANCE.md).

Top-level Python files are implementation modules retained while their
maintained scenarios are adopted by the registry. Their filenames are not a
public interface; user automation should call `flagquantum-benchmark`.

## Result semantics

Use this result layout:

| Directory | Contents |
| --- | --- |
| `benchmarks/results/local/` | Single-device fast-path results. |
| `benchmarks/results/comparison/` | Cross-framework comparison payloads. |
| `benchmarks/results/smoke/` | Development, replicated, or environment smoke checks. |
| `benchmarks/results/scalability/` | Release-gate scalability payloads only. |

Audit result semantics with:

```bash
python -m benchmarks.audit_results --input benchmarks/results
python -m benchmarks.audit_results \
  --input benchmarks/results/scalability --require-scalability
```

Rank-local replicated kernels, data-parallel throughput checks, and smoke tests
must keep `scalability_claim_allowed=false` and must not use
`claim_evidence_type="production_training_benchmark"`. Release payloads also
require capacity and sharded optimizer-update evidence.

Production promotion is a two-step operation: the executor writes measurements
and a raw log, then `tools/seal_runtime_evidence.py` captures the commit,
workload hash, command, device UUIDs, topology, rank mapping, backend, warmup,
iterations, seeds, and log checksum. It signs the immutable envelope using
`FQ_EVIDENCE_SIGNING_KEY`. Candidates remain under
`benchmarks/results/smoke/release_candidates/` until verification succeeds.

Before an eight-GPU release run, validate the environment without printing
secret material:

```bash
python tools/check_release_evidence_environment.py --world-size 8
```
