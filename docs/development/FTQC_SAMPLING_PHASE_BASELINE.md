# FTS-02C: Compiled memory sampling phase observations

## Scope and acceptance

The user selected baseline collection first on 2026-10-24; the proposed 1.5x
throughput and 2x capacity targets remain disabled. This delivery fills a W6
measurement gap in the [master plan](../roadmap/FTQC_SIMULATION_MASTER_PLAN.md).
It adds a maintained benchmark observer, not a sampler, decoder, QEC public API,
approximation, FTQC package, performance gate or capability promotion. It reuses
`compile_memory_sampler`, the existing PyMatching logical-prediction batch method,
and the benchmark registry/metadata/atomic writer. The independent #812 Ising
comparison and #811 provider work are outside this delivery.

The earlier [47 initial observations](FTQC_002_CPU_BASELINE.md) retain their
original source revision and numerical values. The source for this branch starts
at main `fa586473e365aa30871e376e7aec7cff76538321`, including independently merged
#813's Stim probability-text precision fix. Nothing here relabels an older
observation as produced by the corrected engine.

## Frozen initial measurement

- One distance-three rotated Surface Z-memory experiment, three rounds; data,
  phase and measurement flip rates 0.005, 0.007 and 0.009; seed 53.
- Initial study: total shots 128/1024/4096, batch size 32, three warmups and ten
  measured repetitions. Invoke a fresh process per shot-count point. Larger
  resource allocations remain separate decisions. The CLI bounds shots to 4096,
  batch size to 128, warmups to three and repetitions to ten.
- Every repetition constructs the memory/DEM/decoder and compiles a fresh handle,
  then reuses that handle across all chunks, including the final short chunk.
  It closes the handle even if sampling, prediction or count reduction raises.
  A failure propagates without writing a partial success report.
- Each repetition restarts the same seed and sample-call sequence. Repeated
  counts are not independent statistical trials. No Wilson formula or adaptive
  estimator is added; the existing fixed-shot statistics example remains the
  authority for that workflow.
- The raw observations include warmups. The median excludes only the explicitly
  labelled warmups; no observations are removed for being slow.

## Timing and memory meaning

Separate timers cover memory construction, DEM construction, decoder preparation
(including mechanism merging), sampler compilation, sampling, logical prediction
and shot-level count reduction. The actual pipeline wall time also includes
context cleanup and observer bookkeeping. It excludes module imports, metadata
collection, JSON writing and file I/O. Throughput uses that actual pipeline wall
time, rather than a sum of phase minima. These observation overheads are not
subtracted or presented as an optimized uninstrumented implementation.

Whole-process maximum RSS is recorded before and after the full invocation. The
after value includes imports, warmups, all repetitions, engine and SDK state;
it is not an exclusive phase peak or proof of bounded total memory. If the
platform has no `resource` measurement, RSS is explicitly unavailable. Separately,
`visible_batch_tensor_bytes_max` counts only detector, observable and predicted
logical tensors live together at the observation point. It excludes temporary
reductions, allocator caches and engine/SDK scratch. Do not treat that number as
process/device peak memory or a capacity certificate. More total shots at fixed
batch size need empirical fresh-process RSS observations, not a symbolic claim.

Source checkout results record a full commit and tracked-tree cleanliness.
Installed-package results explicitly leave those fields unavailable. Source and
installed-distribution versions are distinct, so an editable environment's old
distribution metadata is not mistaken for the executed source version.

## Runnable public-user workflow

Use a CPU environment with Torch and the `stim`/`pymatching` extras installed.
No GPU, credentials or remote jobs are required. Save this as
`/tmp/qec_sampling_reference.py` and run
`PYTHONPATH=. python /tmp/qec_sampling_reference.py` from the checkout, or run
`python /tmp/qec_sampling_reference.py` with an installed build and its extras.

```python
from flagquantum.qec import (
    DetectorErrorModel, PhenomenologicalNoise, PyMatchingDecoder,
    RotatedSurfaceCode, build_memory_circuit, compile_memory_sampler,
)

memory = build_memory_circuit(RotatedSurfaceCode(distance=3), rounds=3)
noise = PhenomenologicalNoise(
    data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
)
model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
decoder = PyMatchingDecoder.from_detector_error_model(model.merge_duplicate_mechanisms())
completed = 0
with compile_memory_sampler(memory, noise=noise, seed=53) as sampler:
    for shots in (8, 8, 1):
        sample = sampler.sample(shots=shots)
        predicted = decoder.decode_batch(sample.detectors)
        assert predicted.shape == sample.observables.shape == (shots, 1)
        completed += sample.shots
print("Completed shots:", completed)
print("Logical observables:", model.num_observables)
```

Expected stdout:

```text
Completed shots: 17
Logical observables: 1
```

The observer for this existing workflow is discoverable through the maintained
CLI. Run each command in a separate process with an external 120-second bound:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python -m flagquantum.benchmarking run qec_memory_sampling \
  --shots 128 --batch-size 32 --warmup 3 --iterations 10 \
  --json-output /tmp/qec-memory-128.json
```

Repeat with 1024 and 4096 shots and separate output files. Expected invariants:
13 raw observations, three warmups; every observation completes the requested
shots in `ceil(shots / batch_size)` chunks with one logical observable. The
metadata must say local CPU, single process and non-release evidence, with
scalability/release claims disabled. Timings, RSS and logical failures are
measured outputs and are not universally pinned. This observer adds no threshold
that can turn a smaller or faster run into a mission-level success.

## Post-#810 mainline acceptance

Main `f42a060bdd3a5f2dc98bded657da9ed9ef63a750` passes
[complete CI, 54/54](https://github.com/flagos-ai/FlagQuantum/actions/runs/40086817430),
[source/single-/required two-GPU checks](https://github.com/flagos-ai/FlagQuantum/actions/runs/40086821308)
and [smoke](https://github.com/flagos-ai/FlagQuantum/actions/runs/40086700639).
Three existing optional manual GPU pools were unselected. QEC, retirement,
Simulation fault/Stim and Ising suites pass with 2,242 passed and one original
inapplicable-case skip. Fresh sdist/wheel-from-sdist contain all 823 package Python
members byte-identical to that exact source. A fresh Python 3.12.14 / Torch 2.14.1
core install passes dependency/native/Bell checks, both master-plan examples and
H-T-H/feedback outside the checkout. No historical twelve-profile rerun is claimed.

The initial local install hit disk exhaustion, and the first recovered download
exceeded its 600-second network bound. Logs were preserved; only disposable task
environments/cache were pruned with manifests. The same default Torch 2.14.1
resolution then completed within a 1200-second download budget using hardlinks.
No CI, numerical or performance criterion was changed to accept it.

| Artifact | SHA-256 |
| --- | --- |
| sdist | `e058a9eab35ddd9c193b03bcead79c28d3bfbd42329e87bd2c43b6f67f049269` |
| macOS arm64 abi3 wheel | `620e8c76811bd3bad4a1fcf4136fcb991bd5c15e39170ffb067a1b052bfe3014` |

This is acceptance of the exact post-#810 snapshot, including independently
delivered #806-809. The later #813 source change is audited and tested on the
new branch rather than inferred from this older acceptance. FTS-02 remains
partial; W1 phase timing, larger construction/shot regimes, mixed-engine
selection, domestic-device allocation and licensed experimental data remain open.
