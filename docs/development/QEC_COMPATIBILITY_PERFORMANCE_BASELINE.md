# QEC compatibility performance baseline

## Scope and ownership

The first compatibility route is complete through #822. The next step measures
where time is spent before selecting an implementation to optimize or replace.
This is a local CPU observation study, not a performance gate, GPU result,
threshold study or approval of a new QEC API. The user's 16 A800 devices have
not been accessed. Device allocation and single-card measurement precede any
multi-card experiment; independent shots do not establish state-capacity scaling.

The maintained `qec_memory_sampling` runner and its d=3 Z historical records
remain unchanged. This one-off study belongs in `benchmarks/`, composes existing
public APIs, and reuses the benchmark metadata/atomic writer. It adds no registry,
sampler, decoder, result authority or installed library export.

## Predeclared matrix and resource budget

Before inspecting outcomes, fix 48 cells: Surface distances 3/5/7/9/13/17,
X/Z readout, native-memory/exported-Stim entrances, and batch sizes 32/128.
Every cell uses three rounds, exactly 1025 requested shots (including a one-shot
tail for both batch sizes), seed 53, independent X/Z probabilities 0.005/0.007
and measurement probability 0.009. X uses the existing certified all-data X
product. Each fresh CPU process runs one warmup and three measured repetitions,
with one thread for Torch/OMP/MKL/OpenBLAS. Each repetition restarts the same
seed/call schedule; counts across repetitions are not independent trials.

One controller runs cells sequentially, at most 30 seconds per process and 600
seconds for the entire matrix. Retain raw stdout/stderr, all warmups and measured
rows, and any timeout/error. Do not change seeds, drop slow cells, add samples,
reduce repetitions or select a more favorable batch after viewing results. A
budget failure is incomplete evidence, not a reason to enlarge the budget
silently. The fixed matrix isolates distance/entry/batch effects at one total
shot count; it does not establish scaling with total shots.

Both entrances start with the same native memory declaration. Export source
with `memory_to_stim_text`; derive the owned DEM through strict conversion of
that original source, then initialize PyMatching separately. The native entrance
uses `compile_memory_sampler`; the exported-Stim entrance uses
`compile_stim_sampler` on the original source. No matching/decomposed graph is
used for sampling, and this is not a corpus of arbitrary external circuits.
Actual positioned Kraus law remains authoritative; nominal decimal precision
limitations in the compatibility plan are unchanged.

## Timing and memory interpretation

Record memory construction, native export, strict DEM construction, decoder
setup, sampler compilation, sampling, decoding and count reduction separately.
Pipeline wall time includes all these phases, handle close and bookkeeping;
it excludes initial process imports, final metadata and report IO. SDK imports
triggered during setup remain in that phase, including the first warmup.
Every repetition constructs fresh objects and reuses one handle over batches;
this measures a complete experiment, not only steady-state sampling. The shared
export needed for the execution-paired DEM is included for both entrances.

Process RSS is the whole fresh process high-water mark, including imports,
warmup, SDK scratch and all repetitions. Visible tensor bytes count only live
detector, observable and prediction tensors; neither metric is an exclusive
phase peak or proof of bounded total memory. Compare absolute phase medians and
pipeline medians, retain all observations, and distinguish cold setup dominance
from the recurring sampling/decoding cost. Do not infer GPU payoff from a CPU
phase percentage or combine phase minima into artificial throughput.

## Reproduction and user workflow

Install `python -m pip install -e '.[stim,pymatching]'`. Commit source changes
before collection; evidence records an exact clean producer revision. Run one
fresh process per matrix cell, for example:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
python benchmarks/qec_compatibility_matrix.py --distance 3 --basis x \
  --entrance native --shots 1025 --batch-size 32 --warmup 1 --iterations 3 \
  --output /tmp/qec-d3-x-native-b32.json
```

The public workflow under measurement is:

```python
from flagquantum.qec import (
    DetectorErrorModel, Pauli, PhenomenologicalNoise, PyMatchingDecoder,
    RotatedSurfaceCode, build_memory_circuit, compile_memory_sampler,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler, memory_to_stim_text, stim_circuit_to_dem_text,
)

code = RotatedSurfaceCode(3)
memory = build_memory_circuit(code, rounds=3, product=Pauli(x_qubits=code.data_qubits))
noise = PhenomenologicalNoise(data_flip=0.005, phase_flip=0.007, measurement_flip=0.009)
source = memory_to_stim_text(memory, noise=noise)
model = DetectorErrorModel.from_stim_text(stim_circuit_to_dem_text(source))
decoder = PyMatchingDecoder.from_detector_error_model(model.merge_duplicate_mechanisms())
for name in ("native", "exported_stim"):
    sampler = (compile_memory_sampler(memory, noise=noise, seed=53)
               if name == "native" else compile_stim_sampler(source, seed=53))
    completed = failures = 0
    with sampler:
        for requested in (8, 8, 1):
            sample = sampler.sample(shots=requested)
            predicted = decoder.decode_batch(sample.detectors)
            completed += sample.shots
            failures += int((predicted != sample.observables).any(dim=1).sum())
    print(name, completed, failures)
```

Both lines must report 17 completed shots. Logical counts are actual
version/platform-scoped observations, not pinned universal outcomes or a
performance acceptance criterion. Retain the executed example output with the
study evidence. The matrix is accepted as a measurement only if every cell
completes its declared workload, metadata/source/RSS meaning are explicit, and
error/cleanup and actual SDK tests pass. Optimization remains a separate PR
selected from the resulting report; no target speedup is enabled here.

The user example was executed on macOS arm64, Python3.12.14, Torch2.13.0,
Stim1.16.0 and PyMatching2.4.0 before study collection:

```text
native 17 0
exported_stim 17 0
```

These are scoped small-sample observations, not a fixed failure-count contract.
