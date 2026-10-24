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

## Collected bounded observations

The clean producer was `d6deab509391bfdbd97ade7d92d05a18c41bdda6`.
[All cell outcomes and raw successful reports](evidence/qec_compatibility_matrix_cpu_20261024.json)
retain all four observations per completed cell, including warmup, source and
versions, exact workload, thread settings and resource meaning. Collection
commands retain their arguments with only the destination path normalized;
original commands/logs remain in the local work evidence. No timing or numerical
observation is replaced. The evidence commit does not change the producer code.

The declared 48 cells were all attempted in 552.34 seconds, within the 600-second
study limit. **36 completed and 12 timed out at the original 30-second process
limit. Complete-matrix acceptance is not met.** All d=13 native cells and all
d=17 cells timed out; their full-pipeline timing, memory peak and shot outcomes
are unavailable. A timeout bounds the entire process (imports plus four runs),
not an individual compilation. Child process groups were terminated/reaped.
The controller retained every outcome and returned failure for the incomplete
study; no retry, budget increase, seed change or extra sample was used.

Completed cells used macOS arm64, Python3.12.14, Torch2.13.0, Stim1.16.0 and
PyMatching2.4.0, one Torch/OMP/MKL/OpenBLAS thread. Table ranges span X/Z and
batch32/128 cell medians; they are not a confidence interval or phase minimum.
Only the three measured repeats per completed cell contribute to its median.

| Distance | Native median pipeline range (s) | Exported-Stim median pipeline range (s) | Native / exported completion |
| --- | --- | --- | --- |
| 3 | 0.110341–0.124346 | 0.059877–0.065700 | 4/4 / 4/4 |
| 5 | 0.396576–0.449270 | 0.204851–0.235704 | 4/4 / 4/4 |
| 7 | 1.011134–1.201008 | 0.517772–0.589179 | 4/4 / 4/4 |
| 9 | 1.914069–2.336420 | 1.014342–1.216978 | 4/4 / 4/4 |
| 13 | unavailable (timeouts) | 3.545308–4.692846 | 0/4 / 4/4 |
| 17 | unavailable (timeouts) | unavailable (timeouts) | 0/4 / 0/4 |

### Bottleneck disposition and next slice

Across the 36 completed cells, the per-cell median share of pipeline wall spent
in construction/export/strict-DEM/decoder-setup/sampler-compile was 93.89%–99.44%.
The corresponding sampling-plus-decoding share was 0.56%–5.83%. These ratios
are computed within each measured row before taking a median; no phase minima
are summed into an artificial pipeline. Count reduction also includes visible
tensor-byte bookkeeping, so it is not a pure product-kernel cost.

The next optimization investigation should profile native export/preparation
and native sampler compilation on a completed distance-nine case, establish
which existing internal work is repeated, then fix one demonstrated cost while
preserving approved APIs, parity/reference semantics and actual noise law. This
phase-level observation alone does not identify a particular function or prove
that repeated work can safely be cached. A cProfile/implementation investigation
is still required before selecting a concrete optimization.

Rerun the identical matrix after a justified optimization, retaining this
original incomplete study and its timeout outcomes. A successful future large
cell cannot be presented as an observed speedup against an unavailable baseline
timing. No GPU-payoff, d=17 throughput, memory scaling, threshold or speedup
criterion is established here. Keep the existing historical d=3 baseline
separate: the strict source-derived DEM and extra export phase make this a
different complete workflow. Do not relabel those older timings.

## Delivery sequence

1. Deliver this measurement/budget-boundary record with actual SDK, CI selection,
   type/format/architecture/API and default-layer evidence. The 12 timeouts
   remain incomplete measurements, not passing cells.
2. Profile one completed d=9 case, change one demonstrated preparation cost,
   and compare the same workload against the preserved producer. Keep public
   contracts/noise/parity semantics unchanged. Rerun the original matrix.
3. Decide on a single internal replacement or single-card experiment only
   after the new profile identifies a remaining bottleneck and an accessible
   device/budget is recorded. Multi-card independent-shot throughput requires
   its own actual evidence and is not single-state capacity scaling.

The integration/benchmark-contract tests are deliberately both marked so the
existing coverage lane that installs Stim/PyMatching actually selects them. No
workflow/dependency policy or scientific assertion is weakened. Initial fixture
context handling and the missing integration selection were caught and fixed
before formal collection; original failure logs remain in work evidence.
