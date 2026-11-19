# Native synthesis performance baseline

The accepted native Notebook library supplies workloads for a follow-up to the
PF-00 arithmetic baseline. This measurement does not rerun 212 Notebooks or
replace their scientific acceptance. It isolates repeated synthesis and execution
of six existing native Python programs before choosing an optimization.

| Workload | Retained native example |
| --- | --- |
| Arithmetic | Two-bit `x*x+y`, uniform inputs, complete clean output |
| QAOA | Five-vertex MaxCut, four layers, eight nonzero parameters |
| QPE | Eight-bit phase register, three data bits, original dense unitary |
| Pauli evolution | Three-qubit first-order Suzuki product |
| qDRIFT | Original 551-term Hamiltonian, 12 qubits, 2,000 samples |
| Block encoding | Seven-qubit checkerboard with packed fields |

## Reproduce

From a checkout installed in a Python environment with CPU Torch:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m benchmarks.synthesis.native_performance \
  --output /tmp/native-performance.json
```

Each workload uses five primary timing samples, three independent RSS samples,
and one diagnostic sample. Ordinary calls have a 60-second deadline; the intrusive
diagnostic has a separate 180-second deadline. Every sample starts a fresh worker and makes two
warmup calls before measurement. Use `--case qaoa` to select one workload. Source
revision, Python/JSON hashes, environment, thread limits, warmups, individual
measurements and numerical errors are retained. Failed samples remain failures;
there is no automatic retry or substitution of smaller programs.

The observer reuses PF-00's subprocess deadlines, process-group cleanup and
external RSS sampling. Compilation includes the existing `synthesize` call and
its target lowering. A separate intrusive run attributes compiler stages C1-C6;
exclusive spans are summed without counting nested calls twice. Primary timings
have no profiler or RSS sampler. Execution measures `fq.run(program)` and state
access after separate execution warmups; it excludes reference construction and
assertions. It does not claim to measure optimizer convergence, finite-shot
sampling, plotting or complete Notebook wall time.

Every accepted sample compares the complete complex state against the existing
independent Circuit, or the mathematical `x*x+y` oracle. Absolute phase, wire
order and all allocated work wires remain observable. Compiled circuit hashes
and resource records must agree across a workload's samples. The qDRIFT sample
retains its fixed sampling sequence and all 2,000 draws.

Compile and execution RSS windows are separate. Their peaks are sampled lower
bounds, not exact allocation maxima; short spikes may be missed. Lifetime
`ru_maxrss` includes imports and warmups and is never subtracted to infer a stage
peak. Process-tree RSS can double-count shared pages. Diagnostic stage memory
includes profiler overhead and is not a substitute for primary memory samples.

## User workflow supported

```python
import flagquantum as fq
from examples.synthesis.algorithms.search_and_optimization.qaoa.qaoa_max_cut.main import (
    build_program,
)

program = build_program()
result = fq.run(program)
print(dict(program.probabilities(result, registers=("v",))))
```

This is the same native `qfunc` path used by the baseline. Performance work must
preserve these probabilities, the complete complex state and resource budgets.
No new user-facing API is introduced.

## Optimization gate

Choose the next change from measured repeated work in a shared production path.
Preserve raw before/after measurements and independent numerical checks. Report
compilation, execution and RSS separately, including regressions and variability.
The PF-01 review threshold remains a median improvement of at least 10% and 20 ms,
or a sampled RSS reduction of at least 10% and 16 MiB. Smaller or noisy differences
are unproven benefits. These are local review thresholds, not CI timing assertions.

Start with Python. Validation reuse needs a proven identity and invalidation
boundary; measurements alone do not authorize skipping validation. A Rust port
is a later, separately measured decision. Routine tests cover observer behavior
and representative full states; the full timing run is an explicit local command.

## First measured change

The initial cohort located repeated checks in Services after construction.
`inspect_execution` already computed and checked the actual executable's resource
estimate. Construction now receives that estimate alongside its checked plan,
instead of immediately inspecting the same plan again solely to obtain the same
costs. The normal `inspect_execution` return type is preserved. No result is
cached across calls, and the final `Program` validation remains in place.

`Program._validate` also avoids hashing the submitted circuit a second time when
it is the exact object just validated by `TargetProduct.validate`. A separately
supplied equal circuit still receives a content comparison. Mutating either a
shared or detached circuit remains a refusal. Plan controls, physical layout,
basis, budgets, result association and mutable nested data retain their checks.
This is local repeated-work removal in Services, not a new trust model or PF-02
snapshot cache.

All six workloads retain identical circuit hashes and resource records, and all
effective before/after samples pass their independent complete-state checks.
Diagnostic plan inspections during synthesis drop from three to two. The local
sequential cohorts measured:

| Workload | Compile before / after | Run before / after | Compile change |
| --- | ---: | ---: | ---: |
| Arithmetic | 10.59 / 11.94 ms | 1.54 / 1.71 ms | 12.7% slower |
| QAOA | 26.23 / 27.47 ms | 6.71 / 7.80 ms | 4.7% slower |
| Matrix QPE | 8.75 / 7.33 s | 3.16 / 3.00 s | 16.2% faster |
| Pauli product | 5.31 / 4.77 ms | 1.69 / 1.54 ms | 10.1% faster |
| qDRIFT | 1.20 / 0.97 s | 0.39 / 0.31 s | 19.5% faster |
| Checkerboard | 5.92 / 5.17 ms | 1.49 / 1.38 ms | 12.7% faster |

These are medians of five fresh workers per side, each with two warmups. QPE and
qDRIFT exceed the time review threshold. Small-case differences are below the
20 ms absolute threshold and do not establish either a speedup or a material
regression. Compile-plus-run medians decrease 11.5% for QPE and 19.5% for qDRIFT.
No workload crosses the combined 10% and 16 MiB RSS regression threshold; no
memory benefit is established. Measurements use one nonexclusive Apple M5 host,
CPU Torch, Python 3.12.14 and fixed thread counts. Sequential cohorts cannot rule
out host-load or thermal effects, so these results are not a universal guarantee.

The [raw bundle](../../benchmarks/synthesis/baselines/native-20261119/manifest.json)
retains all pilot failures and the initial 54-sample attempt. That attempt had
one qDRIFT sample whose numerical check passed but process-group cleanup failed;
it is not relabeled successful. After allowing a completed worker up to two
seconds for normal teardown, a full separate nine-sample qDRIFT cohort passed.
The comparator selects that entire corrective cohort, rather than replacing a
single inconvenient measurement. The other five before cohorts and all six
after cohorts are complete. Timeouts still trigger prompt forced cleanup.

Recalculate the comparison without running the quantum workloads:

```sh
python benchmarks/synthesis/baselines/native-20261119/compare.py
```

The bundle stores raw hashes, actual production/observer sources, per-sample
state errors, ranges, call counts and memory points. The comparison refuses
failed selected samples, inconsistent resources, incomplete cohorts and
unexpected source changes. Historical AO-00 and PF-00 baselines remain intact.
