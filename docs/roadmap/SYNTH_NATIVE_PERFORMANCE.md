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
