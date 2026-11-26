# Integrated numerical compiler controls

Two independently merged optimizations now compose on integrated main
`4c8805d2882c4593e5ddfed95d03a7a353d02fd4`: the fixed 4-by-4 product
implementation from PR #1276 and fixed-singleton matrix deferral from PR #1288.
This record checks their composition through complete existing compilation
workflows. It adds no production implementation, API, dependency, routing policy,
acceptance threshold or benchmark fixture. The path is Compiler-owned
`single_device_fast_path`. The [upstream reuse record](COMPILER_UPSTREAM_REUSE.md)
remains the account of actual Qiskit/Cirq adaptations; these two changes remove
native Python overhead rather than porting another upstream algorithm.

## Source isolation and independent acceptance

The [complete archive](evidence/compiler-integrated-numerical-controls.json)
retains the exact driver, source hashes, paired outputs, metrics, adapter audits,
raw timing samples and separate Python allocation observations. Candidate code is
integrated main above. For each baseline observation, only
`one_qubit_optimization.collapse_one_qubit_runs` and
`two_qubit_optimization._matmul` are replaced with the function definitions parsed
from `11745ab11070765974b90f5a8f512bf4c30576b1`. Loaded aliases are replaced
within a temporary context and restored afterward. All other implementation and
harness code is identical. This isolates the composition of these two changes;
it is not a comparison between two complete historical checkouts.

The unchanged clean SDKBench checkout is pinned to
`9aa9260920984e2ca05bcb293abb885fccd9c144`. Its fixtures, adapters, stage timers,
resource budgets and independent routing evaluator are unchanged. The previously
inspected nonce is reused. Widths 6/8/12, five application families and four
topologies run through both ordinary restored and native target-artifact
workflows: 120 pairs and 240 independent acceptances. Every pair has identical
complete normalized operation/layout digests, quality metrics and adapter audit
counters. Identity endpoint layouts are explicitly required. Eighty six-wire
executions use full-operator acceptance up to global phase; 160 larger executions
use three probes only. This is regression evidence on inspected inputs, not a
fresh prospective study or proof for arbitrary inputs.

## Repeated workflow observations

After paired acceptance, separate target-artifact observations use two warmups
and nine retained samples per variant, alternating execution order. The timed
region is the complete adapter `execute("route", ...)` call; fixture generation,
bundle loading and independent evaluation are outside that region. Separate
tracemalloc observations include fixture, bundle, adapter, compilation and
independent acceptance; they are not isolated compiler peaks or process RSS.

| Inspected case | Baseline / integrated median | Python traced peak bytes, baseline / integrated |
| --- | ---: | ---: |
| 6-wire qft, grid | 106.818 / 94.299 ms | 35,364,735 / 35,364,869 |
| 6-wire repetition, ring | 119.534 / 117.975 ms | 35,365,257 / 35,364,091 |
| 12-wire qft, grid | 584.759 / 567.378 ms | 35,364,676 / 35,364,314 |

Observed medians are lower in these three runs. Traced peaks are dominated by
whole-observation setup and acceptance (approximately 35 MB) and differ by less
than 1.2 KB; they do not establish a meaningful compiler memory reduction.
No local test run overlapped the measurements. Python 3.12.14 on the shared
macOS development host uses one numerical thread. This host is neither reserved
nor qualified, and these observations do not establish statistical significance,
a general speedup, cross-SDK ranking, process RSS, accelerator behavior or
scalability. One-shot times in the paired rows are not substituted for repeated
workflow timings. The separate native-CZ selector's resource-admission gap
remains unresolved; this control does not authorize its production admission.
Formal host-qualified SDKBench remains deferred by the user. No machine, service
or kernel was restarted or changed.

## Verification and existing user workflow

On the exact integrated revision, the seeded sparse `pr-default` tier passes
169 unit tests and `pr-runtime` passes 42 integration tests. These local counts
cover the tests present in the sparse checkout, not the whole repository.
The two implementation PRs passed their respective full-source CI before merge;
this document does not treat those separate runs as a full-source test of their
composition.

The existing local public workflow needs no hardware or credentials:

```python
import flagquantum as fq
from flagquantum.compiler import optimize

source = fq.Circuit(3).h(0).h(1).h(2).to_ir()
assert optimize(source).instructions == source.instructions
```

Expected: independent gates remain unchanged. Existing same-wire and two-qubit
folds retain their operator, phase, parameter and validation contracts.

```sh
python tools/ci/ci_tier.py pr-default
python tools/ci/ci_tier.py pr-runtime
```

To reproduce the archived experiment, use its exact source pins and driver,
adjust its temporary harness/output paths, and set `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. The run used a separate
600-second process-group supervisor with TERM/KILL cleanup. Do not install an
outer signal alarm over SDKBench's stage timer. Preserve unsuccessful results;
repeating inspected cases cannot turn them into prospective evidence.
