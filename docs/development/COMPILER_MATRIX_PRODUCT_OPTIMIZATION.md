# Fixed-size compiler matrix products

The private two-qubit matrix product retains its four products, their order,
Python's built-in `sum`, and the existing 4-by-4 representation. It constructs
those four products directly instead of creating an inner generator for each
output element. This reduces interpreter work in block composition without
changing routing, target synthesis, tolerances, evidence, dependencies or public
contracts. The path classification is `single_device_fast_path`.

Retaining `sum` matters: replacing it with chained addition need not preserve
Python-version-specific floating-point accumulation. This change is not a new
Qiskit/Cirq algorithm port. Existing synthesis and routing reuse remains owned by
the compiler, with its existing attribution and boundaries.

## Correctness evidence

[Raw production-source checks](evidence/compiler-matrix-products.json) record:

- 2,400 deterministic matrix pairs on each of Python 3.10.21, 3.11.16 and 3.12.14.
  Each candidate result has the same packed real/imaginary double bytes as the
  exact historical function on that interpreter. The corpus includes real-only,
  mixed and complex values, signed zeros, extreme scales, infinities and NaNs.
  This proves the inspected results, not every possible NaN payload or input.
- 60 inspected application/topology pairs and 120 independent acceptances:
  widths 6/8/12, QAOA/Ising/hopping/QFT/repetition, line/ring/grid/sparse.
  Normalized operations and final layouts, all acceptance metrics, selected
  strategy, restored SWAP count, rewrite count and maximum rewrite membership
  agree with the control.
- SDKBench stays at `9aa9260920984e2ca05bcb293abb885fccd9c144` with a clean
  checkout. The existing acceptance helper includes native target verification,
  serialization and endpoint identity layout checks. Six-wire acceptance uses
  full operators; larger cases use the existing three probes. These are inspected
  development controls, not held-out qualification or a formal participant run.

The control is the exact private function extracted from main commit
`b3ed982a0d90a05ec69b093723aa1b6e83630463`. Other compiler components stay at the
candidate revision. The initial raw reports record that checkout parent in
`implementation_commit`; `candidate_source_sha256` identifies the modified working
source actually executed. They do not claim to measure pristine historical main
against an entire new release. Workflow timings were collected while local tests
could also run, and must not be interpreted as performance evidence.

Reproduce with a local checkout containing the baseline Git object:

```bash
python -m benchmarks.compiler_matrix_products --output /tmp/matrix-products.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m benchmarks.compiler_matrix_products \
  --sdkbench-root /path/to/unchanged-sdkbench \
  --output /tmp/matrix-products-workflows.json
```

The first command needs only the standard library. The second needs the normal
compiler dependencies and the pinned SDKBench source. It reuses
`benchmarks.compiler_cz_identity_prefix._observe` so the independent acceptance
path is not duplicated. A failed check exits nonzero; the last successful
checkpoint retains `complete=false`. This driver does not implement SDKBench's
formal resource-qualified execution profile.

## Preliminary performance observations

[Exploratory raw samples and drivers](evidence/compiler-matrix-product-prototype.json)
retain the earlier prototype, its exact driver sources and the separate memory
observations. On this shared macOS host, with two warmups and seven alternating
retained samples per variant:

| Inspected workflow | Baseline median | Prototype median | Python traced peak, baseline / prototype |
| --- | ---: | ---: | ---: |
| 8-wire Ising, line | 171.631 ms | 170.609 ms | 3,166,778 / 3,226,880 bytes |
| 12-wire QFT, grid | 656.926 ms | 640.850 ms | 10,116,749 / 10,116,531 bytes |
| 12-wire QAOA, sparse | 181.340 ms | 177.110 ms | 3,094,304 / 3,093,960 bytes |

The corresponding local median differences are approximately 0.6%, 2.4% and
2.3%. The microbenchmark's roughly halved matrix-product time is not a claim of
halved compilation latency. No statistical significance or stable speedup is
established. The Ising allocation observation increased by 60,102 bytes; Python
allocation peaks are not process RSS. Profile cumulative times include profiler
overhead and cannot be added to unprofiled wall time. The production-source
checks above establish output agreement, not an independently qualified speedup.

## Verification and remaining limits

The focused compiler selection initially reported 1,616 passes and 26 failures.
Twenty-four failures came from omitted tracked evidence fixtures in the sparse
checkout; after restoring the original files, all 126 tests in the two affected
evidence modules passed. Two comparison anchors correctly rejected installed
Qiskit 2.6.2 because the repository certifies lanes 2.0 and 2.5. Both passed with
the already-cached Qiskit 2.5.0. Thus all 1,642 distinct selected cases passed
across these runs. Two earlier collection attempts also retained their missing
helper/example diagnostics. No test, threshold or certified lane was changed.

The selection includes unit composition/synthesis, symbolic and layout cases,
v1-v4 serialization and tamper cases, native physical evidence, compiler user
examples and benchmark contracts. Raw test output is retained in
[the validation record](evidence/compiler-matrix-product-validation.json).
The benchmark driver also passed focused mypy with imports skipped; this is not
a whole-repository type certification. Black, Ruff, architecture boundaries,
dependency policy and the public API migration baseline passed.
Full `pr-default` and `pr-runtime` certification is still required in complete
CI checkouts; the local focused selection is not a claim that those full tiers
passed. The formal Linux SDKBench campaign remains deferred. No machine restart,
GPU rerun or performance-gate waiver is part of this change.

An existing user workflow supported by this private optimization remains:

```bash
python -m examples.circuits.compiler.compiler_synthesis
```

It checks one-qubit, two-qubit and state-preparation synthesis against the shipped
runtime, reports target-basis membership, demonstrates explicit refusals and
ends with `FlagQuantum compiler synthesis check passed`.

## CI provenance vocabulary correction

The first PR head failed quality, CPU 3.12 and coverage shard 174 because the
evidence used two undeclared revision field names. The reports now reuse
`implementation_commit` for the compiler checkout and `source_commit` for the
external SDKBench checkout. Recorded revisions and numerical observations are
unchanged. The validation record retains the original document hashes, name
mapping and exact historical benchmark source. Historical `runner_sha256` values
continue to bind that source, while future runs emit the declared names. The
repository revision vocabulary and its test are unchanged.

## Integration update and unresolved worker exit

The branch now includes verified main `6db1fbda8a168599e5c0f155b9fb14052cf01342`,
including CI worker-exit diagnostics from PR #1275 and guarded target routing
plan reuse from PR #1269. This is a normal merge, not a change to the matrix
product implementation or its acceptance thresholds.

The previous exact-head coverage shard 130 in run `44952823962` reported one
worker as not properly terminated while executing
`test_complete_native_source_and_circuit[2-0.1274075]`; 308 other tests passed.
The available output does not establish its root cause. No test, timeout,
resource limit or variance threshold is waived. Including the integrated worker
diagnostics makes a subsequent failure observable instead of blindly repeating
the same diagnostic surface. The original failure remains in the run history.

The [post-merge standard-library check](evidence/compiler-matrix-products-main-sync.json)
completes 2,400 deterministic packed-double comparisons on Python 3.12.14 at
`466df8dbfb04196d5a50582cc35d2a9f8e21b2da`, with no mismatch. It is numerical
regression evidence only, not a rerun of the workflow timing or a full CI pass.
The reproduction commands above continue to apply.


## Integrated stack-watchdog verification

The branch now includes main `a2e131363aba76d67ef71ef155fff34b50786eb9`,
which contains the reviewed Python stack watchdog (PR #1281) and complete-operator
routing controls (PR #1283). Both prerequisite PR heads passed 246 exact-head
checks: 243 success and three optional skips. Main synchronization is a normal
merge at `021690dd4d11d32750a484b3634597017b8f70af`; the private matrix-product
implementation itself is unchanged.

[The synchronized record](evidence/compiler-matrix-products-watchdog-sync.json)
retains 2,400 deterministic packed-double comparisons on Python 3.12.14 and all
60 paired application/topology checks with 120 unchanged independent acceptances.
Normalized delivered outputs, routing/layout/resource metrics and rewrite
statistics agree. The separate five-file compiler selection passes 302 tests
(260 unit and 42 integration), including the newly integrated complete-operator
routing regressions. These are focused results, not complete repository tiers.
The paired workflow timings may overlap that test run and are not performance
observations.

The existing complete Rainbow original-source catalog reference also passes
under coverage.py 7.17.2 `sysmon` and the integrated Python watchdog with a
one-second diagnostic trigger: one selected test passed in 139.25 seconds,
33 deselected. That reference invokes all three original/native/independent
complete-state checks at `k=0,1,2`, including clean private environment tests.
The other catalog references are absent from this sparse checkout; they are not
claimed as executed. A Python stack snapshot was observed and execution completed.
This does not prove the root cause of prior Linux worker exits or replace full
Linux CI. The first invocation lacked coverage in the base interpreter and exited
before pytest; using the existing cached 7.17.2 package resolved this setup issue.
No dependency, assertion or resource threshold changed.

Reproduction of this diagnostic, with coverage available:

```sh
COVERAGE_CORE=sysmon python -m coverage run --source=flagquantum -m pytest   tests/integration/test_synthesis_native_catalog.py -k original_source_reference   -q -p tools.ci.pytest_stack_watchdog --fq-stack-dump-after=1   -o faulthandler_timeout=0
```

Use an external bounded supervisor; the completed diagnostic used a 360-second
process budget and 180-second no-output budget. The one-second trigger requests
a diagnostic snapshot, not test termination. CI retains its original diagnostic
trigger and job deadlines. Black, Ruff, English, local links and revision checks
pass. New full-source exact-head CI is required after this synchronization; no
prior failure is relabeled as successful and no gate waiver is requested.
