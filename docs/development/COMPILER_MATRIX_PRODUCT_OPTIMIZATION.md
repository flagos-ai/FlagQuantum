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
python -m examples.compiler_synthesis
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
