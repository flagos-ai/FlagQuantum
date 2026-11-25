# Avoid solving an unchanged two-qubit fallback twice

This is a `single_device_fast_path` compiler implementation change. It changes
no public API, optimization level, routing policy, target admission, numerical
acceptance, serialized result, or dependency.

## Work removed

The existing target-independent block pass first tries a complete block. If
that fails, it tries the maximal all-two-qubit sub-runs separated by local gates.
This preserves narrower cancellation opportunities when the larger product has
no shorter spelling in the declared opcode vocabulary.

When the entire block contains only two-qubit gates, its sole sub-run is exactly
the rejected block. Trying that sub-run cannot reveal another replacement. The
existing call-local cache already avoids a second numerical solve for immutable
Python scalar blocks, but still rebuilds and looks up their keys. Tensor
parameters and caller-owned matrices deliberately bypass that cache and could
reach the numerical solver twice.

The pass now skips only a fallback whose source positions equal the already
rejected complete block. It still solves the complete block once, including its
identity check. A block containing local gates still tries its genuinely narrower
sub-runs. Collection, multiplication order, small-component guards, exact-phase
checks, vocabulary restrictions, metadata, and replacement positions are
unchanged. No tensor, matrix, result, or rejection is cached across invocations.
This removes duplicate work inside the existing native collector; it imports no
new upstream algorithm or code.

## Verification and observed limits

Against parent `cf5fc36c29c3f9162b6b58054f8163c98944e33d`, 2,400 seeded programs
produce identical complete rewritten IR, including metadata. Widths are 2/4/6;
the generator includes all supported two-qubit families, local gates, measurement
and barrier boundaries, Python scalar angles, static scalar tensors and trainable
scalar tensors. These are inspected development inputs, not held-out tasks.

Independent runtime-matrix tests compare complete operators with their original
phase. They cover both operand orders, a declined mixed-family block, narrower
cancellation across a local member, and a dense matrix changed between pass
invocations. All six new cases pass. The parent control passes three and fails
three solver-count assertions: the static tensor blocks are solved twice, and a
declined dense singleton is also redundantly solved. The semantic checks retain
the same expected outputs. The existing two-qubit tests plus new cases pass
31 tests.

The expanded focused selection, including application CZ checks and two-qubit
decomposition conformance, passes 63 tests. The unchanged native SDKBench adapter
suite passes all 60 tests with FlagQuantum from this source and Qiskit 2.6.2 /
Cirq 1.7.0. These are correctness/conformance checks, not comparative timings.
The local full default tier did not pass: disk exhaustion during a separate
diagnostic clone caused checkout failures followed by test temporary-file write
errors. Its terminal summary is 9,457 passed, 368 skipped, 6 failed and 2,160
errors before completion. The failed clone's generated contents were removed;
the test log is retained. This run cannot establish whole-tier health or identify
each failure's cause. A complete CI run is still required before merge.

The unchanged SDKBench evaluator at
`9aa9260920984e2ca05bcb293abb885fccd9c144` also accepts all 120 application
workflow cases: five families, widths 6/8/12, line/ring/grid/sparse targets,
ordinary-restored and explicit target-artifact compilation. Every parsed result
record equals the retained result for the parent main tree. The fixed public
development nonce is `01` repeated 32 times. Six-qubit cases compare complete
operators up to a global phase; widths 8/12 check three output probes and do not
prove universal equivalence. Maximum semantic error is
`6.072657351995283e-14`. The [application records](evidence/two-qubit-fallback-applications.json)
retain acceptance, audit and numerical fields. No resource qualification or
cross-SDK comparison is inferred from these checks.

One warmed local diagnostic uses 41 alternating samples per implementation:

| Inspected pass input | Parent median | Candidate median | Parent Python peak | Candidate Python peak |
| --- | ---: | ---: | ---: | ---: |
| 100 declined scalar-tensor blocks | 5.222 ms | 2.939 ms | 25,596 bytes | 25,516 bytes |
| 100 declined Python-float blocks | 0.768 ms | 0.650 ms | 31,998 bytes | 31,942 bytes |
| Mixed blocks with narrower folds | 8.408 ms | 8.646 ms | 71,300 bytes | 71,276 bytes |

The third control retains all its solves and is approximately 2.8% slower in
this observation. All raw samples are retained; no timing threshold or universal
speedup is inferred. These are timings and Python allocation peaks of one pass,
not whole-compiler latency, process RSS, capacity, hardware fidelity, or an SDK
comparison. The implementation preserves narrower opportunities even where it
removes no numerical work.

The initial differential generator incorrectly treated dynamic directives as
ordinary registered gates, then omitted their existing `is_dynamic` marker.
Those two attempts failed while constructing inputs, before completing any
comparison. The final generator uses the repository's established directive
representation; the 2,400-case result is from that corrected generator. No
production validation rule or test expectation was relaxed.

The benchmark [report](evidence/two-qubit-fallback-development.json) retains
complete samples, Python/platform details and both source fingerprints.
Reproduce with the parent Git object available locally:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_two_qubit_fallback \
  --baseline cf5fc36c29c3f9162b6b58054f8163c98944e33d
python -m pytest tests/unit/test_compilation_two_qubit_fallback.py \
  tests/unit/test_compilation_two_qubit_optimization.py -q
```

## Existing user workflow

```python
import flagquantum as fq
import torch
from flagquantum.compiler import optimize

source = fq.Circuit(2, dtype=torch.complex128)
source.gate("rxx", (0, 1), params={"theta": torch.tensor(0.17, dtype=torch.float64)})
source.gate("rzz", (0, 1), params={"theta": torch.tensor(0.23, dtype=torch.float64)})
result = optimize(source)
assert tuple(item.name for item in result.instructions) == ("rxx", "rzz")
```

The product has no shorter admitted target-independent spelling and remains
unchanged. Its rejected block is solved once per pass invocation. Formal
SDKBench qualification remains a separate requirement under the unchanged
participant-neutral protocol; this diagnostic makes no ranking claim.
