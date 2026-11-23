# Constant scalar block reuse: rejected development candidate

The ordinary compiler's two-qubit block solver already reuses results for blocks
with immutable Python scalar parameters. Numeric parameters in ordinary circuit
construction often arrive as scalar Torch tensors. A candidate extended this
existing per-pass cache to their current numeric values. It is **not integrated**:
the development measurements found useful repeated-block gains together with
control-program regressions. Production behavior and public contracts are unchanged.

## Mechanism and semantic checks

The candidate accepts only exact `torch.Tensor` objects with zero dimensions,
without gradients, and with finite real values accepted by the existing scalar
reader. Tensor subclasses, symbolic parameters, batches, and matrix overrides do
not enter the cache. Keys retain exact numeric spellings, signed zero, canonical
opcodes, and operand positions relative to the ordered pair. Only numbers enter
the key; tensor identity is not a cache key. Every block reads its values again,
and the dictionary is discarded at the end of each invocation.

The candidate's 39 focused tests passed. They cover complete operators through all
16 computational-basis inputs of a four-qubit program, reversed pairs, metadata,
mutation between blocks and compilations, nonfinite mutation, trainable values,
shared-parameter gradients, symbolic values, and existing block-fold behavior.
An initial attempt had five failures because invalid test values were rejected
while constructing IR, before reaching the candidate. Those tests were corrected
to exercise the existing admission boundary rather than bypass it.

This experiment reuses FlagQuantum's existing solver and numeric reader. No
upstream implementation was copied. The existing collection/consolidation
relationship to Qiskit is documented in the
[upstream reuse audit](COMPILER_UPSTREAM_REUSE.md).

## Paired observations

The [complete evidence](evidence/compiler-constant-scalar-reuse.json) retains
the exact candidate patch, tests, driver, all timing samples, Python allocation
samples, input hashes, source hashes, and the rejection decision. The baseline is
`c3154a8c4eb71172d05e999e1d88348fb40e9b2a`.

Four arms distinguish this candidate from the separate commutation-pair candidate
at `9d1ce6c6c1ca4803abb7036d18dabd13285b7421`: baseline, scalar reuse alone,
commutation reuse alone, and both. Each arm has one warmup and nine timing samples
in rotated and reversed order. All 22 inputs produced exactly the baseline's
exported operation list and final layout in every arm. This pins CZ count and
dependency depth to the same exported circuit; it does not replace independent
numerical acceptance.

| Inspected input | Scalar / baseline time | Both / baseline time |
| --- | ---: | ---: |
| DNN, eight wires, grid | 0.873 | 0.968 |
| DNN, eight wires, line | 0.884 | 0.955 |
| QAOA, six wires, grid | 0.895 | 0.873 |
| QAOA, six wires, line | 0.910 | 0.880 |
| Simon, six wires, grid | 0.968 | 0.960 |
| Simon, six wires, line | 0.960 | 0.952 |

Sixteen seeded controls span two, four, eight, and twelve wires with 10, 50, 200,
and 800 operations, using random H, RX, RY, RZ, CX, and CZ gates on line targets.
Scalar-only time ratios range from 0.909 to 1.150. The eight-wire, 800-operation
control regresses from 0.11427 to 0.13137 seconds at the median. Combined ratios
range from 0.962 to 1.149. Gains from the two candidates are not additive: both
are slower than scalar-only reuse on the inspected DNN inputs.

Three separate Python allocation measurements per arm on the six inspected inputs
give scalar-only median peak ratios of approximately 0.9997–1.0059 and combined
ratios of 1.0001–1.0082. These are `tracemalloc` allocations, excluding native
allocations and process RSS. They cannot establish a process-memory bound.

## Decision and limits

This shared macOS observation is neither a dedicated-host SDKBench campaign nor
an unseen-program test. The six external inputs were already inspected. There is
no confidence interval, formal performance acceptance, participant ranking, or
scalability claim. All samples are retained; no favorable rerun replaces them.

The observed tradeoff is insufficient to enable scalar reuse by default. Reading
and constructing keys for nonrepeating values adds work; the present measurements
do not isolate how much of each regression comes from that work, allocation, or
host noise. A further candidate must establish a general admission or reuse
mechanism and measure its own overhead on varied workloads, rather than dispatch
on these input names or tune against their timings. The rejected patch and tests
remain reproducible evidence, not an additional implementation to maintain.

The production workflow this study supports remains:

```python
import flagquantum as fq
from flagquantum.compiler import optimize

circuit = fq.Circuit(2)
circuit.h(1)
circuit.cz(0, 1)
circuit.h(1)
optimized = optimize(circuit.to_ir())
assert [(gate.name, gate.wires) for gate in optimized.instructions] == [("cx", (0, 1))]
```

Full default and integration tiers were not run for the rejected implementation.
The report's own validation and exact-head CI status must be read separately from
the candidate's focused semantic tests.

The report-only default tier completed in 1,077.05 seconds with 21,549 passed,
796 skipped, eight failures and six setup errors. One failure exposed this
report's unregistered `commutation_revision` field; it is corrected to a nested
`revision` field without changing the recorded source or samples. The other
evidence-origin failure includes repository-history availability mismatches.
Six launch tests and six MPS setup errors encounter blocked local socket/address
operations in the sandbox. These results are not a passing default tier.
