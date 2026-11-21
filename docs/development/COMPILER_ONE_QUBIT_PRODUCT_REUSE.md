# Reuse numeric products within local gate runs

This compiler change is a `single_device_fast_path` implementation improvement.
It changes no public API, optimization level, target admission, phase policy,
symbolic-parameter behavior, or serialization contract.

## Work removed

`collapse_one_qubit_runs` previously read each candidate gate's numeric matrix
when admitting it to a run and then read it again when composing the completed
run. It also retained all closed runs and copied the output list before writing
their replacements.

The traversal now retains the admitted matrix product for each open wire. It
finishes a run when an interfering instruction arrives and writes the replacement
at the run's original positions. Gates on spectator wires retain their positions.
Closed runs and a second output list are no longer retained. Each open wire holds
at most one 2-by-2 product; the output and open position lists still require space
proportional to source length.

A singleton does not perform a matrix multiplication. When a second gate joins,
the traversal starts with the original identity multiplication and then uses the
same left-associated multiplication order as `_run_product`. The existing Euler
emitter, absolute-phase reconstruction check, strict length reduction, vocabulary
decline, and small-component and trainable-input guards remain authoritative.
Reconstruction of the proposed replacement is still performed independently of
the retained source product.

Products exist only within the current traversal. No key, cross-pass cache,
source tensor, mutable matrix, or parameter value is retained for a later call.
The pending-wire organization follows the existing compiler traversal used by
[`_cz_local_cleanup`](COMPILER_CZ_CIRQ_REUSE.md). This change copies no additional
upstream implementation and introduces no dependency.

## Development evidence

Against parent `67a4f3f3d8ad59a6f91c4e72378f5c83973ce230`, 2,400 seeded programs
with widths 2, 4, and 6 have identical complete rewritten IR, including metadata.
Independent runtime-matrix tests cover every complex amplitude of interleaved
runs, reversed CX operands, protected parameters, and dense operands changed
between passes. A parent control passes 17 of the 18 new tests and fails the
matrix-read count regression: it reads each named source matrix twice rather
than once.

The retained local diagnostic uses 41 alternating warmed samples for two
1,000-instruction patterns. These inspected patterns are not held-out tasks.

| Pattern | Parent median | Candidate median | Parent Python peak | Candidate Python peak |
| --- | ---: | ---: | ---: | ---: |
| Mixed local runs | 3.818 ms | 2.404 ms | 64,816 bytes | 57,296 bytes |
| Singletons separated by CX | 1.016 ms | 1.018 ms | 81,600 bytes | 19,116 bytes |

The mixed-run pass is faster in this local observation. The singleton timing
difference does not establish a performance change. These are pass timings and
Python allocation peaks, not whole-compiler throughput, process RSS, quantum
capacity, cross-SDK results, or a superiority claim. The initial eager-product
prototype made singleton processing approximately 18% slower and was rejected;
the final path avoids that multiplication, with a focused regression test.

The [raw samples and source fingerprint](evidence/one-qubit-product-reuse-development.json)
are retained. Reproduce from a checkout containing the parent Git revision:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m benchmarks.compiler_one_qubit_product_reuse \
  --baseline 67a4f3f3d8ad59a6f91c4e72378f5c83973ce230
python -m pytest tests/unit/test_compilation_one_qubit_product_reuse.py \
  tests/unit/test_compilation_one_qubit_optimization.py -q
```

## Existing user workflow

```python
import flagquantum as fq
from flagquantum.compiler import optimize

source = fq.Circuit(2).h(0).h(1).h(0).h(1).cx(1, 0)
result = optimize(source)
assert tuple(gate.name for gate in result.instructions) == ("cx",)
```

The output and its exact phase are unchanged. The implementation avoids repeated
numeric reads while constructing that output. Formal comparative qualification
continues to require SDKBench's unchanged participant-neutral protocol.
