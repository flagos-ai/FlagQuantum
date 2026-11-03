# PF-01: reuse decision fields within one Services inspection

This second Services slice follows revision 9 sections 13.5–13.6 and
[#1021](https://github.com/flagos-ai/FlagQuantum/pull/1021), merged as
`f820a1f7eaa3990bed1bcf30f3adfd5199c23edd` after 57 successful checks and
three configured skips. It continues PF-01; it does not introduce PF-02 caching,
change Runtime's public properties or enable Rust migration.

## User workflow

The existing native program still prepares, computes, composes and runs:

```python
import flagquantum as fq
from flagquantum import synth
from flagquantum.synth import Const, Output, QNum, allocate, hadamard_transform


@synth.qfunc
def energy(x: Const[QNum], y: Const[QNum], result: Output[QNum]) -> None:
    result |= x*x+y


@synth.qfunc
def main(x: Output[QNum], y: Output[QNum], result: Output[QNum]) -> None:
    allocate(2, x)
    allocate(2, y)
    hadamard_transform(x)
    hadamard_transform(y)
    energy(x, y, result)


program = synth.synthesize(main, memory_limit_bytes=64 * 1024**2)
result = fq.run(program)
outcomes = dict(program.probabilities(result, registers=("x", "y", "result")))
assert outcomes.keys() == {(x, y, x*x+y) for x in range(4) for y in range(4)}
assert all(abs(p - 1/16) < 1e-12 for p in outcomes.values())
```

The matching Qmod and independently derived Circuit comparisons are in the
[preceding delivery](SYNTH_PYTHON_PERFORMANCE_PF01_DELIVERY.md#existing-user-workflow)
and the PR description. They are existing runnable workflows, not proposed APIs.
This change does not alter arithmetic, selected implementations, physical
layouts or circuit resources. It improves Python work in checked delivery and
result readout. The small native/Qmod example still uses 16 qubits, eight
workspace qubits, 81 operations and depth 59; the hand derivation still uses
nine qubits, one workspace, 15 operations and depth 11.

## Scope and semantic boundary

Only `services/_synthesis_execution.py` changes production code. `_program`
already obtains a complete, freshly validated canonical payload from Runtime's
owning `plan_to_dict(plan)` function. Previously it then read `mode`, `backend`,
`device`, `batch_size`, `precision` and `target` through plan properties. Each
property invokes `to_dict()` and performs full contract validation again.

The check now reads those fields from the decision mapping it already owns.
The raw `state_mode`, `world_size`, allocation bytes, RuntimeConfig, instruction
layers, analysis dimensions, basis, layout and budgets are still checked.
Actual layers still come from Runtime's authoritative `plan_execution_program`.
The full serialized payload is released before reconstruction as in #1021.
There is no global, object-ID or cross-call cache. Each new inspection/resource
access/result association performs its existing checks. Runtime's public
getters and serialization schema/identity/error vocabulary remain unchanged.

A new negative test constructs a valid sealed two-batch plan whose subclass
reports `batch_size == 1`. Inspection must reject the sealed profile rather
than trusting the virtual property. Existing serializer-override, mutable raw
configuration, sealed metadata/decision/fingerprint/extension, actual layer,
layout, budget and result-shape negatives remain enabled. The profile continues
to admit exactly one full local CPU Torch statevector.

## Reproducible comparison

Before: `f820a1f7eaa3990bed1bcf30f3adfd5199c23edd`.
After: `40a421eba6486d1c9ffad4217d398018d780ee57`.
Both are clean source pins. The original AO/PF-00 and #1021 receipts are retained
unchanged; this comparison uses a separate pair of archives.

The unchanged PF-00 driver measures ten fixed configurations: three small
logical entrances, native large energy/shared-square, native constant product,
native basis/line, Qmod line and a native budget refusal. Each side uses five
fresh timing processes with two warmups, three separate unprofiled RSS processes
and one intrusive diagnostic process per configuration, plus five cold starts.
GC and all contract checks remain enabled; Torch/BLAS/OpenMP threads are one.

E0 includes checked `synth.synthesize` delivery. E1 adds the first `fq.run` and
named readout where the fixed state budget permits execution. Independent
mathematics checks basis/nonzero-XOR/cleanup; small executions compare all
complex amplitudes and named outcomes. Large-state execution exclusions remain
explicit. Intrusive call counts identify the mechanism and are not natural
stage-time percentages. RSS is a separate OS-sampled compiler-window metric;
lifetime high-water and planned quantum-state bytes retain distinct scopes.

At each source pin run:

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m benchmarks.synthesis.performance --production-revision <full-SHA> \
  --output /tmp/pf01-decision.json --samples 5 --warmups 2 --memory-trials 3 \
  --case energy-2x2:scalar:inline:logical:success \
  --case energy-2x2:qfunc:inline:logical:success \
  --case energy-2x2:qmod:inline:logical:success \
  --case energy-8x8:qfunc:inline:logical:success \
  --case shared_square-8x8:qfunc:inline:logical:success \
  --case constant_product-5:qfunc:inline:logical:success \
  --case energy-2x2:qfunc:inline:basis:success \
  --case energy-2x2:qfunc:inline:line:success \
  --case energy-2x2:qmod:inline:line:success \
  --case energy-2x2:qfunc:inline:logical:refused
```

The host is nonexclusive; no thermal/frequency control or cross-host result is
claimed. The review line remains at least 10% and 20 ms in E0/E1, or at least
10% and 16 MiB in RSS. Missing trials cannot become zero costs or disappear into
successful-subset medians. A material other-axis regression requires review.
No wall-clock threshold is asserted in ordinary CI.

## Results and validation

Both complete primary cohorts contain 81 independently verified observations
and nine expected refusals, plus five cold starts. None failed or timed out.
Canonical circuit hashes, all resource counters and refusal diagnostic codes
match. The comparison includes every original sample and rejects mismatched
measurement code, protocol, environment, execution scope or selected circuits.

| Native qfunc workload | E0 before / after median | E1 before / after median |
| --- | --- | --- |
| Energy, 2+2 bits, logical | 29.754 / 19.079 ms | 94.528 / 68.197 ms |
| Energy, 8+8 bits, logical | 943.421 / 594.381 ms | Excluded by state budget |
| Shared square, 8+8 bits | 1496.023 / 997.954 ms | Excluded by state budget |
| Constant product, 5 bits | 74.746 / 50.284 ms | Excluded by state budget |
| Energy, 2+2 bits, basis | 239.715 / 178.496 ms | 477.260 / 344.803 ms |
| Energy, 2+2 bits, line | 744.570 / 475.264 ms | 1539.272 / 881.871 ms |

Six E0 groups and all six executable E1 groups meet the 10% + 20 ms review
line. Native large energy improves by 37.00%/349.04 ms and shared square by
33.29%/498.07 ms. The small logical E0 groups improve by approximately 35–38%,
but their 10–12 ms absolute reductions do not meet the E0 review line. Their
complete E1 reductions do meet it. The budget-refusal control shows no material
timing change; its diagnostic remains unchanged.

The small native diagnostic shows plan conversion/validation calls 37 → 10,
IR reconstruction 41 → 14, content hashing 86 → 59 and Instruction construction
3769 → 1582. These are intrusive diagnostic counts, separate from primary timing.
They support the audit finding: nine redundant validated property reads per
`_program` call, three such calls during synthesis. The remaining validation
and actual-layer reconstruction stay intact.

### Memory review

Primary shared-square sampled RSS shows a material increase: 180.922 →
223.734 MiB (23.66%, 42.81 MiB). That unfavorable observation is retained.
The exact same workload was reviewed with five before/five after fresh workers,
alternating AB/BA order and importing immutable source snapshots. All ten
independently verified their circuit and cleanup. Compiler-window RSS medians
were 214.281 → 223.547 MiB (4.32%, 9.27 MiB); lifetime high-water medians were
224.672 → 223.609 MiB. The original material regression did not reproduce at
the review threshold. The smaller increase remains visible; this does not prove
identical allocations or eliminate host variability. No RSS improvement is
claimed. No other primary group crosses the material regression line.

The RSS tool now accepts one successful identifier from the existing fixed
inventory, so the actual flagged expression can be reviewed without inventing
new workloads. Unknown/refused/conflicting selections fail before worker launch.
Earlier default and `--qmod-line` commands remain supported.

The new hash-bound artifacts are listed in
[`pf01-decision-manifest.json`](../../benchmarks/synthesis/baselines/pf01-decision-manifest.json).
Reproduce the comparison and exact-workload RSS review with:

```bash
python -m benchmarks.synthesis.performance_compare \
  --before benchmarks/synthesis/baselines/pf01-decision-before.json.gz \
  --after benchmarks/synthesis/baselines/pf01-decision-after.json.gz \
  --output /tmp/pf01-decision-comparison.json
python -m benchmarks.synthesis.performance_rss_review \
  --before f820a1f7eaa3990bed1bcf30f3adfd5199c23edd \
  --after 40a421eba6486d1c9ffad4217d398018d780ee57 \
  --case shared_square-8x8:qfunc:inline:logical:success \
  --output /tmp/pf01-decision-rss-review.json
```

The RSS controller source is `5c4d214a406daea1f707642148f9b0d3e0a437ad`;
the manifest binds that revision and its script hash to the raw review. Fetch
the complete recorded commits when validating from a shallow checkout.

Final affected-scope acceptance: **259 passed in 28.59 s**, including complete
small basis/XOR domains, relative complex phases, clean workspace, actual
nonidentity routing, mutable/sealed/serializer tampering, the new batch-property
override and invalid RSS-selection negatives. Strict mypy passed for 898
package/benchmark sources, with Ruff, Stable API, architecture, ownership and
documentation-link checks also passing. Broad local `pr-default` and exact-head
remote CI are recorded separately in the PR description and delivery ledger;
they are not substituted by scope acceptance or benchmark observations.

Remaining owner-domain work and any PF-02 reuse require a separate residual
profile and trust/invalidation argument. This slice does not declare the whole
PF track complete or change the AO-03/04/05 circuit-resource targets.
