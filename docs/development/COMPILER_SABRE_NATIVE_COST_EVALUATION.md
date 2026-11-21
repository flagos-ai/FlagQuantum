# Reducing target-CZ routing evaluation work

The benchmark-only target-CZ-aware SABRE selector now avoids identical plans and
reuses native lowering and exact matrix cost estimates within one selection.
Production routing and public contracts remain unchanged. The earlier decision
against default admission remains in force.

## Responsibility and safety

This is a `single_device_fast_path` development experiment. It uses the existing
SABRE search, native translator, Qiskit-derived KAK/Weyl engine and bounded native
synthesis. See [upstream reuse](COMPILER_UPSTREAM_REUSE.md) and the
[original experiment](COMPILER_SABRE_NATIVE_COST.md).

A profile on integration commit
`100b81b7fdb2c67414b395c27535f7cc065d3208` found that three independent selection
calls each rebuilt physical previews and repeated numeric/KAK work. On the
6-wire QFT grid control, 27 routing-trial calls used about 15 ms under profiling;
15 preview constructions used about 333 ms and six realizations about 249 ms.
These are overlapping profiler totals, including profiler overhead, not a
partition of unprofiled latency. Reducing routing search itself was not the
first opportunity indicated by those observations.

The implementation preserves deterministic trial order and skips plans identical
to the incumbent or an earlier candidate. Only distinct alternatives trigger
native evaluation. A closure shares translated, numerically validated source
instructions by source index and physical operands. Its lifetime is one selector
call for one source and coupling map. A second call gets new storage. Exact
packed real/imaginary matrix bytes identify repeated Weyl entangler-count
calculations; no rounding, approximate key or tolerance change is involved.
Storage is bounded by the incumbent and four existing trials and is released
after selection. This does not reuse synthesized executable outputs.

Actual bounded synthesis still checks the incumbent and shortlisted alternative.
The selected route must reduce realized CZ count without increasing delivered
two-body dependency depth. Existing growth budgets and residual checks remain
operative. Independent routing replay, complete artifact emission, serialization
and SDKBench acceptance still execute. No global or cross-call cache is added.
Skipping an identical candidate can change shortlist search behavior on circuits
outside these inspected controls; this experiment has not established universal
policy equivalence or admission for arbitrary targets.

## Validation and observations

The unchanged SDKBench checkout is pinned to
`9aa9260920984e2ca05bcb293abb885fccd9c144`. The already inspected nonce is
`01` repeated 32 times. Widths 6/8/12, five application families and four
topologies give 60 paired complete workflows and 120 independent acceptances.
All pairs retain identical complete normalized output hashes, evaluator metrics,
initial/final identity layouts, restored SWAP counts and rewrite statistics.
No comparison fixture, evaluator, tolerance or SDK-specific limit was changed.

One hundred thirty-five focused unit and physical-evidence integration tests pass, covering the zero-growth counterexample,
reversed operands, restored layouts, unsupported symbolic/annotated sources,
source/topology isolation the existing native optimization suite, and complete physical-evidence replay.

Two warmups and seven alternating retained samples per variant, followed by
separate Python allocation observations, compare the previous experimental
selector with the new evaluation implementation on the same production source:

| Inspected workflow | Previous median | Reused evaluation median | Reduction | Python traced peak, previous / reused |
| --- | ---: | ---: | ---: | ---: |
| 6-wire QFT, grid | 376.329 ms | 231.150 ms | 38.6% | 2,248,620 / 2,159,084 bytes |
| 6-wire repetition, ring | 361.836 ms | 225.314 ms | 37.7% | 2,241,217 / 2,414,529 bytes |
| 12-wire QFT, grid | 1,122.140 ms | 631.966 ms | 43.7% | 10,119,125 / 10,116,941 bytes |

The repetition control allocates 173,312 more peak traced bytes. This cost must
not disappear behind the timing improvement. These are shared-host development
observations, not qualified latency, process RSS, statistical significance,
SDK rankings or held-out results. They compare two experimental implementations;
they do not demonstrate an improvement over production defaults. Historical
measurements in the original study are not substituted as contemporaneous controls.

The first validation attempt used an outer process alarm, which conflicted with
SDKBench's own compile-stage timer and failed. The completed run uses a separate
supervisor with a 480-second process timeout; SDKBench's timer is unchanged.
An intermediate invocation without the required import environment also failed
before validation. Neither setup failure is counted as an acceptance.

[Raw paired outputs, samples, profile and exact drivers](evidence/sabre-native-cost-evaluation.json)
record the measured integration revision and source hashes. The benchmark-source
hash identifies the uncommitted implementation measured at that revision.
The subsequent parent report-size fix changes JSON formatting only.

## Reproduction and admission

The existing user-facing native artifact workflow is unchanged. Run the complete
benchmark with a clean pinned SDKBench checkout:

```bash
PYTHONPATH=.:/path/to/sdkbench/src python -m benchmarks.compiler_sabre_native_cost \
  --sdkbench-root /path/to/sdkbench --output /tmp/sabre-native-cost.json
python -m pytest tests/unit/test_compiler_sabre_native_cost.py \
  tests/unit/test_native_block_optimization.py -q
```

Expected: 60 inspected comparisons, no independent acceptance failure or guarded
quality regression, and 93 focused passing tests. The runner enables the policy
only in its scoped diagnostic context. Importing it does not enable a compiler
policy. The raw archive also retains the exact paired/timing driver and historical
selector source; its `/private/tmp` paths describe the measured local setup and
must be adjusted for another checkout. Preserve the frozen cases and budgets.

Before production admission, further work must establish affordable end-to-end
cost against a contemporaneous production control, broader unsupported-case and
failure behavior, and frozen prospective evidence. Formal host qualification
remains deferred; no machine reboot or kernel upgrade is needed for this work.
