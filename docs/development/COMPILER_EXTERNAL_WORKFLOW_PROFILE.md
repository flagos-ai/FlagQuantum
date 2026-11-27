# External application compilation cost diagnosis

This report locates work for the next native compiler change. It changes no
production behavior, target admission, SDKBench protocol or capability level.
All cases are previously inspected development inputs. The profile is not a
cross-SDK speed comparison, a resource-qualified benchmark, or held-out evidence.

## Exact inputs and observation boundary

FlagQuantum revision `96e972b9cc1dff728455dcc9ba84b48f730d328d` includes the
separately approved source/v4 capacity changes under review in PRs #1326/#1327.
It is an unmerged candidate, not a verified integrated release. SDKBench remains
unchanged at `5ba03de33f8a9193f86b26d2cb3fa841ede7c98a`.

Both existing FlagQuantum workflows receive each of the six external DNN, QAOA
and Simon line/grid fixtures. The first diagnostic profiles the first adapter
execution in a fresh process. The second performs two independently accepted
warmups before profiling another execution, still in a fresh process per case.
The complete cProfile records include all functions, call counts, self and
cumulative time. Profile collection stops before independent numerical acceptance.
The adapter's existing deadline and output policy remain unchanged.

Every worker has a separate 20-second outer wall budget, bounded process-group
cleanup, one requested numerical thread and observed operating-system RSS.
There is no hard memory enforcement or exclusive host reservation on macOS.
All 24 profiled executions and 24 warmups pass unchanged acceptance. Six-wire
QAOA/Simon use complete operators; eight-wire DNN uses three probes only.
No returned result or source was repaired, and no outcome was dropped.

## Attribution after warmup

Times below are instrumented cumulative milliseconds. Nested functions overlap:
columns must not be summed. These are single observations, not latency estimates.
The two workflows have different delivered obligations and are kept separate.

| Input | Ordinary adapter | Fixed-point optimization | Automatic routing | Artifact adapter | Native legalization (3 calls) | Baseline replay (2 calls) | IR value encoding |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EXT-02-GRID-08 | 564.904 | 375.020 | 32.689 | 3803.140 | 1440.961 | 1078.565 | 554.611 |
| EXT-02-LINE-08 | 501.781 | 360.911 | 26.477 | 4014.396 | 1489.811 | 1129.219 | 548.805 |
| EXT-03-GRID-06 | 151.676 | 109.911 | 9.257 | 1086.372 | 396.002 | 311.155 | 113.369 |
| EXT-03-LINE-06 | 139.827 | 94.936 | 10.498 | 1213.652 | 418.511 | 321.548 | 126.666 |
| EXT-04-GRID-06 | 28.562 | 16.989 | 3.516 | 225.812 | 68.115 | 51.067 | 27.250 |
| EXT-04-LINE-06 | 27.208 | 16.983 | 2.847 | 216.124 | 66.746 | 50.017 | 25.034 |

On these ordinary workflows, fixed-point optimization is the larger cumulative
region; automatic routing is a smaller one. DNN grid has three two-qubit block
collapse passes, three one-qubit collapse passes and six commutation analyses.
The pending structural pair reuse in PR #1323 is absent from the measured source;
its complete-workflow effect must be measured separately after integration.

The artifact workflow performs native legalization once to produce its baseline
and twice more during the separately invoked proposal/verifier replay checks.
DNN grid records 3,096 top-level translation calls, 25,272 recursive expansion
calls and 22,032 Euler synthesis calls. It also records 1,824,802 IR value-encoder
calls and 104,573 instruction constructor calls. Native legalization and IR
encoding are overlapping regions, not independent costs. This indicates that
retained-baseline translation and repeated representation work deserve scrutiny
before further routing-only latency tuning.

## Next implementation decision and protected invariants

Inspect the existing translation search and mature target-directed decomposition
patterns before proposing another optimization. A useful next candidate must
remove measured work within the existing Compiler boundary and preserve concrete
target predicates, rule ordering, parameter and metadata ownership, complete
source/target/identity checks, independent replay and all resource budgets.
Do not remove replay, cache mutable IR hashes across calls, assume opcode names
imply target support, or introduce a new persistent registry to fit these inputs.

Earlier parameterless translation reuse and matrix-reader caching were rejected
by their declared whole-workflow controls; this profile does not reverse those
decisions. See [translation reuse](COMPILER_TRANSLATION_REUSE_CONTROL.md),
[matrix readers](COMPILER_NATIVE_MATRIX_READS.md) and
[upstream reuse](COMPILER_UPSTREAM_REUSE.md). No new upstream code was copied in
this diagnostic, and no dependency or license boundary changes.

A future candidate needs independently verified numerical outputs, mutation and
unsupported-case coverage, varied generated inputs, complete-workflow unprofiled
paired timing/RSS, and prospective neutral validation. Profiler overhead and cold
imports can distort times; the first-execution record is retained separately to
make that distinction inspectable. No statistical significance, speedup or
causal explanation beyond the observed call graph is claimed.

## Existing contributor workflow and evidence

```bash
python -m examples.target_aware_compilation
python -m pytest tests/unit/test_native_block_optimization.py \
  tests/integration/test_native_block_physical_evidence.py -q
```

The existing workflow compiles for legal target edges, restores logical placement
and checks equivalent local CPU execution. It introduces no new user API.

The [compressed JSON evidence](evidence/compiler-external-workflow-profile.json.gz)
retains both original reports in full, original JSON SHA-256 values, driver
sources, every worker outcome, configuration, fixture digest, warmup acceptance,
RSS and every cProfile function record. Temporary paths describe the measured
setup; changing locations for reproduction must preserve pinned source and
fixture identities. Decompression requires only Python's standard `gzip` module.

Archive SHA-256: `ae02c5b20ca3654c618723faa9103031f72ec5b15be9ae27448d603e5c04c94e`; size 544,190 bytes.

## Report verification

Fourteen documentation-link and public-evidence privacy tests passed (6.74 s),
and `python -m examples.target_aware_compilation` completed with valid edges,
restored mapping and local statevector execution. Archive completeness, embedded
source/report identities, English report text and credential-pattern scanning
passed. All cold, warm and warmup metrics agree within each case/workflow.
These focused checks do not certify full default/runtime tiers or current-head
CI. Those remain mandatory before integration.
