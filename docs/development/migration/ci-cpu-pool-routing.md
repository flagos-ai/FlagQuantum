# Coverage CPU pool routing

Prerequisite: qualified PR [#1352](https://github.com/flagos-ai/FlagQuantum/pull/1352),
head `9a3f580d7979115412adb686ed9fea6f8ec450aa`. Its three workflows pass on
attempt 1: 432 successful checks and four existing conditional skips. All 384
CPU reports are retained and verified. Whole executed test-partition maxima are
Runtime 485 seconds, coverage 586, unit 464, and distributed CPU 425, including
preparation/upload and excluding queue.

## Problem and resulting routing

The previous completed workflow reached 211 simultaneous executing jobs, rather
than a global limit of 35. Its job labels constrained 32 Runtime and 28 coverage
partitions to the A800 CPU pool; four ordinary coverage partitions also landed
there because their general CPU selector included A800. Long-case queues remain
after the other pools finish.

Ordinary coverage partitions now select the existing BW or MC CPU pool, in an
approximately two-to-one split matching the observed pool sizes. They therefore
do not consume the A800 pool while its expensive cases wait. Runtime routing is
unchanged. Existing expensive A800 cases, including Maxwell and both complete
logical-cat studies, retain their original source, admitted profile, and workload.

Two existing complete autocallable cases receive explicit BW targets:

| Original case suffix | Existing target | Observed test phase |
| --- | --- | --- |
| `test_complete_native_source_and_circuit[2-0.22]` | `fq-bw1100-15-cpu16-validation` | 81.439 seconds |
| `test_complete_native_source_and_circuit[2-0.1274075]` | `fq-bw1100-16-cpu16-validation` | 81.467 seconds |

Both belong to `tests/integration/test_synthesis_native_autocallable.py`. Their
full original inputs, complete state/workspace assertions, and tolerances remain
unchanged. They keep one pytest worker and four numerical threads under the
already-admitted four-CPU/24-GiB profile. Targeted cases remain on distinct hosts,
preserving the existing test that prevents serial heavy cases from accumulating
on one host. The earlier five explicit BW routes remain intact.

The [qualification record](ci-cpu-pool-routing-qualification.json) contains the
actual proposed-host results, source pin, locked image, resource values and memory
events. Both cases passed with no OOM, but observed memory reached the quota;
this is not a memory-headroom or capacity certificate. Offline preparation and
test phases are not claimed to be whole GitHub-job timings. Initial standalone
collection failed because its snapshot omitted a benchmark verification helper;
that zero-call result is retained separately. The corrected immutable closure
includes that helper from the same commit. No scientific code was edited.

## Preserved boundaries

The change only affects coverage matrix routing. It does not change the workflow,
trust rules, fork PR hosted fallback, eight hosted concurrency buckets, runtime
CPU budget code, dependency profiles, runner registrations/services/labels/quotas,
duration hints, partition counts, scientific inputs, safety limits, or timeouts.
No product Python or native source changes, and no public API or migration-ledger
change, are required.

The planner prefix through writing the plan files has the same AST as the
qualified parent after normalizing one local loop-variable rename. That rename
separates a batch-list variable from a batch-key variable so the tool passes
strict typing; it changes no planning computation. A dual replay over the actual previous 38,563-node union,
canonically ordered for both implementations, produces byte-identical 256 plan
files. Only routing metadata changes: 26 A800, seven explicit BW targets, 148
ordinary BW, and 75 ordinary MC partitions. This replay does not claim recovery of
the old collection order or execution of the next CI head.

Local acceptance includes all 29 existing planner tests on the final tool,
strict tool typing, and the complete default selection: 21,921 passed and 864
existing conditions, with the same 22,785 nodes and outcomes as the qualified
parent. All 1,264 product Python files remain byte-identical to that parent;
this routing-only change does not require rebuilding the native wheel. This
change's own exact-head CI remains pending. Acceptance requires
all original scientific singletons and call phases, all selected nodes once,
existing profile admission and host separation, and measured whole test-partition
times. A scheduling prediction or standalone profile cannot replace those checks.

## Existing user workflow

```python
import flagquantum as fq

result = fq.run(fq.Circuit(2).h(0).cx(0, 1))
print(result.state.abs().square())
# tensor([[0.5000, 0.0000, 0.0000, 0.5000]])
```

This CI change supports the same user workflow while reducing unrelated work in
the constrained CPU pool. It introduces no numerical optimization or scalability
claim. The architecture migration, including the prepared Runtime retirements
and the directly #1084-dependent work, remains open; Rust stays deferred.
