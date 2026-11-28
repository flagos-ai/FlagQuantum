# CPU pool routing and bounded artifact transport

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

Coverage matrix routing, the main source-pack base, and coverage-download retry
boundaries change.
Trust rules, fork PR hosted fallback, eight hosted concurrency buckets, runtime
CPU budget code, dependency profiles, runner registrations/services/labels/quotas,
duration hints, partition counts, scientific inputs, safety limits, and test timeouts
remain unchanged.
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

The first routing commit collected the same 38,563 coverage nodes in byte-identical
256 partition lists. All 384 original CPU reports are retained, digest-checked,
and bound to actual merge `a10b64e3aeaa247b024cb1b6fc0851d96379fb94`:
128 Runtime partitions execute 14,259 selected nodes once; 256 coverage partitions
account for all selected nodes and preserve all 83 previous scientific singletons
and their existing call phases. All actual coverage job labels, runner names,
and admitted resources are verified, including the 26/7/148/75 route split.
Whole maxima are Runtime 672, coverage 593, unit 445, and distributed CPU 489
seconds. Three Runtime jobs exceed 600 seconds because of source transfer.
Coverage aggregation and the dependent completion gate fail on transport;
425 other Main jobs and both other workflows pass. This first routing revision
is therefore not an all-green CI qualification.

All 1,264 product Python files remain byte-identical to qualified #1352; no native
source changes or new native-wheel build are claimed. The final workflow revision
passes 21,926 local default cases with 864 existing conditions; all 22,785 original
nodes/outcomes match, with only five new Git source-topology cases. All 162 focused
tests, package typing, and architecture/API/format/documentation gates pass.
Its own exact-head CI remains required, including complete original science and
whole-job timing checks.

## Incremental source transfer

On the first routing commit, three successful Runtime partitions exceeded the
600-second whole-job target: 616, 617, and 672 seconds. Their initial incremental
source artifact downloads took 278, 389, and 350 seconds respectively. Their test
phases were 284, 175, and 255 seconds, compared with 283, 175, and 254 on the
qualified parent. Passing scientific tests therefore did not establish the
whole-job timing target.

For a pull-request merge, the main producer now prefers the previous PR head
(`HEAD^2^`) as its incremental tree base. This avoids retransmitting unchanged
objects from earlier PR commits that returning consumers already cache. An
explicit `FQ_CI_SOURCE_BASE` still takes precedence, including failing closed for
an invalid explicit base. A missing second-parent ancestor, a linear checkout,
or a non-PR event retains the existing first-parent history fallback.

The payload still carries the exact merge commit and Git path proof for the
source-repair helper. Consumer digest checks, exact-SHA checks, complete-tree
verification, bounded cache-miss repair, complete-source fallback, and final
checkout verification remain unchanged. A cold or older cache still requires
those existing repair/fallback paths; a previous PR tree is not assumed present.
The full-source pack producer, artifact format, and workflow permissions remain
unchanged. The separate local-GPU workflow producer is unchanged.

Replaying the actual first routing CI merge object produces a 67,845-byte
incremental pack instead of 2,157,948 bytes (96.86% smaller). Both producers yield
the same 77,338,581-byte full pack and SHA-256. This is a measured payload
reduction for that immutable object, not a download-speed certificate or a claim
about a later CI run. Five new real-Git topology cases verify the PR base,
explicit override, non-PR and linear fallbacks, invalid-base rejection, and a
complete cold checkout from the full artifact. Together with the existing source
restoration and planner cases, all 61 focused tests pass. Complete default
selection and this revision's exact-head CI must pass independently.

## Coverage download retry boundary

The original aggregate job started downloading coverage at 01:55:02 UTC on
2026-11-28. After six minutes, the enclosing composite action timed out. Its
nested retry began at 02:01:02 and immediately timed out within the same expired
step. The integrity validator, coverage combine, and policy gate never ran.
This failure is retained as transport failure, not a passing coverage policy.

The first coverage-data step now has a two-minute limit and permits a separate
workflow-level retry on failure, unless the job is cancelled. That retry has its
own three-minute limit and cannot ignore failure. Both use the existing isolated
artifact downloader with identical exact-source pattern, merge setting and
strict digest checks. Each invocation stages into a separate private directory.
The validator and combiner select only the successful invocation's output;
all 256 files, SQLite integrity, combine, and coverage policy remain mandatory.
The normal success path skips the additional download. Neither scientific test
budgets nor the aggregate job's existing 20-minute limit change.

GitHub's [composite action schema](https://github.com/actions/runner/blob/main/src/Runner.Worker/action_yaml.json)
does not provide a per-nested-step timeout. The two limits therefore belong to
workflow steps; the shared composite and third-party download action are
unchanged. The aggregate's total download allowance becomes five minutes across
two independent steps, instead of a six-minute step that could exhaust its own
retry. Local contract tests verify the retry condition, matching inputs, required
second-attempt failure, isolated selected paths, and preserved complete-data gate.
Actual recovery from a stalled transfer still requires the new workflow run.

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
