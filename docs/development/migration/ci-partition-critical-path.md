# Test partition critical path

Status: implemented in the working tree with full local-source acceptance;
exact new published-head CI and its elapsed-job acceptance remain pending. A passing correctness gate alone does
not certify the ten-minute job goal.

## Observed baseline

PR 1343 head `32f1d3fff711966acd72454ba94ee7ffeb01288c` passed all applicable
checks in CI run `45968406496`, attempt 1. The original source artifact binds
actual tested merge `d7de090aad93322b1a46b02a719f5b0676f9bfc2`.
The complete coverage selection contains 38,359 nodes.

| Test group | Original partitions | Longest completed job |
| --- | ---: | ---: |
| Coverage | 192 | 43.92 minutes |
| Runtime integration | 8 | 28.90 minutes |
| Python 3.12 smoke/unit | 1 | 10.88 minutes |
| Distributed CPU contracts | 6 | 9.15 minutes |

All 192 successful coverage logs are retained externally by job ID and SHA-256.
Their printed slowest-50 phase reports provide 4,136 positive per-node duration
sums, each verified against the original collected plan. This is a bounded timing
sample, not timings for every selected node. The scheduling-data provenance
retains the source, tested merge, complete job IDs and original log hashes.
Original logs and measurements are never overwritten with candidate outcomes.

The three previous tail partitions each contained 625 nodes. Their actual
pytest wall times were 2,211.56, 2,435.94 and 2,484.56 seconds. Each service
reported four allocated CPUs and four single-thread test workers. Individual
full cases took 2,189.54 seconds for Maxwell and approximately 1,200 seconds
for each complete network-resiliency comparison. More ordinary shards alone
cannot shorten an indivisible computation.

## Implementation

- Coverage uses up to 256 nonempty partitions. This is within the documented
  [GitHub matrix limit](https://docs.github.com/en/actions/reference/limits).
  Updated measured weights replace stale estimates; all collected nodes still
  occur exactly once. Cases measured at 120 seconds or more receive sole-node
  slots when capacity permits. If a requested small fan-out cannot isolate
  every case, the remaining cases still execute without omission.
- A sole heavy case runs through one test worker with the CPU share already
  admitted by the existing affinity, cgroup, memory and registration policy.
  Threads replace parallel test workers; the service quota is not increased.
- The existing pytest plugin retains every completed setup/call/teardown phase
  in a source-bound `test-durations.json`, including serial and xdist execution.
  Worker-local state is merged by the controller. Counts and exit status expose
  incomplete or failed runs; the telemetry never changes report outcomes.
  This avoids relying only on the printed slowest-50 sample for later updates.
  Reports record the actual execution profile, worker and thread counts.
- Runtime integration uses 128 measured partitions with the original bounded
  parallel-worker, single-thread configuration for mixed cases. Python 3.12 smoke/unit uses four partitions. Marker filters,
  command selections and all applicable checks remain unchanged. Runtime
  launchers retain one thread per rank. Hosted concurrency remains spread
  across eight bounded buckets. Existing distributed contract partitions are
  retained; their baseline jobs already finished within ten minutes.
- Fixed CPU Toffoli uses the existing exact permutation implementation from
  PR 1253 head `fdf1e2c77b57b5bb14d00f431e97843bc90b2623`, with its original
  complete CPU mathematical, gradient and input-ownership tests. The CPU path
  enters the existing fixed-permutation dispatcher. Explicit matrices and GPU
  catalog routing retain their previous paths. Only owned states without
  gradients reuse their storage. No new public API, dependency or native ABI
  is introduced.

The fan-out refusal test now uses 257 instead of 193 because 256 is the new
supported cap. Its rejection assertion is unchanged. This authorized CI limit
change is the only replaced original parameter value in that test; it is not
an omitted scientific scenario or a weakened coverage threshold. Concurrency
fixtures now exercise all 256 rows rather than the former 192.

## Acceptance completed so far

- All 168 focused planning, quota, trust/routing, tier, shard and CPU-Toffoli
  tests pass in the actual Git checkout.
- Replaying the complete original 38,359-node corpus yields 256 nonempty
  partitions, no added or removed nodes and no duplicate assignment. All 75
  cases measured at 120 seconds or more have sole-node slots. The largest
  ordinary partition contains 253 nodes. This replay is explicitly labeled as
  uncommitted working-tree data, not the next CI plan.
- One unchanged 17-qubit, 2,383-operation Maxwell walk, containing 1,952 CCX
  operations, gives exactly identical full output bytes before and after the
  optimization. Local macOS arm64 cProfile wall times were 1.3603 and 0.3365
  seconds. These are diagnostics, not Linux CI or scalability evidence.
- The untouched complete Maxwell test passes from the newly built and separately
  installed candidate wheel in 239.64 seconds locally. It executes every one of
  1,372 walk selections, 1,373 phase rows and 3,272,222 operations at the original
  width, preserving all projection, norm, error, fidelity and independent
  classical assertions. All 278 loaded product modules come from that wheel.
- The original 24-qubit, twenty-layer reference binding was profiled separately
  without changing its input or 4,724 operations. Its complete output and
  profile are retained as original evidence; it is not a replacement oracle
  generated by the optimized implementation.

## Required published acceptance

The target is that each test partition's entire executed job, including setup
and upload, should finish within 600 seconds when practical. Existing finite
60-minute safety limits, manual full-acceptance budgets, original workload sizes,
marker selections, assertions and coverage policy remain unchanged. The target
is not implemented by cancelling cases at ten minutes or classifying them as
optional.

Before publication, complete the actual-source regressions, quality gates,
source-to-wheel binding and installed-artifact checks for the combined change.
Then inspect the new run's actual source and collected plan, verify every
unchanged original scenario and the explicitly updated CI boundary parameter,
and measure every completed partition by its real job timestamps. Retain the
new resource, per-test and job-duration evidence. Investigate any remaining
outlier; do not call the ten-minute objective achieved from this replay or from
local timings.


## Final local-source validation

The complete actual-checkout suites pass: Unit/API/Team 19,516/972,
QEC 2,801/329, root/local integration 1,592/223 and benchmark contracts
1,096/50 (passed/skipped). All eleven quality checks, four supplementary
gates and real optional JAX 135/6 pass. The unchanged public API, qubit
vocabulary, owner rules and architecture limits remain enforced.

The first fourteen formal failures and original logs are retained, followed
by 187 passing corrective scenarios and the complete successful rerun.
The fixes retain the required-check name policy and original module-size
ceiling. Full timing metadata records all six deliberate subprocess cases,
including the real failed and skipped cases, under both serial and xdist;
it does not convert an underlying failure into success.

This local acceptance does not certify the 600-second Linux job objective.
The published run must still preserve the actual complete selection and
satisfy all required gates, and its real job durations determine the
remaining performance work.


## Final installed-artifact acceptance

The final native wheel
`114b07f5a99eb97293af843608d76da91017584640029f6b19a47494533210b1`
matches all 1,265 actual-checkout product Python files. Its separate installation
passes 197 JAX/CPU cases (six existing hardware skips), 724 public/compatibility
cases (seven skips and one source-only AST case covered by the full source
suite), all 735 original output comparisons, native loading, cold imports and
the CLI. A fresh full-amplitude 17-qubit walk matches the original bytes.

The complete 1,372-walk Maxwell acceptance was run on the earlier native wheel
70bdfb37d2. All final mathematical product ASTs match that wheel; the only product
Python difference is a private explanatory docstring moved into its README.
All 126 native and build-input files match the original preflight, and the same
offline builder and flags were used. The final wheel independently passes binary
loading and the fresh walk; the complete Maxwell case was not rerun for that
documentation-only product difference. Original scientific inputs and assertions
remain untouched. The final exact-source Linux CI is still required.


## Preserve the QEC scientific thread contract

The first published run exposed a real CI context regression: the exclusive
Runtime profile admitted twelve Torch threads, while the existing matched QEC
initialization, training and validation workflows require exactly one. Their
original guards and refusal assertions stay unchanged. The QEC test context now
sets one Torch thread before each case and restores the caller's allocation in
`finally`, including after failures. Other suites retain their admitted thread
budget. No workload, scientific fixture, refusal or numerical threshold changes.

The original failed Runtime job logs are retained. An actual subprocess
starts at three threads, executes the real QEC fixture around successful and
intentionally failing cases, and verifies restoration in a sibling suite. Its
real exit status remains one with the deliberate failure. The initially failing
matched families and the context regression pass 89 cases under a twelve-thread
parent allocation. The full QEC suite passes 2,801/329 under that allocation (passed/skipped).
New exact-head CI and full-job timing still require qualification.


The first run also showed mixed Runtime partitions exceeding 600 seconds under
one-worker threading. Runtime therefore increases again from 64 to 128 slots
and restores its original bounded parallel workers with one thread per worker.
The exclusive threaded profile is reserved for measured sole-node coverage
cases. The new fan-out preserves the exact tier and marker selector, hosted
buckets, once-only real launchers and safety limits. The capacity policy tests
verify all 128 rows and the original mixed-worker admission command. Completed
successful shards from the first run are useful measurements, not qualification
of its failed head or proof that every final job meets the target.


## Runtime discovery and execution are separate phases

The f8bd44f7e run's actual 38,450-node coverage plan preserves every earlier
38,449 node exactly once and adds the previously missing thread-restoration
case. All 75 originally measured heavy cases remain alone. No new scientific
failure was observed before this refinement, but Runtime slots 5 and 12 still
exceeded 600 seconds despite the increased fan-out.

Each ordinary xdist worker was collecting the entire Runtime tier before
selecting its small partition. One original successful 143-node Runtime report
records 263.27 seconds of pytest wall time and 1,673.85 process CPU seconds,
while completed test phases sum to 221.87 seconds and the longest case is
136.37 seconds. This is not a precise collection-time census; it motivates
removing the known repeated full-tier discovery and retaining measurements of
both phases. Increasing partitions alone also adds repeated setup to the
bounded A800 pool.

`ci_tier.py pr-runtime --plan-shard` first runs the unchanged integration marker
and original duration-balanced shard selector serially with `--collect-only`.
The source-bound selection records the original complete marker inventory's
count and sorted node-ID digest, shard count/index, selected count and exact
node-file digest. The execution pass verifies that manifest, requests only
those node IDs and explicitly disables a second shard filter. All 128 original
partition assignments retain their existing balancing policy.

Mixed partitions retain their original bounded workers and single-thread
configuration. Only an actual sole-node selection with an existing duration
hint of at least 120 seconds requests the already established exclusive profile
from `ci_cpu_budget.py`. The budget owner rechecks the same host/cgroup/memory
admission; the runner does not copy or increase its quota formula. QEC's
per-case one-thread fixture still takes precedence for its scientific workflow.
Real two-rank launchers remain once-only and one-thread-per-rank.

The serial collection JUnit report, original admission and selection remain in
the resource artifact, including original optional-module skips and collection
errors. Exclusive execution also retains a copy of the original collection
before writing its new resource report. The actual execution JUnit path follows
the updated report directory. Discovery records are not counted as executed
cases. Collection, budget and test failures retain their real exit statuses;
there is no success fallback, lowered assertion or ten-minute cancellation.

Controlled end-to-end regressions exercise three original partitions with a
real failed and skipped case, excluded slow/unit bodies, retained optional
collection skips and a module whose import proves that execution workers no
longer rediscover unselected files. They also verify inherited shard-option
reset, source/content refusal, original selector/CLI status, actual budget-owner
invocation and refusal, single-worker overrides, resource retention and JUnit
relocation. Synthetic admission fixtures establish orchestration behavior, not
hardware capacity or Linux timing. New exact-head CI must still verify the
complete original selection and each full executed-job duration.
