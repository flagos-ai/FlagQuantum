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


## Full-cohort timing update and qualified heavy-case routing

Run `45998519970` at head `93d221add32858ac1254ed146ea4ada39cb20928`
produced all 256 original coverage execution reports. Their artifact digests
were verified, and their recorded node sets cover the complete 38,464-node
source-bound plan exactly once. This is a full phase-timing census, unlike the
earlier printed slowest-50 sample. Original reports, including failed outcomes,
remain retained separately from candidate execution.

Nine successful whole coverage jobs exceeded 600 seconds. Eight already had a
single case, including the complete Maxwell, network-resiliency and projection
embedding workloads; further fan-out cannot divide those cases. One mixed
partition contained 218 nodes and took 911 seconds. Scheduling weights now
increase from measured phase sums of at least one second; earlier larger weights
remain intact. Replaying the complete original plan still produces 256 nonempty
partitions with every node exactly once, 83 sole-heavy partitions and a maximum
ordinary predicted serial weight of 155.5545 seconds. Predictions are scheduling
inputs, not candidate elapsed-job acceptance.

Only sole cases with a measured scheduling weight of at least 300 seconds use
the existing `flagquantum-a800-cpu` pool. Both A800 hosts passed the existing
read-only locked-image qualification in run `45999529800`, with
`export-bundle=false`, at the same source head. No images were installed or
pulled, and no runner labels, services, CPU quotas or admission formulas changed.
The replay routes 32 partitions to this pool; ordinary partitions retain their
existing CPU pool. Fork PRs retain hosted execution, eight bounded hosted groups
and the same cached-image requirement for trusted runners. The workflow retains
its original finite safety budgets; 600 seconds is an elapsed-job target.

The failed coverage partition exposed a test import-order dependency:
`test_lindblad_integrators.py` accessed `fq.lindblad` without importing that
submodule. Earlier broad collection happened to import its historical facade.
These internal tests now explicitly import the authoritative Runtime evolution
module and retain all original scenarios and assertions. The complete file
passes independently (48 cases). No public export or compatibility facade is
added. The failed original execution remains failed evidence.

New published-head acceptance must verify all actual selected node inventories,
root execution reports, source bindings and whole-job durations. Neither the
old run's successful shards nor this replay certify the candidate as all green
or within ten minutes.


## Runtime recipient qualification and live capacity

After coverage planning completed in run `46006910591`, the active-job count rose
from approximately 35 to 208, including 188 coverage partitions. The pre-plan idle
period was a dependency boundary. Later Runtime work still queued behind the
A800-only route while ordinary CPU slots became idle.

Read-only qualification runs `46010078828` and `46010700562` at exact source
`f42066f001fb1f6aaabd7a18ff33e03a1701cddd` verify all four existing locked profiles
on every one of 126 online BW1100 registrations. Eight representative BW slots
passed first, followed by all remaining 118 exact selectors. The four MC550
representatives failed on missing core-py3.10; their later profiles were not
executed or inferred to pass. Offline and unqualified slots are excluded from
new Runtime recipients. The [qualification record](ci-cpu-cache-qualification.json)
retains each selector, actual runner, run/job IDs, source workflow and image-lock
binding. No image is downloaded/imported and no runner configuration is changed.

The first 32 of the existing 128 Runtime partitions retain the A800 pool. Replay
of the original 14,260-node inventory confirms that these include all thirty
actual sole-heavy selections. The remaining 96 receive distinct qualified BW
selectors, balanced twelve per each of eight hosts. Existing admission measures
their actual four-CPU service budgets; the collector and actual execution reports
must record those real resources rather than assuming twelve workers. LPT,
selection, count, marker semantics, finite safety limits and scientific workloads
are unchanged. Forks retain hosted execution with eight bounded groups. This
expands placement capacity; actual candidate whole-job timing remains required.

The same original run contains twelve diagnosed coverage failures after all their
tests passed: the runner timed out fetching the official upload Action archive
before artifact upload. Original logs and reports are retained as failed evidence.
This is distinct from a test failure and cannot be reclassified as green.

The self-hosted security job spent approximately nineteen minutes preparing
source and installing dependencies, exhausting its twenty-minute job limit about
one minute after starting audit. Run that fresh, network-dependent job on hosted
Ubuntu in bucket 2, preserving the same complete requirements snapshot, strict
no-resolver audit, sixty-second request timeout, eight-minute audit step, Bandit,
SBOM checks and twenty-minute total safety budget. This correction changes its
execution placement rather than weakening the audit or increasing its timeout.


## Qualified parent and artifact SDK follow-up

Parent PR 1344 head `f3329a61bf9f867647f93b5cd2eb81789dc03109` passes
432 checks with four conditional skips; all three applicable workflows succeed
on attempt 1. Its actual source tree and all 384 original CPU report artifacts
are retained and verified. Runtime executes the original 14,260 selected nodes
once; coverage executes the 38,468 planned nodes once. Main CI job timestamps
show a peak of 213 overlapping assigned jobs.

| Group | Partitions | Maximum whole-job seconds | Jobs above 600 seconds |
| --- | ---: | ---: | ---: |
| Runtime | 128 | 759 | 4 |
| Coverage | 256 | 832 | 6 |
| Python 3.12 unit | 4 | 451 | 0 |
| Distributed CPU | 6 | 571 | 0 |

Those durations include setup, tests and upload, excluding queue time. A green
correctness gate does not establish the remaining elapsed-time target. Several
late artifact steps spend 203–331 seconds fetching/preparing the official action
and uploading reports. The existing bounded upload wrapper now uses the same
pinned official SDK locally. Its raw upstream metadata, MIT license, bundle,
Git object identities and SHA-256 values are retained under the wrapper's
`vendor/` directory. The three-attempt graph, final failure, service protocol,
inputs, outputs, overwrite and retention semantics remain unchanged. Source
restoration must include the complete local action before the first upload;
regression tests restore its exact entry point and bytes from a fresh archive.
The SDK is absent from the inspected Python source distribution and wheel.
Real successor CI upload/download and source-binding acceptance remain pending.

The native-fermion Python 3.12 job on `fq-a100-cpu10` started at 14:15:20 UTC
and reached its real-driver test step at 14:49:46 UTC: 34 minutes 26 seconds
before scientific execution. This lane intentionally installs fresh certified
science dependencies, so it now uses hosted Ubuntu networking in existing
bounded buckets 3 and 4. Both original Python/Torch/PySCF/h5py combinations,
fresh isolated environments, exact source restoration, real RHF tests and the
45-minute finite safety budget remain unchanged. The hosted ceiling stays at
eight shared groups. This routing change still needs actual CI timing evidence.

The remaining five compute-heavy coverage outliers already own sole-test slots
and twelve admitted threads: the complete Maxwell case spends 635 seconds in
its call; four projection-embedding cases each spend 695–734 seconds repeating
the module fixture. Additional fan-out alone cannot shorten them. Their original
workloads, complete amplitudes, optimizer budgets and assertions remain intact;
fixture/resource optimization requires separate measured verification.


## Source publication placement

The first successor head `b9b4ad6da30217ff53fd707684e1f2123dfba9ec`
reached a source-preparation network tail before any CPU partition could start.
Main CI run `46027287200`, source job `140601846799` on
`fq-bw1100-11-cpu13`, spent 15:43:04–15:46:50 UTC in API restoration and
then entered the existing Git checkout fallback. Its original job remains
observed until superseded by this concrete routing correction; log evidence
must determine the exact restoration error, rather than attributing it to the
new SDK without proof.

Both main and GPU source publishers now use hosted Ubuntu in the existing
shared bucket 5. Source publication needs Git/GitHub networking rather than a
scientific image or private accelerator. The existing hosted checkout path,
read-only permissions, credential persistence prohibition, HTTP/1.1 setting,
30-minute finite budget, exact-commit packing, unique per-run artifact keys,
source digest checks and complete/incremental consumers remain unchanged.
Trusted and fork sources both remain on hosted publication capacity. Scientific
CPU/GPU consumers keep their existing qualified routes and source checks.
The eight-group hosted ceiling is unchanged. Actual new-head publication and
whole-job timing still require CI verification.


## Ordinary single-GPU placement and frozen campaigns

At successor head `2d77f841`, ordinary `one-gpu-local` failed twice on the 171
host before scientific execution because no physical device satisfied the
existing idle threshold. Original jobs `140609266692` and `140615033371`
remain failed evidence. The same commit's two-GPU job `140609268286` passed on
172, with its digest-verified original hardware artifact recording NVIDIA
A800-SXM4-80GB devices, driver 580.126.20, Torch 2.13.0+cu130, and actual
selected devices 0 and 1. No GPU process was stopped or reconfigured.

Ordinary single-GPU CI now uses this already provisioned A800 172 host.
The precision-frontier and QNG-ablation manual inputs still select the original
171 host and physical device zero; their frozen commands and environments do
not change. Single- and two-GPU CI retain the same accelerator concurrency lock,
so their real computations and measurements cannot overlap. The one-GPU
20-minute budget, idle memory/utilization thresholds, all correctness tests,
benchmark warmups/repetitions/inner loops, performance evaluation and evidence
collection remain unchanged. This is a resource-placement change, not a relaxed
admission check or a performance/scalability promotion. The new head must pass
the original single-GPU tests and measurement gate on the selected hardware.


## Spread serial embedding fixtures without changing scientific work

The original four projection-embedding reports each used 761–803 process CPU
seconds in 698–737 wall seconds: about 1.09 cores despite a twelve-core
admission. Their utilization of that allowance was about nine percent. The
Maxwell report differs: 4,561 process CPU seconds in 638 wall seconds uses about
7.15 cores, so it retains its existing A800 placement. These measurements are
original parent-run evidence, not predictions relabeled as new timings.

In the successor run, the final embedding partition did not start until
16:21:11 UTC, even after nearly all other pools finished. Four unchanged full
embedding cases now use four distinct, online and idle, positively qualified
BW registrations: cpu16 on hosts 10, 11, 12 and 14. Their existing source-bound
qualification receipt covers all four locked images, including coverage-py3.12.
The planner refuses the override if that profile or any target receipt is
missing. Only already-sole partitions receive the override; mixed partitions
keep their existing placement. Other heavy cases and all Maxwell work remain
under the existing A800 rule.

The five-field plan schema, every selected node and per-shard digest, full
chemistry/state/energy/100-query optimizer workflows, assertions, independent
references, coverage policy and finite timeouts remain unchanged. The matrix
adds a private explicit selector; forks still use hosted capacity and all
self-hosted cached-image/resource checks stay enabled. Resource admission must
report the BW service's actual four-core budget rather than the A800 allowance.
No service, label, quota, dependency or numerical algorithm is changed. Actual
whole-job timing and all original scientific results on these targets remain
required before calling the bottleneck resolved.


## Rootless preparation and full-width placement

Offline environment preparation reads the complete Docker security-options
query before selecting the container user. Under `pipefail`, a producer writing
after `grep -q` has already matched can receive SIGPIPE and make a positive
rootless match appear false. A query failure now stops preparation rather than
selecting a host UID without knowing the daemon's identity. Rootless container
UID zero maps to the unprivileged daemon user; it does not grant host root.
The original failed preparation and deterministic regression outcomes are
retained separately from successful verification.

The complete wide-state glued-trees oracle has an existing qualified BW slot,
just as the four serial embedding workloads do. It remains a sole test with its
original full amplitudes, width, oracle and finite budget. Qualification still
checks the locked coverage profile and exact runner selector; trust rules,
resource shares and other heavy A800 routes are unchanged. No service, runner
label, image or quota is modified. The placement must be accepted through the
new head's actual scientific result, resource report and entire job duration;
a cache qualification or scheduling replay alone cannot certify performance.


## Follow-up after qualified PR #1346: fermion encoding assembly

PR #1346 head `77f4e2d3b1e9a7851991f9ed58ff3e1b29685301` passed
432 checks with four unchanged conditional skips, on the first attempt of all
three workflows. All 384 raw CPU reports are retained. Runtime128 accounts for
14,260 original nodes and coverage256 for 38,518 selected nodes once. Their
whole-job maxima are 539 and 747 seconds respectively, with five scientific
singletons still above 600 seconds. Maxwell takes 691 seconds after waiting
999 seconds from plan readiness. Green CI is not ten-minute acceptance.

The original PBE job's 300-second stack is inside fermion encoding's repeated
full Pauli-sum canonicalization. The follow-up replaces only that accumulated
sum rebuild, inside the existing Algorithms encoder, with updates to the
factors touched by each original image. Ladder mapping, per-image addition
order, leading complex zero, finite rejection, exact-zero removal, final
canonical ordering, public signature and result type remain unchanged. There
is no new dependency, export, package, native ABI, compatibility layer or Rust.

Before activation, a temporary definition preserved all 59 original complete
encoder outcomes and six overflow/cancellation outcomes from the independently
installed prerequisite. The nine additional contract/Fock-reference cases pass
on the unchanged original implementation. The representative local 16-mode,
1,024-source-term diagnostic takes 3.03 seconds originally and 0.088 seconds
with the proposed fold; this is not a full chemistry or Linux performance claim.
Formal source, installed-candidate, full original independent PBE workflows and
exact published-head CI acceptance remain required. No original scientific
input, optimizer budget, marker, oracle or timeout is changed.

The original complete 17-qubit, 2,383-operation Maxwell walk also retains
identical complete output bytes across fifteen local executions using one,
two, four, eight and twelve threads. Local differences are small and do not
qualify a Linux thread-policy change. The isolated counter fixture is likewise
insufficient to attribute the full Maxwell runtime to statistics. Retain the
original scientific workflow while investigating its actual Linux cost.

The actual candidate now passes all 65 original outcomes, including exact
coefficient bytes and overflow/refusal results, from both source and its own
new native installation. The full actual-source default selection passes
21,896 tests with 852 conditional skips in 488 seconds including startup.
All 22,739 earlier cases remain and all nine additional cases pass. One
previously passing historical Git-revision lookup is skipped by its existing
60-second environment guard; neither that test nor its bound was changed, and
its raw reason is retained. The complete fermion unit suite passes 568/1,
and all seventeen native integration/training modules pass 164 tests. The
installed suite has exactly the same 733 outcomes, 732 passed and one missing
optional HDF5 dependency, with all 523 loaded product modules verified from
its own installation. Both native extensions load there, and all 1,265 product
Python files match source, wheel and installation. The golden native energy
example still reports 1.300000 for all three encodings.

Candidate wheel SHA-256:
`c6db9050d0939e0a0294ba6baf2444351bf6c575123b6b1d70fd9efc49327dd7`.
All eleven quality gates and the applicable fixed hooks pass. This local
qualification does not establish the four full original PBE timings or the new
published-head CI; those remain mandatory. The latest #1084 increment through
`d4802f79b864b3ae25c3e03efc66f6c06ab6dfb8` has sixteen SHA-verified
paths and no candidate overlap; both relocation ledgers retain the same
seventeen unfiltered changed-incoming diagnoses. Integrated G0 remains open.


## CPU Toffoli layout follow-up after PR #1349

PR #1349 qualified commit `b0e5856c2335acfcad9a662edbd8378148de0f65` on
all three workflows at attempt 1: 432 successful checks and four existing
conditional skips. Its 384 original CPU reports preserve 14,260 Runtime and
38,540 coverage selection nodes once. Whole executed job maxima were Runtime
540 seconds, unit 423 seconds and distributed CPU 426 seconds. Only Maxwell
remained above 600 seconds, at 705 seconds; it waited another 873 seconds
after the coverage plan was ready.

The follow-up changes only the existing Simulation CPU Toffoli calculation.
For medium state vectors, grouping untouched axes bounds the indexing rank and
swapping two cached target views halves the temporary amplitude buffer. The
owned non-gradient path copies one target half, then exchanges the two views.
The differentiable path clones the admitted input before either write. Every
caller continues to use the same Simulation defining function, with its original
signature, input admission, dtype, output shape and buffer-ownership rules.

The direct-axis layout remains for fewer than twelve qubits and for twenty or
more qubits. Local calibration found grouped views slower for the smallest
states and for the measured four-thread large streaming case. Both layouts
are permanent numerical choices in the same owner. These local timings do not
qualify the Linux full workload or establish a universal hardware crossover.

Before changing the implementation, 891 complete original outcomes were retained
from the qualified independent native installation, including raw tensor and
gradient bytes, strides, conjugation, input mutation, storage aliasing and
public admission refusals. The actual modified source reproduces all of them
exactly. The 62 existing independent dense/gradient/ownership references and
24 added seventeen-qubit batched-layout/gradient cases pass on both the original
and the modified source. The new cases use the existing generic matrix action
and the unchanged Toffoli matrix as their independent numerical oracle.

An independent, immutable-source hardware check of the unchanged original
Maxwell node on the qualified BW17 CPU16 four-CPU/24-GiB profile passed all
original assertions, but took 940 seconds including preparation. Its container
reached its 24-GiB hard limit with 12,794 max-limit events and no OOM kills.
This result does not justify moving Maxwell to BW for the ten-minute goal.
It does not isolate which construction, compilation or execution allocation
caused that pressure. Source files were hash-verified before and after the
read-only test phase, with no temporary numerical override.

The final complete source default selection passes 21,921 tests with 864
conditional skips in 410 seconds including startup. All 22,785 collected cases
remain: the earlier 22,761 and the 24 added independent references. One
historical Git-revision check changes from its existing environment skip to a
pass; no previously passing case regresses. The relevant native/backend and
complex-matrix integration modules pass all 188 cases. Together with the 86
Toffoli cases, the independently installed native wheel passes the same 274
cases, and reproduces all 891 complete original outcomes exactly.

All 1,265 product Python files match source, wheel and installation, both native
extensions load from that installation, and all 532 loaded product providers
resolve there. The BMM test fixture now locates its unchanged kernel through
the loaded package, allowing the same assertions to run outside the repository.
Initial source-relative fixture errors and sandbox local-interface failures
were retained; the full installed selection passed after correcting those test
environment constraints. No optional dependency resolution is claimed.
Candidate wheel SHA-256:
`6c7dcc3b2cea9de23523ed9078fb31ccca93c70b59cfba85ac3c36ece7ffb2ea`.

The follow-up still requires its own exact-head CI with the full original
scientific node and coverage; the predecessor's green checks do not qualify
this change or establish its Linux performance. Original
widths, amplitudes, phases, optimizer budgets, draws, clocks, markers, tolerances,
assertions, safety limits and coverage thresholds remain unchanged. Native
source and ABI remain unchanged; no definition relocation is recorded. Rust
remains deferred until the incoming branch and architecture migration close.
