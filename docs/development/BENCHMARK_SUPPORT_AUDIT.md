# Complete benchmark support source audit

## Ownership and interface before implementation

Frozen Kickmix main.perf.cc/perf.perf.h are K-0324/K-0325. Benchmarking owns
adaptive repeated invocation, immutable timing samples, named marginal rates,
separate source goals, ordered selection and result presentation. The existing
Compiler spin SI renderer, Ecosystem argument conversion and accepted physical
increment workload are reused. No new production execution path is introduced.

The shared timing owner invokes the actual body once per measured repetition.
Its monotonic clock reports integer nanoseconds. Total repetition and batch-limit
arithmetic use the source uint64 wrap semantics. Result decoration returns
immutable successors. Registration uses explicit ordered cases carrying names and callbacks.
Invocation-owned results replace process startup globals. Measured durations remain separate from the model
or upstream performance goals.

The source lookup function duplicates test.test.cc. The preceding complete
shared-test-support batch verifies all three read prefixes before write prefixes,
file creation and exact errors. Existing Python file ownership is reused.

## Boundaries retained before acceptance

The source empty-result diagnostic sits inside its result loop, so an empty
benchmark exits successfully without a diagnostic. Source CLI accepts target zero,
then a report with a performance goal converts a nonfinite logarithmic deviation
to int. UBSan reproduces the undefined conversion. Native CLI parsing retains zero
and the measurement owner produces the original zero sample. Goal comparison
refuses that undefined conversion. Defined no-goal reports retain the original
zero, signed-zero, NaN and infinity output. K-0324 retains these original boundaries.

The reference remains byte-identical. Deterministic timing comparisons use a
harness-provided steady_clock symbol with a controlled monotonic clock. Real
source and native clock observations are recorded separately. Controlled clock
results establish algorithmic order, not physical performance parity.

## Complete definition disposition

| Original definition | Native owner and evidence |
| --- | --- |
| BenchmarkResult and show_rate overloads | Immutable TimingSample, with_rate and ordered rate tuples. All eight signed/unsigned integral overloads, double and the char16 template route retain double conversion. |
| goal_nanos / goal_micros / goal_millis | with_goal preserves each successive division and floating rounding, including subnormal-scale and large inputs. |
| RegisteredBenchmark, registry, startup macros and cleanup | Explicit ordered native cases retain names, callbacks and invocation-owned samples, replacing static initialization, the sentinel and heap ownership. Duplicate selections retain invocation order. |
| benchmark_go | measure retains adaptive batch limits, source uint64 wrapping, nanosecond conversion and body exceptions. _next_repetitions isolates the same early stop and both source diagnostics. |
| resolve_testdata_file_path | The source body is byte-identical to test.test.cc. Its complete lookup corpus and scoped filesystem evidence are reused. Native maintained workloads own explicit paths. |
| si2 | Existing Compiler SI renderer gains a private hundred-range truncation option. Default spin output retains whole units. Complete original benchmark output uses truncation to tens. |
| find_benchmarks | select_benchmarks retains trailing-star prefixes, exact matching, duplicate filters, empty filters and complete missing-filter diagnostics. |
| main option parsing | Existing Ecosystem source scalar/argument owners supply first-match flags, terminators, float32 target rounding and bounds. The native runner uses the FlagQuantum command identity. |
| main result loop | format_timing retains every source-defined report, marginal rate order and logarithmic deviation marker. Native Python exceptions replace source process exits or uncaught C++ exceptions. |
| Existing physical workload | The native runner calls the accepted physical_increment generator, checks 1270/1274 records and reports real generation timings separately from the original 580/610 ns goals. |

## Acceptance evidence and remaining scope

All four source/header/direct argument dependency blobs are checked before
compilation. The complete original entry point is included with its main symbol
renamed only by the harness. A controlled external steady_clock definition makes
adaptive observations deterministic. Original source contents remain unchanged.
The host C++ clock symbol replacement is test instrumentation. Native and source
real-clock observations remain distinct from those controlled observations.

The corpus contains 414 complete observations. It covers 237 SI boundaries,
15 command routes, 10 adaptive runs, 145 complete reports, five goal-unit vectors
the complete integral/double rate-overload vector and duplicate-name callbacks
producing multiple results. The source's zero-cost
callback is optimized into aggregate arithmetic. Its huge repetition overflow
route is compared with native planner-state replay. Normal callbacks are invoked
for every actual native repetition. The 1e15 limit, early stop, zero target,
multiple batches and two state/error paths preserve original behavior. The entire
corpus regenerates byte-for-byte.

All 234 concentrated tests and 55 local register integration tests pass. Strict
mypy passes six Python files. Black, Ruff, team scope, architecture, internal IR
and protected API checks pass. Both timing/report owners and the actual native
runner have 100% measured statement and branch coverage with coverage omissions
disabled for this measurement. The existing coverage configuration remains intact.
Dependency scanning retains 720 existing findings with zero touched findings.

The empty-result source diagnostic remains unreachable. The source accepts a zero
timing target and can report nonfinite values without a goal, which native also
preserves. With a goal, UBSan confirms conversion of NaN to int at original
main.perf.cc line 162. Native refuses that undefined goal comparison. The
[2026-11-28 review](G2_FUNCTIONAL_MIGRATION_DISPOSITIONS.md) accepts K-0324's
complete defined-domain migration with these original defects retained.
**K-0325 is accepted** for the complete
declaration and adaptive-engine scope. C++ runtime/clock portability and matched
native/C++ performance remain unclaimed.

The timing framework itself emits zero quantum primitives, measurements or work
bits and declares no quantum analytical model. The real 256-bit increment workload
has independently recorded primitive histograms and component-boundary records.
Existing complete complex-amplitude, measurement-feedback and clean-work evidence
is reused through the local register integration suite. The resource receipt keeps
actual primitive counts, boundary markers, measured times, source targets and the
absence of an independent quantum model in separate fields.
