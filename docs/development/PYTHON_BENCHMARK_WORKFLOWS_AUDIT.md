# Complete Python benchmark workflows

## Source and owners

The frozen source is Kickmix revision
`9a0c11a4ee7697c082fbf8f2e1e8bf4a71fcb453`, Google LLC, Apache-2.0.
Complete parents K-0248 through K-0251 cover array_perf.py, benchmark_suite.py,
circuit_builder_perf.py and run_all_perf.py. All four source files were read
and Git-blob verified. Existing accepted physical construction and numerical
owners supply their direct dependency closure.

Verification owns timing, registration, reporting, discovery and benchmark
workloads. Compiler owns scalar/array binding, CCX dispatch, addition, component
marks, workspace and mutable physical circuits. Simulation owns execution. The
workloads continue through these authoritative production owners. Runtime and
Core contracts retain their existing responsibilities.

## Complete timing source

The existing TimingSample stores numerical samples for both source timing
policies. Its existing negative-one C++ unset-goal default is retained. Python
results use None for an absent goal and retain a distinct name. This preserves
explicit Python negative-one goals, which fail the original logarithm.
C++ report and adaptive policy consumers retain their existing behavior.

The Python policy uses float monotonic seconds, floor division for adaptive
repetition estimates, a hundredfold batch limit and eight-way body unrolling
above fifty calls. It retains zero/negative target results, clock-driven batch
ordering, exceptions and per-invocation mutable callback state. Reporting uses
Python round at SI and comparison boundaries. Goal conversion multiplies by the
source unit constants. Decorator registration supports direct and curried forms,
retains duplicate rates in their original order, snapshots mutable units after
the callback, and overwrites inner result names and goals as the source does.

TimingSuite owns the two current workload groups. Existing private numerical
TimingSample and report/adaptive owners are reused. Foreign result repr and the
source result's str hook are mapped to explicit format_suite_timing. Numeric
fields, name, goal, rates and rendered output remain independently comparable.
Empty, zero-time and invalid-goal formatting failures retain source exception
types and messages. The existing C++ policy remains separately callable.

The original oracle contains 23 SI boundaries, 32 complete formatted reports,
five exact failures, seven controlled-clock traces and five registration cases.
The native tests additionally verify both decorator forms, registry order,
callback-mutated names/units/rates, duplicate rates, main output, empty suites,
body exceptions and conflicting goal units.

Timing infrastructure adds zero physical primitives, measurements or workspace.
Its analytical model is None. Source performance goals are recorded separately
from native measurements and are independent of construction equivalence.
