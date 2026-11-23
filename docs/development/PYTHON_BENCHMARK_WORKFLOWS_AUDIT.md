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

## Array workload

The complete native array workload retains the original 100 quantum identities,
100 classical identities and 100 alternating Boolean inputs. The existing
operand-array binder constructs each measured result. Setup occurs once per
benchmark invocation, while the body is timed repeatedly. The source group name
km.array is replaced by native OperandArray. This naming adaptation is explicit
in rendered-oracle comparison.

All 300 input and output elements match the unchanged source at one and three
body repetitions. Array construction emits no quantum primitives, measurement
or workspace. The analytical model is None. Original timing goal 2.8 microseconds
is retained separately. A five-millisecond local timing run observes approximately
144.53 microseconds per native call, including typed conversion. K-0248 remains
open for the original performance target.

## Complete stateful builder workloads

The four CCX cases and 256-bit in-place addition use the existing native operand
binder, typed CCX emitter, physical-trace lowering, addition binding, workspace
pool and mutable KMX builder. The measured native boundary includes binding,
lowering, serialization and append. No cached gate stream replaces generation.
The adder retains MSB-first native ports over source LSB-first register layout,
original register names, component marks, reset/measurement/feedback order and
reusable clean-wire order. The shared workspace owner releases all 254 wires
after every body invocation.

At one and three consecutive repetitions, complete source KMX text agrees
byte-for-byte, including source operand IDs, marks and feedback. Ten actual
three-lane executions match original quantum masks, exact phases and executed
operation counts. The 254 adder workspace masks are all zero after each workload.
The batch reruns accepted small-width full-complex/all-applicable-measurement-
branch addition and control consumers plus normal circuit.append(op) workflows.
Large-width stochastic execution and small-width exhaustive branch evidence
retain separate scopes.

| Single-body workload | Static physical primitive records | Actual three-lane primitive counts | Analytical model |
| --- | --- | --- | --- |
| Scalar CCX | CCX 1 | CCX 3 | None |
| Quantum CCX array | CCX 256 | CCX 768 | None |
| Literal-control CCX array | CX 170 | CX 510 | None |
| Mixed-control CCX array | X 85, CX 129, CCX 42 | X 170, CX 216, CCX 126 | None |
| 256-bit addition | R 254, HMR 254, CX 1779, CZ 254, CCX 255 | R 762, HMR 762, CX 5337, CZ 377, CCX 765 | None |

The adder has two additional static component INFO_TAG records, six actually
executed over three lanes. It uses 512 allocated logical wires, 254 reusable
clean wires and one reused feedback bit. CCX cases allocate no temporary wires.
Conditional execution accounts for the difference between static and actual
mixed-control/feedback counts. Three-repetition counts are recorded separately
in the original execution oracle. No QEC model is inferred from opcode totals.

| Workload | Original goal in microseconds | Observed native microseconds per call |
| --- | --- | --- |
| Scalar CCX | 0.28 | 24.14 |
| Quantum CCX array | 0.95 | 2780.67 |
| Literal-control CCX array | 0.70 | 1608.29 |
| Mixed-control CCX array | 3.30 | 5857.29 |
| 256-bit addition | 3.30 | 42175.75 |

These local CPU observations use five-millisecond target runs and include the
native construction boundary above. The original goals are declared targets,
separate from measured extension timings. K-0250 remains open for those targets.
There is no throughput, SIMD or distributed scalability equivalence claim.
