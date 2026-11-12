# Register score lane reservations

## Ownership and interface

`drawer/register_lanes.py` owns the minimum-lane heap, reservations and geometric
open/closed intervals. `register_score.py` owns dependency columns, register
identities and word widths over completed lifetimes.
`component_score.py` projects validated Core calls and retains their native
register grouping. Core register objects are reused as presentation metadata.
Simulation owns complex amplitudes and measurement execution.

`RegisterLaneManager` supports `reserve_n`, `unreserve` and `maybe_reserve`.
The release predicate receives the original call ordinal and Core `Register`.
Boundary locations are the strings `input` and `output`. The hook additionally
receives the row-major element index, including `()` for scalars. Predicates
may inspect the pool and must leave its allocation and reservation state to the
manager. Hooks can reserve lanes. Each independent layout owns its manager.

```python
from flagquantum.drawer.component_score import component_register_score
from flagquantum.drawer.register_lanes import LaneLocation, RegisterLaneManager
from flagquantum.core.ir.program.registers import Register


class SpacedInputs(RegisterLaneManager):
    def maybe_reserve(
        self, location: LaneLocation, register: Register, index: tuple[int, ...]
    ) -> None:
        if location == "input" and register.name == "a" and index == ():
            self.reserve_n(1, lambda where, reg: isinstance(where, int))


score = component_register_score(record, lane_manager=SpacedInputs())
```

The selected `record` comes from the existing component builder. The normal
execution consumer continues to use `circuit.append(op)` through its existing
binding. Display lane numbers express vertical spacing only.

## Locked call chain

The reference is Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, file
`qualtran/drawing/musical_score.py`, Git blob
`3863bb66cdba17c4654657e275a8485d94b8da75`.

| Reference behavior | Native consumer |
| --- | --- |
| `reserve_n` removes the lowest available lanes | `RegisterLaneManager.reserve_n` holds lanes in insertion-ordered reservations |
| `unreserve` checks each predicate, releasing successful reservations before checking later ones | Native predicate order and observed available-lane snapshots match |
| `new` calls `unreserve` once before the whole register | Shared layout preserves Core array groups, including empty arrays |
| `new` invokes `maybe_reserve` after each element allocation | Native `new` retains scalar and row-major element order |
| RIGHT ports allocate before LEFT ports free | Shared layout retains the established operation order |
| THRU ports reuse predecessor positions | THRU ports generate no allocation, release or predicate calls |
| `free` releases each element and then invokes the hook | Native `free` exposes the newly available lane to the hook |
| `free` never calls `unreserve` | Final outputs and consumed inputs retain that cadence |
| Final output register order controls release hooks | Core return identities and the root signature order reach the shared layout |

The original source remains frozen. Tests instrument the supplied source manager's
documented hook at the test boundary. Production imports use native types only.

## Verified scope and explicit adaptations

The native default capacity grows with the graph, preserving existing drawings.
The reference default capacity is 100. An explicit `max_n_lines=128` provides
identical finite heaps for callback and geometry comparisons. Capacity and
reservation counts reject negative, boolean and noninteger values. Oversized
finite reservations fail before taking lanes. A predicate exception restores the
available heap so retrying cannot return an already released lane twice. The
source can partially consume or release its heap on those failures. These error
transactions are deliberate native adaptations, tested separately from parity.

Hook exceptions propagate. A fresh manager provides an independent retry.
External callback-owned state retains its usual caller ownership. Foreign `LineManager` objects remain an open protocol boundary. Native hooks
can inspect geometric open/closed intervals through `hlines`. Native subclasses support `maybe_reserve` and the
context-aware `new_y_for` hook documented below.

Grouped metadata must cover every element, agree with word widths, and preserve
whole-register THRU lifetimes. Invalid metadata is refused before allocation.
Styling can customize captions and glyphs while retaining connectivity and Core
register metadata.

## Acceptance

`tests/unit/test_drawer_register_lanes.py` compares finite heap snapshots,
reservation predicate order, hook order, full horizontal/vertical geometry and
glyph records for Add widths 1, 2, 3, 5, 16 and 32. Scalar, vector, matrix and
empty-array boundary cases retain row-major indices. Further cases cover fresh
empty arrays, lowest-lane reuse, RIGHT-before-LEFT order, capacity, zero-size
reservations, invalid requests, callback failure/retry and metadata validation.
Boundary identity and dtype-label normalization follows the existing component
score comparison. Port identities retain native names.

Run the focused comparison in the locked environment.

```bash
python -m pytest tests/unit/test_drawer_register_lanes.py -q
```

Existing drawer, exact-adjoint and CPU/tensor regressions exercise primitive
operand order, full complex amplitudes, all forced Add measurement histories,
work restoration and ordinary append consumers. Resource accounts retain their
separate layers.

| Account | Existing negative-control MultiAnd consumer |
| --- | --- |
| Actual forward primitives | H6, T6, Tdg6, CX18, S3, X2 |
| Actual erasure primitives | H3, measurement3, conditional CZ3, reset3, X2 |
| Source forward model | AND3 and Clifford2, equivalent T count 12 |
| Source adjoint model | Clifford5 and measurement3 |
| Physical storage | Four controls and three work/output wires, source peak seven |
| Added by display reservation | Zero primitives, physical work wires, measurements or feedback |

Reserved display lanes contribute spacing. Physical storage continues to follow
the bound operation. Q-0747 remains open for its general foreign and wire-symbol protocol boundaries. Javascript animation remains a
separate dependency. Evidence uses the prefix `g2-score-reservations-`.

Local acceptance on 2026-11-13 uses implementation commit `153637f2`.
There are 41 new cases, 70 focused passes in 2.19 seconds, 343 concentrated
passes in 15.69 seconds and 86 independent notebook/CPU/tensor passes in
8.95 seconds. The concentrated suite reports eight existing parser warnings;
the integration suite reports two existing dependency/deprecation warnings.
Strict typing of four changed Python files, Black, Ruff, Compiler scope,
architecture, internal IR and protected API checks pass. The locked source blob
matches. The new lane owner covers all 55 statements and 22 branches. Combined
coverage for the three production owners is 275/281 statements and 103/110
branches, or 97% combined. Missing paths are existing layout validation and
component admission refusals.

The first integration invocation exposed an unguarded temporary launcher during
Cotengra multiprocessing spawn. A guarded standalone launcher fixes that test
entry and the full 86-case rerun passes. Both logs are retained with the evidence
prefix, including `integration-guarded`. The previous default-suite baseline
limitations remain in the main progress ledger.


## Context-aware coordinate policies

The source passes the current instance, register and optional array index into
`LineManager.new_y`. Native `RegisterLaneManager.new_y_for` receives the original
call ordinal or input boundary, the Core register and that same index cadence.
Scalars receive `None`. Arrays receive their row-major tuple. Empty arrays have
no coordinate calls. The subsequent `maybe_reserve` scalar index remains `()`.
THRU ports retain their coordinates without new allocation.

The native method has a distinct name because existing no-argument `new_y`
subclasses already own heap policies. Its default calls that existing method.
An overridden `new_y_for` owns selection and consumption of the chosen available
lane. Reservations continue through `new_y` because no register element exists
for a reservation. This extension is owned by Drawer and uses Core metadata.

A minimal context-sensitive policy can reserve spacing before the first input.

```python
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.register_lanes import LaneLocation, RegisterLaneManager
from flagquantum.drawer.register_score import layout_register_score


class SpacedCoordinates(RegisterLaneManager):
    def new_y_for(
        self,
        location: LaneLocation,
        register: Register,
        index: tuple[int, ...] | None = None,
    ) -> int:
        if location == "input" and register.name == "q" and index is None:
            self.reserve_n(1, lambda where, reg: where == "output")
        return super().new_y_for(location, register, index)


score = layout_register_score(
    {"q": 1}, ("q",), (), lane_manager=SpacedCoordinates()
)
assert score.lines[0].lane == 1
```

`tests/unit/test_drawer_lane_coordinates.py` compares a replacement policy that
chooses the highest or lowest available lane from all three context values.
The original source runs unchanged, with its supported method replaced at the
test boundary. Add widths 1, 2, 3, 5, 16 and 32, both reservation modes, scalar,
vector, matrix and empty boundaries match every allocation, available-heap
snapshot, reservation callback and completed port/line geometry. Existing
no-argument policies, failure propagation and graph immutability are covered.
Only boundary identity/dtype captions use the existing comparison normalization.

Implementation `13f92bdf` passes 24 new cases, 65 focused cases in 2.22 seconds,
367 concentrated cases in 18.02 seconds and 86 independent notebook/CPU/tensor
cases in 10.70 seconds. No cases skip. The suites retain eight and two existing
warnings. Strict typing of both changed Python files, Black, Ruff, scope,
architecture, internal IR and protected API checks pass. The lane owner covers
all 57 statements and 22 branches. The frozen source blob remains unchanged.
Evidence uses `g2-score-coordinates-`.

Coordinate selection adds zero physical primitives, work wires, measurements or
feedback. Existing execution regressions retain their amplitude, work-restoration
and append checks. The actual MultiAnd and source-model accounts above continue
separately. Q-0747 remains open for general foreign and wire-symbol protocol boundaries.
Partial-line observations are accepted in the lifecycle section below. The distinct native hook name is an explicit integration adaptation.


## Partial horizontal-line lifecycle

`RegisterLaneManager.hlines` stores immutable `LaneSegment` values with the
source geometric fields `y`, `seq_x_start`, `seq_x_end` and `flavor`. An open
segment has `seq_x_end=None`. The flavor uses the source's textual quantum or
classical value. Canonical Core metadata also preserves the classical identity
of zero-width classical words. The existing completed `RegisterLine` carries
register identity and word width for the layout and its rendering consumers.
These records serve the callback observation and semantic layout responsibilities.

The layout sets the manager's `seq_x` cursor to -1 at inputs, to the current
sequential column before each operation and to the output column at final
release. Direct manager callers set this cursor explicitly. Its initial value
is zero. The cursor integrates the source's per-position coordinates into the
native allocation entry points.

Allocation evaluates release predicates once per register, selects each lane,
adds its open interval and calls `maybe_reserve`. Array elements follow row-major
order. Release returns each lane to the heap, calls `finish_hline` to replace the
unique open interval with a closed value, then invokes `maybe_reserve`. Closed
values persist when that same lane is reused. A saved callback snapshot retains
its original immutable values. Coordinate policies see the state before the new
interval opens. Reservations allocate spacing without creating intervals.

Callbacks can inspect the interval collection to choose reservations and their
release predicates. The layout owns the cursor, while the manager owns interval
membership. Callbacks preserve those ownership boundaries. Each independent
layout owns a fresh manager.

```python
from flagquantum.core.ir.program.registers import Register
from flagquantum.drawer.register_lanes import (
    LaneLocation,
    LaneSegment,
    RegisterLaneManager,
)
from flagquantum.drawer.register_score import layout_register_score


class ObservedIntervals(RegisterLaneManager):
    def __init__(self) -> None:
        super().__init__()
        self.snapshots: list[frozenset[LaneSegment]] = []

    def maybe_reserve(
        self, location: LaneLocation, register: Register, index: tuple[int, ...]
    ) -> None:
        self.snapshots.append(frozenset(self.hlines))


manager = ObservedIntervals()
score = layout_register_score({"q": 1}, ("q",), (), lane_manager=manager)
assert manager.snapshots == [
    frozenset({LaneSegment(0, -1)}),
    frozenset({LaneSegment(0, -1, 0)}),
]
assert score.lines[0].end == 0
```

The original source blob remains
`3863bb66cdba17c4654657e275a8485d94b8da75`. Implementation `9b273636`
compares every coordinate, allocation hook, reservation predicate and
before/after-finish snapshot using actual source `LineManager` methods.
Add widths 1, 2, 3, 5, 16 and 32 cover ordinary reservations and a replacement
policy driven by partial intervals. Source Add(1) has no allocation after its
inputs, so its newly reserved spacing gets no later predicate evaluation.
Scalar/vector/matrix/empty arrays, quantum/classical/zero-width metadata,
same-lane reuse and immutable saved snapshots have direct source comparisons.
Completed manager geometry also matches every line in the native score.

Invalid `free` calls have an explicit native preflight. Element count mismatch,
duplicate lanes and missing/ambiguous open intervals fail before any release.
The source can partially mutate its heap and intervals for these malformed
calls. All three failure classes execute against the original source in tests,
and the native valid retry restores the complete pool. Missing or ambiguous
`finish_hline` calls reject atomically in both implementations. Exceptions from
caller hooks propagate, with a fresh manager providing an independent retry.

There are 49 new cases, 114 focused passes in 1.88 seconds, 416 concentrated
passes in 15.23 seconds and 86 independent notebook/CPU/tensor passes in
8.27 seconds, with zero skips. The concentrated and integration sets report
eight and two existing warnings. Strict typing of three Python files, Black,
Ruff, scope, architecture, internal IR and protected API checks pass. The lane
owner covers all 82 statements and 26 branches. Combined lane/layout coverage
is 97% across 244 statements and 88 branches. Uncovered paths are existing
layout validation and implicit-borrow branches.

Evidence uses `g2-score-lifetimes-`. Initial test-harness failures for the source
`idx` keyword and the Add(1) predicate expectation are retained. Both are fixed
in the accepted tests. Production and reference implementations were compared
through the same original methods throughout.

Geometric observation adds zero physical gates, work wires, measurement or
feedback. Existing exact-adjoint, all-branch amplitude/work-restoration and
append regressions run with the concentrated set, alongside independent CPU/TN
execution. The actual primitive and source-model accounts in the resource table
above stay separately reported. Classical and zero-width metadata tests exercise
the manager directly. Component graph admission continues to have its documented
quantum-only boundary. Q-0747 remains open for general foreign and wire-symbol
protocol admission. Q-0748's original JSON asset is tracked independently.
