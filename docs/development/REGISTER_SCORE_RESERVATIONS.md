# Register score lane reservations

## Ownership and interface

`drawer/register_lanes.py` owns the existing minimum-lane heap and reservations.
`register_score.py` owns dependency columns and completed lifetimes.
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
External callback-owned state retains its usual caller ownership. Foreign
`LineManager` objects, overridden `new_y` policies and hooks inspecting source
partial `HLine` sets remain open protocol boundaries. The supported subclass
extension is the native `maybe_reserve` hook.

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
the bound operation. Q-0747 remains open for its general foreign protocol and
custom partial-line/new-coordinate behavior. Javascript animation remains a
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
