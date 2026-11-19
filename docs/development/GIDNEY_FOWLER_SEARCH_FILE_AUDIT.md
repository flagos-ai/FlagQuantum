# Complete Gidney-Fowler budget and search source closure

The full source is frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Parent | Source file | Frozen blob | Native owner |
| --- | --- | --- | --- |
| Q-0912 | surface_code/gidney_fowler_model.py | 3452b58e02553c2995751898a5cbcfea30e2aee9 | qec/resources/physical_search.py |
| Q-0913 | surface_code/gidney_fowler_model_test.py | 006dbb7289064df1ed4b88d23018c55718da121a | tests/unit/test_gidney_fowler_search_file.py |

## All definitions and direct dependencies

The source backwards-compatibility `get_ccz2t_costs` is supplied directly by the
accepted `PhysicalCostModel.estimate` composition, with its source three summary
fields mapped to `failure_probability`, `physical_qubits` and `duration_hours`.
`qubit_hours` has the same multiplication. No redundant compatibility wrapper or
summary type is introduced.

Source budget inversion maps to `estimate_gidney_fowler_from_budget`, retaining
all defaults, custom routing overhead, optional factory and explicit data-layout
override. It subtracts factory error before selecting data distance from the
spacetime-volume budget. An explicit layout bypasses the budget-derived distance.
Source factory/data iterators map to `iter_ccz_factories` and
`iter_routing_data_layouts`, preserving order, ranges and optional parallel
replicas. The full grid search maps to `search_gidney_fowler_resources`, using
the existing immutable physical estimate, factory and layout interfaces.

The direct dependency closure reuses the complete CCZ factory, parallel factory,
physical model, hardware assumptions, shared scheme and configurable routing
layout. These calculations own no quantum registers, measurements or scratch
space. Source primitive/channel amplitudes and workspace evidence remain at
those implementations. Physical scalar costs are not counted as emitted gates.

## Source comparison and retained boundaries

Full ordered iterator output is compared for one, two and four replicas. Budget
inversion and explicit overrides agree exactly for four routing overheads. All
source spreadsheet and grid examples are compared, including the complete
6,665,400,000-Toffoli THC case with four factories, optimizing duration. The
selected distances, total errors, durations and footprints agree exactly.
The original three source tests run unchanged.

One-pass layout iterators retain a source defect. The nested source loop consumes
the layout iterator for the first factory and never revisits later factories.
The reproduction supplies an expensive factory first and a cheaper factory
second. An iterator selects the first factory in both implementations, while a
reusable tuple reaches the cheaper second factory. This is recorded behavior,
not a claim of an exhaustive Cartesian search for one-pass layout inputs.
Equal objective values retain the first admissible pair.

Empty demand or zero width in budget inversion retains source division by zero.
A factory exceeding its budget and a grid with no admissible candidate preserve
ValueError. Budget inversion keeps the shared source floating-rounding behavior
recorded independently under Q-0924. Existing native physical-parameter validation
continues to delimit the supported hardware domain. Q-0613 diagrams remain open.

All 312 concentrated resource regressions and three unchanged original source
tests pass. Both changed Python files pass strict mypy, Black and Ruff, with
Core-team scope, architecture, IR and API checks passing. Q-0912 and Q-0913
close as complete source/test modules with the recorded iterator defect.
