# Complete physical cost model source closure

All source assets are frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Parent | Source file | Frozen blob | Native owner |
| --- | --- | --- | --- |
| Q-0919 | surface_code/physical_cost_model.py | 4134fe0c77629987e309e0f94fab520e5423ab1f | qec/resources/physical_cost.py |
| Q-0920 | surface_code/physical_cost_model_test.py | 6db1fd031fa4f39815e152f85125b91650fe39c2 | tests/unit/test_physical_cost_model_file.py |
| Q-0918 reference asset | surface_code/physical_cost_model.ipynb | 5cb5070d0ea667fcb51b1fe972258d8761f8240d | Original execution and 48-scenario native matrix |

## Complete definitions and ownership

The Core-owned `PhysicalCostModel` composes the accepted `PhysicalParameters`,
`SurfaceCodeDataLayout`, `MagicStateFactory` and `SurfaceCodeScheme` boundaries.
The source fields retain their names. `logical_error_model` binds and caches the
scheme. Native `estimate` replaces source `_get_physical_cost_base_quantities`
with the existing immutable `PhysicalResourceEstimate`, retaining separate data
and factory components. Its default 128-entry cache matches the source cache
bound and repeated queries reuse one complete aggregation.

All four source query methods map to native `n_cycles`, `duration_hours`,
`n_physical_qubits` and `error`. Source literature constructors map to
`gidney_fowler` and `beverland`, retaining defaults, custom factory distances,
three Beverland layouts, unknown-layout rejection and subclass construction.
AlgorithmSummary input is supplied through the existing concrete
`ModeledResources` gate categories and peak width. Rotation layers do not enter
this physical aggregation, matching the source.

Direct dependencies have complete factory, layout and scheme interfaces from
preceding batches. Both real Gidney-Fowler and Beverland physical consumers now
use this composition, and their duplicate `_from_components` aggregator is
removed. Existing signatures, defaults, result classes, fields and input
validation remain covered by API and regression checks. Parallel replicas use
the accepted factory implementation, while single factories avoid an additional
floating division. A replacement implementation exercises both actual consumers.
No Stable Core export is added.

## Scientific comparison

The aggregation follows the source order. Cycles are the integer maximum of
production and consumption. Space adds the two footprints. Data error is
calculated over those same cycles, then added to factory error as the source
first-order approximation. Duration preserves the source multiplication and
hour conversion order. No error clipping or confidence reinterpretation is added.

The full matrix covers both factory families with all four layouts, empty and
mixed demand, all query methods and both model constructors. All 48 original
notebook combinations are additionally compared for integer cycle counts,
duration, footprint and error. Full 15-to-1 complex density and per-channel
comparisons reuse the preceding frozen-source evidence. Small numerical density
rounding accumulates with the notebook's 400-million-T demand, so the total error
comparison records an absolute 8e-7 bound derived from 2e-15 per T state. It does not assert bitwise numerical
equality. The observed maximum absolute total-error difference is
3.9968028886505635e-7. Exact cycle/time/space comparisons remain exact.

These scalar models emit no circuit or new measurement branches. Primitive gate
counts, QEC categories and physical estimates remain separate. The original
notebook test is run against the unchanged, hash-verified asset with an isolated
Python kernel. Native notebook presentation remains a separate Q-0918 delivery.

The scheme's strict budget rounding boundary Q-0924 and unrelated Q-0613
symbolic/bounded component diagrams remain open. Existing native validation uses
its established ValueError policies. Cache-dependent replacement tests clear the
aggregation cache before injecting reference formulas so cached evidence cannot
bypass the replacement.

All 288 concentrated resource regressions pass. The unchanged original notebook
test passes in the isolated kernel. All six changed Python files pass strict
mypy, Black and Ruff, with Core-team scope, architecture, IR and API checks
passing. Q-0919 and Q-0920 close as complete source/test modules.
