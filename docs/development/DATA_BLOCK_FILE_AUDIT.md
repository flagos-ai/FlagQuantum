# Complete surface-code data block source closure

The complete source is frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Parent | Source file | Frozen blob | Native owner |
| --- | --- | --- | --- |
| Q-0908 | surface_code/data_block.py | c0237613c2e1ef7d56726ba33c5469712b84a66f | qec/resources/data_layout.py |
| Q-0909 | surface_code/data_block_test.py | 86931572a58dfcd0aaa6dc9b88ba3008d13f8eec | tests/unit/test_data_block_file.py |

## Ownership and complete definition mapping

The Core team's QEC resource domain owns the geometry and physical model.
`SurfaceCodeDataLayout` retains the three abstract source members with native
names `distance`, `tiles` and `steps_per_magic_state`. Its concrete methods model
T-plus-CCZ consumption, physical qubits and storage error, using the existing
shared gate categories and logical-error model. Symbolic demand stays symbolic.
No model type or method is added to Stable Core exports.

| Source definition | Native definition | Geometry and throughput |
| --- | --- | --- |
| SimpleDataBlock | RoutingDataLayout | Configurable routing overhead, zero consumption cycles |
| CompactDataBlock | SingleHallwayDataLayout | ceil(1.5n) tiles, nine steps per state |
| IntermediateDataBlock | TwoRowDataLayout | ceil(2n) tiles, five steps per state |
| FastDataBlock | SquareDataLayout | ceil(2n + sqrt(8n) + 1) tiles, one step per state |

`SquareDataLayout.tile_count` maps the distance-independent source helper.
`from_error_budget` retains the original per-tile budget inversion and subclass
construction. All layout distances and routing parameters are immutable.
Source arithmetic domain failures remain observable. The existing `DataBlock`
facade retains its constructor, defaults, input validation and concrete demand
conversion, and delegates geometry and consumption to these four implementations.
The Gidney-Fowler estimator exercises an independent source-backed replacement
of that implementation without changing its consumer.

## Evidence and accounting

The complete source files are hash checked. The parity matrix covers all four
layouts across distances 0, 3, 7 and 21 and widths 0, 1, 7, 100, 1318 and 12581.
Custom routing includes negative, zero, fractional and greater-than-one overhead.
Empty and symbolic gate budgets, both QEC schemes, error-budget construction,
subclass construction, immutable fields and original domain failures are covered.
The original three source tests run unchanged.

This file estimates physical scalars and owns no quantum registers, measurement
branches or workspace. Primitive emission, complete complex amplitudes and clean
ancillas retain the existing algorithm/primitive evidence. Measured primitive
counts and declared QEC costs remain separate from these physical assumptions.
The shared scheme's strict floating budget boundary under Q-0924 remains recorded
independently, without changing reference formulas or adding a success tolerance.

The batch passes 256 concentrated physical-resource regressions and all three
unchanged source tests. All three changed Python files pass strict mypy, Black
and Ruff. Core-team scope, architecture, IR and API checks pass.
Q-0908 and Q-0909 close as complete source files.
