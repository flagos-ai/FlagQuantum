# Complete magic-state factory source closure

All source files are frozen at Qualtran v0.7.0 revision
`bcb17bb25360ed68ce3bc084645db3e929799389`.

| Parent | Complete source | Frozen blob | Native owner |
| --- | --- | --- | --- |
| Q-0914 | magic_state_factory.py | 8f6020554bb004567c7e77eab771a3446557ac51 | qec/resources/magic_state_factory.py |
| Q-0906 | ccz2t_factory.py | 937748981f673d46d081a42b1904a2e3cfbd7529 | qec/resources/ccz_to_t_factory.py |
| Q-0907 | ccz2t_factory_test.py | 5fd35ce68b0caaa4a1267a82e8c0c11d33258772 | tests/unit/test_magic_state_factory_files.py |
| Q-0917 | multi_factory.py | 3d57ab743ff595cb3c7fdccdf0a033b22cfeedde | qec/resources/parallel_factory.py |

## Definitions and direct dependencies

The abstract factory has the original three abstract methods for qubits,
production cycles and cumulative error. `CCZ2TFactory` maps to `CCZToTFactory`.
Source `distillation_l1_d` and `distillation_l2_d` become
`level_one_distance` and `level_two_distance`, retaining defaults 15 and 31.
All eight level-zero, level-one and level-two error methods retain their source
formulas, evaluation order and the shared logical-error callable.

`MultiFactory` maps to `ParallelFactory`, retaining its base factory and
`n_factories` fields. It scales physical footprint, ceil-divides cycles and
forwards cumulative error. Its native base contract is exercised with both the
concrete CCZ factory and parallel factory composition.

The direct dependencies are the already accepted shared surface-code scheme
and existing gate-category budget conversion. The actual Gidney-Fowler
estimator now calls the complete CCZ factory. Its existing concrete demand
conversion and physical aggregation retain their previous boundaries.
An implementation-replacement test verifies the same consumer against an
alternative factory subclass and observes the production/footprint calls.

## Source comparison

All eight distillation stages and footprints agree exactly for four distance
pairs and five physical error rates. Full demand conversion and parallel rules
agree for six gate budgets and four replica counts, including odd T demands,
rotations, Ands, swaps and the original 100-million-gate example. Symbolic CCZ
error agrees as an expression. Unbound symbolic production cycles and zero
replica division preserve the original TypeError and ZeroDivisionError.
The original CCZ test runs unchanged and passes.

These files estimate scalar physical resources. Circuit registers, measurement
branches and clean workspace belong to the existing primitive emitters. Existing primitive emission and
logical QEC accounting remain separately verified at their owners. Existing
Gidney-Fowler, Beverland, 15-to-1 density-matrix and resource-report tests are
included in the batch regression.

The batch passes 124 physical-resource regression cases. All five changed Python
files pass strict mypy, Black and Ruff, with Core-team scope, architecture, IR
and API checks passing. The unchanged source test passes.

## Disposition

Q-0914, Q-0906, Q-0907 and Q-0917 close as complete files. Original arithmetic
failures above are explicit behavior, with direct reproductions. Upstream
floating budget inversion recorded under Q-0924 remains open independently.
The unchanged physical-model and algorithm evidence from the preceding verified
batches is reused.
