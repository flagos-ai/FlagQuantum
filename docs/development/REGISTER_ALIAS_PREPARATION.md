# Dense and sparse alias preparation

The [locked table preprocessing](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/linalg/lcu_util.py)
rounds cumulative probabilities and assigns alternates in two donor
passes. Dense precision is absolute in the coefficient scale. The
sparse factory uses normalized precision and retains mapping order.
The dense-to-sparse factory applies the source absolute zero tolerance.
Tests compare the integer tables, precision width and normalization
directly against the unchanged factories.

The [locked preparation circuit](https://github.com/quantumlib/Qualtran/blob/bcb17bb25360ed68ce3bc084645db3e929799389/qualtran/bloqs/state_preparation/state_preparation_alias_sampling.py)
prepares uniform rows, applies Hadamards to sigma, loads the alias
record, compares keep <= sigma, and swaps alternate with selection.
The sparse path uses an independent row index and also loads selection
from the table. Selection, sigma, alternate, keep, comparison flag and
the sparse row index are logical outputs. The adjoint restores the
matching prepared registers to zero. Temporary work returns to zero
in both directions, and classical slots may be reused.

Singleton addresses occupy zero qubits. The native composition handles
these by omitting empty-width lookup words and using the shared constant
lookup. Tests cover dense and sparse singleton restoration separately
from the source comparisons.

| Case | T + T† | Measurements | Rz | Clean work | Slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense weights (0, 1, 2), mu=1 | 42 | 7 | 2 | 4 | 2 |
| Sparse weights {3: 1, 0: 3}, mu=1 | 18 | 1 | 0 | 2 | 1 |

The emitted T-family, measurement and rotation counts match the source
model in these cases with seven T-family gates per exact controlled
swap and separate accounting for explicit Rz. Tests check the source
parent stage order, QROM and comparator dependencies, the exact swap
stream, the complete source complex state, small raw measurement
histories and inverse cleanup through the append entry point.


## Native QEC profiles (G2-ALIAS08)

`source_dense_alias_preparation_cost(table, legacy_shims=True)` and the sparse
counterpart pass the selected profile to uniform preparation, QROM and
LessThanEqual. `LinearCombinationPreparation.resource_model(legacy_shims=True)`
provides the same native path for coefficient PREPARE. The default remains the
modern profile. The source's abstract CSwap category is retained; profile
selection does not synthesize that swap or change an emitted instruction.

Dense/sparse table cases and one-, three- and eight-bit LCU coefficient cases
compare every source gate category and logical peak in both directions.
Structured LCU, precontrolled LCU and repeated-root legacy tests now obtain all
child costs from FlagQuantum factories; the temporary test-only original-child
model adapter was removed. Existing full complex-state and clean-work tests
continue to exercise the unchanged emitters. Models never invoke these emitters.

```sh
python -m examples.algorithms.linear_combination_controls --repeated --legacy-profile
```

The runnable append example retains the same coherent state and inverse checks,
prints the chosen model Clifford classification, and reports 28 model T
equivalents versus 34 actual IR T/T-dagger instructions. Both profiles retain
five measurements and 13 peak logical qubits. Profile choice changes source
Clifford accounting, not the circuit or the explicitly separate IR report.
Singleton model boundaries and controlled-uniform preparation are not extended
by this profile propagation.

This adapter adds 28 tests; the final combined compiler/algorithm regression
passes 1,105 tests without skips. Strict types, formatting, lint, ownership and
architecture checks pass. Both reference trees remain unchanged.
