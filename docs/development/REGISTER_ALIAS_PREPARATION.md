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


## Table-free model profiles (G2-ALIAS09)

`AliasPreparationShape.resource_model(legacy_shims=True)` now propagates the
source QEC profile to shape-only QROM and concrete uniform preparation. For a
symbolic uniform dimension or comparison width, it preserves the original
symbolic declaration and applies the shared positive-And cleanup charge to its
declared multiplicity. The default remains unchanged. The legacy flag selects
QEC classification; it does not switch to the different legacy T-complexity
folding rates or infer any primitive gate stream.

All seven categories and source peak widths are compared for concrete/symbolic
row counts and precision, dense/sparse shapes, both profiles and both source
directions. Replacement and no-emission tests verify the shared QROM boundary.

The source's symbolic declarations and concrete reconstruction are distinct
queries. With normalization equal to row count and precision 0.125, the symbolic
factory retains a Sympy integer for mu even after it simplifies to 3. For L=5,
substitution of that symbolic declaration gives Clifford counts 179/215 in the
modern/legacy profiles. Reconstructing the factory with Python integer 5 gives
165/197. Native queries match both unchanged original results, and preserve the
14/18 differences. Do not normalize types or optimize either reference formula
and then claim it represents the other original call chain.

```sh
python -m examples.algorithms.alias_resource_profiles
python -m examples.algorithms.alias_resource_profiles --symbolic
```

The default example uses five rows and three probability bits. It reports
modern/legacy categories and model T-equivalent budgets for dense preparation
and sparse preparation in a 16-element selection domain. The symbolic option
prints formal declarations in L and N; its dimensions retain the source shape
conditions. Neither example emits or simulates a circuit. Actual data, gate
streams, complex amplitudes and work cleanup remain the separate executable
alias path covered by the concentrated regression.

G2-ALIAS09 adds 25 parameterized/contract/witness cases, following the ten new
G2-RES36 shared-QROM cases. Final regression: 1,312 passed, no skips. Strict
types, formatting, lint, scope and architecture checks pass.
