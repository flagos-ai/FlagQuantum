# Complete k-XOR instance source

Qualtran v0.7.0 `bcb17bb25360ed68ce3bc084645db3e929799389` supplies
`optimization/k_xor_sat/kxor_instance.py`, frozen blob
`70f08821a68cb1c708f6544ba7cf5170dc223e0e`. The complete file maps to
`flagquantum.algorithms.kxor_instance`. It closes Q-0614 and Q-0615 and supplies the
classical data dependency for instance loading and Kikuchi algorithms.

## Ownership and interface

`Constraint` owns a sorted tuple of zero-based variable indices and its signed
right-hand side. `KXorInstance` owns a multiset of constraints, variable/clause
widths, symbolic declarations and classical preprocessing. Domain consumers
import this direct Algorithms module. Existing neutral scalar arithmetic and
shape-only sequence descriptors provide symbolic lengths. The planted/random
factories retain the original NumPy RNG stream and scalar values.

The complete source includes scope sorting, random and planted constraints,
symbolic constructors, constraint evaluation, default maximum RHS, instance
size, subsetting, variable-index width, grouping, RHS width, big-endian scope
packing, brute-force Kikuchi sparsity and the original example. All are retained.
`attrs.evolve` preserves the parent's maximum RHS when taking a subset, including
an empty subset. Grouping sorts by summed RHS with stable insertion-order ties.
Cancelled scopes remain in that list and in the original sparsity calculation.

Packed scopes concatenate `ceil(log2(n))` bits per index in sorted order. Their
first index occupies the most significant chunk. Data remains classical at this
boundary. The following scope-loader batch owns registers, clean workspace and
operations consumed through `circuit.append(op)`.

## Source parity and retained boundaries

Exhaustive four-variable pairs of two-variable scopes cover both RHS signs,
repeated scopes, all sign assignments, all subset orderings used in the fixture,
packed values and all possible Kikuchi subset widths. Symbolic lengths, RHS
symbols, expression widths, symbolic subsets and concrete instances containing
symbolic constraints match the frozen source. Fixed-seed random and planted
factories match each constraint, grouped RHS and final RNG state.

The complete original test file, blob
`9b80e6b70b5af53a111792989981fa4667e7cb11`, runs unchanged for all three original
planted advantages. Its original owner runs first. The same loaded test function
then runs with the native class supplied through its module global. The source
files on disk remain hash-checked. This reuses all original 100-instance,
1,000-clause statistical checks for each advantage on both implementations.

The original planted sign formula has expectation `-rho`, despite its comment
claiming `rho`. At `rho=1` an all-positive planted assignment produces RHS `-1`.
Parity deliberately retains the formula and registers the discrepancy here.
Empty instances without an explicit maximum RHS raise the original `ValueError`.
Evaluating an empty scope retains NumPy's invalid-index failure. Arbitrary invalid
indices, non-sign assignments and overflow of NumPy scalar packing inherit the
original behavior and carry no extended-input correctness claim.

## Resource and execution evidence

This file performs classical preprocessing. Its measured quantum primitive
inventory and quantum resource model are inapplicable. Subsequent oracle audits
will report their actual instructions, complex amplitudes, measurement histories,
work cleanup and independent model counts. This file's exhaustive constraint
values and packed scopes form the inputs to those comparisons.

## Validation

The batch regression passed 288 tests with three retained original notebook/IPython
warnings. Black, Ruff, strict mypy for both changed Python files, Algorithms scope,
architecture, multi-level IR and protected API checks passed. Native local variable
names follow repository style. The original field `Constraint.S` and its constructor
remain available. Native `scope_as_int` names its argument `scope`.
