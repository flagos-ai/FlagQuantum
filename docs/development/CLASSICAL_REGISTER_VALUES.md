# Classical integer register values

Simulation owns value binding and numeric-domain checks. The native
`classical_registers.py` and `integer_values.py` consume plain names, values,
word widths and dimension tuples. They do not introduce a new Register/Soquet
class, modify Core contracts, execute a graph or allocate quantum storage.

The pinned reference is unmodified Qualtran v0.7.0 commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, specifically
`ClassicalSimState._update_assign_from_vals`, `_Int`, `_UInt` and
`_MontgomeryUInt` range checks. Tests use the original simulator constructor to
initialize assignments on its real signature and an empty instruction graph.
They verify initialization only, not graph execution.

## Interfaces and source policies

`bind_integer_register_values(name, values, width, shape=(), signed=False)` binds
one named input to an insertion-ordered mapping from row-major index tuples to
original element objects. The empty index `()` represents a scalar word.

1. Fetch the named value; missing input raises ValueError chained from KeyError.
   Extra named inputs remain ignored, as in the original single-register loop.
2. A nonempty shape takes the array branch. Use `np.asanyarray`, require exact
   shape, apply original range checks, and bind indices in product/range order.
   Zero-length axes produce no element assignments but still require the field.
3. A scalar Sympy expression bypasses domain validation and is retained by
   identity. Other scalar values use the shared integer-word validator. A zero-
   dimensional ndarray is not silently converted into a scalar integer.

Shapes must be concrete and enumerable; even a Sympy integer shape dimension
retains the original shape-property failure. A scalar word width can be an
unchanged Sympy expression because value-domain comparisons are a different
operation from enumerating register elements. Original symbolic comparison
results and unresolved comparison errors remain visible.

`validate_integer_word_array` intentionally checks ranges only, while
`validate_integer_word_value` also checks scalar integer type. Both share word
bounds for nonnegative Python widths or symbolic integer expressions, including
source zero-width classical behavior. Width metadata is never cast. The later
Negate composite still keeps its separate zero-width and symbolic gate/value
boundaries; extending the numeric validator does not repair those failures.

## Diagnostic acceptance is not quantum executability

Original array validation does not enforce integer dtype. A floating array,
NaN, a masked high element or some object-array values can therefore pass this
initial binding. The native path retains that behavior and element types. The
same floating value is rejected in a scalar register. `asanyarray` preserves
masked subclasses; it must not be replaced with `asarray` or an integer cast.

These are source diagnostic/value-protocol semantics. They do not certify that
an array can be encoded as computational-basis words or executed by a quantum
backend. Stochastic classical distributions, full graph propagation and foreign
type/Soquet semantics remain separate unfinished migration work.

## Runnable integer example

```sh
python -m examples.algorithms.classical_register_values
```

The example binds unsigned two-bit words `[[1,2],[3,0]]` in row-major order and
uses the existing XorK value model. `circuit.append(XorEachWord())` agrees with
output `[[0,3],[2,1]]`. The eight data wires need no work or measurement wires.
The primitive scan is four X gates; the independent source model is four
Clifford, zero T and zero measurements. No logical storage estimate is substituted
for the actual declared wire count.

## Acceptance

G2-VAL01 adds 145 tests: 27 scalar/array width-boundary tests and 118 register-
binding/composition tests. They cover original result types and insertion order,
error classes and causes, input identity, missing fields, strided arrays, masked
subclasses, symbolic values/widths/shapes, literal zero and empty axes. A composed
nine-wire test retains an entangled spectator and compares every complex
amplitude across all 512 inputs, both XOR applications, original initialized
values and XorK outputs. There are no hidden auxiliary wires.

The concentrated suite passes 541 tests without skips, including existing
Negate intermediate validation, arithmetic, lookup and bitwise regressions.
Thirty-six paired original/native NumPy warnings remain visible. Three
production/example files pass strict typing; Black, Ruff, Simulation scope,
architecture and the runnable example pass. No pinned source or public API was
changed. The complete classical interpreter and G2 are not declared finished.


## Collecting register outputs (G2-VAL02)

`collect_integer_register_value(assignments, width, shape=..., signed=...)`
ports the original `_get_in_vals` and integer NumPy storage selection into the
same Simulation module. Keys are native row-major index tuples rather than
foreign graph endpoints. Scalars return the original object at `()` without
copying or revalidation. Arrays use signed or unsigned 8/16/32/64-bit NumPy
storage at the original width thresholds, and object storage above 64 bits.
Missing entries propagate `KeyError`; assignment conversions propagate NumPy's
exceptions. Collection performs no second domain check. Symbolic width comparisons
retain the source relational failure for arrays; scalars bypass dtype selection.
Even an empty array selects its dtype before traversing its indices.

This helper intentionally does not preserve the incoming array subclass: source
output assembly allocates a fresh plain ndarray. Input binding still preserves
`asanyarray` subclasses. Finite-field arrays, arbitrary custom dtypes, graph-edge
routing, stochastic dispatch and the full classical graph interpreter remain
open; the integer helper does not claim those capabilities.

The runnable `examples.algorithms.classical_register_values` now binds, transforms
and collects the two-dimensional values before comparing with `circuit.append(op)`.
The existing coherent test feeds all 512 complex amplitudes through this collection
path, including an untouched spectator, and checks exact double-XOR restoration.
The quantum path remains four X/Clifford gates, zero T, measurements or work wires;
collection is a classical numerical operation and adds no quantum resources.

Acceptance: 100 new source assembly cases, 369 concentrated passes, no skips.
Eight existing paired source/native NumPy overflow warnings remain visible.
Checks include dtype thresholds, missing indices, empty axes, symbolic scalar
identity, symbolic width errors, conversion failures and 65-bit exact values.
The pinned `classical_sim.py` blob is verified against the original inventory:
`a1933c22023d54fd7683a5a1aa29ac595a046e24`. Source code is not modified.
