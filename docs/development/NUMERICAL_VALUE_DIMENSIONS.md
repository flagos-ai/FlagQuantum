# Numerical shape and length descriptors

Simulation owns `value_dimensions.py`: unknown numerical arrays and sequences,
and their shape/length queries. The module ports `Shaped`, `HasLength`, `shape`
and `slen` from unmodified Qualtran v0.7.0, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`, to native numerical names. It does not
define Core register signatures or Compiler graph types.

`SymbolicArrayShape(shape)` stores the exact tuple supplied by the caller. Only
the tuple container is checked; dimensions are not converted or validated.
It always reports unknown values, even with concrete dimensions.
`SymbolicSequenceLength(length)` retains the original object without coercion.
It deliberately has no Python `__len__`, which cannot express symbolic or
negative diagnostic lengths. `value_length` selects the first array dimension,
stored sequence length, or ordinary `len`, in that order. `value_shape` returns
the original shape object. Empty array shapes retain the source `IndexError`
when queried for length.

Existing classical activation and controlled-value consumers accept the native
array descriptor and reject unknown activation values before executing a child.
Compiler consumers continue to receive plain shape tuples or counts. Production
Simulation imports no Compiler, Runtime, Core register hierarchy or foreign SDK.

Run the user example without installing Qualtran:

```sh
python -m examples.algorithms.numerical_value_dimensions
```

The example estimates the compute-only predicate from a 2x2 unknown shape, then
uses a known 2x2 pattern with `circuit.append(op)` and the existing controlled-X
emitter. The complete known operation emits 12 T, 3 measurements and 3 work wires;
9 declared wires include a spectator. The predicate-only model has 7 logical
qubits; the complete upstream controlled-X model has 8, excluding the spectator.
These scopes are distinct and are not presented as interchangeable estimates.

`tests/unit/test_value_dimensions.py` compares constructor failures, object
identity, symbolic classification, first-dimension length, and both QEC profiles
directly to the pinned source. The concrete matrix pattern follows the public
source control factory and matches its primitive stream. Full complex amplitudes
and clean work wires are checked over all 8 forward and 64 round-trip measured
histories. Existing activation and controlled-callback replacement tests remain
part of the concentrated regression. Control-pattern conversion is a separate
module; this slice does not implement the full upstream control specification.
