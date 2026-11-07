# Numerical symbolic parameter classification

Simulation needs a numerical parameter predicate independent of Compiler's graph
identities and Core's protected contracts. `value_parameters.py` owns the exact
`is_symbolic` dispatch from unchanged Qualtran v0.7.0 `symbolics/types.py` at
`bcb17bb25360ed68ce3bc084645db3e929799389`.

Zero or multiple arguments use `any` and its original short-circuit order. One
argument checks `sympy.Basic` first, then retrieves `is_symbolic` once and invokes
it if present. Otherwise the result is false. A single checker result is returned
without a boolean cast; its type, identity and errors remain visible as upstream.
There is no integer fast path: an integer subclass can implement this protocol.

The owner is shared by numerical integer, bounded integer, fixed-point and
Montgomery codecs/value models. It has no gate or register semantics, imports no
Compiler/Runtime implementation and emits no quantum instructions. No symbolic
execution, dtype hierarchy or public Core API is introduced.

Sympy remains optional for native numerical execution. If the Sympy package itself
is absent, classification still uses the object protocol; no Sympy objects can be
constructed through that absent package. An import failure inside an installed
Sympy package is propagated rather than hidden. Source parity is checked with the
pinned dependency installed, and the separate native no-Sympy behavior is tested
explicitly. No dependency manifest is changed.

This corrects the earlier private width helper, which skipped integer callbacks
and coerced single-checker results. The source classifier rejects symbolic integer
subclasses at signed encoding and modular numeric boundaries even when their
Python arithmetic could run. Source error order must not be replaced by eager
coercion or a guessed concrete dimension.

## Verification and existing user workflows

The predicate is an internal numerical dependency; users continue to construct
operations with `circuit.append(op)`. The fixed-point and bounded-address golden
paths exercise its consumers without introducing a symbolic-object API:

```sh
python -m examples.algorithms.classical_fixed_point_values
python -m examples.algorithms.classical_bounded_integer_values
pytest -q tests/unit/test_value_parameters.py
```

Forty-two new cases compare the unchanged classifier directly: ordinary values,
NumPy and Sympy objects, integer subclasses, arbitrary single-checker results,
variadic call order, short-circuiting, noncallable attributes and callback errors.
Single-result identity and type are checked without coercion. A Sympy subclass
whose checker access would fail proves that Basic recognition occurs first.

Consumer checks compare signed integer encoding, fixed-point shape metadata,
bounded shape validation, Montgomery conversion/domain and the classical
Montgomery window against the unchanged upstream methods using symbolic integer
subclasses. Separate import-boundary tests verify native execution without Sympy
and propagation of a failure inside an installed dependency. Existing coherent
gate/measurement/work and resource tests are rerun for the affected consumers;
the predicate itself adds no gates, qubits or measurements.

Acceptance: 42 new protocol cases and 1,980 concentrated passes without skips;
195 original/native arithmetic warnings remain visible. Six production files
pass changed-surface strict typing. Formatting, ownership, architecture and the
fixed-point, bounded SELECT and Montgomery examples pass. Pinned symbolic,
numerical, decomposition and budget source blobs remain unchanged. Remaining
source shape/length abstractions are not claimed as ported by this predicate.
