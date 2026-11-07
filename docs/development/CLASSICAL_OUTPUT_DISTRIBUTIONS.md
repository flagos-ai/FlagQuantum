# Stochastic integer-register output binding

The existing Simulation `bind_integer_register_values` now completes the scalar
distribution branch from the unchanged Qualtran v0.7.0
`ClassicalSimState._update_assign_from_vals`, commit
`bcb17bb25360ed68ce3bc084645db3e929799389`. It reuses the accepted integer bounds,
distribution resolver and output collector rather than defining another binder.

An optional `resolve_distribution` callback supplies the graph-owned instance
identity and existing RNG or fixed-value policy. Only a scalar
`ClassicalDistribution` invokes it. Its result is passed immediately to the shared
integer validator and retained by identity on success. There is exactly one
resolution: a sampled distribution does not trigger another draw, and a sampled
symbolic expression does not get the preexisting symbolic-input bypass. With no
resolver, nondeterministic scalar values are rejected.

Ordinary scalar symbolic expressions retain the original early bypass. Shaped
ports keep the existing asanyarray, exact-shape and array-range-validation path;
their elements are never sampled, including when an object array contains a
distribution. Missing fields, invalid shapes and array validation failures occur
before any resolver call. The resolver owns sampling errors and fixed-map lookup
errors, which propagate unchanged. No implicit instance ID or RNG is invented.

The callback does not change graph ownership. This module still binds one
integer register's row-major index keys; it does not traverse or schedule a graph,
create signatures or choose execution backends. The graph owner validates other
dtype families separately. This slice does not equate the diagnostic input domain
of every classical bit type with an unsigned integer word.

Run the updated user example:

```sh
python -m examples.algorithms.classical_measurement_values
```

Its numerical measurement output now flows through the shared scalar binder and
collector before phase resolution. The same `circuit.append(op)` copy/erase path
remains authoritative for quantum behavior: zero T, one measurement, one clean
work wire, and three declared wires including a spectator. Both full complex
quantum branches and trajectory output are checked by the existing measurement
tests. Model leaf counts and emitted copy/clear/feedback instructions remain
separate; no compiler gate recipe changes in this slice.

Reference tests invoke the actual original assignment method with original random,
fixed and banned handlers. They compare signed and unsigned return types and
aliases, exact RNG state, invalid fixed results, sampled symbolic values, missing
instance IDs and shaped rejection without sampling. A replacement test proves
resolution precedes the shared validator exactly once. End-to-end tests use the
new action dispatcher, scalar binding, collector and phase helper against the
original complete MeasureX leaf graph, with exact seeded outcomes and phases.
