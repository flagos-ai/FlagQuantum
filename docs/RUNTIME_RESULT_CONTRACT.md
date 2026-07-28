# Runtime result contract

`flagquantum.runtime.ExecutionResult` is the stable result returned by
user-facing execution APIs.

## Canonical fields

- `value`: differentiable observable or loss value;
- `state`: materialized tensor state when the selected mode naturally returns
  one;
- `samples`: sampled outcomes;
- `plan`, `accuracy`, `metrics`, `provenance`, `runtime`, `compatibility`:
  structured execution metadata.

`state` is an attribute, not a method. Backend-native objects returned by
`run_native` may continue to expose `state()` for compatibility.

## Summary projection

`summary()` always preserves the structured sections above. Runtime metadata is
also projected to the top level for historical distributed consumers:

```python
import flagquantum as fq

program = fq.Circuit(n_qubits=2).h(0).cx(0, 1)
result = fq.run(program)
summary = result.summary()

assert summary["runtime"]["mode"] == summary["mode"]
assert summary["has_plan"] is True
```

Runtime keys cannot overwrite canonical fields. Conflicts are listed in
`runtime_projection_conflicts`.

## Backend-native compatibility

Normalized legacy results retain a private reference to the native result:

- `to_statevector()` delegates when full-state materialization is supported;
- backend-specific read-only attributes such as `shards` remain accessible;
- runtime summaries remain live, so communication/materialization counters
  update after delegated operations.

New code should prefer canonical fields and use `run_native` only when it
intentionally depends on backend-specific behavior.
