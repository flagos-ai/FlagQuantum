# API Change Proposal 048: Connected Twin-region model composition

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** The
user authorized the next digital-twin slice after connected structural coverage:
compose compatible local Twin models into one runnable regional prediction model.

## Problem

`compose_connected_region()` proves only that validated local cells form a
connected structural footprint. It deliberately cannot predict a circuit wider
than one source cell. Applications need an offline regional model without
silently treating local validation bounds as joint-circuit accuracy.

## Decision

- Add `compose_region_twin(...)` for the existing sequence of `(twin, support)`
  cells and return an immutable `TwinRegionModel`.
- Reuse `compose_connected_region()` as the structural gate.
- Remap every local profile, gate-noise rule, and readout rule into one canonical
  regional wire order.
- Require identical overlapping qubit calibrations, gate durations, scoped noise
  channels, and readout errors. Reject conflicts before returning a model.
- Compose unscoped gate noise only when every source cell declares the same
  channel for that gate.
- Never infer correlated noise between cells.
- Require the caller's complete physical mapping to exactly match the composed
  region wire order and reject circuits outside the region's topology or limits.
- Return an ordinary `TwinPrediction`. Do not expose a regional evidence report,
  TV error bound, confidence level, provider operation, or accuracy claim.

## Public API

```python
import flagquantum as fq

twin_a = fq.twin.load_twin("cell-a-twin.json")
support_a = fq.twin.load_circuit_support("cell-a-support.json")
twin_b = fq.twin.load_twin("cell-b-twin.json")
support_b = fq.twin.load_circuit_support("cell-b-support.json")

region_twin = fq.twin.compose_region_twin(
    [(twin_a, support_a), (twin_b, support_b)]
)

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
prediction = region_twin.predict(
    circuit,
    physical_qubits=(20, 27, 34),
)

print(prediction.twin_probabilities)
print(prediction.total_variation_from_ideal)
```

The code is fully offline and submits no QPU task. The last value compares the
composed model with noiseless simulation; it is not Twin-to-QPU accuracy.

## Compatibility

The change is additive. Existing Twin names, signatures, serialized schemas,
and evidence semantics remain unchanged. No persistence schema is introduced
for `TwinRegionModel` in this slice.

## Acceptance

- Compatible overlapping cells deterministically predict one wider connected
  circuit.
- Mapping order, topology, operation, instruction-count, and depth violations
  fail closed before prediction.
- Calibration, duration, channel, readout, target, capture-time, identity, or
  connectivity conflicts are rejected.
- The public contract, guide, release notes, capability boundary, executable
  example, and focused tests all describe model output without regional accuracy.
