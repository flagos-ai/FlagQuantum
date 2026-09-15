# API Change Proposal 047: Connected Twin-region structural coverage

## Status

**Implementation authorized; additive to the frozen Twin v1 contract.** The
user authorized the next digital-twin slice after topology-qualified circuit
support: compose prospectively validated cells into a connected structural
region without turning local evidence into a joint accuracy claim.

## Problem

`TwinCircuitSupport` correctly qualifies one frozen Twin mapping by directed
physical couplers and circuit depth. Applications that hold several overlapping
cells still cannot ask whether an explicitly mapped circuit stays inside their
connected union. Treating those local cells as one statistically verified
multi-qubit model would be invalid because independently measured error bounds
do not establish joint-circuit accuracy.

## Decision

- Add `compose_connected_region(...)` for a sequence of `(twin, support)` cells.
- Reject mismatched Twin/evidence identities, different provider/backend
  targets, different calibration capture times, duplicate cells, disconnected
  unions, and cells with no common operation support.
- Add immutable `TwinConnectedRegion` with deterministic identity and an
  explicit `coverage_report(...)` operation.
- Require a complete logical-to-physical mapping for every checked circuit.
- Preserve directed couplers and report missing qubits and interactions.
- Use the minimum instruction-count and circuit-depth limit and the intersection
  of operation support across source cells.
- Add immutable `TwinRegionCoverage` with only `covered` and `out_of_scope`
  outcomes.
- Keep `tv_error_bound` and `confidence_level` fixed to `None`; local evidence
  is never averaged, added, or promoted into region accuracy.
- Perform no provider access, hardware submission, model update, routing
  decision, or agent/MCP operation.

## Public API

```python
import flagquantum as fq

twin_a = fq.twin.load_twin("cell-a-twin.json")
support_a = fq.twin.load_circuit_support("cell-a-support.json")
twin_b = fq.twin.load_twin("cell-b-twin.json")
support_b = fq.twin.load_circuit_support("cell-b-support.json")

region = fq.twin.compose_connected_region(
    [(twin_a, support_a), (twin_b, support_b)]
)

circuit = fq.Circuit(3).h(0).cx(0, 1).cx(1, 2)
coverage = region.coverage_report(
    circuit,
    physical_qubits=(20, 27, 34),
)

print(coverage.status)
print(coverage.covered_qubits)
print(coverage.covered_directed_couplers)
print(coverage.missing_qubits)
print(coverage.missing_directed_couplers)

assert coverage.tv_error_bound is None
assert coverage.confidence_level is None
```

The four input files come from existing prospective cell-validation workflows.
The complete example is `examples/twin_connected_region.py`. It is entirely
offline and submits no QPU task.

## Interpretation

`covered` means that the explicit mapping, directed two-qubit interactions,
operation names, instruction count, and circuit depth fit inside the
conservative intersection of the supplied cells. It does not mean the composed
circuit has been run on hardware or that its output-distribution error is
bounded. Region-level accuracy requires a separate prospective validation of
the complete mapped circuit or region.

## Compatibility

The change is additive. Existing Twin types, functions, signatures, status
literals, serialized schemas, and measurement-distribution semantics remain
unchanged.

## Acceptance

- Overlapping cells from one target and capture form a deterministic connected
  region.
- A fully mapped circuit on declared directed couplers reports `covered`.
- Missing qubits, reverse or absent couplers, unsupported operations, excessive
  instruction count, excessive depth, and operations wider than two qubits fail
  closed as `out_of_scope`.
- Cross-target, cross-capture, duplicate, identity-mismatched, and disconnected
  compositions are rejected before a region is returned.
- Every coverage report has no TV error bound or confidence level.
- API contract, guide, limitations, release notes, tests, and executable example
  describe the same structural-only behavior.
