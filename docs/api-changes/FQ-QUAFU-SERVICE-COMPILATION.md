# Quafu service compilation through fq.run

## Decision and authorization

The repository owner explicitly approved this journey in the implementation
conversation: omit the local compiler for direct Quafu submission; preserve
explicit QSteed compilation. Approval covers this narrowly scoped behavior
extension. Status: implemented candidate, pending review and release.

## User journey and compatibility

`fq.run(circuit, target="quafu:Baihua", shots=1024)` submits logical OpenQASM
with `options.compiler="quarkcircuit"`. No local QSteed or Quark SDK is loaded,
and no local calibration lookup is required. Optional `target_qubits` is
validated structurally; the service validates physical availability/topology.
FlagQuantum still lowers and serializes its IR and inserts measurement bases;
it does not perform device-specific routing in this path.

The public signature, exports, serialized schemas and existing successful calls
remain unchanged. This previously rejected argument combination is now accepted.
`compiler="qsteed"` retains local plugin compilation and `options.compiler=None`
submission. No deprecated argument or removal window is needed. The root API is
appropriate because both paths return the existing ExecutionResult, support the
same output requests, and represent the same execute-on-target journey.

## Evidence and validation

The direct path converts Quafu's classical-MSB-left count strings to
FlagQuantum's measurement-wires-left-to-right ordering before expectation
aggregation. The existing explicit-precompilation path is unchanged. An
asymmetric fixture covers this boundary; Bell-only counts cannot detect it.

Service-compiled results record `compiler=None`, `compilation_location="service"`
and `service_compiler="quarkcircuit"`. Submission digests identify submitted
artifacts, not the circuit ultimately executed by the service. Returned
`transpiled` data remains provider evidence; local tests cannot certify hardware
execution or physical mapping chosen by the service.

Tests block local compiler/SDK imports and calibration requests, exercise the
actual HTTP adapter with a fake transport, check counts and grouped expectations,
and reject invalid mappings/shots before submission. Existing precompiled
provider and root execution contracts are retained. API snapshots are not
regenerated. The Quafu guide and unreleased notes document the new journey.
