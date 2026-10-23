# API Change 074: Dynamic Circuit V1 implementation evidence

## Status

Implemented and merged. This document records the implementation-stage evidence;
the construction API was subsequently promoted to stable on 2026-10-23 under
[`API_CHANGE_PROPOSAL_062`](API_CHANGE_PROPOSAL_062_DYNAMIC_CIRCUIT_PROMOTION.md).
The repository owner authorized this work on 2026-10-21 with the exact directive
`approve API-CHANGE-074-DYNAMIC-CIRCUIT-V1-IMPLEMENTATION`.

That approval includes the experimental behavior change from `RuntimeError` to
`ValidationError` for a classical bit read before measurement. It does not add or
promote a public name. At that stage `DynamicCircuit` remained candidate-stable;
`run_dynamic` and backend assessment remain experimental after its later promotion.

## User-facing behavior

The supported path remains:

```python
import flagquantum as fq
from flagquantum.dynamic import DynamicCircuit

circuit = DynamicCircuit(2)
circuit.h(0)
circuit.measure(0, classical_bit=0)
circuit.conditional("x", 1, classical_bit=0)

result = fq.experimental.dynamic.run_dynamic(circuit, shots=128, seed=7)
```

A condition that reads a bit before an earlier measurement defines it now raises
`flagquantum.errors.ValidationError` during construction. A dynamic program
loaded directly from IR is checked again before trajectory execution. Calling the
distributed statevector path with any of the four classical-control metadata keys
raises `CapabilityError` with
`dynamic_conditional_execution_unsupported` and points to `run_dynamic`.

The executable user example is
`examples/local/dynamic_validation.py`.

## Implementation inventory

- `flagquantum/core/classical_control.py` owns the four-key internal contract,
  key-presence detection, classical width, metadata validation, and ordered
  measurement-to-condition def-use validation.
- `DynamicCircuit._append_dynamic` validates the complete instruction prefix
  before mutating the builder.
- `run_dynamic` repeats validation for directly reconstructed IR.
- Both shot-time defensive checks use `ValidationError`.
- `run_distributed` refuses classical control before compilation or matrix lookup.
- The public `CircuitIR` dataclasses, `IR_VERSION`, root exports, and dynamic
  public signatures are unchanged.

The proposal draft used `n_wires` for the new internal validator. The live
repository's completed qubit-vocabulary gate rejects every newly introduced
non-private wire-named parameter, so implementation uses `n_qubits`. This is an
internal keyword with no released caller and keeps the change aligned with the
current repository contract.

## Contract preservation

The three frozen files match `origin/main` byte for byte. The command

```console
$ git hash-object docs/public_api_v1.json contracts/public-api-v1-candidate.json contracts/dynamic-circuit-v1-candidate.json
7392e144e8e8e421ae4af3665b3d06c124a26c93
2bcb1514cbf3e9748309629403253a58f8a26ac6
31903f94b987fe608322987248d3b2653c7cf29e
```

produced the same three hashes when each file was read from `origin/main`.
`IR_VERSION` remains `"1.0"`. The conformance test pins these field lists:

```text
CircuitIR: ['n_wires', 'instructions', 'version', 'dtype', 'shape', 'observables', 'measurements', 'metadata']
Instruction: ['name', 'wires', 'params', 'matrix', 'metadata']
MeasurementNode: ['kind', 'wires', 'shots', 'metadata']
ObservableNode: ['name', 'wires', 'coefficient', 'metadata']
```

The candidate-signature checker is included in the passing regression command
below. The approval is recorded here, rather than by changing the frozen dynamic
candidate contract, because the proposal separately requires that contract to
remain byte-identical.

## Def-use and distributed refusal evidence

`tests/unit/test_classical_control_contract.py` checks all of the following:

- highest measured bit `0`, condition on bit `1`:
  `ValidationError: classical bit 1 was read before measurement`;
- condition on bit `0` before its later measurement:
  `ValidationError: classical bit 0 was read before measurement`;
- wire `0` measured into bit `1`, condition on bit `0`:
  `ValidationError: classical bit 0 was read before measurement`;
- malformed, negative, non-binary, wrong-type, non-canonical, duplicate, and
  mixed condition metadata;
- key-presence classification for `is_dynamic`, `classical_bit`, `conditions`,
  and `condition_clauses`, including false or empty values;
- refusal of each of those keys by `run_distributed` before `compile_program` is
  called.

The exact distributed exception is:

```text
CapabilityError: dynamic_conditional_execution_unsupported: distributed execution does not carry classical feedback; use fq.experimental.dynamic.run_dynamic(..., shots=...)
```

The previous unconditioned-state or `KeyError: 'measure'` outcomes are therefore
unreachable through `run_distributed`. No numeric tolerance was introduced by
this change.

## Verification

The complete focused dynamic, exact density-matrix, candidate-contract, and
hybrid-compiler command produced:

```text
collected 161 items
157 passed, 4 skipped, 3 warnings in 6.69s
```

The four skips are optional Qiskit tests. The warnings are the existing
`n_wires` deprecation warning in dynamic conformance. The proposal's two hybrid
compiler files account for 50 of the passing tests. The documentation
source-of-truth checker also completed with exit code zero.

The repository language checker produced:

```text
Checked 3339 paths; 0 files violate the language policy.
Binary images and non-UTF-8 files require separate visual review.
```

Backend assessment produced these exact blocker tuples for the measured valid
program:

```text
unsupported simulator: ('backend_does_not_declare_dynamic_circuit_support',)
Braket IQM profile: ()
```

## Scope that remains open

These are explicit gaps, not implied implementation:

- `assess_dynamic_backend(circuit, backend)` has no request or rank parameter.
  Adding one would change the experimental public signature and was not approved
  by this proposal. Multi-rank execution is protected by the runtime
  `CapabilityError`, but no multi-rank blocker tuple can be measured through the
  current assessment signature.
- Proposal 062 names a static batched baseline but records neither a benchmark
  command nor an accepted numeric threshold. This change does not touch the
  static execution path, but no performance pass is claimed without that missing
  authority.
- The roadmap conflict about moving `run_dynamic` out of `experimental` remains
  unresolved. This implementation follows proposals 062 and 074 and leaves it
  experimental.
- Phase 4 is now approved by API change 072 and its internal levels live under
  `flagquantum/core/ir/`, not the obsolete `flagquantum/_compiler/` path measured
  by the original proposal. Those levels still refuse dynamic public IR because
  their measurement def-use representation is not implemented. This change is a
  validated bridge over existing metadata, not a substitute Program IR.
- `CAPABILITY_NAMES` has no conditional-execution fact and therefore no runtime
  protocol handler or conformance test for one. The proposal's refusal to add an
  unconsumed vocabulary name stands.
- No loop, `while`, or repeat-until-success construct is added. Hybrid capture
  still raises `while is unsupported`.

These gaps prevent a stability promotion, distributed scalability claim, or
hardware parity claim. They do not weaken the local dynamic V1 safety boundary
implemented here.
