# Qubit keyword migration

Implementation candidate on `qubit-naming-migration`; pending API/release review.

New code uses `fq.Circuit(n_qubits=2)`, `fq.probabilities(qubits=(0,))`,
`fq.samples(qubits=(0,))`, `fq.counts(qubits=(0,))` and
`fq.RuntimePolicy(observable_qubits=(0,))`. Cloud deployment profiles use
`CloudBackendProfile(n_qubits=...)` and expose `profile.n_qubits`. Remote
providers list hardware and simulators with `list_devices(n_qubits=...)`.

Old count aliases and `wires` / `observable_wires` keywords emit
DeprecationWarning. Supplying both selection keywords is an error, including
explicit None. Positional selection and the Observable overload are unchanged.
The proposed schedule is deprecation in 0.3.x and removal in 0.4.0, subject to
release-owner review; this PR does not bump the package version.

The runtime helpers that are importable but not reachable from `fq.*` are renamed
outright, with no forwarder, because no documented `fq.*` path reaches them:

| Before | After |
|---|---|
| `runtime.planner.estimates.estimate_state_bytes(n_wires=…)` and its five siblings | `n_qubits=` |
| `runtime.planner.execution_policy.estimate_execution_state_bytes(n_wires=…)` | `n_qubits=` |
| `runtime.planner.noise_calibration.NoiseSelectorCalibration.estimate_seconds(n_wires=…)` | `n_qubits=` |
| `runtime.planner.topology.rank_ownership(n_wires=…)` | `n_qubits=` |
| `runtime.measurements.validate_measurements(requests, n_wires=…)` | `n_qubits=` |
| `runtime.measurements.execute_measurements(output, requests, n_wires=…)` | `n_qubits=` |
| `runtime.dynamic.circuit.DynamicCircuit.measure(wire=…)` / `.reset(wire=…)` | `qubit=` |
| `runtime.dynamic.circuit.DynamicCircuit.conditional(wires=…)` | `qubits=` |

`fq.MeasurementResult.wires` is a result field of an exported class, so it is the
one runtime name that keeps a forwarder: the field is `qubits`, and `wires` is a
deprecated property that warns and returns it. `fq.MeasurementResult` is also why
`execute_measurements` builds `MeasurementResult(qubits=…)` rather than a keyword
the dataclass no longer declares.

RuntimePolicy writes schema `flagquantum.runtime_policy`, version `2.0`, with
`observable_qubits`. Its reader accepts the prior unversioned payload with
`observable_wires`, including module state dictionaries. Conflicting selection
fields and unsupported versions are rejected. Older package versions do not
understand the new writer; checkpoint compatibility is backward-reading, not
forward-reading.

IR and backend-native payload fields are unchanged. The deprecated
`observable_wires` property remains available on RuntimePolicy, while
`CloudBackendProfile.n_wires` and its constructor keyword remain compatibility
aliases during the same migration window. `discover_backends(n_wires=...)`
likewise delegates to `list_devices(n_qubits=...)` with a deprecation warning.
The affected candidate signatures
are updated for this explicitly requested migration; the historical baseline
and checker remain unchanged.

## Scope of the rename

The migration covers four surfaces, all measured by
`tools/census_wire_vocabulary.py` and reconciled against
`contracts/qubit-vocabulary-contract.toml` on every run:

| Surface | Count | Disposition |
|---|---:|---|
| parameters on the public function surface | 341 | renamed; 11 are already deprecated aliases and are deleted at 0.4.0 |
| public attribute and property names | 144 | 122 renamed, 22 excluded as payload keys |
| module-level public definition names | 10 | renamed |
| string literals | 1291 | reported only, never ledgered |

An attribute is renamed whether it is a field, a property or method
(`Circuit.n_wires`), or an instance attribute assigned in a method
(`TextDrawer().wire_order`). It is excluded only when its spelling reaches a
payload, and each exclusion names the class or method that witnesses it plus the
schema event that retires it — `IR_VERSION` for the IR keys, or the owning payload
schema's own version bump. See
[the attributes authorization](../api-changes/FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md)
and [its decision record](../development/API_CHANGE_PROPOSAL_067_QUBIT_VOCABULARY_ATTRIBUTES.md).

Aliases are owed only where the name is reachable from `fq.*`. On the attribute
surface that is `fq.Circuit.n_wires`, `fq.MeasurementResult.wires`, and
`fq.OutputRequest.wires`; everything else is renamed in place in the same release.

## Spellings the census cannot see

The scanner reads names — parameters, attributes, definitions — and reports
string literals without judging them. Further user-visible spellings are
reachable exactly the way a parameter is, and none is on a ledger. They are
listed here so that "the ledger is clean" is not read as "no user-visible `wire`
is left":

| Spelling | Where a user meets it | Disposition |
|---|---|---|
| `wire_options`, `show_wire_labels`, `active_wire_notches` | keyword arguments to `Circuit.draw(**kwargs)` and `draw_mpl(**kwargs)`, which forward to the drawers instead of declaring a parameter | **migrated**: the canonical spellings are `qubit_options`, `show_qubit_labels`, and `active_qubit_notches`; the old three are translated by `flagquantum.drawer.mpl_drawer.resolve_legacy_options` and warn, exactly as a parameter alias does |
| `n_wires`, `wires` on a *legacy* device object | the two spellings a third-party qdev reports, read by `flagquantum.drawer.ir_adapter.to_drawable_circuit` | **accepted, not published**: read at one boundary and immediately re-expressed as `n_qubits`/`qubits`, so no renderer ever meets them |
| `wires` | the keyword a captured hybrid program must use — `qp.H(wires=...)`, `qp.measure(wires=...)`, `qp.reset(wires=...)` — required by the capture layer, which rejects any other keyword | open; renaming it changes the source language, not a signature |
| `wire_start`, `wire_end`, `owned_wires` | dictionary keys returned by `runtime.planner.topology.rank_ownership` | kept; no reader in the package builds them into a qubit-named contract, and the parameter the caller passes is already `n_qubits` |
| `max_marginal_wires` | a measurement-metadata key: written by `observables` into a request and read by `runtime.measurements` out of it | kept; it crosses a request boundary, so it moves only with a request-schema version |
| `per_sharded_wire_gate` | the value of `communication_frequency` in a candidate-plan scoring payload | kept; no reader anywhere in the package, so renaming it would change evidence without a consumer to migrate |
| `n_wires` | the metric key in `ExecutionResult(metrics={"n_wires": …})` built by the backend adapters | kept; a metric key is part of a comparison payload |

The capture keyword is the remaining open work. Renaming it would break every
hybrid program the capture layer can read, so it is a decision for the
hybrid-language owner rather than for this migration. The last rows are payload
keys the census reports as literals, and they move only when the payload that
carries them is versioned.

### Reading a legacy spelling is not the same as publishing one

Two of the rows above split a distinction a name-based rule cannot express, and
the drawer is where it became load-bearing:

* **A keyword this package declares or documents** (`show_wire_labels`) is a name
  we chose, so it moves — with a forwarder and a warning, because the caller who
  wrote it was writing something we published.
* **A caller's own spelling** (`wires` on a third-party device object) is not ours
  to change. Whatever a legacy object calls its width, the drawer reads it at one
  boundary and immediately re-expresses it as `n_qubits`/`qubits`. Acceptance is
  not publication: nothing this package returns, renders, or documents uses the
  old spelling, and no renderer below the boundary has to know it exists.
* **A frozen payload key** (`CircuitIR.n_wires`) stays until the payload that
  carries it is versioned, however many readers would prefer otherwise.

The second row is why `to_drawable_circuit` accepts two spellings of each of the
two things a device reports while everything below it emits one. Reading only the
published spelling saw a legacy device as having no qubits at all, and the text
renderer then called `min()` on an empty sequence. The fix belongs at the
boundary, not in the ten renderers that consume an operation entry.
