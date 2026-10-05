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
aliases during the same migration window, as does `OpenQASMImport.n_wires`. `discover_backends(n_wires=...)`
likewise delegates to `list_devices(n_qubits=...)` with a deprecation warning.
The affected candidate signatures
are updated for this explicitly requested migration; the historical baseline
and checker remain unchanged.

## The executor surface

The distributed executors under `flagquantum.runtime.executors` are the fourth
slice. `fq.runtime` is not an attribute of the `fq` package, so no name in this
slice is reachable from `fq.*` and none of them keeps a forwarder: every rename
below is a hard rename in the same release.

| Before | After |
|---|---|
| `runtime.executors.mps.*` and `runtime.executors.jax.*`: `n_wires`, `wire`, `wires`, `left_wire`, `wire_shards`, `adjoint_wires`, `observable_wires` | the `qubit` spellings |
| `runtime.executors.statevector.*`: `wire_layout`, `persistent_wire_layout`, `preferred_local_wires`, `local_physical_wire`, `sharded_physical_wire`, `logical_wire`, `wires` | `qubit_layout`, `persistent_qubit_layout`, `preferred_local_qubits`, `local_physical_qubit`, `sharded_physical_qubit`, `logical_qubit`, `qubits` |
| `runtime.executors.statevector.layout.StatevectorLayoutSwap`: `local_logical_wire`, `local_physical_wire`, `sharded_logical_wire`, `sharded_physical_wire` | the four `…_qubit` spellings |
| `runtime.distributed.{scale_profile,training_profile}`: `wire_layout`, `n_wires` | `qubit_layout`; `n_wires` is **kept** on the two scale cases, whose `as_dict` payload carries it as a key |
| `runtime.executors.mps.records.MPSReverseTapeRecord`: `wires` | **kept** as a serialized field; the `build_mps_reverse_tape_record` parameter is `qubits` |
| `runtime.executors.jax.common.rank_for_wire`, `validate_observable_wires`; `runtime.executors.statevector.forward.communication_aware_wire_layout`; `runtime.executors.statevector.program_cache.remap_instruction_wires` | `rank_for_qubit`, `validate_observable_qubits`, `communication_aware_qubit_layout`, `remap_instruction_qubits` |

Payload keys do not move with the attribute that fills them. The executors' own
`summary()` dictionaries still carry `"n_wires"`, `"wire_layout"`, and
`"logical_to_physical_wires"`, and `execute_torch_distributed_statevector_reverse`
still reports `"observable_wires"` in its `summary()`. Each is a key in a payload
that leaves the process, so it moves only with that payload's schema.

`flagquantum.runtime.executors` is sparsely sliced: several of its modules —
`jax/statevector/kernels.py`, `statevector/reverse_adjoint_kernels.py`,
`statevector/forward_rzz_segment.py`, `mps/canonicalization.py`,
`jax/mps/shards.py` — declare nothing this slice owns and yet read a plan or a
shard this slice renamed. They were caught up in the same change, and the only
check that sees them is a test run, because the reference reaches them through an
`Any`. A rename whose reach is wider than its file list is not verified by
`mypy --strict` alone.

## The algorithms surface

`flagquantum.algorithms` is the fifth slice. `fq.algorithms` is not an attribute
of the `fq` package and no algorithm name appears in `fq.__all__`, so no name in
this slice is reachable from `fq.*` and none of them keeps a forwarder: every
rename below is a hard rename in the same release.

| Before | After |
|---|---|
| every `algorithms` parameter and field named `wire`, `wires`, `n_wires` | the `qubit` spellings |
| `n_counting_wires`, `n_evaluation_wires`, `n_support_wires`, `n_item_wires`, `n_a_wires`, `n_b_wires`, `n_embedding_wires` | `n_counting_qubits`, `n_evaluation_qubits`, `n_support_qubits`, `n_item_qubits`, `n_a_qubits`, `n_b_qubits`, `n_embedding_qubits` |
| `counting_wires`, `evaluation_wires`, `declared_wires`, `data_wires`, `left_wires`, `right_wires`, `purification_wires` | the `…_qubits` spellings |
| `HamiltonianTerm.max_wire` | `HamiltonianTerm.max_qubit` |
| the private helpers `_as_wire_tuple`, `_resolve_wires`, `_zero_wires`, `_constant_wire`, `wire_tuple` | `_as_qubit_tuple`, `_resolve_qubits`, `_zero_qubits`, `_constant_qubit`, `qubit_tuple` |
| the private constants `_GROVER_WIRE_LIMIT`, `_PHASE_ORACLE_WIRE_LIMIT`, `_CENTROID_WIRE_LIMIT`, `_MAX_SUPPORT_WIRES`, `_MAX_ITEM_WIRES`, `_MAX_DATA_WIRES` | the `QUBIT` spellings |

This slice owns no payload. Its `persisted_attribute_count` is zero, so it adds
no `[attribute_exclusions]` row and no `definition_retirement` name, and the
private bucket the gate re-measures drops by the 41 private `wire` parameters the
same rename reaches.

`flagquantum/algorithms` is sparsely sliced too, and the files it does not own
fall into three groups. Seven modules declare nothing any slice owns —
`error_mitigation.py`, `feature_selection.py`, `kmedians.py`, `optimization.py`,
`quantum_kernel.py`, `qubo.py`, `spsa.py` — and `qubo.py` reads
`HamiltonianTerm.wires` and `Hamiltonian.n_wires`, so it was caught up while its
own `wire`-named locals and prose were left for whoever owns them. Two files
outside the package read the same two declarations and were caught up with it:
`deployment/cloud.py` and `simulation/mps/tebd.py`. Both sit behind a local
`n_wires` of their own, which is why a per-file replacement of the spelling would
have renamed a name this slice does not own.

`flagquantum/algorithms/primitives/qft.py` reads `step.wire` on `_Hadamard`, a
private class. A private name is not on the ledger and no gate counts it, but it
is the same attribute and moves with the rename.

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
surface that is `fq.Circuit.n_qubits`, `fq.MeasurementResult.qubits`, and
`fq.OutputRequest.qubits`; everything else is renamed in place in the same release.

## Spellings the census cannot see

The scanner reads names — parameters, attributes, definitions — and reports
string literals without judging them. Further user-visible spellings are
reachable exactly the way a parameter is, and none is on a ledger. They are
listed here so that "the ledger is clean" is not read as "no user-visible `qubit`
is left":

| Spelling | Where a user meets it | Disposition |
|---|---|---|
| `qubit_options`, `show_qubit_labels`, `active_qubit_notches` | keyword arguments to `Circuit.draw(**kwargs)` and `draw_mpl(**kwargs)`, which forward to the drawers instead of declaring a parameter | **migrated**: the canonical spellings are `qubit_options`, `show_qubit_labels`, and `active_qubit_notches`; the old three are translated by `flagquantum.drawer.mpl_drawer.resolve_legacy_options` and warn, exactly as a parameter alias does |
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
