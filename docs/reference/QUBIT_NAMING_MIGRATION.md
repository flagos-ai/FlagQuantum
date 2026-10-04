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

## The simulation surface

`flagquantum.simulation` is the sixth slice: its CPU cores, the code the executors
call. `fq.simulation` is not an attribute of the `fq` package and none of the 104
public names this slice's files declare is reachable as `fq.<name>`, so no name
here keeps a forwarder and every rename below is a hard rename in the same
release.

| Before | After |
|---|---|
| every `simulation` parameter named `wire`, `wires`, `n_wires` | the `qubit` spellings |
| `first_wires`, `second_wires`, `rzz_first_wires`, `rzz_second_wires`, `second_wire`, `terminal_wires`, `global_wires`, `local_wires`, `target_wires`, `gate_wires`, `raw_wires`, `occupied_wires`, `ordered_wires`, `relabeled_wires`, `normalized_wires`, `concatenated_wires`, `moved_wires`, `built_wires`, `other_wires`, `swap_wires`, `max_wires` | the `…_qubits` / `…_qubit` spellings |
| `CollapseOperator.wires` | `CollapseOperator.qubits` |
| the definitions `native_cpu_rotation_tile_wires`, `native_cpu_forward_rotation_tile_wires`, `infer_n_wires_from_dense_state` | `native_cpu_rotation_tile_qubits`, `native_cpu_forward_rotation_tile_qubits`, `infer_n_qubits_from_dense_state` |
| the private names `_validate_wires`, `_step_wire_groups`, `_swap_wires`, `_apply_wire_permutation_gather`, `_rotation_segment_tile_wires`, `component_by_wire`, `by_wire`, `invalid_n_wires` | the `qubit` spellings |

Two spellings in the slice's own files are **kept**, and both are neighbours'
names rather than this slice's:

* `EvolutionPlan.n_wires` is written into `EvolutionPlan.to_dict()`, read back by
  `flagquantum/lindblad/_plan.py`, and is the slice's only
  `[attribute_exclusions]` row. The field keeps the old spelling and gains a
  `n_qubits` property, the same shape the IR keys use.
* `_StatevectorFusedGateStep.wires`, `_StatevectorDenseRegion.wires`, and the
  other `Step`/`Region` variants are declared in
  `flagquantum/simulation/statevector/program.py`, a module no slice owns. This
  slice reads them and does not rename them; they move with whatever slice owns
  `program.py`. For the same reason
  `flagquantum/simulation/statevector/wire_permutation.py` keeps its module name:
  a file name is not a name on the function surface, and renaming it would move an
  import path across two unowned modules.

`flagquantum/simulation` is sparsely sliced — the contract names 17 of its modules
— and this slice's public functions are called from far outside it. Fourteen
modules were caught up: the nine `runtime/executors/statevector` files that drive
the renamed kernels, `simulation/statevector/{local,operations}.py`,
`lindblad/_plan.py`, `qec/sampling.py`, and `algorithms/core.py`, which is WQ-5's
file and only has the imported name changed. The discipline was to repair exactly
the call sites `mypy --strict` names and nothing else: `operations.py` and
`local.py` keep every wire-named local and every `request.wires` read of their own.

## The accelerator surface

`flagquantum.simulation.jax` and `flagquantum.simulation.tensor_network` are the
seventh slice: the numerical cores behind the `jax` and `tensor_network` runtime
modes. Neither is an attribute of the `fq` package, so no name here keeps a
forwarder and every rename below is a hard rename in the same release.

| Before | After |
|---|---|
| every parameter named `wire`, `wires`, `n_wires` in `jax/{mps,statevector,tensor_network}/kernels.py` | the `qubit` spellings |
| `local_wires`, `local_gate_wires`, `n_local_wires`, `sharded_wires`, `touched_sharded_wires`, `left_wire`, `left_wires`, `dense_observable_wires`, `observable_wires` | the `…_qubits` / `…_qubit` spellings |
| `DenseIslandPlan.n_wires`, `DenseIslandPlan.island_for_wire`, `MPSBondProfile.n_wires`, `MPSConfig.dense_observable_wires`, `CompiledMPSOperation.wires` | the `qubit` spellings |
| `MPSState.n_wires`, `StaticMPSProgram.n_wires`, `MPSPlanningMixin.n_wires`, `SiteKernelStats.triton_wire_probability_calls`, `SiteKernelStats.wire_probability_fallback_calls` | the `qubit` spellings |
| `TensorNetworkState.n_wires`, `TensorNetworkState.dense_observable_wires`, and the `n_wires` / `wires` / `observable_wires` fields of `CompiledTNProgram`, `CompiledTNObservableProgram`, `TensorNetworkContractionPlan`, `TensorNetworkExpectationPlan` | the `qubit` spellings |
| the definition `jax_basis_indices_for_wires` | `jax_basis_indices_for_qubits` |
| the private names `_validate_wire`, `_wire_probabilities`, `_local_z_diagonal`, `_split_pair`, `_validate_observable_wires`, `_local_layers`, `_product_state`, `_validate_inputs`, `_normalize_bitstring`, `_initial_state_tensors` | the `qubit` spellings |

One spelling in the slice's own files is **kept**, and it is a frozen payload key
rather than this slice's name:

* `TEBDResult.n_wires` is written into `TEBDResult.to_dict()` and read back by
  `TEBDResult.from_dict()`, and is the slice's only `[attribute_exclusions]` row.
  The field keeps the old spelling and gains a `n_qubits` property, the same shape
  `EvolutionPlan.n_wires` uses.

Three more spellings are kept because the name is a *catalog identity* or a
neighbour's payload key, not a qubit name. The module
`simulation/mps/wire_probability_dispatch.py` and its `_mps_wire_probabilities`
helper share the `wire_probability` token with the Triton kernel's `kind` string
`wire_probabilities` and the semantic id `mps.measurement.wire_probabilities.local`
— identities that are asserted by
`tests/unit/test_mps_wire_probability_catalog_dispatch.py` and recorded in
`benchmarks/results/local/mps_wire_probability_dispatch_a800.json`. The slice's own
`SiteKernelStats` counters are not catalog identity and do move. `Instruction.wires`
and `CircuitIR.n_wires` are declared in `flagquantum/core/ir.py` and are the
`IR_VERSION = "1.0"` payload; the MPS path reads them and passes
`CompiledInstruction(wires=…)` for the same reason.

Bare-name string literals that name a qubit index or collection move with the
rename — the slice's own summaries and in-package interfaces — but only where the
module that *writes* the key is one the rename reached. Two summaries keep
`"observable_wires"` because they are written by
`runtime/executors/statevector/reverse.py` and `runtime/executors/jax/kernel.py`,
which are outside this slice's reach, and the tests that read them are unchanged.

`flagquantum.simulation` is sparsely sliced and this slice's public functions are
called from outside it, so twenty-two executor and neighbour modules were caught
up under the same discipline as WQ-6 — repair the call sites `mypy --strict` names
and nothing else. Ten of them also lose a private wire-named parameter, which is
why `boundary.measured_private` drops from 308 to 275. Two of those helpers,
`_initialize_jax_mps_rank_tensors` and `_jax_mps_boundary_protocol`, had their
declarations renamed in `jax/mps/{shards,canonicalization}.py` and their call sites
had to move with them.

## The compiler, tooling, and measurement surface

The compiler, benchmarking, kernel, noise, remote-provider, deployment and twin
modules are the last slice, and the widest: thirty-one files, fifty-five public
parameters, seventeen public attribute names, two module-level definitions, and
forty-nine private parameters. `boundary.measured_private` falls from 275 to 226,
which is the largest single drop of the migration and the reason this slice could
not be split further without leaving a file half-renamed.

The user-visible names it settles are the ones a compiler-facing user types:

| Old | New |
|---|---|
| `fq.CloudBackendProfile(n_wires=...)` | `fq.CloudBackendProfile(n_qubits=...)` |
| `discover_backends(n_wires=...)` | `discover_backends(n_qubits=...)` |
| `CouplingMap.line(n_wires=...)`, `CouplingMap.ring(n_wires=...)` | `... (n_qubits=...)` |
| `DirectedCouplingMap(n_wires, edges)` | `DirectedCouplingMap(n_qubits, edges)` |
| `NoiseModel.add(gate, channel, wires=...)` | `... qubits=...` |
| `NoiseModel.apply_readout_probabilities(..., n_wires=...)` | `... n_qubits=...` |
| `KrausChannel.n_wires` | `KrausChannel.n_qubits` |
| `Layout.n_wires` | `Layout.n_qubits` |
| `PhysicalCircuitPlan.logical_wire_count` / `.physical_wires` | `.logical_qubit_count` / `.physical_qubits` |
| `TwinPrediction.n_wires` | `TwinPrediction.n_qubits` |
| `synthesize_one_qubit(..., wire=...)`, `synthesize_two_qubit(..., wires=...)` | `... qubit=...`, `... qubits=...` |
| `expectation_z_from_counts(..., wires=...)` | `... qubits=...` |
| `validate_deployment_routing_plan(..., n_wires=...)` | `... n_qubits=...` |

Two of these are not mistakes a caller can catch at import time; both are
`TypeError` at the call, which is why the slice also moved the keywords its own
twenty-two followers pass. `flagquantum/runtime/measurements.py` is the one that
matters most: it is the sole caller of
`NoiseModel.apply_readout_probabilities`, it is outside this slice's file set, and
`mypy --strict` did not report it because the two sites pass through an
`Any`-typed `noise_model`. The AST walk in the slice's follower pass found it.

The compiler's *reported* refusal messages moved with the parameters, because a
`pytest.raises(match=...)` pins the text: "layout must place at least one logical
qubit", "physical qubit {p} is outside the layout", "physical qubit {p} holds no
logical qubit", "to_logical_order needs one value per physical qubit", "SWAP qubit
pair {pair} is outside a {n}-qubit layout", "routing metadata reports
logical_qubit_count as {n}, which does not fit a {m}-qubit program", "No coupling
path between qubits {a} and {b}", "Directed coupling map requires a positive qubit
count", "Coupling qubit count must be an integer", "correlated readout matrix size
must match qubits", and "routed CircuitIR retains nonlocal two-qubit
instructions". 634 string literals still carry a wire spelling; they are reported
by the census and not judged by it.

### The names this slice keeps

Five payload keys, and only these, still spell a wire. Each is a serialized field
whose reader is the IR or the vendor, not a caller:

| Key | Owner | Why it stays |
|---|---|---|
| `CircuitIR.n_wires` | `flagquantum/core/ir.py`, `IR_VERSION` | the accepted IR schema; a rename is a version bump, not a slice |
| `Instruction.wires`, `MeasurementNode.wires`, `ObservableNode.wires` | same | same |
| `TwinPrediction.n_wires` | `flagquantum/twin/prediction.py` | the persisted twin payload a saved model is read back through |
| `routing["logical_wire_count"]` | the routing metadata a route writes into `CircuitIR.metadata` | read by `final_layout`, which validates it as recorded candidate evidence |
| `n_wires` | the Quafu vendor request body | the provider's spelling, not ours |
| `FQ_CPU_SINGLE_WIRE_ELEMENTWISE` | `benchmarking/statevector_cpu_paths.py` | a kernel-switch environment variable already published to benchmarking users |

The `wire_probabilities` *catalog kind*, its semantic id
`mps.measurement.wire_probabilities.local`, the implementation id
`FQKI-TRITON-MPS-007-A` and the module name `mps_wire_probabilities` also stay:
they are catalog identity, not an API, and the private delegate
`_mps_wire_probabilities` is declared in a module no slice owns. The *implementation*
identity `fused_mps_wire_probabilities` did move, because it names a Python symbol
that `flagquantum.kernels.catalog.implementations` is imported to resolve.

The benchmark CLI keeps `--n-wires` and `--marginal-wires`. A flag is neither a
parameter nor a keyword, so no ledger counts it, and the same flag family is
emitted by modules this slice does not own; renaming it here would leave the
benchmark suite with two spellings of one switch.

## Scope of the rename

The migration covers four ledgers plus a report, all measured by
`tools/census_wire_vocabulary.py` and reconciled against
`contracts/qubit-vocabulary-contract.toml` on every run:

| Surface | Count | Disposition |
|---|---:|---|
| parameters on the public function surface | 341 | renamed; 11 are kept as deprecated forwarders and are deleted at 0.4.0 |
| public attribute and property names | 144 | 122 renamed, 22 excluded as payload keys; 5 of the 122 keep a forwarder until 0.4.0 |
| module-level public definition names | 10 | renamed |
| documented keyword arguments | 17 | 13 reworded; 4 kept, in two documents, each with a recorded reason |
| string literals | 634 | reported only, never ledgered |

An attribute is renamed whether it is a field, a property or method
(`Circuit.n_wires`), or an instance attribute assigned in a method
(`TextDrawer().wire_order`). It is excluded only when its spelling reaches a
payload, and each exclusion names the class or method that witnesses it plus the
schema event that retires it — `IR_VERSION` for the IR keys, or the owning payload
schema's own version bump. See
[the attributes authorization](../api-changes/FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md)
and [its decision record](../development/API_CHANGE_PROPOSAL_067_QUBIT_VOCABULARY_ATTRIBUTES.md).

Aliases are owed only where the name is reachable from `fq.*`. On the attribute
surface that is `fq.Circuit.n_wires`, `fq.MeasurementResult.wires`,
`fq.OutputRequest.wires`, `fq.CloudBackendProfile.n_wires` and
`fq.OpenQASMImport.n_wires`; everything else is renamed in place in the same
release. `OpenQASMImport.n_wires` is the odd one of the five: `fq.from_openqasm`
builds the object and already takes `n_qubits`, so the class is given no
constructor alias and `n_wires` remains readable only because the OpenQASM import
proposal documents it that way.

## The documentation surface

The documentation is measured as its own surface rather than folded into the
parameter one, and the reason is that the two boundaries are drawn differently. The
parameter boundary is decided by which module a name is declared in, so it is
reproducible; deciding which prose *describes the package* is not. Matching module
names in `docs/**` turns `flagquantum.compiler` into its entire subtree and yields a
number that looks precise but is decided by prefix matching. So this surface asks a
smaller, answerable question instead: **which keyword arguments do the documents
tell a reader to write?**

That is an instruction. A reader copies it into a script and expects it to run, and
after a rename a document that still shows the old spelling is the one place a
copy-paste is the discovery mechanism — nothing else in the gate can see it, because
the Python-side ledgers read signatures, and the generated pages render *from* the
code so they never quote an example by hand.

| Rule | Effect |
|---|---|
| a `name=value` keyword argument whose name contains `wire`, in any tracked Markdown file that is not a dated record | counted; it is an instruction to a reader |
| a keyword written inside a code span, as in a parenthesised `` (`wires=`) `` | counted. A reader reads a code span the same way. The matcher used to require that the name not follow a backtick, and that blind spot hid a contract row until this slice |
| `docs/api-changes/**`, `docs/development/**`, `benchmarks/results/**`, and this file | skipped; their job is to quote what the tree said on their date, and the gate must not punish them for it |
| the bare noun `wire` | not counted, and not forbidden. The documentation has to be able to say that CUDA-Q orders wire zero as the least-significant bit and to describe a `--n-wires` flag, and a rule that forbade those would be switched off rather than obeyed |
| a CLI flag, an attribute read, a comparison | not counted; a flag is neither a parameter nor a keyword, `info.wires = 1` reads a caller's object, and `n_wires == 2` is not an argument |

Two documents are exempt, for opposite reasons, and each exemption names the file,
the reason, and the whole multiset of spellings that file may show — so a fourth
example cannot appear inside either without its own record.

[The hybrid compilation plan](../roadmap/PYTHON_HYBRID_COMPILATION_EXECUTION_PLAN.md)
quotes the source language a captured program is written in.
`flagquantum/compiler/_hybrid/capture.py` rejects every keyword but `wires`, so
rewording the example would make it unparseable and the plan wrong. The plan's
`wire=` occurrences were a different case: the capture layer rejects `wire=` too, so
they were corrected to `wires=` rather than exempted.

[The API reference](../reference/API.md) documents `fq.Circuit(n_wires=2)`, which
still constructs and still warns, and which `[aliases].removal_version` says will be
deleted at 0.4.0. Rewording that occurrence would delete the compatibility promise
rather than complete the migration, so the promise keeps its spelling until it
expires.

## Spellings the census cannot see

The scanner reads names — parameters, attributes, definitions — and reports
string literals without judging them. Documented keyword arguments are a fourth
ledgered surface, listed above. Ten further user-visible spellings reach a user
and reach no ledger, because each is a value rather than a name, or a name
declared outside the boundary. They are listed here so that "the ledger is clean" is
not read as "no user-visible `wire` is left":

| Spelling | Where a user meets it | Disposition |
|---|---|---|
| `wire_options`, `show_wire_labels`, `active_wire_notches` | keyword arguments to `Circuit.draw(**kwargs)` and `draw_mpl(**kwargs)`, which forward to the drawers instead of declaring a parameter | **migrated**: the canonical spellings are `qubit_options`, `show_qubit_labels` and `active_qubit_notches`; the drawer translates the old three through `flagquantum.drawer.mpl_drawer.resolve_legacy_options` and warns, exactly as a parameter alias does |
| `n_wires`, `wires` on a *legacy* device object | the two spellings a third-party qdev reports, read by `flagquantum.drawer.ir_adapter.to_drawable_circuit` | **accepted, not published**: read at one boundary and immediately re-expressed as `n_qubits`/`qubits`, so no renderer ever meets them |
| `wires` | the keyword a captured hybrid program must use — `qp.H(wires=...)`, `qp.measure(wires=...)`, `qp.reset(wires=...)` — required by the capture layer, which rejects any other keyword | open; renaming it changes the source language, not a signature |
| `wire_start`, `wire_end`, `owned_wires` | dictionary keys returned by `runtime.planner.topology.rank_ownership` | kept; no reader in the package builds them into a qubit-named contract, and the parameter the caller passes is already `n_qubits` |
| `max_marginal_wires` | a measurement-metadata key: written by `observables` into a request and read by `runtime.measurements` out of it | kept; it crosses a request boundary, so it moves only with a request-schema version |
| `per_sharded_wire_gate` | the value of `communication_frequency` in a candidate-plan scoring payload | kept; no reader anywhere in the package, so renaming it would change evidence without a consumer to migrate |
| `n_wires` | the metric key in `ExecutionResult(metrics={"n_wires": …})` built by the backend adapters | kept; a metric key is part of a comparison payload |
| `--n-wires`, `--wires`, `--wire-layout`, `--marginal-wires`, `--capacity-wires`, `--start-wires`, `--step-wires`, `--stop-wires`, `--worker-wire`, `--max-reference-wires`, `--fq-dense-observable-wires` | command-line flags, declared by 66 `argparse` calls over 11 distinct names in 59 files across `benchmarks/`, `tools/` and `flagquantum/` | open; a flag is a typed interface, so it migrates the way keyword arguments did — publish the qubit spelling, keep the old one as a hidden deprecated alias. 51 of the 66 declarations are `--n-wires` |
| `n_wires`, `wires`, `wire`, `wire0`, `wire1`, `capacity_wires`, `dense_observable_wires` | parameters of helper functions in `benchmarks/` (141 sites in 60 files), `tools/` and `examples/` | open; `[boundary]` is the package, so the parameter ledger does not read these files at all |
| `wire`, `wires`, `n_wires` | Python docstrings and comments inside the package (285 occurrences over 131 docstrings in 59 files, plus 54 over 44 comment lines in 18 files) | open; this is what `help()` prints, so it is user-facing, but it is prose attached to a name rather than a name, so no ledger reads it |

The capture keyword is the remaining open work. Renaming it would break every
hybrid program the capture layer can read, so it is a decision for the
hybrid-language owner rather than for this migration. The four rows after it are
payload keys the census reports as literals, and they move only when the payload
that carries them is versioned.

The last two rows are the ones that were invisible until the documentation slice
went looking for stale examples, and they are the reason the next slice is a CLI
surface rather than another name ledger. A flag and a helper parameter in
`benchmarks/` are both reachable by a user, and neither is a declaration the census
reads, because `[boundary]` stops at the package. The docstring row is the same
problem one level in: `help()` prints it, but it is prose attached to a name rather
than a name.

The three tiers are therefore: **ledgered** — parameters, attributes, definitions
and documented keywords, reconciled on every run and fail-closed; **reported** — the
632 string literals, which the gate prints and never judges; and **listed** — the ten
rows above, which are measured here by hand and are the honest answer to "what is
left". A reader who wants to know whether the migration is finished should read all
three, not the first one alone.

Documented keyword arguments are no longer invisible: they are the fourth ledgered
surface, reconciled on every run. See
[the documentation surface](#the-documentation-surface) above.

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
