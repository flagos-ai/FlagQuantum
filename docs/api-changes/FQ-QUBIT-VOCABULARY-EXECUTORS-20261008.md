# WQ-4: the runtime executor slice of the qubit vocabulary migration

One document per approved change to the Stable Core public API. This one records
the fourth slice of
[Quit calling them wires](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md),
[The other two doors](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md), and
[the runtime slice](FQ-QUBIT-VOCABULARY-RUNTIME-20261007.md) as it lands.

It proposes no new capability, no export, no default, and no schema. `IR_VERSION`
stays `"1.0"`. It renames names on surfaces the three predecessor documents
already decided and invents no pair of its own. Slice membership is not decided
here either: it is declared by `[[slices]] files` in
`contracts/qubit-vocabulary-contract.toml`, and the gate recomputes each slice's
counts from the frozen baseline on every run.

This is the first slice with **no user-facing alias at all**. Its whole surface
lives under `flagquantum.runtime`, which is not an attribute of the `fq` package
(`hasattr(fq, "runtime")` is `False`), so a forwarder would have no legal caller
outside the package and would be permanent weight against engineering decision
principle 1 (no permanent compatibility debt). Every rename below is a hard
rename in the same commit.

## Decision and authorization

The authorization is the one the user gave for the program: *"the user-facing
surface of FlagQuantum says `qubit`, never `wire`"*, restated in the
[integral document](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md). WQ-4 invents no
new scope: the contract records this document as `executor_authorization` under
`[verification]`, alongside the entries the three predecessor slices left, and
the gate checks that the path exists on every run.

WQ-4 owns **32 files** and is measured at **49 canonical parameters**, **41
ledgered attribute names**, **4 definition names**, and **4 payload-key
exclusions**. The 41 attributes split by declaration kind into 37 `field`, 3
`member`, and 1 `instance`.

The integral document scopes WQ-4 as `runtime/executors/`. The slice also carries
`runtime/distributed/scale_profile.py`, `runtime/distributed/training_profile.py`,
and `runtime/trajectories/result.py`, which name the profiles and the Monte Carlo
record the executors consume. They were folded into this slice rather than left
unassigned, because the gate requires every baseline site to belong to exactly
one slice; two of the three contribute exclusions only.

## What the slice renames

### 49 parameters

The parameters fall into five groups.

| Group | Before | After |
|---|---|---|
| Distributed MPS state and shards (`mps/distributed_state.py`, `mps/state.py`, `mps/reverse_*.py`, `mps/records.py`, `mps/compiled_training.py`, `mps/reverse_planning.py`) | `n_wires`, `wire`, `left_wire`, `wires`, `adjoint_wires` | `n_qubits`, `qubit`, `left_qubit`, `qubits`, `adjoint_qubits` |
| Distributed statevector layout, forward, and gather (`statevector/layout.py`, `statevector/forward.py`, `statevector/forward_executor.py`, `statevector/local_execution.py`) | `n_wires`, `wires`, `wire_layout`, `persistent_wire_layout`, `preferred_local_wires`, `local_physical_wire`, `sharded_physical_wire`, `logical_wire` | `n_qubits`, `qubits`, `qubit_layout`, `persistent_qubit_layout`, `preferred_local_qubits`, `local_physical_qubit`, `sharded_physical_qubit`, `logical_qubit` |
| JAX executors (`jax/common.py`, `jax/kernel.py`, `jax/planning_core.py`, `jax/{mps,statevector,tensor_network}/**`) | `n_wires`, `wire`, `wires`, `wire_shards`, `observable_wires` | `n_qubits`, `qubit`, `qubits`, `qubit_shards`, `observable_qubits` |
| Distributed training profiles (`distributed/scale_profile.py`, `distributed/training_profile.py`) | `wire_layout` | `qubit_layout` |
| Monte Carlo result record (`trajectories/result.py`) | `wires` | `qubits` |

`observable_wires` appears in the JAX and MPS executors as the tuple a caller
asks an expectation over; it is the same concept `RuntimePolicy.observable_wires`
and `fq.expectation(..., wires=…)` already carry, and it moves with them.

### 41 attribute names

| Owner | Before | After | Kind |
|---|---|---|---|
| `statevector.models.DistributedStatevectorPlan` | `n_wires`, `sharded_wires` | `n_qubits`, `sharded_qubits` | field |
| `statevector.models.StatevectorGatePlan` | `wires`, `sharded_wires_touched` | `qubits`, `sharded_qubits_touched` | field |
| `statevector.models.StatevectorExecutionSegment`, `StatevectorFusionBlock` | `wires` | `qubits` | field |
| `statevector.forward.TorchDistributedStatevectorResult` | `wire_layout`, `logical_to_physical_wires` | `qubit_layout`, `logical_to_physical_qubits` | field |
| `statevector.gather.DistributedStatevectorGatherResult` | `wire_layout`, `logical_to_physical_wires` | `qubit_layout`, `logical_to_physical_qubits` | field |
| `statevector.reverse.TorchDistributedStatevectorGradientResult` | `observable_wires`, `logical_to_physical_wires` | `observable_qubits`, `logical_to_physical_qubits` | field |
| `statevector.layout.StatevectorLayoutSwap` | `local_logical_wire`, `local_physical_wire`, `sharded_logical_wire`, `sharded_physical_wire` | the four `…_qubit` spellings | field |
| `statevector.layout.PersistentStatevectorLayoutPlan` | `n_wires` | `n_qubits` | field |
| `mps.distributed_state.DistributedBoundarySync` | `left_wire`, `right_wire` | `left_qubit`, `right_qubit` | field |
| `mps.distributed_state.DistributedShardPlan` | `wires` | `qubits` | field |
| `mps.distributed_state.DistributedMPSState` | `n_wires` | `n_qubits` | member |
| `mps.distributed_state.ShardedMPSState` | `n_wires` | `n_qubits` | instance |
| `mps.state.RankOwnedMPSState` | `n_wires` | `n_qubits` | field |
| `mps.state.MPSPartition` | `wires` | `qubits` | field |
| `mps.compiled_training.MPSTrainingStep` | `observable_wires` | `observable_qubits` | field |
| `mps.records` / `jax.mps.training_records` | `JAXMPSRankShardState::wires`, `JAXShardedMPSParameterGateAssignment::wires`, `JAXShardedMPSParameterFlowPlan::n_wires`, `JAXShardedMPSTrainingPlan::n_wires` | the `qubit` spellings | field |
| `jax.kernel.JAXQuantumKernel` | `n_wires`, `observable_wires` | `n_qubits`, `observable_qubits` | field |
| `jax.planning_core.JAXDistributedQuantumPlan` | `n_wires` | `n_qubits` | field |
| `jax.{mps,statevector,tensor_network}` result records | `n_wires`, `observable_wires` | the `qubit` spellings | field |
| `tensor_network.state.DistributedTensorNetworkState` | `n_wires` | `n_qubits` | member |
| `tensor_network.state.DistributedTensorNetworkExpectation` | `observable_wires` | `observable_qubits` | field |
| `runtime.trajectories.result.MPSMonteCarloResult` | `n_wires` | `n_qubits` | member |

One site changes declaration kind: `ShardedMPSState::n_wires` was an attribute
assigned in `__init__`, so the declaration-kind ledger records it as `instance`.
The kind is the scanner's evidence for *how* the name reaches the screen, not a
property of the rename, and the gate asserts the kind distribution per slice so a
site cannot be moved to a quieter door.

### 4 definition names

| Before | After |
|---|---|
| `runtime.executors.jax.common.rank_for_wire` | `rank_for_qubit` |
| `runtime.executors.jax.common.validate_observable_wires` | `validate_observable_qubits` |
| `runtime.executors.statevector.forward.communication_aware_wire_layout` | `communication_aware_qubit_layout` |
| `runtime.executors.statevector.program_cache.remap_instruction_wires` | `remap_instruction_qubits` |

## Payload keys this slice keeps

Four WQ-4 rows are excluded from the rename because their spelling is a key of a
payload that leaves the process. Each keeps its witness, evidence, reason, and
removal condition in `[attribute_exclusions]`:

| Key | Owner | Witness | Evidence |
|---|---|---|---|
| `n_wires`, `persistent_wire_layout` | `distributed.scale_profile.FlagOSStatevectorScaleCase` | its own `as_dict` | `serialization_method` |
| `n_wires` | `distributed.training_profile.FlagOSTrainingCase` | its own `as_dict` | `serialization_method` |
| `wires` | `mps.records.MPSReverseTapeRecord` | its own payload builder | `serialization_method` |

The conversion is deliberately partial: the MPS reverse tape record keeps a
`wires` field while `build_mps_reverse_tape_record(wires=…)` becomes
`build_mps_reverse_tape_record(qubits=…)`. A *parameter* is a name a caller
writes; a *field* is a key that is serialized. Renaming the parameter and keeping
the field is the same split WQ-3 made for `ExecutionPlan`: the in-process reader
sees the domain noun, the payload keeps the pinned key.

The same split holds inside a single function: `"n_wires"` stays a dictionary key
in the executors' `summary()` payloads while the attribute that fills it says
`n_qubits`, so

```python
{"n_wires": self.n_qubits, "sharded_wires": tuple(self.sharded_qubits)}
```

is the intended shape and not an oversight. The audit is
`tools/census_wire_vocabulary.py`, which reports a *literal* separately from a
*name*; the gate forbids the reverse direction, so an exclusion cannot be parked
once its payload schema does move.

## The catch-up the slice owes

A slice renames a declaration; nothing renames its readers. Three classes of
reader live outside the slice's file list and each had to be caught up, in a
separate commit, with the test suite as the evidence:

1. **Test consumers.** `tests/unit/test_statevector_forward.py`,
   `test_statevector_full_state_gather.py`, `test_statevector_persistent_layout.py`,
   `test_statevector_reverse.py`, `test_mps_initial_tensor_shapes.py`,
   `test_mps_site_broadcast.py`, `test_mps_training.py`,
   `tests/distributed/statevector_persistent_forward_executor.py`, and
   `tests/distributed/statevector_layout_swap_executor.py` read the renamed
   attributes and pass the renamed keywords.
2. **Unsliced executors.** `flagquantum/runtime/executors/**` is sparsely sliced:
   `jax/statevector/kernels.py`, `statevector/reverse_adjoint_kernels.py`,
   `statevector/forward_rzz_segment.py`, `mps/canonicalization.py`, and
   `jax/mps/shards.py` declare nothing WQ-4 owns, yet read a plan or a shard that
   WQ-4 renamed. `mypy --strict` cannot see these because the reference arrives
   through an `Any`; only a whole-directory `pytest` run finds them.
3. **A module line ceiling.** `architecture.toml` sets
   `default_module_line_ceiling = 1250` with no exception in the tree.
   `statevector/reverse_adjoint_sweep.py` sat exactly at 1250 and the longer
   spelling pushed it to 1253. The name is not the problem, the duplication was:
   two copies of
   `plan.n_qubits - wire - 1 - len(plan.sharded_qubits)` became one `rank_bit_count`
   local, and the file is now 1249 lines. No ceiling was raised.

The lesson the slice writes down: a rename whose reach is wider than its file
list is only verified by running the tests that read it. WQ-2's over-reach was
caught by `mypy` after the push and WQ-4's by `pytest` after the rename; both were
found by a check the slice had to be told to run.

## Compatibility

| Surface | Compatibility |
|---|---|
| `runtime.executors.*` parameters and attributes | hard rename; a caller passing `n_wires=` or reading `.wire_layout` gets `TypeError` / `AttributeError`, which is the honest failure for a name with no compatibility promise |
| `runtime.distributed.*` training and scale profiles | payload keys unchanged; the dataclass fields that build them keep `n_wires` where the row is excluded |
| plan and result payloads | their keys are unchanged; `logical_to_physical_wires` and `observable_wires` stay keys in `summary()` |
| `fq.Circuit`, `fq.CircuitIR`, `fq.MeasurementResult`, `fq.*` helpers | untouched by this slice |
| `docs/operator_manifest.json` | untouched |

Nothing here changes a numerical result; the rename is textual on both sides of
every bind.

## Acceptance Tests

1. `python tools/check_qubit_vocabulary.py` exits 0 and reports
   `79 of 341 baseline sites retired, 11 kept as deprecated aliases` with WQ-4 at
   `retired 49 remaining 0`; `68 of 122 attribute sites retired, 3 kept as
   deprecated aliases` with WQ-4 at `retired 41 remaining 0`; and
   `4 of 10 definition names retired` with WQ-4 at `retired 4 remaining 0`.
2. `python tools/check_architecture.py` reports `architecture boundaries passed`,
   and `flagquantum/runtime/executors/statevector/reverse_adjoint_sweep.py` is
   1249 lines, below the unraised 1250 ceiling.
3. `python -m mypy --strict --python-version 3.12 --ignore-missing-imports
   flagquantum` reports `Success: no issues found in 624 source files`.
4. `python -m pytest tests/unit/test_census_wire_vocabulary.py
   tests/unit/test_qubit_vocabulary_contract.py -q` passes with the live counts
   `262` canonical parameters, `11` aliases, `372` internal, and `54` ledgered /
   `22` excluded attributes.
5. `python -m pytest tests -q` passes for every test this slice touched;
   `tests/unit/test_statevector_forward.py`,
   `test_statevector_full_state_gather.py`, `test_statevector_persistent_layout.py`,
   `test_statevector_reverse.py`, and `test_statevector_training.py` are run
   explicitly, because they read the renamed plan and result objects.
6. `python tools/public_api_snapshot.py` reports `public API migration baseline
   passed`, and neither `docs/public_api_v1.json` nor
   `contracts/public-api-v0.2-baseline.json` is edited.
7. `python tools/validate_required_checks.py` still passes: this change adds tests
   and contract rows, never a CI job.
8. `python tools/check_team_scope.py --require-classified --base d529e9753`
   classifies every changed path, and `contracts/qubit-vocabulary-contract.toml`
   declares WQ-4's owner as `simulation + runtime`: 29 of the slice's 32 files sit
   under `flagquantum/runtime/executors/**`, which `team-ownership.toml` assigns
   to `simulation`, while `flagquantum/runtime/distributed/**` and
   `flagquantum/runtime/trajectories/result.py` belong to `runtime`.

Status: landed. The rename is WQ-4's and no other slice may retire a site this
ledger assigns to WQ-4.
