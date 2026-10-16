# WQ-7: the accelerator slice of the qubit vocabulary migration

One document per approved change to the Stable Core public API. This one records
the seventh slice of
[Quit calling them wires](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md),
[The other two doors](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md),
[the runtime slice](FQ-QUBIT-VOCABULARY-RUNTIME-20261007.md),
[the runtime executor slice](FQ-QUBIT-VOCABULARY-EXECUTORS-20261008.md),
[the algorithms slice](FQ-QUBIT-VOCABULARY-ALGORITHMS-20261009.md), and
[the simulation CPU slice](FQ-QUBIT-VOCABULARY-SIMULATION-CPU-20261009.md) as it
lands.

It proposes no new capability, no export, no default, and no schema. `IR_VERSION`
stays `"1.0"`. It renames names on surfaces the six predecessor documents already
decided and invents no pair of its own. Slice membership is not decided here
either: it is declared by `[[slices]] files` in
`contracts/qubit-vocabulary-contract.toml`, and the gate recomputes each slice's
counts from the frozen baseline on every run.

Like WQ-4, WQ-5, and WQ-6, this slice has **no user-facing alias at all**. Its
whole surface lives under `flagquantum.simulation`, and `hasattr(fq, "simulation")`
is `False`: neither `flagquantum.simulation.jax` nor
`flagquantum.simulation.tensor_network` is reached by an attribute of the package
`__init__`, and none of the public names the slice's 17 files declare is reachable
as `fq.<name>`. A forwarder would have no legal caller outside the package and
would be permanent weight against engineering decision principle 1 (no permanent
compatibility debt). Every rename below is a hard rename in the same commit.

## Decision and authorization

The authorization is the one the user gave for the program: *"the user-facing
surface of FlagQuantum says `qubit`, never `wire`"*, restated in the
[integral document](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md). WQ-7 invents no new
scope: the contract records this document as `accelerator_authorization` under
`[verification]`, alongside the six entries the predecessor slices left, and the
gate checks that the path exists on every run.

WQ-7 owns **17 files** and is measured at **78 canonical parameters**, **18
ledgered attribute names**, **1 definition name**, and **1 payload-key
exclusion**. The attributes split 13 `field`, 4 `member`, 1 `instance`.

The slice is the second to reopen code the *executors* call rather than the
executors themselves: `flagquantum/simulation/jax` and
`flagquantum/simulation/tensor_network` are the numerical cores behind the `jax`
and `tensor_network` runtime modes, so the rename leaves the slice through 18
spellings in 10 modules no slice owns. That catch-up is described below, and it is
the whole reason this slice's diff is wider than its file list.

## What the slice renames

### 78 parameters

The parameters fall into five groups, read here from the frozen ledger's 78 rows.

| Group | Modules | Renamed parameters | Sites |
|---|---|---|---:|
| JAX MPS kernels | `jax/mps/kernels.py` | `is_zz_z_chain_hamiltonian`, `jax_mps_apply_adjacent_chain_scan.spec`, `jax_mps_apply_two_adjacent`, `jax_mps_apply_two_remote`, `jax_mps_initial_open_boundary_tensors`, `jax_mps_initial_padded_stack`, `jax_mps_z_sum`, `jax_mps_z_values`, `jax_sharded_mps_z_sum`, `parse_zz_z_chain_hamiltonian` (`n_wires`, `wires`, `wire`, `left_wire`, `observable_wires`) | 11 |
| JAX statevector kernels | `jax/statevector/kernels.py` | `jax_accumulate_all_to_all_statevector_delta`, `jax_apply_local_statevector_gate`, `jax_apply_matrix_to_batched_local_state`, `jax_basis_indices_for_wires`, `jax_combine_pair_exchanged_statevector`, `jax_gate_basis_in_for_delta_and_local_input`, `jax_local_positions_for_gate_input`, `jax_rank_mask_for_touched_delta`, `jax_sharded_statevector_loss`, `jax_sharded_statevector_rank_loss` (`n_wires`, `wires`, `wire`, `local_wires`, `local_gate_wires`, `n_local_wires`, `sharded_wires`, `touched_sharded_wires`, `observable_wires`) | 27 |
| JAX tensor-network kernels | `jax/tensor_network/kernels.py` | `jax_tensor_network_expectation_product_ops`, `jax_tensor_network_hamiltonian_expectation`, `jax_tensor_network_loss_from_output`, `jax_tensor_network_nodes_from_circuit`, `jax_tensor_network_pauli_string_expectation`, `jax_tensor_network_z_sum`, `jax_tensor_network_z_values` (`n_wires`, `wires`, `observable_wires`) | 10 |
| MPS programs and state | `mps/{brickwork,dense_island,entrypoints,observables,state,static,tebd}.py` | `compiled_local_z_zz`, `DenseIslandPlan.equal_width`, `DenseIslandPlan.island_for_wire`, `DenseIslandState.apply_local`, `run_mps`, `mps_heisenberg_local_scan`, `mps_z_zz_local_scan`, `MPSState.apply_channel_trajectory`, `MPSState.apply_one`, `MPSState.apply_parametric_one`, `MPSState.apply_swap`, `MPSState.apply_two`, `MPSState.apply_two_bucket`, `MPSState.apply_two_remote`, `MPSState.expectation_z`, `MPSState.expectation_z_and_nearest_neighbor_zz`, `MPSState.expectation_z_sum`, `MPSState.from_statevector`, `MPSState.zero`, `StaticMPSProgram.apply_cx_mpo_real_imag`, `StaticMPSProgram.apply_one`, `StaticMPSProgram.apply_ry_real_imag`, `StaticMPSProgram.apply_rz_real_imag`, `StaticMPSProgram.apply_two`, `StaticMPSProgram.compile`, `run_tebd` (`n_wires`, `wires`, `wire`, `left_wire`, `left_wires`, `dense_observable_wires`) | 26 |
| Tensor-network programs | `tensor_network/{entrypoints,local,state}.py` | `run_tensor_network`, `run_local_tensor_network`, `TensorNetworkState.__init__`, `TensorNetworkState.expectation_z` (`dense_observable_wires`, `wires`) | 4 |

### 18 attribute names

| Site | Before | After | Kind |
|---|---|---|---|
| `mps/dense_island.py::DenseIslandPlan` | `island_for_wire` | `island_for_qubit` | `member` |
| `mps/dense_island.py::DenseIslandPlan` | `n_wires` | `n_qubits` | `field` |
| `mps/models.py::CompiledMPSOperation` | `wires` | `qubits` | `field` |
| `mps/models.py::MPSBondProfile` | `n_wires` | `n_qubits` | `field` |
| `mps/models.py::MPSConfig` | `dense_observable_wires` | `dense_observable_qubits` | `field` |
| `mps/planning.py::MPSPlanningMixin` | `n_wires` | `n_qubits` | `member` |
| `mps/site_kernels.py::SiteKernelStats` | `triton_wire_probability_calls` | `triton_qubit_probability_calls` | `field` |
| `mps/site_kernels.py::SiteKernelStats` | `wire_probability_fallback_calls` | `qubit_probability_fallback_calls` | `field` |
| `mps/state.py::MPSState` | `n_wires` | `n_qubits` | `member` |
| `mps/static.py::StaticMPSProgram` | `n_wires` | `n_qubits` | `field` |
| `tensor_network/models.py::CompiledTNObservableProgram` | `n_wires` | `n_qubits` | `field` |
| `tensor_network/models.py::CompiledTNObservableProgram` | `wires` | `qubits` | `field` |
| `tensor_network/models.py::CompiledTNProgram` | `n_wires` | `n_qubits` | `field` |
| `tensor_network/models.py::TensorNetworkContractionPlan` | `n_wires` | `n_qubits` | `field` |
| `tensor_network/models.py::TensorNetworkExpectationPlan` | `n_wires` | `n_qubits` | `field` |
| `tensor_network/models.py::TensorNetworkExpectationPlan` | `observable_wires` | `observable_qubits` | `field` |
| `tensor_network/state.py::TensorNetworkState` | `dense_observable_wires` | `dense_observable_qubits` | `instance` |
| `tensor_network/state.py::TensorNetworkState` | `n_wires` | `n_qubits` | `member` |

`TensorNetworkContractionPlan` and `TensorNetworkExpectationPlan` are the two
records a user builds by hand to drive the tensor-network path, and their field
order is asserted by `tests/unit/test_tensor_network_plan_surface.py`; the rename
moves names only, never order or default. `MPSState.n_wires` and
`TensorNetworkState.n_wires` are the member reads the slice's own refusal messages
and the executor family both depend on.

### 1 definition name

| Before | After |
|---|---|
| `jax/statevector/kernels.py::jax_basis_indices_for_wires` | `jax_basis_indices_for_qubits` |

It is imported by `flagquantum/runtime/executors/jax/array_conversions.py` and
`flagquantum/runtime/executors/jax/statevector/kernels.py` under the local alias
`_jax_basis_indices_for_wires`; both the import and the local alias move, because
a private alias that keeps the retired spelling is the state this migration exists
to remove.

### 33 private parameters, unledgered

The ledger counts the public function surface, so the `wire`-named parameters of
this slice's private helpers are not rows in it. They move anyway, because a slice
that renames `n_wires` in `MPSState.zero` and leaves `_validate_wire(wire=…)` alone
has renamed a word rather than a concept. Fifteen are declared by the slice's own
files (`DenseIslandState._local_z_diagonal`, `MPSState._split_pair`,
`MPSState._validate_wire`, `MPSState._wire_probabilities`, `tebd.py::_local_layers`,
`tebd.py::_validate_inputs`, `TensorNetworkState._validate_observable_wires`, and
seven more) and eighteen by the executor modules the slice is called from
(`jax/mps/{canonicalization,execution,gradient_ownership,shards}.py`,
`jax/statevector/kernels.py`, `mps/execution.py`, `mps/sampling.py`,
`mps/wire_probability_dispatch.py`). This is why `boundary.measured_private` drops
from 308 to 275: the gate re-measures the private bucket and refuses a stale
number.

## The one payload key this slice keeps

`simulation/mps/tebd.py::TEBDResult::n_wires` is written into `TEBDResult.to_dict()`
and read back by `TEBDResult.from_dict()`. The spelling therefore reaches a
serialized schema and moves only when that schema is versioned. It is the slice's
single row in `[attribute_exclusions]`, and the gate checks that the row names a
witness class, the evidence, and the removal condition.

The field therefore stays `n_wires` and the constructor keeps passing it by the old
keyword:

```python
@dataclass(frozen=True)
class TEBDResult:
    n_wires: int
    ...

    @property
    def n_qubits(self) -> int:
        """Return the public qubit count; ``n_wires`` remains serialized internally."""
        return self.n_wires
```

The same reasoning applies to the two IR readers the slice calls.
`mps/tebd.py` builds `Instruction(name=..., wires=...)` and the MPS path reads
`instruction.wires` in its refusal messages; `core/ir.py`'s `wires` field is a
frozen `IR_VERSION = "1.0"` payload key owned by no slice. WQ-7 passes the old
keyword there deliberately, exactly as it reads `ir.n_wires` from the IR objects it
is handed. `runtime/module.py`'s `CompiledInstruction(wires=instruction.wires)`
keeps its keyword for the same reason: the class extends `Instruction`, whose
payload key is frozen.

## The payload keys this slice does rename

A bare-name string literal that names a qubit index or collection is renamed when
the slice's rename reaches the module that writes it and the key is not enumerated
in the contract's exclusions. The keys that move are the slice's own summaries and
in-package interfaces:

| Module | Before | After |
|---|---|---|
| `mps/dense_island.py`, `mps/models.py`, `mps/state.py`, `mps/static.py`, `tensor_network/state.py` | `"n_wires"` | `"n_qubits"` |
| `mps/state.py`, `tensor_network/state.py` | `"observable_wires"` | `"observable_qubits"` |
| `mps/models.py`, `tensor_network/entrypoints.py` | `"dense_observable_wires"` | `"dense_observable_qubits"` |
| `mps/state.py`, `mps/static.py` | `metadata={"wire": …}` | `metadata={"qubit": …}` |
| `mps/state.py`, `mps/static.py` | `metadata={"wires": …}` | `metadata={"qubits": …}` |
| `runtime/executors/tensor_network/state.py` | `"observable_wires"` | `"observable_qubits"` |

The rule is the writer's, not the reader's. Two summaries that carry the same
label are **not** renamed, because the module that writes them is outside this
slice's reach: `runtime/executors/statevector/reverse.py` and
`runtime/executors/jax/kernel.py` both still emit `"observable_wires"`, and the
tests that read those summaries are unchanged. Renaming one and not the other is
the honest outcome of the rule that whoever declares a name renames it — a slice
may rename what it declares, not what
it reads.

## Renamed refusal messages

Four user-visible messages changed their wording because the word was the concept:

| Module | Before | After |
|---|---|---|
| `mps/state.py` | `"wire index out of range"` | `"qubit index out of range"` |
| `mps/state.py` | `"observable wire index out of range"` | `"observable qubit index out of range"` |
| `mps/state.py` | `"expectation_z wire index out of range."` | `"expectation_z qubit index out of range."` |
| `mps/state.py` | `"expectation_z_sum wire index out of range."` | `"expectation_z_sum qubit index out of range."` |
| `mps/dense_island.py` | `f"wire {wire} is outside the plan"` | `f"qubit {qubit} is outside the plan"` |
| `jax/mps/kernels.py` | `"JAX MPS z_sum wire index out of range."` | `"JAX MPS z_sum qubit index out of range."` |

Messages are not a ledger surface, so no row counts them. They are recorded here
because `pytest.raises(..., match=…)` pins them, and five test modules had to
follow.

## The catch-up a rename owes

`flagquantum/simulation` is sparsely sliced — the contract names 17 of its modules
— and the slice's public functions are called from outside it. Five classes of
reader had to follow, and the discipline was the same in each case: **repair the
call sites `mypy --strict` names, and nothing else.**

1. **Ten executor and neighbour modules call the renamed functions.**
   `runtime/executors/jax/{array_conversions.py,mps/{canonicalization,execution,gradient_ownership,gradients,result,shards}.py,statevector/{execution,kernels}.py,tensor_network/{execution,gradients}.py}`,
   `runtime/executors/mps/{distributed_state,execution}.py`,
   `runtime/executors/tensor_network/{execution,state}.py`,
   `runtime/module.py`, `runtime/trajectories/result.py`,
   `simulation/mps/{canonical,local,sampling,wire_probability_dispatch}.py`, and
   `simulation/tensor_network/observables.py` pass the old spellings as keywords
   or read the old attribute names. Only the reached name moves: `runtime/module.py`
   keeps `CompiledInstruction(wires=…)`, and `runtime/executors/jax/kernel.py`
   keeps its own summary key.
2. **Two helpers in `jax/mps` had their declarations renamed and their call sites
   had to move with them.** `_initialize_jax_mps_rank_tensors(n_wires=…)` and
   `_jax_mps_boundary_protocol(left_wire=…, right_wire=…)` are declared in
   `jax/mps/shards.py` and `jax/mps/canonicalization.py`; the first pass renamed
   the callers' keywords without the declarations, which `mypy` reported as
   `Unexpected keyword argument`. Both sides now agree.
3. **`runtime/executors/mps/distributed_state.py` mirrors the boundary protocol.**
   Its `summary()` rows `"left_wire"` and `"right_wire"` are renamed with the
   protocol they copy, and the consumers
   (`flagquantum/runtime/parity.py`, `tests/distributed/runtime_modes.py`,
   `tests/test_jax_distributed_plan.py`) read the container keys, not these, so
   they are unaffected.
4. **Sixteen test modules name the renamed parameters, attributes, definitions,
   and messages.** Seven pass a retired spelling as a keyword
   (`test_jax_statevector_exchange_kernels.py` alone has 49), five name a renamed
   attribute or definition, and five pin a renamed refusal message with
   `pytest.raises(..., match=...)`. The tool that finds the first group is
   AST-driven rather than textual: it retargets a keyword only when the callee is
   a function this slice renamed *and* the keyword is one of that function's
   retired parameters. Its second version also matches a method by the last
   component of its qualified name, because a receiver typed `Any` — which is what
   a protocol-typed test fixture is — hides the keyword from every checker.
5. **`tools/census_wire_vocabulary.py` keeps citing `wire_probability_dispatch`.**
   The module name, the semantic id `mps.measurement.wire_probabilities.local`,
   and the private helper `_mps_wire_probabilities` are a *catalog identity*, not a
   qubit name: they are keys into the kernel catalog and are asserted by
   `tests/unit/test_mps_wire_probability_catalog_dispatch.py`, recorded in
   `benchmarks/results/local/mps_wire_probability_dispatch_a800.json`, and
   documented in `flagquantum/kernels/README.md`. They stay, and the slice's own
   `SiteKernelStats` counters — which are *not* catalog identity — move.

Four spellings inside the slice's *own* files were deliberately **not** renamed,
because the name belongs to a neighbour or to a frozen payload:

* `MPSState`/`TensorNetworkState` read `instruction.wires`, `ir.n_wires`, and
  `lowered.n_wires`. Those fields are declared in `core/ir.py` and
  `runtime/module.py`, modules no slice owns.
* `simulation/mps/wire_probability_dispatch.py` keeps its module name and its
  `_mps_wire_probabilities` helper. A file name is not a name on the function
  surface, and the helper's name shares a token with the catalog's `kind` string.
* `mps/tebd.py` keeps `TEBDResult::n_wires`, the slice's excluded payload key, and
  therefore also keeps the `n_wires=n_qubits` keyword its constructor is given.
* `tensor_network/models.py` and `mps/models.py` keep `max_marginal_wires` and
  `asymmetric_wire_order` where those are read from a versioned execution-policy
  payload.

`python tools/check_team_scope.py --require-classified --base ae8910bfc` reads the
change as `58 changed paths: (protected) 1, (shared) 18, runtime 2, simulation 37`.
`owner_team` stays `simulation` because the slice's own 17 files and 39 of the 58
changed paths are simulation code; the two `runtime` paths are the executor
modules that call into the renamed kernels, and `check_team_scope.py` accepts them
as classified rather than unowned. The alternative — leaving the catch-up to a
later slice — would have left the package failing `mypy --strict` on ten modules
at this commit.

## Verification

Each command was run at the commit that carries the rename, in
`/tmp/n19/work/fq-wq7` on branch `feat/qubit-vocabulary-accelerators`.

| # | Command | Result |
|---|---|---|
| 1 | `python -m black --check flagquantum tests tools` | all files unchanged |
| 2 | `python -m ruff check flagquantum tests tools` | `All checks passed!` |
| 3 | `python -m mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum` | `Success: no issues found in 624 source files` |
| 4 | `python tools/check_architecture.py` | `architecture boundaries passed` |
| 5 | `python tools/check_qubit_vocabulary.py` | exit 0; `286 of 341` parameters retired, `102 of 122` attributes retired, `8 of 10` definitions retired; WQ-7 is `78 / 18 / 1` with 0 remaining |
| 6 | `python tools/public_api_snapshot.py` | `public API migration baseline passed`, and neither `docs/public_api_v1.json` nor `contracts/public-api-v0.2-baseline.json` is edited |
| 7 | `python tools/validate_required_checks.py` | still `validated 6 externally configured required checks`: this change adds tests and contract rows, never a CI job |
| 8 | `python tools/check_team_scope.py --require-classified --base ae8910bfc` | `58 changed paths: (protected) 1, (shared) 18, runtime 2, simulation 37` |
| 9 | the CI marker lane | no new failure; see the PR body for the exact run |

Status: landed. The rename is WQ-7's and no other slice may retire a site this
ledger assigns to WQ-7.
