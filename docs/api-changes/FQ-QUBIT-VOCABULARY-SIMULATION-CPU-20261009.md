# WQ-6: the simulation CPU slice of the qubit vocabulary migration

One document per approved change to the Stable Core public API. This one records
the sixth slice of
[Quit calling them wires](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md),
[The other two doors](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md),
[the runtime slice](FQ-QUBIT-VOCABULARY-RUNTIME-20261007.md),
[the runtime executor slice](FQ-QUBIT-VOCABULARY-EXECUTORS-20261008.md), and
[the algorithms slice](FQ-QUBIT-VOCABULARY-ALGORITHMS-20261009.md) as it lands.

It proposes no new capability, no export, no default, and no schema. `IR_VERSION`
stays `"1.0"`. It renames names on surfaces the five predecessor documents already
decided and invents no pair of its own. Slice membership is not decided here
either: it is declared by `[[slices]] files` in
`contracts/qubit-vocabulary-contract.toml`, and the gate recomputes each slice's
counts from the frozen baseline on every run.

Like WQ-4 and WQ-5, this slice has **no user-facing alias at all**. Its whole
surface lives under `flagquantum.simulation`, and `hasattr(fq, "simulation")` is
`False`: the package `__init__` reaches the simulation package through no
attribute, and none of the 104 public names the slice's files declare is reachable
as `fq.<name>`. A forwarder would have no legal caller outside the package and
would be permanent weight against engineering decision principle 1 (no permanent
compatibility debt). Every rename below is a hard rename in the same commit.

## Decision and authorization

The authorization is the one the user gave for the program: *"the user-facing
surface of FlagQuantum says `qubit`, never `wire`"*, restated in the
[integral document](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md). WQ-6 invents no new
scope: the contract records this document as `simulation_authorization` under
`[verification]`, alongside the five entries the predecessor slices left, and the
gate checks that the path exists on every run.

WQ-6 owns **17 files** and is measured at **77 canonical parameters**, **1
ledgered attribute name**, **3 definition names**, and **1 payload-key
exclusion**. The single attribute is a `field`, so the kind split is 1 `field`,
0 `member`, 0 `instance`.

The slice is the first to reopen a file the *composition* of which matters: the
simulation CPU cores are the code the executors call, so the rename leaves the
slice through 47 spellings in 16 packages modules no slice owns. That catch-up is
described below, and it is the whole reason this slice's diff is wider than its
file list.

## What the slice renames

### 77 parameters

The parameters fall into six groups, read here from the frozen ledger's 77 rows.

| Group | Modules | Renamed parameters | Sites |
|---|---|---|---:|
| Dense-state kernels | `density_matrix.py` | `apply_kraus_density`, `apply_unitary_density`, `expand_operator`, `expectation_z_density` (`n_wires`, `wires`) | 7 |
| Lindblad evolution | `lindblad.py` | `amplitude_damping(wire=…)`, `evolve_density_matrix`, `plan_density_matrix_evolution` | 3 |
| Native CPU adjoint | `native_cpu/adjoint.py` | `fused_cx_rotation_segment_adjoint`, `fused_rotation_adjoint_`, `fused_rotation_segment_adjoint_`, `fused_rzz_segment_adjoint_` (`n_wires`, `wires`, `wire`, `second_wire`, `first_wires`, `second_wires`, `rzz_first_wires`, `rzz_second_wires`) | 12 |
| Native CPU permutation and rotation | `native_cpu/permutation.py`, `native_cpu/rotation.py`, `native_cpu/rzz.py` | `compact_cpu_cx_adjoint_auxiliary_bytes`, `compact_cx_permutation_images`, `fused_clifford_matching_out`, `fused_cx_adjoint_inplace_`, `use_compact_cpu_cx_adjoint_cycles`, `use_compact_cpu_cx_mapping`, `fused_hadamard_block_adjoint_`, `fused_rotation_block_adjoint_`, `fused_rotation_block_forward_`, `fused_static_clifford_layer_`, `native_cpu_forward_rotation_tile_wires`, `fused_rzz_segment_forward_` | 20 |
| Pauli and Hamiltonian algebra | `pauli.py`, `statevector/adjoint.py` | `pauli_product_density_expectation`, `pauli_product_operator`, `pauli_product_statevector_expectation`, `z_expectation_adjoint_chunk`, `z_expectation_chunk`, `z_hamiltonian_chunk`, `z_hamiltonian_weights` | 9 |
| Statevector programs and sampling | `statevector/{clifford_matching,double_single,dynamic_noise,fixed_layer_cpu,noisy,product_state,single_qubit_cpu,split_real_imag}.py`, `stabilizer/engine.py` | `apply_native_clifford_matching`, `apply_double_single_gate`, `double_single_pauli_term_expectation`, `run_double_single_statevector`, `apply_dynamic_bit_flip`, `apply_native_fixed_one_qubit_layer`, `apply_amplitude_damping_batched`, `apply_kraus_batched`, `apply_matrix_batched`, `expectation_z`, `execute_product_state_program`, `product_state_execution_is_beneficial`, `apply_single_qubit_matrix_cpu`, `apply_gate_pair`, `pauli_term_expectation`, `sample_noisy_measurements(terminal_wires=…)`, `sample_stabilizer(wires=…)` | 26 |

### 1 attribute name

| Site | Before | After | Kind |
|---|---|---|---|
| `simulation/lindblad.py::CollapseOperator` | `wires` | `qubits` | `field` |

`CollapseOperator` is the one simulation record that is part of an argument a
user writes: `fq.CollapseOperator("amplitude_damping", rate, (0,))` takes the
qubit by position, but a caller who reads back `operator.wires` is reading a name
this migration owns. The class is not serialized into any versioned payload, so
the rename needs no exclusion.

### 3 definition names

| Before | After |
|---|---|
| `native_cpu/adjoint.py::native_cpu_rotation_tile_wires` | `native_cpu_rotation_tile_qubits` |
| `native_cpu/rotation.py::native_cpu_forward_rotation_tile_wires` | `native_cpu_forward_rotation_tile_qubits` |
| `pauli.py::infer_n_wires_from_dense_state` | `infer_n_qubits_from_dense_state` |

The first two are re-exported from `flagquantum/simulation/native_cpu/__init__.py`
and read by the executor's forward and reverse sweeps; the third is imported by
`flagquantum/algorithms/core.py`, which WQ-5 owns. All three are internal
definition names on the public module surface, so the module list moves with them.

### 23 private parameters, unledgered

The ledger counts the public function surface, so the 23 `wire`-named parameters
of this slice's private helpers are not rows in it. They move anyway, because a
slice that renames `n_wires` in its public signature and leaves
`_fused_rotation_segment_adjoint_result(n_wires=…)` alone has renamed a word
rather than a concept. This is why `boundary.measured_private` drops from 331 to
308: the gate re-measures the private bucket and refuses a stale number.

## The one payload key this slice keeps

`simulation/lindblad.py::EvolutionPlan::n_wires` is written into
`EvolutionPlan.to_dict()`. The record is read back by `flagquantum/lindblad/_plan.py`,
a private reader that reconstructs the dataclass from the payload, so the spelling
reaches a serialized schema and moves only when that schema is versioned. It is
the slice's single row in `[attribute_exclusions]`, and the gate checks that the
row names a witness class, the evidence, and the removal condition.

The field therefore stays `n_wires` and gains the public spelling as a property:

```python
@dataclass(frozen=True)
class EvolutionPlan:
    n_wires: int
    ...

    @property
    def n_qubits(self) -> int:
        """Return the public qubit count; ``n_wires`` remains serialized internally."""
        return self.n_wires
```

The same reasoning applies to the two IR constructors the slice calls.
`Instruction(name=..., wires=(0,))` in `statevector/split_real_imag.py` and
`MeasurementNode("sample", selected, shots=shots)` in `stabilizer/engine.py` are
constructions of `flagquantum/core/ir.py` records whose `wires` field is a frozen
`IR_VERSION = "1.0"` payload key owned by no slice. WQ-6 passes the old keyword
there deliberately, exactly as it reads `ir.n_wires` and `instruction.wires` from
the IR objects it is handed.

## Two user-visible descriptor spellings

`lindblad.py::_descriptor_matrix` turns a user's dictionary into a matrix, and the
dictionary keys are the user's:

| Before | After |
|---|---|
| `{"wires": (0, 1), "matrix": ...}` | `{"qubits": (0, 1), "matrix": ...}` |
| `{"wire": 0, ...}` | `{"qubit": 0, ...}` |

The mapping branch keeps its two-key fallback chain, so a dictionary that names
only the operator still reads (`item.get("qubit", (0,))`), but both spellings in
that chain are the qubit ones: this slice ships no alias, so the old key stops
being read. These are not parameters or attributes, so no ledger row counts them;
they are recorded here because they are the one place in this slice where a `wire`
spelling a user types was renamed rather than kept.

## The catch-up a rename owes

`flagquantum/simulation` is sparsely sliced — the contract names 17 of its
modules — and the slice's public functions are called from outside it. Four
classes of reader had to follow, and the discipline was the same in each case:
**repair the call sites `mypy --strict` names, and nothing else.**

1. **Thirteen executor and neighbour modules call the renamed functions.**
   `runtime/executors/statevector/{forward_sweep,forward_rzz_segment,reverse_adjoint,reverse_adjoint_kernels,reverse_adjoint_rotation_segment,reverse_adjoint_sweep,split_real_imag,split_real_imag_precision,split_real_imag_double_single,split_real_imag_device_double_single}.py`,
   `simulation/statevector/{local,operations}.py`, `lindblad/_plan.py`, and
   `qec/sampling.py` pass the old spellings as keywords. Only the keyword moves:
   `local.py` keeps its own wire-named locals and its own `request.wires` read,
   and `_plan.py` keeps `payload["n_wires"]`, which is the payload key.
2. **`simulation/native_cpu/__init__.py` re-exports the two renamed natives.**
   Both the import and the `__all__` string move.
3. **`algorithms/core.py` imports `infer_n_wires_from_dense_state`.** WQ-5 owns
   that file, and its own ledger says nothing about this import, so the change is
   confined to the imported name.
4. **Twelve test files name the renamed parameters and definitions.** Nine of
   them pass a retired spelling as a keyword (`test_native_cpu_adjoint.py` alone
   has 58), three name one of the three renamed definitions, and four assert a
   renamed refusal message with `pytest.raises(..., match=...)`. The tool that
   finds the first group is AST-driven rather than textual: it retargets a keyword
   only when the callee is a function this slice renamed *and* the keyword is one
   of that function's retired parameters, so a keyword that names a neighbouring
   function's own parameter is left alone.
   `tools/census_wire_vocabulary.py` keeps citing
   `infer_n_wires_from_dense_state`: it is the docstring's example of the old
   spelling, not a call.

Three spellings inside the slice's *own* files were deliberately **not** renamed,
because the name belongs to a neighbour:

* `_StatevectorFusedGateStep.wires`, `_StatevectorDenseRegion.wires`, and the
  other `Step`/`Region` variants are declared in
  `flagquantum/simulation/statevector/program.py`, a module no slice owns. A slice
  may rename what it declares, not what it reads.
* `flagquantum/simulation/statevector/wire_permutation.py` keeps its module name.
  A file name is not a name on the function surface, and renaming it would move
  an import path across two unowned modules; the module's private helpers keep
  their spellings for the same reason.
* `self.n_wires` stays: it is the excluded `EvolutionPlan` field.

`tools/check_team_scope.py` reads the change as `(protected) 1, (shared) 15,
algorithms 1, core 1, simulation 31`, so the contract declares WQ-6's owner as
`simulation + algorithms + core` rather than `simulation` alone. The alternative —
leaving the catch-up to a later slice — would have left the package failing
`mypy --strict` on 16 modules at this commit.

## Verification

Each command was run at the commit that carries the rename, in
`/tmp/n19/work/fq-wq6` on branch `feat/qubit-vocabulary-simulation-cpu`.

| # | Command | Result |
|---|---|---|
| 1 | `python -m black --check flagquantum tests tools` | all files unchanged |
| 2 | `python -m ruff check flagquantum tests tools` | `All checks passed!` |
| 3 | `python -m mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum` | `Success: no issues found in 624 source files` |
| 4 | `python tools/check_architecture.py` | `architecture boundaries passed` |
| 5 | `python tools/check_qubit_vocabulary.py` | exit 0; `208 of 341` parameters retired, `84 of 122` attributes retired, `7 of 10` definitions retired; WQ-6 is `77 / 1 / 3` with 0 remaining |
| 6 | `python tools/public_api_snapshot.py` | `public API migration baseline passed`, and neither `docs/public_api_v1.json` nor `contracts/public-api-v0.2-baseline.json` is edited |
| 7 | `python tools/validate_required_checks.py` | still `validated 6 externally configured required checks`: this change adds tests and contract rows, never a CI job |
| 8 | `python tools/check_team_scope.py --require-classified --base 57c0b8338` | every changed path is classified: 1 protected, 15 shared, 1 `algorithms`, 1 `core`, 31 `simulation` |
| 9 | the CI marker lane | no new failure; see the PR body for the exact run |

Status: landed. The rename is WQ-6's and no other slice may retire a site this
ledger assigns to WQ-6.
