# WQ-8: the compiler, tooling, and measurement slice of the qubit vocabulary migration

One document per approved change to the Stable Core public API. This one records
the eighth and last slice of
[Quit calling them wires](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md),
[The other two doors](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md),
[the runtime slice](FQ-QUBIT-VOCABULARY-RUNTIME-20261007.md),
[the runtime executor slice](FQ-QUBIT-VOCABULARY-EXECUTORS-20261008.md),
[the algorithms slice](FQ-QUBIT-VOCABULARY-ALGORITHMS-20261009.md),
[the simulation CPU slice](FQ-QUBIT-VOCABULARY-SIMULATION-CPU-20261009.md), and
[the accelerator slice](FQ-QUBIT-VOCABULARY-ACCELERATORS-20261010.md) as it lands.

It proposes no new capability, no export, no default, and no schema. `IR_VERSION`
stays `"1.0"`. It renames names on surfaces the seven predecessor documents
already decided and invents no pair of its own. Slice membership is not decided
here either: it is declared by `[[slices]] files` in
`contracts/qubit-vocabulary-contract.toml`, and the gate recomputes each slice's
counts from the frozen baseline on every run.

This slice closes the ledger. Before it, 55 canonical parameters, 17 attribute
names, and 2 definition names were still spelled `wire`; after it, the live scan
finds **no** canonical parameter, **no** `field` attribute, and **no** `instance`
attribute at all. `[boundary].measured_canonical` stays `341`,
`[attribute_ledger].measured_canonical` stays `122`,
`[attribute_exclusions].measured_persisted` stays `22`, and
`[boundary].measured_deprecated_aliases` stays `11`, because a rename moves rows
from the ledger to the retirement tables rather than deleting either.

## Decision and authorization

The authorization is the one the user gave for the program: *"the user-facing
surface of FlagQuantum says `qubit`, never `wire`"*, restated in the
[integral document](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md). WQ-8 invents no new
scope: the contract records this document as `compiler_authorization` under
`[verification]`, alongside the seven entries the predecessor slices left, and the
gate checks that the path exists on every run.

WQ-8 owns **31 files** and is measured at **55 canonical parameters**, **17
ledgered attribute names**, **2 definition names**, and **5 payload-key
exclusions**. The attributes split 9 `field`, 7 `member`, 1 `instance`; two of the
`member` rows are the slice's deprecated forwarders described below.

The 31 files are the compiler and its neighbours, the benchmarking harnesses that
drive them, the noise and device-profile model, the Triton kernels the compiler
talks to, the remote-QPU provider stack, and one `twin` payload:

| Group | Modules |
|---|---|
| benchmarking | `batched_statevector_corpus`, `batched_statevector_memory`, `differentiable_simulator_corpus`, `external_simulator_compare`, `simulator_compare`, `simulator_workload_corpus`, `statevector_cpu_paths`, `statevector_local` |
| compiler | `directed_topology`, `layout`, `one_qubit_synthesis`, `openqasm`, `openqasm_import`, `physical_plan`, `qcis`, `routing`, `topology_legalization`, `two_qubit_synthesis` |
| deployment | `cloud`, `routing_evidence` |
| kernels | `triton/hva_forward_tangent`, `triton/mps_wire_probabilities`, `triton/statevector_gates` |
| noise | `channels`, `device_profile`, `model` |
| remote | `qpu/azure`, `qpu/contracts`, `qpu/execution`, `qpu/http` |
| twin | `prediction` |

## What the slice renames

### 55 parameters

The parameters fall into six groups, read here from the frozen ledger's 55 rows.

| Group | Renamed parameters | Sites |
|---|---|---:|
| Benchmarking harnesses | `build_workload`, `run_case`, `run_benchmark`, `build_parameter_batch`, `build_cx_chain`, `build_diagonal_chain`, `build_marginal_circuit`, `build_mixed_chain`, `build_rotation_chain`, `build_two_wire_diagonal_chain` (`n_wires`, one `marginal_wires`) | 25 |
| Compiler algorithms | `DirectedCouplingMap.__init__`, `CouplingMap.__init__`, `CouplingMap.line`, `CouplingMap.ring`, `CouplingMap.neighbors`, `Layout.map_wires`, `remove_layout_restore.rename`, `synthesize_one_qubit_matrix`, `synthesize_two_qubit`, `emit_openqasm` (`n_wires`, `wires`, `wire`, `result_wires`) | 12 |
| Noise and device profiles | `DeviceNoiseProfile.calibration_for`, `DeviceNoiseProfile.duration_for`, `NoiseModel.add`, `NoiseModel.add_readout`, `NoiseModel.add_correlated_readout`, `NoiseModel.apply_readout_probabilities` (`wire`, `wires`, `n_wires`) | 6 |
| Triton kernels | `single_qubit_matrix`, `cx_sequence`, `ry_rz_pair`, `heisenberg_hva_forward_tangents` (`n_wires`, `wire`) | 7 |
| Remote QPU providers | `AzureQuantumProvider.__init__`, `azure_backend_profile`, `QuantumProvider.discover_backends`, `counts_result`, `HttpQuantumProvider.__init__` (`n_wires`, `default_n_wires`) | 5 |
| Deployment evidence | `expectation_z_from_counts`, `build_deployment_routing_evidence`, `validate_deployment_routing_plan` (`wires`, `n_wires`) | 3 |

The new spellings follow the one substitution rule the contract states under
`[naming]`, so the whole table is a root substitution and not a list of choices:
`n_wires → n_qubits`, `wires → qubits`, `wire → qubit`,
`marginal_wires → marginal_qubits`, `result_wires → result_qubits`,
`default_n_wires → default_n_qubits`. Three compound names move by the same rule
in the files that declare them: `coupling_n_wires → coupling_n_qubits`,
`logical_wire_count → logical_qubit_count`, and `map_wires → map_qubits`.

`logical_qubits` is the one substitution output the contract used to forbid
outright. It left `[naming] forbidden` in this slice, because the rule now reaches
`logical_wires` and the name is ordered and unambiguous — `physical_qubits` on the
matching row. `qubit_indices`, `qubit_ids`, and `qubit_labels` remain forbidden:
each is a synonym for a list this package already calls `qubits`.

### 17 attribute names

The count splits 9 `field`, 7 `member`, 1 `instance`. Fifteen are hard renames:

| Kind | Site | After |
|---|---|---|
| field | `compiler/directed_topology.py::DirectedCouplingMap::n_wires` | `n_qubits` |
| field | `compiler/physical_plan.py::MappingTransition::physical_wires` | `physical_qubits` |
| field | `compiler/physical_plan.py::PhysicalCircuitPlan::coupling_n_wires` | `coupling_n_qubits` |
| field | `compiler/physical_plan.py::PhysicalCircuitPlan::logical_wire_count` | `logical_qubit_count` |
| field | `compiler/physical_plan.py::PhysicalInstructionRecord::logical_wires` | `logical_qubits` |
| field | `compiler/physical_plan.py::PhysicalInstructionRecord::physical_wires` | `physical_qubits` |
| field | `compiler/qcis.py::QCISInstruction::wires` | `qubits` |
| field | `compiler/routing.py::CouplingMap::n_wires` | `n_qubits` |
| field | `compiler/topology_legalization.py::TopologyLegalizationResult::logical_wire_count` | `logical_qubit_count` |
| member | `compiler/layout.py::Layout::map_wires` | `map_qubits` |
| member | `compiler/layout.py::Layout::n_wires` | `n_qubits` |
| member | `deployment/cloud.py::DeploymentPackage::n_wires` | `n_qubits` |
| member | `noise/channels.py::KrausChannel::n_wires` | `n_qubits` |
| member | `noise/model.py::CorrelatedReadoutError::n_wires` | `n_qubits` |
| instance | `remote/qpu/http.py::HttpQuantumProvider::default_n_wires` | `default_n_qubits` |

The two remaining rows are the slice's deprecated forwarders, and both are
reachable from `fq.*` because `fq.deployment` and `fq.compiler` are behind
`fq.CloudBackendProfile`'s constructor:

| Alias site | Forwards to | Authorization |
|---|---|---|
| `deployment/cloud.py::CloudBackendProfile::n_wires` | `n_qubits` | [Qubit terminology migration](FQ-QUBIT-NAMING-20260913.md), which names this property and its constructor keyword explicitly |
| `compiler/openqasm_import.py::OpenQASMImport::n_wires` | `n_qubits` | [OpenQASM import subset and `fq.from_openqasm`](FQ-OPENQASM-IMPORT-20261002.md) |

`fq.from_openqasm(...).n_wires` is documented in `tests/test_openqasm_import.py`
and in the OpenQASM proposal, so the property keeps its old spelling and gains
`n_qubits`, the same shape `fq.Circuit.n_wires` uses. The class is deliberately
**not** given a constructor alias: `OpenQASMImport` is built by `fq.from_openqasm`,
which already takes `n_qubits`, so a second keyword would be a caller's second way
to say one thing.

`CloudBackendProfile` additionally keeps its two constructor aliases
`CloudBackendProfile.__init__::n_wires` and `CloudBackendProfile.simulator::n_wires`.
Both were already live before this slice, which is why
`[boundary].measured_deprecated_aliases` stays `11`.

### 2 definition names

| Before | After |
|---|---|
| `benchmarking/statevector_cpu_paths.py::build_two_wire_diagonal_chain` | `build_two_qubit_diagonal_chain` |
| `kernels/triton/mps_wire_probabilities.py::fused_mps_wire_probabilities` | `fused_mps_qubit_probabilities` |

The second one is a **catalog symbol**. `kernels/catalog/implementations.py`
records it as the `symbol` of implementation `FQKI-TRITON-MPS-007-A`, and
`tests/unit/test_kernel_catalog.py` asserts that every recorded symbol resolves to
a function in the module it names, so the catalog row had to move with the
function. The semantic id `mps.measurement.wire_probabilities.local`, the
implementation id `FQKI-TRITON-MPS-007-A`, the module name
`kernels/triton/mps_wire_probabilities.py`, and the evidence test ids in
`kernels/catalog/evidence.py` all stay: they are stable identities, not names a
user types to reach a qubit.

`benchmarks/mps_wire_probability_dispatch.py` imported the old symbol and called
`MPSState._wire_probabilities`, which WQ-7 renamed to `_qubit_probabilities` while
leaving this file alone because it is outside every slice. It is repaired here,
because a rename that leaves its only external caller broken is not finished.

### 49 private parameters, unledgered

The ledger counts the public function surface, so the `wire`-named parameters of
this slice's private helpers are not rows in it. They move anyway, because a slice
that renames `NoiseModel.add(wires=…)` and leaves `_validate_wire(wire=…)` alone
has renamed a word rather than a concept. They are declared by the slice's own
files (`Layout._validate_wire`, `Layout._validate_wire_layout`,
`device_profile.py::_validated_qubit_index`, `routing.py::_require_multi_wire_device_local`,
`openqasm.py::_gate`, `two_qubit_synthesis.py::_swap_phase_sequence`, and the
benchmarking harnesses' own `_build_*` helpers) and by the follower modules below.
Five module-level constants also move: `DEFAULT_N_WIRES → DEFAULT_N_QUBITS`,
`DEFAULT_MARGINAL_WIRES → DEFAULT_MARGINAL_QUBITS`,
`_MULTI_WIRE_OPERAND_PAIRS → _MULTI_QUBIT_OPERAND_PAIRS`,
`_reported_wire_counts → _reported_qubit_counts`, and
`_require_multi_wire_device_local → _require_multi_qubit_device_local`. This is why
`boundary.measured_private` drops from 275 to 226: the gate re-measures the private
bucket and refuses a stale number.

### 634 message strings

`message_strings` was `1291` and is `634`; 94 of the remaining strings are in this
slice's own files, and every one of them is a frozen key, a CLI flag, a data path,
or a vendor spelling. The strings that changed are the ones where the word *was*
the concept:

| Module | Before | After |
|---|---|---|
| `compiler/routing.py` | `"Coupling wire count must be an integer."` | `"Coupling qubit count must be an integer."` |
| `compiler/routing.py` | `f"Coupling wire {wire} is outside [0, {self.n_wires - 1}]."` | `f"Coupling qubit {qubit} is outside [0, {self.n_qubits - 1}]."` |
| `compiler/routing.py` | `"Coupling map has fewer wires than the circuit."` | `"Coupling map has fewer qubits than the circuit."` |
| `compiler/routing.py` | `f"No coupling path between wires {start} and {goal}."` | `f"No coupling path between qubits {start} and {goal}."` |
| `compiler/directed_topology.py` | `"Directed coupling map requires a positive wire count."` | `"Directed coupling map requires a positive qubit count."` |
| `compiler/layout.py` | `"layout must place at least one logical wire"` | `"layout must place at least one logical qubit"` |
| `compiler/layout.py` | `f"physical wire {physical} holds no logical wire"` | `f"physical qubit {physical} holds no logical qubit"` |
| `compiler/one_qubit_synthesis.py` | `"wire must be a non-negative integer"` | `"qubit must be a non-negative integer"` |
| `compiler/two_qubit_synthesis.py` | `"wires must name exactly two wires"` | `"qubits must name exactly two qubits"` |
| `compiler/openqasm.py` | `"OpenQASM result wires must be unique in-range integers."` | `"OpenQASM result qubits must be unique in-range integers."` |
| `deployment/cloud.py` | `"Requested wires are outside the measured register."` | `"Requested qubits are outside the measured register."` |
| `noise/model.py` | `"readout wires must be unique"` | `"readout qubits must be unique"` |
| `noise/model.py` | `"a wire can have only one readout error rule"` | `"a qubit can have only one readout error rule"` |
| `noise/device_profile.py` | `f"device profile has no calibration for wire {wire}"` | `f"device profile has no calibration for qubit {qubit}"` |
| `twin/prediction.py` | `"n_wires must be positive"` | `"n_qubits must be positive"` |

Messages are not a ledger surface, so no row counts them. They are recorded here
because `pytest.raises(..., match=…)` pins them, and the test modules that pin them
had to follow.

### The CLI flags this slice keeps

`--n-wires` and `--marginal-wires` are not parameters and no census sees them. They
stay: the same flag family is emitted by benchmarking modules no slice owns, so
renaming these two would leave the package with two spellings for one flag.
`statevector_cpu_paths.py` therefore reads `args.n_wires` and
`args.marginal_wires` while every in-package call moves to `n_qubits`. The two
strings that reproduce a command line
(`differentiable_simulator_corpus.py`, `simulator_workload_corpus.py`) keep the
flag spelling for the same reason.

## The five payload keys this slice keeps

Each is a `[attribute_exclusions]` row. A row names the class or method that
witnesses the spelling, the evidence the census found, and the event that retires
it; the gate re-measures all three on every run.

| Site | Witness | Evidence | Why it stays |
|---|---|---|---|
| `noise/device_profile.py::GateDuration::wires` | `DeviceNoiseProfile` | `contained` | the device-profile payload is read back by key |
| `noise/device_profile.py::QubitNoiseCalibration::wire` | `DeviceNoiseProfile` | `contained` | same payload |
| `noise/model.py::NoiseRule::wires` | `NoiseModel` | `contained` | `NoiseModel.to_dict()` / `from_dict()` round-trip the rule set under `"wires"` |
| `noise/model.py::ReadoutRule::wires` | `NoiseModel` | `contained` | same payload |
| `twin/prediction.py::TwinPrediction::n_wires` | `TwinPrediction` | `serialization_method` | `to_dict()` writes `"n_wires"` and `from_dict()` reads it |

The fields therefore keep their spellings and the constructor keeps passing them by
the old keyword, while every qubit-named accessor around them moves:

```python
@dataclass(frozen=True)
class TwinPrediction:
    n_wires: int
    ...
    def to_dict(self) -> dict[str, object]:
        return {..., "n_wires": self.n_qubits, ...}
```

`DeviceNoiseProfile` is the shape this repeats: the payload key `"wires"` and the
`GateDuration.wires` / `QubitNoiseCalibration.wire` fields keep their names, while
the validator parameters (`calibration_for(qubit)`, `duration_for(qubits)`), the
locals, and every refusal message move. `object.__setattr__(self, "wires", qubits)`
keeps the *field name* on the left of the string and takes the renamed local on the
right; the mirror-image mistake in `compiler/routing.py` and
`compiler/directed_topology.py` — where the dataclass field is named `n_qubits` but
`__post_init__` still wrote `"n_wires"` — produced an `AttributeError` at runtime
that `mypy --strict` cannot see, and is fixed here.

## The catch-up a rename owes

`flagquantum.compiler`, `flagquantum.noise`, `flagquantum.deployment`,
`flagquantum.remote`, and `flagquantum.kernels` are sparsely sliced — the contract
names 31 modules across them — and the slice's classes are read from outside it. In
addition to the 31 owned files, **22 follower modules** under `flagquantum/**`, one
benchmark script, and four test modules had to follow, and the discipline was the
same as in WQ-6 and WQ-7: **repair the call sites `mypy --strict` names, and nothing
else.**

Three classes of reader were caught:

1. **`CouplingMap` / `DirectedCouplingMap` / `Layout` consumers.** `compiler/{sabre,
   layout_planning,native_gate_legalization,target_emission,target_artifact,
   compilation_evidence}.py` read `coupling_map.n_qubits` and
   `layout.n_qubits`, and pass `Layout.map_qubits(...)`.
2. **`NoiseModel` callers.** `runtime/dynamic/_noise.py`, `qec/noise.py`,
   `compiler/compilation_evidence.py`, and `twin/{model,region_model}.py` call
   `model.add(..., qubits=…)`, `add_readout(qubits=…)`,
   `apply_readout_probabilities(n_qubits=…)`, and read `rule.wires` off the
   untouched payload field.
3. **Provider and dispatcher adapters.** `remote/qpu/{braket,calibration,quafu}.py`,
   `remote/jobs.py`, `runtime/dynamic/deployment.py`,
   `simulation/statevector/{single_qubit_matrix,ry_rz,cx_sequence}_dispatch.py`,
   `simulation/mps/wire_probability_dispatch.py`, and `algorithms/error_mitigation.py`
   read `plan.coupling_n_qubits`, `device.n_qubits`, `instruction.wires`, and pass
   `n_qubits=` to the renamed factories.

Three keyword classes were deliberately **not** moved, because the keyword is a
serialized key rather than the slice's name:

* `physical_wires=`, `logical_wires=`, and `logical_wire_count=` in the
  compilation-evidence bundle, whose payload keys are pinned by
  `contracts/compilation-evidence-bundle-v1-candidate.json`.
* `result_qubits=` **is** moved: it is `emit_openqasm`'s own parameter, not a payload
  key, so the rename reaches it and its argument expression moves with it.
* `n_wires=` in `remote/qpu/quafu.py`'s `_submission_options` / `_service_options`:
  the helper is private, but its parameter feeds the Quafu provider payload and the
  keyword still travels across that boundary.

`Instruction.wires`, `CircuitIR.n_wires`, `MeasurementNode.wires`, and
`ObservableNode.wires` are declared in `flagquantum/core/ir.py` and are the
`IR_VERSION = "1.0"` payload, so the slice reads them and never writes them.

### Two call sites `mypy --strict` cannot see

`mypy --strict` covers `flagquantum/**` and not `tests/**`, and inside the package
it has two blind spots that this slice hit:

* **A receiver typed `Any`.** `runtime/measurements.py` holds its noise model as
  `Any`, so `self.noise_model.apply_readout_probabilities(probabilities,
  n_wires=…)` on line 96 and the same call on line 258 type-checked after the
  rename and failed at runtime with `TypeError: … got an unexpected keyword
  argument 'n_wires'`. Neither `mypy` nor a grep for `def` finds those; only an
  AST sweep that pairs every keyword argument with the callee's declared
  parameters does. The slice ran one over `flagquantum/**` and it reported exactly
  these two.
* **A constructor called by class name.** The census records a constructor
  parameter as `path::Class.__init__::name`, so a follower table keyed on the
  ledger's qualname matches nothing at a call site that says
  `AzureQuantumProvider(n_wires=32)`. Stripping the trailing `__init__` before
  matching is what found 25 of this slice's keywords in `tests/**`.

The same pairing has to be run over `tests/**` with the slice's own classes as
the key, since a test-tree attribute name is not evidence about its receiver:
`.n_wires` is still correct on a `CircuitIR` and retired on a `Layout`, and both
appear in the same file. The rule this migration settles on is that only a
*call-site keyword* of a callee the slice renamed, a *`match=` regex* of a message
the slice now spells differently, or an *attribute read* of a name only the slice
declares may move — never a `def test_*` name, never a bare-name dictionary key,
and never another slice's spelling.

## Verification

Each command was run at the commit that carries the rename, in
`/tmp/n19/work/fq-wq8` on branch `feat/qubit-vocabulary-compiler-tooling`.

| # | Command | Result |
|---|---|---|
| 1 | `python -m black --check flagquantum tests tools` | all files unchanged |
| 2 | `python -m ruff check flagquantum tests tools` | `All checks passed!` |
| 3 | `python -m mypy --strict --python-version 3.12 --ignore-missing-imports flagquantum` | the three errors this tree already carried at `e5f0348b1`, and no fourth |
| 4 | `python -m compileall -q flagquantum` | exit 0 |
| 5 | `python tools/check_qubit_vocabulary.py` | exit 0; `341 of 341` parameters retired, `117 of 122` attributes retired, `10 of 10` definitions retired; WQ-8's three rows read `55 retired / 0 remaining`, `15 retired / 2 remaining`, `2 retired / 0 remaining`, and the two attributes still counted are the forwarders `[attribute_aliases]` declares |
| 6 | `python tools/public_api_snapshot.py` | `public API migration baseline passed`; neither `docs/public_api_v1.json` nor `contracts/public-api-v0.2-baseline.json` is edited, because no name in `stable_exports` is in this slice |
| 7 | `python tools/validate_required_checks.py` | still `validated 6 externally configured required checks`: this change adds tests and contract rows, never a CI job |
| 8 | `python tools/check_team_scope.py --require-classified --base e5f0348b1` | see the PR body for the exact class counts |
| 9 | the CI marker lane | no new failure; see the PR body for the exact run |

Status: landed. This slice closes the ledger: no canonical parameter, no `field`
attribute, and no `instance` attribute remains on the live surface, and the five
live attributes are the deprecated forwarders that `[aliases]` and
`[attribute_aliases]` record for removal at `0.4.0`. The rename is WQ-8's and no
other slice may retire a site this ledger assigns to WQ-8.
