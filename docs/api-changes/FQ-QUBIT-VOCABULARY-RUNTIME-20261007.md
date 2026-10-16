# WQ-3: the runtime slice of the qubit vocabulary migration

One document per approved change to the Stable Core public API. This one records
the third slice of
[Quit calling them wires](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md) and
[The other two doors](FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md) as it lands.

It proposes no new capability, no export, no default, and no schema. `IR_VERSION`
stays `"1.0"`. It renames names on surfaces both predecessor documents already
decided, and it retires the last `wire` spelling on the planner and measurement
entry points a runtime caller reads.

## Decision and authorization

The user's authorization for this program is *"the user-facing surface of
FlagQuantum says `qubit`, never `wire`"*, restated in the
[integral document](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md). The naming table is
fixed once in that document and reused by every slice; this slice invents no new
pair. Slice membership is not decided here either — it is declared by
`[[slices]] files` in `contracts/qubit-vocabulary-contract.toml`, and the gate
recomputes each slice's counts from the frozen baseline on every run.

WQ-3 owns ten files and is measured at **14 canonical parameters**, **2 ledgered
attribute names**, **0 definition names**, and **6 payload-key exclusions**.

## What the slice renames

Fourteen wire-named parameters, in three groups:

| Entry point | Before | After |
|---|---|---|
| `runtime.planner.estimates` (6 estimators) | `n_wires` | `n_qubits` |
| `runtime.planner.execution_policy.estimate_execution_state_bytes` | `n_wires` | `n_qubits` |
| `runtime.planner.noise_calibration.NoiseSelectorCalibration.estimate_seconds` | `n_wires` | `n_qubits` |
| `runtime.planner.topology.rank_ownership` | `n_wires` | `n_qubits` |
| `runtime.measurements.validate_measurements` | `n_wires` | `n_qubits` |
| `runtime.measurements.execute_measurements` | `n_wires` | `n_qubits` |
| `runtime.dynamic.circuit.DynamicCircuit.measure` | `wire` | `qubit` |
| `runtime.dynamic.circuit.DynamicCircuit.reset` | `wire` | `qubit` |
| `runtime.dynamic.circuit.DynamicCircuit.conditional` | `wires` | `qubits` |

The six estimators are `estimate_state_bytes`, `estimate_mps_bytes`,
`estimate_density_bytes`, `estimate_stabilizer_bytes`,
`estimate_tensor_network_bytes`, and
`estimate_tensor_network_working_set_bytes`; their internal call chain
(`estimate_execution_state_bytes`, `estimate_seconds`, `rank_ownership`) moves in
the same commit, so no caller has to hold both spellings.

One attribute name is renamed, and one is aliased:

| Owner | Before | After | Alias |
|---|---|---|---|
| `runtime.planner.candidate_plans.CircuitAnalysisView` | `n_wires` | `n_qubits` | no — the protocol is not reachable from `fq.*` |
| `fq.MeasurementResult` | `wires` | `qubits` | yes — forwarding property with `DeprecationWarning` |

Eight message strings that name the count are reworded with the parameter, so a
refusal a user reads does not name the vocabulary the release is retiring:
`measurement qubits …`, `measurement qubits must be unique`, `measurement
postselect metadata must be a qubit-to-bit mapping`, and the marginal and
Pauli-basis limit sentences.

## Why only `MeasurementResult` gets an alias

Aliases are owed only where the old spelling is reachable from `fq.*`; everywhere
else an alias has no legal caller outside the package and would be permanent
weight against engineering decision principle 1.

`fq.MeasurementResult` is in `fq.__all__` and its `__module__` is
`flagquantum.runtime.result`, so `result.measurements[0].wires` is a documented
result field of an exported class. `fq.runtime` is not an attribute of `fq`
(`hasattr(fq, "runtime")` is `False`), so `runtime.measurements` and
`runtime.planner` are reachable only by module path. That is the same T3 boundary
the integral document draws: hard rename in the same release, no forwarder.

The forwarder is a `@property`, not a field, and that has a consequence the gate
records rather than ignores: `MeasurementResult::wires` moves from declaration
kind `field` to `member`. A field's name can only reach a payload through
`asdict`; converting it to a property removes that door, and the slice's
`attribute_field_count` drops from 1 to 0 while `attribute_member_count` rises
from 1 to 2. The total stays at 2, so a conversion cannot be used to hide a site.

## Payload keys this slice keeps

Six WQ-3 rows are excluded from the rename because their spelling is a key of a
payload that leaves the process. Each keeps its witness, evidence, and removal
condition in `[attribute_exclusions]`:

| Key | Owner | Witness |
|---|---|---|
| `n_wires`, `wire_usage` | `CircuitAnalysis` | the `ExecutionPlan` payload |
| `shardable_wires` | `ExecutionPlan` | its own `to_dict` |
| `wires` | `LayerPlan` | the `ExecutionPlan` payload |
| `n_wires` | `NoiseSelectorCalibrationRecord` | its own serialization method |
| `observable_wires` | `RuntimePolicy` | `RuntimePolicy.from_dict`, which reads it as an input key |

`CircuitAnalysis` gains an `n_qubits` property and `ExecutionPlan.summary()`
reports the count through it, so the plan payload keeps its pinned keys while the
in-process reader sees the domain noun. The gate forbids the other direction: a
row whose witness stops reading the spelling is an error, so an exclusion cannot
be parked once the payload schema does move.

Four further spellings are payload literals the census reports without ledgering
them; they are listed in
[the migration reference](../reference/QUBIT_NAMING_MIGRATION.md) under *Spellings
the census cannot see* rather than renamed here:
`rank_ownership`'s `wire_start` / `wire_end` / `owned_wires` return keys,
the `max_marginal_wires` measurement-metadata key,
`"per_sharded_wire_gate"` as a candidate-plan evidence value, and the `"n_wires"`
metric key the backend adapters build into `ExecutionResult.metrics`.

## Compatibility

| Surface | Compatibility |
|---|---|
| `fq.MeasurementResult.wires` | deprecated property, removed at 0.4.0 with the eleven shipped parameter aliases and the two WQ-2 attribute aliases |
| `runtime.planner.*`, `runtime.measurements.*`, `runtime.dynamic.circuit.*` | hard rename; a caller passing `n_wires=` or `wire=` gets `TypeError`, which is the honest failure for a name that had no compatibility promise |
| plan JSON, measurement metadata, metric dictionaries | unchanged |
| `result.wires` in an exported script | still works, with a warning |

Nothing here changes a numerical result. `MeasurementResult` is constructed
positionally or with `qubits=` everywhere in the package, and the two
reconstructions in `ExecutionResult.to()` were updated with it.

## Acceptance Tests

1. `python tools/check_qubit_vocabulary.py` exits 0 and reports
   `30 of 341 baseline sites retired` with WQ-3 at `retired 14 remaining 0`, and
   `27 of 122 attribute sites retired, 3 kept as deprecated aliases` with WQ-3 at
   `retired 1 remaining 1`.
2. `MeasurementResult.qubits` carries the value and `dataclasses.asdict` has no
   `wires` key; reading `.wires` warns and returns `qubits`.
3. Passing `n_wires=` to a renamed planner estimator, or `wire=` to
   `DynamicCircuit.measure`, raises `TypeError`.
4. The gate fails when a slice's `attribute_field_count`, `attribute_member_count`,
   or `canonical_count` disagrees with the ledger — demonstrated by the WQ-3
   conversion itself, which the slice had to declare.
5. `python -m pytest tests/unit/test_census_wire_vocabulary.py
   tests/unit/test_qubit_vocabulary_contract.py -q` passes with the live counts
   `311` canonical parameters and `95` ledgered / `22` excluded attributes.
6. `python tools/validate_required_checks.py` still passes: this change adds tests
   and contract rows, never a CI job.
7. `python tools/public_api_snapshot.py` reports `public API migration baseline
   passed`, and neither `docs/public_api_v1.json` nor
   `contracts/public-api-v0.2-baseline.json` is edited.

Status: landed. The rename is WQ-3's and no other slice may retire a site this
ledger assigns to WQ-3.
