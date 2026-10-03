# The other two doors: attribute names and definition names

One document per proposed change to the Stable Core public API. This one extends
[Quit calling them wires](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md) from
parameters to the two name surfaces that document recorded as *owned gaps*, and
it answers that document's Open Question 1.

It changes no behaviour. It decides which names may move and which may not, and
it puts both decisions under a gate.

## Decision and authorization

The user's authorization for this program is *"the user-facing surface of
FlagQuantum says `qubit`, never `wire`"*
([integral document](FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md), *Decision and
authorization*). That sentence is about the surface a user sees, not about the
syntax a user types. `result.n_wires`, `circuit.n_wires`, `GateInfo.n_wires`, and
`infer_n_wires_from_dense_state` are all on that surface, and none of them is a
parameter.

The integral document recorded them as measured, unledgered gaps:

| Surface | Integral document | This document |
|---|---:|---|
| public attribute and property names containing `wire` | 102, **not gated**, owned by WQ-2 | **144**, gated, split by declaration kind |
| module-level public definition names containing `wire` | not measured | **10**, gated |
| string literals containing `wire` | 1291, reported only | unchanged: still reported only |

The 102 has grown to 144 because the original scan could only see one of the
three ways a class declares a name. Section *Why the count moved from 102 to 144*
records the measurement error, in the same spirit as the integral document's four
corrections.

Nothing here is a new Stable Core surface. This document, like the integral one,
renames existing names and adds no export, field, default, or serialized key.

## Why the count moved from 102 to 144

The original attribute scan read annotated assignments inside public classes:
`wires: tuple[int, ...] = ()`. That finds every **field**, and it cannot find a
name that no assignment in the class body declares.

The most-read attribute in the package is the first one it missed, because it is
a property:

```python
@property
def n_wires(self) -> int:
    return self._n_wires

@n_wires.setter
def n_wires(self, value: int) -> None:
    self._set_declared_count("n_wires", value)
```

It is not visible to the parameter ledger either, because no user calls it with a
keyword. `TextDrawer().wire_order` is missed for a third reason: it is assigned to
`self` inside `__init__`, so it is neither an annotation nor a definition.
Measured on the WQ-1 base commit, adding the two missing kinds moves the surface
from 102 names to 144:

| Declaration kind | How a class declares it | Renameable | Payload key | Total |
|---|---|---:|---:|---:|
| field | annotation or assignment in the class body | 81 | 21 | 102 |
| member | property, method, or the setter beside it | 31 | 1 | 32 |
| instance | `self.name = ...` inside a method | 10 | 0 | 10 |
| **all** | | **122** | **22** | **144** |

The ten instance attributes are `MPLDrawer.n_wires`, `MPLDrawer.wire_map`,
`TextDrawer.n_wires`, `TextDrawer.show_all_wires`, `TextDrawer.wire_order`,
`TextDrawer.wire_map`, `TextDrawer.reverse_wire_map`,
`HttpQuantumProvider.default_n_wires`, `ShardedMPSState.n_wires`, and
`TensorNetworkState.dense_observable_wires`. None is a key of any payload, so all
ten are renameable and none needs an exclusion.

The same error hid ten module-level names, which are neither parameters nor class
attributes: `rank_for_wire`, `validate_observable_wires`,
`remap_instruction_wires`, `jax_basis_indices_for_wires`,
`communication_aware_wire_layout`, `native_cpu_rotation_tile_wires`,
`native_cpu_forward_rotation_tile_wires`, `build_two_wire_diagonal_chain`,
`fused_mps_wire_probabilities`, and `infer_n_wires_from_dense_state`. Renaming
their parameters and leaving their own names alone would leave a user looking at
`infer_n_wires_from_dense_state(qubits)`. All ten are functions; no class name
contains `wire`.

## Problem and affected user journey

The integral document's screen showed a result attribute spelled `wires`. The
member and instance surfaces show the same problem in places the user cannot
escape: `Circuit.n_wires` is read by every script that prints a circuit width, and
the one text-drawing entry point is reached through a parameter spelled
`wire_order`.

`fq` does not re-export the drawer: `hasattr(fq, "draw")` is `False`, so the
reproducer imports the real route, `flagquantum.drawer.draw`, rather than a
spelling that would raise `AttributeError` before demonstrating anything.

```python
import flagquantum as fq
from flagquantum.core.operator_schema import gate_info
from flagquantum.drawer import draw
from flagquantum.simulation.pauli import infer_n_wires_from_dense_state

circuit = fq.Circuit(n_qubits=2).h(0).cx(0, 1)
circuit.n_wires                              # property; "wire" comes back
draw(circuit, wire_order=(1, 0))             # a parameter (WQ-2 renames it)
fq.plan(circuit).analysis.n_wires            # CircuitAnalysis.n_wires, excluded
infer_n_wires_from_dense_state(amplitudes)   # a module-level function name
gate_info("h").n_wires                       # GateInfo field; not aliased
```

A user who greps the package for `n_qubits` and finds `Circuit.n_wires` next to
`Circuit.n_qubits` has learned that the two words are interchangeable, which is
the opposite of what the program is for. Rule 9 of `AGENTS.md` asks a stable name
to state its domain meaning directly; two spellings of one concept is the failure
that rule names.

## Naming

The integral document's rule is unchanged and is not restated per site: root
substitution, `wire → qubit`, `wires → qubits`. Applied to this surface:

| Old | New | Kind |
|---|---|---|
| `n_wires` | `n_qubits` | field, member, instance |
| `wires` | `qubits` | field |
| `wire`, `x_wires`, `z_wires`, `data_wires`, `ancilla_wires`, `cnot_wires`, `ancilla_wire` | `qubit`, `x_qubits`, … | field |
| `wire_convention`, `max_wire` | `qubit_convention`, `max_qubit` | field |
| `wire_order`, `show_all_wires` | `qubit_order`, `show_all_qubits` | parameter and instance |
| `wire_map`, `reverse_wire_map` | `qubit_map`, `reverse_qubit_map` | instance |
| `default_n_wires`, `dense_observable_wires` | `default_n_qubits`, `dense_observable_qubits` | instance |
| `rank_for_wire`, `remap_instruction_wires`, `infer_n_wires_from_dense_state` | root substitution inside the identifier | definition |

The four prohibited alternatives (`qubit_indices`, `qubit_ids`, `qubit_labels`,
`logical_qubits`) are checked against this surface as well, not only against
parameters. A class that avoided `n_wires` by declaring `n_qubit_indices` would
otherwise satisfy the letter of the migration and break its purpose.

## Alias policy: why attributes mostly do not get one

**Aliases are for the names a user already reaches through `fq.*`.** WQ-1's rule
was that an alias is owed wherever `AGENTS.md` rule 8 protects the name; the same
rule applied to attributes yields a much shorter list, because most of this
surface is not exported.

Measured by importing `flagquantum` and intersecting the live attribute scan with
`dir(fq)`, exactly seven wire-named attributes are reachable from the root
module. Four are payload keys this document excludes, so three take an alias, and
each alias belongs to the slice that owns the file:

| Owner | Slice | Old → New | Alias |
|---|---|---|---|
| `fq.Circuit` | WQ-2 | `n_wires` → `n_qubits` | **yes** — stays as a property that warns and forwards |
| `fq.OutputRequest` | WQ-2 | `wires` → `qubits` | **yes** |
| `fq.MeasurementResult` | WQ-3 | `wires` → `qubits` | **yes** |
| `fq.ExecutionPlan` | WQ-3 | `shardable_wires` → `shardable_qubits` | no — excluded payload key |
| `fq.RuntimePolicy` | WQ-3 | `observable_wires` → `observable_qubits` | no — the alias already shipped; `from_dict` still reads it |
| `fq.CircuitIR` | WQ-2 | `n_wires` → `n_qubits` | no — excluded IR key |
| `fq.Instruction` | WQ-2 | `wires` → `qubits` | no — excluded IR key |

`flagquantum.core.GateInfo.n_wires` is a near miss: it is a user-visible object
inside `fq.plan` results but is not reachable from `fq.*` (`hasattr(fq,
"GateInfo")` is `False`) and is frozen in no checked-in baseline, so it is renamed
outright rather than aliased. `fq.ExecutionResult` is
exported but declares no wire-named attribute, so it appears nowhere in this
table; `ExecutionOptions` and `TrainingResult` are in the same position, and
`qec/**`, `algorithms/**`, `drawer/**`, and `simulation/**` are renamed by root
substitution without aliases because no user reaches them through `fq.*`.

The forwarding property reuses the machinery `RuntimePolicy` already ships for
`observable_wires`: return the canonical attribute after
`warn_qubit_alias("n_wires", "n_qubits")`, whose message already names the removal
version. Removal is at **0.4.0**, the single window the eleven shipped parameter
aliases publish; opening a second window for attributes would put two deprecation
schedules in one release.

### A shimmed rename is a third state, and the ledger records it as one

`[attribute_ledger] canonical` records names nobody has renamed yet;
`[[attribute_retirement.sites]]` records names that are gone. A rename that keeps a
forwarder is neither, so WQ-2 adds `[[attribute_aliases.sites]]`, structurally the
same table as the parameter-side `[aliases] declared`:

```toml
[attribute_aliases]
removal_version = "0.4.0"
sites = []

[[attribute_aliases.sites]]
site = "flagquantum/circuit.py::Circuit::n_wires"
replacement = "n_qubits"
```

The gate reads three claims from it, all failing closed:

- the name must be a live declaration in the scan, so a slice cannot publish a
  removal window for a forwarder it never wrote;
- `replacement` must be `replacement_name(site)`, the same rule the parameter and
  retirement tables use;
- the name must not also appear in `[[attribute_retirement.sites]]`, because "gone"
  and "still reachable" cannot both be true.

An aliased name is **not** counted as retired: it stays in the ledger, stays in the
slice's `remaining` column, and the report prints the aliased total separately
(`... 0 of 122 attribute sites retired, 0 kept as deprecated aliases`). A reviewer
reading `remaining 28` for WQ-2 therefore learns that the rename is done and that
two of the old spellings deliberately still answer. When 0.4.0 deletes the three
forwarders, their rows move from the alias table to the retirement table and the
count reaches the baseline.

One obligation comes from the mechanism, not the ledger: a forwarding `@property`
is not a dataclass field. `OutputRequest.wires` is currently the stored field, so
becoming the forwarder moves it from `field` to `member`, and the slice's
`attribute_field_count`/`attribute_member_count` are re-measured from the scan once
in that commit. `fq.Circuit.n_wires` is already a property, so its kind does not
move.

## Why a member or an instance attribute can be a payload key at all

The field rule is structural: a class that declares a serialization method, or
calls `asdict(self)`, turns its own field names into keys, and `asdict` recurses
into field types. That rule cannot apply to a member or an instance attribute,
because `asdict` walks *fields* and neither kind is one — which is exactly why the
earlier class-level rule wrongly excluded `KrausChannel.n_wires` and
`MPSState.n_wires` (the latter is not even a dataclass).

So those two kinds use their own evidence rule: the spelling is a payload name
only when one of the **owning class's own serialization methods** contains it as a
string literal. Under that rule exactly one member is excluded —
`RuntimePolicy.observable_wires`, whose `from_dict` still accepts the legacy key —
and no instance attribute is. `Circuit.n_wires` is correctly *not* excluded even
though its spelling appears in the in-process `circuit_param` dict, because no
serializer of `Circuit` emits it.

## A generated document is not a payload

The exclusion test is "does renaming this spelling change what something outside
the process reads?" A file regenerated by an in-repo tool fails that test, because
CI fails loudly until it is regenerated, and the regeneration is part of the same
change. Two sites sit on that line and are deliberately **not** excluded:

- `OperatorSchema.wire_convention` is emitted as a literal key by
  `operator_manifest()`, which writes `docs/operator_manifest.json`. That function
  is a module-level builder, not a method of `OperatorSchema`, and the JSON is a
  build artifact: `.github/workflows/ci.yml` runs
  `python tools/operator_manifest.py --check`, so a rename that forgot to
  regenerate the file cannot merge. A runtime payload such as `CircuitIR` has no
  such check, which is why it is excluded and this one is not.
- `GateInfo.n_wires` is a plain field. Nothing serializes `GateInfo`, and it is
  not reachable from `fq.*` — `flagquantum.core` and `flagquantum.operators` are
  both importable paths but neither is an attribute of `fq`, so
  `hasattr(fq, "GateInfo")` is `False`.

## Compatibility

- **No Stable Core export changes.** `fq.__all__` keeps its membership;
  `docs/public_api_v1.json` freezes export names only.
- **`IR_VERSION` stays `"1.0"`.** The four IR keys were already excluded as
  parameters; the same four spellings are also *fields*, and they stay excluded
  for the same reason. `MeasurementContract.wires` joins them: it is not an IR
  key, but `ExecutionRecordContract` builds its payload with `dataclasses.asdict`,
  which recurses into it.
- **One member exclusion is a deprecation alias, not an IR key.**
  `RuntimePolicy.observable_wires` is read by `from_dict`, so its spelling is an
  accepted input key of the `flagquantum.runtime_policy` payload. It is already
  documented in `QUBIT_NAMING_MIGRATION.md`; it is recorded here because it is
  excluded by the member rule, not by an exception.
- **All 22 exclusions, with the payload each one belongs to.** Twenty-one are
  fields, one is a member, and none is an instance attribute — an instance
  attribute has never reached a payload:

  | Payload | Sites | Kind | Removal condition |
  |---|---|---|---|
  | FlagQuantum IR | `CircuitIR.n_wires` | field | `IR_VERSION` is raised and the payload migrates in the same release |
  | FlagQuantum IR | `Instruction.wires`, `MeasurementNode.wires`, `ObservableNode.wires` | field | same |
  | execution record | `MeasurementContract.wires` | field | the `ExecutionRecordContract` payload schema is versioned and migrated |
  | execution plan | `ExecutionPlan.shardable_wires` | field | the `ExecutionPlan` payload schema is versioned and migrated |
  | execution plan | `LayerPlan.wires`, `CircuitAnalysis.n_wires`, `CircuitAnalysis.wire_usage` | field | same |
  | noise model | `NoiseRule.wires`, `ReadoutRule.wires` | field | the `NoiseModel` payload schema is versioned and migrated |
  | device profile | `GateDuration.wires`, `QubitNoiseCalibration.wire` | field | the `DeviceNoiseProfile` payload schema is versioned and migrated |
  | runtime policy | `RuntimePolicy.observable_wires` | **member** | the `RuntimePolicy` payload schema is versioned and migrated |
  | scale profile | `FlagOSStatevectorScaleCase.n_wires`, `…persistent_wire_layout`, `FlagOSTrainingCase.n_wires` | field | the respective profile schema is versioned and migrated |
  | MPS reverse tape | `MPSReverseTapeRecord.wires` | field | the `MPSReverseTapeRecord` payload schema is versioned and migrated |
  | noise calibration | `NoiseSelectorCalibrationRecord.n_wires` | field | that record's schema is versioned and migrated |
  | simulation | `EvolutionPlan.n_wires`, `TEBDResult.n_wires` | field | the respective result schema is versioned and migrated |
  | twin prediction | `TwinPrediction.n_wires` | field | the `TwinPrediction` payload schema is versioned and migrated |

  Every row is a row, not a pattern: `n_wires` is a payload key in `CircuitIR` and
  an ordinary field in `DirectedCouplingMap`, which is why rule 8 of `AGENTS.md`
  forbids settling this with a suffix rule.
- **`qec/**` is renamed without aliases.** The domain belongs to the `core` team
  but the vocabulary belongs to this program; `qec` is not reachable from `fq.*`.

## Migration example

```python
import flagquantum as fq
from flagquantum.drawer import draw
from flagquantum.simulation.pauli import infer_n_qubits_from_dense_state

circuit = fq.Circuit(n_qubits=2).h(0).cx(0, 1)

width = circuit.n_wires        # DeprecationWarning: use n_qubits; removal 0.4.0
width = circuit.n_qubits       # the supported spelling

result = fq.run(circuit, outputs=fq.probabilities(qubits=(0, 1)))
result.wires                   # DeprecationWarning: use qubits
result.qubits                  # the supported spelling

draw(circuit, qubit_order=(1, 0))           # was wire_order
draw(circuit, show_all_qubits=True)         # was show_all_wires
psi = infer_n_qubits_from_dense_state(psi)  # was infer_n_wires_from_dense_state
```

## Deprecation and removal

| Item | Value |
|---|---|
| First deprecation version | 0.3.x (current development line; `warn_qubit_alias` already ships this text) |
| Planned removal version | 0.4.0, the same window as the eleven shipped parameter aliases |
| Alias coverage | the three reachable attribute owners in *Alias policy*; everything else is renamed in place |
| Compatibility tests | `tests/unit/test_qubit_vocabulary_contract.py` (ledger conformance and mutation), `tests/unit/test_census_wire_vocabulary.py` (scanner) |

## Documentation and tooling impact

- `tools/census_wire_vocabulary.py` gains two surfaces and one declaration kind:
  members, module-level definitions, and `self.name = ...` instance attributes.
  The module docstring now names three attribute doors instead of two.
- `tools/check_qubit_vocabulary.py` gains `_definition_errors` and reconciles the
  attribute surface **per declaration kind**, so a field converted into a property
  moves obligation without changing a total and without going unnoticed.
- `contracts/qubit-vocabulary-contract.toml` gains `[attribute_ledger]`,
  `[attribute_exclusions]`, `[attribute_retirement]`, `[attribute_aliases]`,
  `[definition_ledger]`, `[definition_retirement]`, and
  per-slice `attribute_field_count` / `attribute_member_count` /
  `attribute_instance_count` / `definition_count`.
- No change to `.github/workflows/ci.yml` or `tools/pre_push.py`: the gate is
  already wired in as a step of the existing `quality` job, and this extension adds
  no job, so the required-check roster is unchanged.

## Owner and approvals

Owner: `integration` (contract, census, gate) with WQ-2 as the implementing slice.
Authorization: the user's whole-surface constraint, restated for this surface by
this document. Review requirements: the nine items listed in
[`PUBLIC_API_PROTECTION.md`](../development/PUBLIC_API_PROTECTION.md), all of which
appear above, plus the acceptance tests below.

## Acceptance Tests

1. `python tools/check_qubit_vocabulary.py` exits 0 and reports
   `0 of 122 attribute sites retired, 0 kept as deprecated aliases` and
   `0 of 10 definition names retired` alongside the parameter totals, with WQ-2 at
   28 attribute sites.
2. The gate fails when a `wire`-named field, member, instance attribute, or
   module-level definition is added to the package, demonstrated against fixture
   packages that declare one of each.
3. The gate fails when an exclusion row's witness, evidence, or declaration kind
   no longer matches the scan — the check that makes a rename impossible while the
   name is still a payload key.
4. The gate fails when a slice's `attribute_field_count`, `attribute_member_count`,
   `attribute_instance_count`, `definition_count`, or `canonical_count` disagrees
   with the ledger.
5. A field converted into a property, or a field that stops being serialized, is
   reported rather than silently reclassified; a member of a private class is not a
   site, so privacy stays private all the way down.
6. `tools/validate_required_checks.py` still passes: this change adds steps and
   tests, never a job.
7. Running the repository's public-API gates after the WQ-2 rename reports no
   Stable Core diff beyond the authorized export-adjacent changes.
8. An aliased row is rejected unless the old spelling is still a live declaration,
   forwards to `replacement_name`, and is absent from the retirement table; the
   report prints the aliased total apart from the retired total, so a shimmed
   rename is never reported as finished work.

Status: proposed. The measured numbers are the WQ-1 base commit's; the rename is
WQ-2's, and no other slice may retire a site this ledger assigns to WQ-2.
