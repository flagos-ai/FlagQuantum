# API Change Proposal 067: The attribute and definition name surfaces

## Status

**Proposed.** This proposal extends
[Proposal 065](API_CHANGE_PROPOSAL_065_QUBIT_VOCABULARY.md) from the parameter
surface to the two name surfaces it did not gate. The authorization record is
[`FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md`](../api-changes/FQ-QUBIT-VOCABULARY-ATTRIBUTES-20261006.md),
which answers
[`FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md`](../api-changes/FQ-QUBIT-VOCABULARY-INTEGRAL-20261005.md)
Open Question 1.

It proposes no capability, no export, no result field, no default, and no
serialized schema change. `IR_VERSION` stays `"1.0"`. It is a rename program's
second half: the integral document migrated the names a user *passes*, and this
one covers the names a user *reads back* and the names a user *imports*.

## 1. What Proposal 065 measured and what it left open

Proposal 065's `[boundary]` defines the program as "every `wire`-named parameter
on the package's public function surface", and the integral document measured that
surface at 341 sites. It also recorded, honestly, two measured gaps it did not
gate:

- **102 wire-named public attribute names**, reported as an owned gap because the
  parameter ledger cannot see `MeasurementResult.wires`;
- **string literals**, 1291 of them, reported and never ledgered.

It did not measure module-level definition names at all.

The gap is not cosmetic. `fq.Circuit.n_wires` is the property every script reads
to print a circuit width, and the integral document's own migration example
(`circuit.n_qubits` in, `result.wires` out) is inconsistent while it stands: a
user renames their input and gets the old noun back.

## 2. Why the attribute count is 144 and not 102

The 102 was measured with a scan that read annotated assignments in public class
bodies. That finds every **field** and nothing else. Two other doors put a name on
the user's screen:

| Door | Example | Seen by a field scan? | Seen by the parameter scan? |
|---|---|---|---|
| field — annotation or assignment in the class body | `MeasurementResult.wires` | yes | no |
| member — property, method, or the setter beside it | `Circuit.n_wires` | no | no |
| instance — `self.name = ...` inside a method | `TextDrawer().wire_order` | no | no |

`Circuit.n_wires` is a property:

```python
@property
def n_wires(self) -> int:
    return self._n_wires

@n_wires.setter
def n_wires(self, value: int) -> None:
    self._set_declared_count("n_wires", value)
```

A property and its setter are one user-visible name, so the census counts one site
there, not two. That detail is a rule, not an accident: `_declared_attributes`
keeps a per-class `seen` set so the two accessors cannot inflate the ledger.

Measured at the WQ-1 base commit, the corrected surface is:

| Declaration kind | Renameable | Payload key | Total |
|---|---:|---:|---:|
| field | 81 | 21 | 102 |
| member | 31 | 1 | 32 |
| instance | 10 | 0 | 10 |
| **all** | **122** | **22** | **144** |

Ten module-level definition names were invisible to all three doors because they
are neither parameters nor class attributes:
`rank_for_wire`, `validate_observable_wires`, `remap_instruction_wires`,
`jax_basis_indices_for_wires`, `communication_aware_wire_layout`,
`native_cpu_rotation_tile_wires`, `native_cpu_forward_rotation_tile_wires`,
`build_two_wire_diagonal_chain`, `fused_mps_wire_probabilities`, and
`infer_n_wires_from_dense_state`. All ten are functions; no public class name
contains `wire`.

This is the same class of error as the four corrections the integral document
records: the ruler was narrower than the surface, and the number it produced read
like a fact about the package.

## 3. Why `AGENTS.md` rule 8 requires a proposal for this

Rule 8 protects "stable exports, signatures, defaults, result fields, enum/Literal
values, documented exception behavior, or serialized public schemas". Two of those
phrases reach this surface:

- **result fields.** `fq.MeasurementResult.wires` is a documented result field of
  a `fq.*` export. It is renameable, and the rename is the change this proposal is
  for.
- **serialized public schemas.** Twenty-one fields on this surface are keys of a
  payload that leaves the process, and one member is an accepted input key. Rule 8
  forbids moving them, and rule 8's companion — `PUBLIC_API_PROTECTION.md` — forbids
  quietly *not* moving them either. So each is recorded as a named exclusion with a
  witness and a removal condition.

The remaining 122 names are on this surface but are not exported from `fq.*`
(`fq.core.GateInfo`, `qec/**`, `drawer/**`, `simulation/**`). Rule 9, "stable API
names must state their domain meaning directly", is the rule that reaches them: a
name that says `wire` in a package whose every parameter says `qubit` is exactly
the avoidable ambiguity rule 9 was written against.

## 4. Relationship to engineering decision principle 1 (no permanent debt)

An exclusion is not a licence. Principle 1 requires a compatibility adapter to
have "a named owner, documented scope, removal condition, and target version", and
the contract gives each row all four:

| Part | Where it lives |
|---|---|
| owner | the slice that owns the file (`WQ-2` … `WQ-8`) |
| scope | one payload key, named by `path::Owner::attribute` |
| removal condition | "`IR_VERSION` is raised" or "the `<Payload>` schema is versioned and migrated" |
| target version | the release in which that schema bump lands |

Nothing is excluded because it is hard. Each exclusion is the **conservative**
direction: renaming a payload key without version negotiation would change an
on-disk format silently, and 17 of the 22 spellings are fields of a dataclass
frozen in `contracts/public-api-v0.2-baseline.json`. The four that are not frozen
are plan-JSON and `asdict` layouts; they are excluded for the same reason, because
they still cross a process boundary.

The gate enforces the conservative direction in both ways. A live payload key that
is missing from the exclusion list is an error, and an exclusion whose witness has
stopped reading the spelling is *also* an error ("stale: its class no longer builds
a payload from that name"). An exclusion cannot be parked.

## 5. Alias policy

An alias is owed where rule 8 protects the name, which on this surface means the
names reachable from `fq.*`. Intersecting the live scan with `dir(fq)` yields
exactly seven, of which four are excluded payload keys:

| Owner | Old → New | Alias |
|---|---|---|
| `fq.Circuit` | `n_wires` → `n_qubits` | yes, forwarding property with `DeprecationWarning` |
| `fq.MeasurementResult` | `wires` → `qubits` | yes |
| `fq.OutputRequest` | `wires` → `qubits` | yes |
| `fq.core.GateInfo` | `n_wires` → `n_qubits` | no — not reachable from `fq.*`, and frozen in no baseline |
| `fq.CircuitIR`, `fq.Instruction` | `n_wires`, `wires` → `n_qubits`, `qubits` | no — excluded payload keys |

The mechanism already exists: `flagquantum/core/_qubit_aliases.py` provides
`warn_qubit_alias(old, new)`, and `RuntimePolicy.observable_wires` already ships
the forwarding-property pattern this proposal reuses. Removal is at **0.4.0**, the
same window the eleven shipped parameter aliases publish. One program, one
deprecation window.

## 6. Why `IR_VERSION` does not change

The integral document's section 4 already answers this for parameters, and the
argument is unchanged: `IR_VERSION` describes the payload's *shape*, and no shape
changes. Every `wire`-named key keeps its spelling, and the four locks on the IR
payload (`CircuitIR.n_wires`, `Instruction.wires`, `MeasurementNode.wires`,
`ObservableNode.wires`) are re-recorded here as attribute exclusions rather than
parameter exclusions, because the same spelling is both.

The one new payload decision is `MeasurementContract.wires`. It is not an IR key,
but `ExecutionRecordContract` builds its payload with `dataclasses.asdict`, which
recurses into it, so it is a key of a payload that leaves the process. It is
excluded, with `ExecutionRecordContract` as its witness.

## 7. Tooling: what makes this checkable

The program is only credible if the two new surfaces are gated, not described.

- `tools/census_wire_vocabulary.py` scans **four** surfaces and names each reason:
  parameters, attribute names (three declaration kinds), module-level definition
  names, and message strings (reported, never ledgered). The module docstring
  records the measurement error that moved 102 → 144, and the one that shows why
  members need their own evidence rule.
- `tools/check_qubit_vocabulary.py` reconciles the contract against a live scan
  and reports progress per slice **per surface**: 341 parameter sites, 122
  attribute sites, 10 definition names. A slice is finished when all three of its
  counters reach zero.
- `contracts/qubit-vocabulary-contract.toml` carries `[attribute_ledger]`,
  `[attribute_exclusions]`, `[definition_ledger]`, `[definition_retirement]`, and
  per-slice `attribute_field_count`, `attribute_member_count`,
  `attribute_instance_count`, `definition_count`.
- Both tools stay wired as **steps** of the existing `quality` job and inside
  `tools/pre_push.py`. No job is added, so
  `tools/validate_required_checks.py`'s six externally configured required checks
  are untouched.

## 8. Evidence rules, per declaration kind

One rule for the whole surface would be wrong, and the measurement shows in which
directions. The field rule is structural: a class that declares a serialization
method, or calls `asdict(self)`, turns its own field names into keys, and `asdict`
recurses into field types (`contained`). Applying that rule to members excluded
`KrausChannel.n_wires` and `MPSState.n_wires`, neither of which can leave the
process — `MPSState` is not even a dataclass. Applying the textual rule instead
("the spelling appears as a string literal somewhere") marked 85 of 102 names as
payload keys and froze 65 that carry no obligation.

So members and instance attributes use their own rule: the spelling is a payload
name only when one of the **owning class's own serialization methods** contains it
as a string literal. Under that rule exactly one is excluded,
`RuntimePolicy.observable_wires`. `Circuit.n_wires` is correctly *not* excluded
even though its spelling appears in the in-process `circuit_param` dict, because no
serializer of `Circuit` emits it.

Definition names need no exclusion rule. A module-level name is reachable only as
a Python identifier; the four places the package spells one as a literal are a
source-level kernel-catalog `symbol` field and two `__all__` lists, and the same
rename updates all four.

## 9. Migration path

One slice, one PR, cut from `main` after WQ-1 landed. WQ-2 renames its 16
parameters, 21 attributes, and 7 instance attributes, and keeps the three aliases.
Later slices retire their own rows.

```python
# before
circuit = fq.Circuit(n_wires=2)
print(circuit.n_wires)
result = fq.run(circuit, outputs=fq.probabilities(wires=(0, 1)))
print(result.wires)
fq.draw(circuit, wire_order=(1, 0), show_all_wires=True)

# after (0.3.x, with deprecation warnings on the alias paths)
circuit = fq.Circuit(n_qubits=2)
print(circuit.n_qubits)
result = fq.run(circuit, outputs=fq.probabilities(qubits=(0, 1)))
print(result.qubits)
fq.draw(circuit, qubit_order=(1, 0), show_all_qubits=True)

# 0.4.0: `circuit.n_wires` and `result.wires` raise AttributeError
```

## 10. Open questions

1. **Should `fq.core.GateInfo.n_wires` be renamed outright rather than aliased?**
   It is not reachable from `fq.*` and is frozen in no checked-in baseline, so no
   compatibility obligation exists. The proposal's answer is yes, rename it — but
   the answer is recorded here rather than assumed, because `GateInfo` is a
   user-visible object inside `fq.plan` results.
2. **Are the 1291 string literals in scope?** They stay reported and never
   ledgered: a literal scan cannot tell a serialized key from a refusal sentence.
   WQ-2's third commit sweeps the ones its own files own, and the rest are recorded
   as a measured-but-ungated surface.
3. **What removes the 22 exclusions?** Each names its own condition, and the
   removal itself is a separate proposal per payload — this one only promises that
   the debt is visible and cannot grow silently.

## Verification

```bash
python tools/check_qubit_vocabulary.py
# Qubit vocabulary contract passed: 0 of 341 baseline sites retired
#   WQ-2 ... remaining 16
# Qubit vocabulary contract passed: 0 of 122 attribute sites retired
#   WQ-2 ... remaining 28
# Qubit vocabulary contract passed: 0 of 10 definition names retired
#   WQ-4 4  WQ-6 3  WQ-7 1  WQ-8 2

python -m pytest tests/unit/test_census_wire_vocabulary.py \
    tests/unit/test_qubit_vocabulary_contract.py -q
# 111 passed

python tools/validate_required_checks.py
# validated 6 externally configured required checks
```

Every number in this proposal is produced by those commands at the WQ-1 base
commit and is reconciled by the gate on every run; none is transcribed by hand.
