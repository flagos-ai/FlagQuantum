# API Change Proposal 066: Construction-time circuit composition

## Status

**Proposed.** This proposal authorizes **two new Stable Core capabilities**,
`Circuit.control` and `Circuit.power`, and records them as members of one
construction-time composition family together with the two members that have
already shipped, `Circuit.compose` and `Circuit.adjoint`. The two shipped members
are re-examined here as the mechanism evidence for the family, not re-authorized;
Section 1.1 states which document owns which member.

It adds no root export, no result field, no default on an existing signature, and
no serialized schema change. `IR_VERSION` stays `"1.0"`. Measured on this checkout:
`fq.__all__` is 36 names and `compose` is not one of them, because all four
operations are methods on a type that is already a stable export.

The authorization records it answers to are
[`FQ-CIRCUIT-COMPOSITION-20261002.md`](../api-changes/FQ-CIRCUIT-COMPOSITION-20261002.md)
(placement) and
[`FQ-CIRCUIT-ADJOINT-20261003.md`](../api-changes/FQ-CIRCUIT-ADJOINT-20261003.md)
(inversion). The machine-checked half is
[`contracts/circuit-composition-contract.toml`](../../contracts/circuit-composition-contract.toml),
whose `[scope]` section already names the two missing operations and states why
they are missing:

```toml
not_provided = ["Circuit.control", "Circuit.power"]
not_provided_reason = "no approved API change proposal; control and power are unplanned, not partial"
```

This document is that proposal. It is written under non-negotiable rule 8 of
`AGENTS.md` ("Treat the Stable Core public API as protected"), and it answers the
four questions the alignment plan asked of `N1-11`.

## 1. What already exists, and what is actually being authorized

The family has four members and two of them have shipped. Stating that plainly
matters, because a proposal that reads as if it were authorizing all four would be
claiming authority over history it does not have.

| Method | State on this checkout | Authority |
|---|---|---|
| `Circuit.compose` | implemented, contracted, tested | `FQ-CIRCUIT-COMPOSITION-20261002.md` |
| `Circuit.adjoint` | implemented, contracted, tested | `FQ-CIRCUIT-ADJOINT-20261003.md` |
| `Circuit.control` | **absent** — `AttributeError` | **this proposal** |
| `Circuit.power` | **absent** — `AttributeError` | **this proposal** |

```
>>> fq.Circuit(2).h(0).control(1, ctrl_qubits=(0,))
AttributeError: 'Circuit' object has no attribute 'control'
>>> fq.Circuit(2).h(0).power(2)
AttributeError: 'Circuit' object has no attribute 'power'
```

The two shipped methods are re-examined here rather than merely cited, because
the question this proposal has to answer first is not "should `control` exist" but
"is the *family* one mechanism". If the family is one mechanism then the two
shipped members are its evidence, and the two proposed members are its
continuation. If it is not, then `control` and `power` need their own proposals
and their own justification. Section 2 answers that with measurement.

### 1.1 A numbering collision this proposal has to repair

`FQ-CIRCUIT-COMPOSITION-20261002.md` forward-references the proposal it expected
to be written for it, in two places:

```text
docs/api-changes/FQ-CIRCUIT-COMPOSITION-20261002.md:12
docs/api-changes/FQ-CIRCUIT-COMPOSITION-20261002.md:239
```

Both name `API_CHANGE_PROPOSAL_065_CIRCUIT_COMPOSITION.md`. That number is
**taken**: `API_CHANGE_PROPOSAL_065_QUBIT_VOCABULARY.md` is the shipped
authorization for the `wire` → `qubit` rename program, an unrelated decision, and
`067` is the shipped authorization for the attribute half of that same rename. A
reader following the composition record's own instruction would arrive at a
document about vocabulary, not about composition.

The numbers in this series are issued in dependency order rather than by plan
table row, so the free number is `066`. This proposal is written as `066` and the
two references above are repointed to it as part of the same change. That is a
link repair and not a rewrite of recorded approval status: the referenced
document did not exist when the record was written, and the record's statement
that composition was authorized by "a proposal to be written" is unchanged.

`065`'s own text is not touched, and neither is `067`'s.

## 2. Question 1 — why this is construction-time composition and not a second IR

The claim is that all four operations are **producers of the program description
the repository already has**, not a new way to describe a program. That is
checkable, and it was checked.

Measured on this checkout, for a program composed from two blocks and compared
against the same program built by appending instructions by hand:

| Claim | Measurement | Result |
|---|---|---|
| The result is the existing IR type | `type(c.to_ir()).__name__`, `isinstance(..., CircuitIR)` | `CircuitIR`, `True` |
| The serialized document is the same document | `composed.to_ir().to_dict() == hand.to_ir().to_dict()` | `True` |
| The document's top-level vocabulary is unchanged | `sorted(to_dict())` | `['dtype', 'instructions', 'kind', 'measurements', 'metadata', 'n_wires', 'observables', 'shape', 'version']` |
| The instruction vocabulary is unchanged | `sorted(to_dict()["instructions"][0])` | `['matrix', 'metadata', 'opcode', 'params', 'wires']` |
| The version pin is unchanged | `IR_VERSION`, `to_dict()["version"]` | `1.0`, `1.0` |
| The framework-neutral identity is unchanged | `semantic_fingerprint(composed)` vs `semantic_fingerprint(hand)` | both `e1a37ba73d1df83ea41338f74df8a6b7f4ef653619f4050bda177a312abecdd0` |
| Composition adds no opcode | composed opcodes vs the 35 registered opcodes | `{'h', 'cx', 'rz'} ⊆ registry`; `{'compose', 'adjoint'} ∩ registry = {}` |

The fingerprint row is the load-bearing one. `semantic_fingerprint` hashes
`ir_version`, `n_wires`, `instructions`, `observables`, and `measurements` and
excludes transport provenance, so two programs collide under it exactly when they
are the same executable program. Composition collides with the hand-built
program, which is the statement "composition is a way to build a program, not a
way to represent one".

`control` and `power` are bound by the same evidence requirement, and that binding
is the substantive decision in this document: **neither may introduce a new
opcode or a new instruction field.** A controlled arbitrary unitary has no opcode
today, and a fractional power of a non-parameterised gate has no opcode today. The
tempting resolutions — a `cu` opcode, a nesting node, a `power` field on
`Instruction` — are all second representations, and each one would have to be
taught to every executor, compiler pass, drawer, and importer. The contracted
resolution is the one `N1-8` already wrote into the contract for the sibling
methods: an operation that cannot be expressed with the opcodes the IR already
has refuses with `CapabilityError` naming the opcode and its qubits, instead of
being represented somehow.

That is a real limitation and this proposal owns it rather than hiding it in a
mechanism: `control` will be able to condition operations whose controlled form
the IR can already express, and will refuse the rest. Widening it later is a
separate proposal, because widening it means adding an opcode.

## 3. Question 2 — the relationship to `AGENTS.md` rules 6 and 8

### 3.1 Rule 6 (keep the user API simple)

Rule 6 requires normal usage to stay centred on `fq.Circuit`, `fq.Module`, `run`,
`plan`, training loops, and deployment packages. All four operations are methods
on `fq.Circuit`, so they add no entry point, no namespace, and no root export:
`fq.__all__` stays at 36 measured names. A user who never composes a program
never meets any of the four.

The rule is served rather than merely respected, because the alternative is worse
for the ordinary user. `qft(3)` returns a three-qubit circuit; a user of a
five-qubit program who wants it at qubits 1..3 has, today, exactly one option, and
it is this:

```python
for instruction in qft(3).to_ir().instructions:
    target.gate(
        instruction.name,
        tuple(qubit + 1 for qubit in instruction.wires),
        params=instruction.params,
    )
```

That loop is four lines of instruction-level plumbing in the middle of what is
otherwise a high-level script, and it is wrong in two measured ways: an
arbitrary-unitary instruction does not survive it at all, and a channel
instruction survives while silently losing `metadata == {"is_channel": True}`.
A method on `Circuit` removes the loop; a helper somewhere else would leave the
user writing it.

### 3.2 Rule 8 (Stable Core protection)

Rule 8 protects *stable exports, signatures, defaults, result fields,
enum/Literal values, documented exception behavior, and serialized public
schemas*. Measured against that list:

- `docs/public_api_v1.json` freezes the root export set and enumerates no
  methods, so adding a method to `Circuit` adds no frozen name.
- `Circuit` is already a stable export, so no export is added or removed.
- No default, result field, enum value, or serialized schema changes — Section 2.
- Two of the four already exist with recorded authorization.

What rule 8 does cover, and what this proposal is therefore for, is that
`docs/development/PUBLIC_API_PROTECTION.md` lists seven requirements for an
additive stable API. Recorded honestly, with the owner of each:

| # | Requirement | State |
|---|---|---|
| 1 | a concrete user journey | Section 5, four journeys |
| 2 | a reason it belongs in Stable Core rather than a namespace or experimental | Section 5.1 |
| 3 | typing and documentation | owned by `N1-4`/`N1-5` |
| 4 | executable behavior contracts | `[scope]`, `[placement]`, `[adjoint]`, and the 17 `[[refusals]]` rows of the composition contract; extended in the same change as each method |
| 5 | API-owner approval | **not given by this document** — see Section 8 |
| 6 | release-note entry | owned by `N1-5`, landed with `power` |
| 7 | updated machine-readable contract after approval | the same change that implements each method |

Rule 8 also forbids the inverse: never update a contract or snapshot merely to
make tests pass. The composition contract's `not_provided_reason` is **not**
edited by this proposal. It is edited by the slice that actually implements each
method, and that edit is legitimate precisely because an approved proposal will
exist at that point — which is what the current text says is missing. Changing it
now, before either method exists, would be deleting an accurate statement to make
a plan look further along.

### 3.3 Rule 9 (names state their domain meaning)

`compose`, `adjoint`, `control`, and `power` are the domain terms. PennyLane
spells the last two `qml.ctrl` and `qml.pow`, and the abbreviations are rejected
here: `control` and `power` are the nouns for the operations, the vocabulary is
closed, and an abbreviation is a second spelling of one concept — the same defect
the vocabulary program exists to remove.

The user-side vocabulary constraint applies to the parameters as well. A control
index is a qubit, so it is spelled with `qubit`; `ctrl_wires` and the
PennyLane-style `work_wires` are not admissible spellings. See Section 4.

## 4. Question 3 — why `IR_VERSION` stays `"1.0"`

`flagquantum/core/ir.py` pins `IR_VERSION = "1.0"`, and the serialized payload
uses `n_wires` and `wires` as **keys**:

```python
fq.Circuit(2).h(0).to_ir().to_dict()
# {'kind': 'flagquantum.circuit_ir', 'version': '1.0', 'n_wires': 2, ...
#  'instructions': [{'opcode': 'h', 'wires': [0], 'params': {}, 'matrix': None,
#                    'metadata': {}}], ...}
```

`n_wires` there is a string in a document, not a parameter name. Changing it is a
schema change, and a schema change requires a version bump, a migration path for
readers of the old schema, and a compatibility window.

All four operations leave that document alone, and Section 2 measures it:
composed and hand-built programs serialize to equal dicts, with the same
top-level keys, the same instruction keys, and the same `version`, and they hash
to the same framework-neutral fingerprint. The four operations are functions from
programs to programs; the set of documents a program can serialize to is
unchanged, so there is nothing for the version to describe.

The constraint this puts on `control` and `power` is the same one Section 2
derives from the "no second IR" requirement, arriving from the other direction:
`IR_VERSION` would have to move if either operation could produce an instruction
the current schema cannot express. It cannot, because it refuses instead. This is
the reason the refusal class is `CapabilityError` and not a fallback
approximation — a fallback would produce a document that lies about which program
was run, which under `AGENTS.md` engineering decision principle 9 (fail closed and
make degradation observable) is worse than a refusal.

The same reasoning is why the vocabulary program's migration
([Proposal 065](API_CHANGE_PROPOSAL_065_QUBIT_VOCABULARY.md)) excludes
`CircuitIR.n_wires` and `Instruction.wires` by name rather than renaming them.
The two decisions are the same decision: a serialized key is an exact-match
version pin.

## 5. Question 4 — why the Stable Core needs these four methods

Four user journeys, all of them program construction, none of them belonging
anywhere but on the type that owns the instruction list.

1. **Place a reusable block on chosen qubits.** `compose` — one of the two shipped
   members (Section 1). A five-qubit program that wants a three-qubit block at
   qubits 1..3 has no method today and must rewrite instruction labels by hand.
2. **Undo a block.** `adjoint` — the other shipped member. Training against an
   inverse, a Loschmidt echo, and undoing a state-preparation block all need "the
   same gates, backwards, each one inverted". The per-opcode rule was declared in
   three places before this family existed.
3. **Apply a block conditioned on other qubits.** `control`. When a controlled
   operation is not one of the named gates, the user is stuck: `Circuit` has no
   attribute containing `contr` at all (measured: the list is empty), so the only
   routes are the named `ccx` and `cswap` opcodes, or a private helper reached by
   full module path:

   ```python
   from flagquantum.algorithms.primitives.oracle import append_multi_controlled_x
   append_multi_controlled_x(circuit, controls=[0, 1], target=2)   # -> ccx
   ```

   That helper is a ladder for one specific opcode, and it is in `algorithms/`,
   which is not an attribute of `fq` — so a user of the Stable Core surface cannot
   reach it or the two others like it without knowing an internal path.

4. **Apply a block to a power.** `power`. This one has the strongest evidence that
   the operation is Core semantics rather than an algorithm's business, because
   the repository already contains three independent implementations of it with
   the same signature and a fourth declaration of that signature:

   | File | Kind |
   |---|---|
   | `flagquantum/algorithms/primitives/types.py:45` | the declared interface, `apply_power_controlled(self, circuit, control, qubits, power)` |
   | `flagquantum/algorithms/amplitude_estimation.py:126` | an implementation |
   | `flagquantum/algorithms/svd.py:390` | an implementation |
   | `flagquantum/algorithms/pca.py:355` | an implementation |

   Three implementations of "append this operator raised to `power`, controlled on
   `control`", each with its own docstring explaining the same thing. That is the
   same shape of defect that `adjoint` closed for three private copies of the
   inverse rule, and it is the replacement evidence principle 10 asks for: the
   boundary is demonstrated by replacing implementations, not by declaring one.

### 5.1 Why Stable Core rather than a namespace or experimental

Because the operation is a property of `Circuit`, which owns the instruction list.
Three candidate alternatives were considered and rejected in the composition
record for `compose`, and the same reasoning applies to `control` and `power`:

- **A free function that writes into a caller's circuit.** Puts the operation
  outside the object that owns the instruction list, so `c.control(...)` — the
  form every comparable framework offers — stays impossible, and the chaining
  style the rest of the API uses (`fq.Circuit(5).h(0).compose(...)`) breaks at
  exactly the point where programs start being reused.
- **A per-factory argument.** `qft(n, *, qubits=None, power=None, control=None)`
  fixes one factory and not the problem: any circuit the user already has — built
  by hand, loaded from OpenQASM, or returned by another factory — stays
  uncontrollable.
- **A namespace or experimental module.** `fq.compiler`, `fq.algorithms`,
  `fq.simulation`, and the rest are not attributes of `fq` at all — measured,
  `fq.algorithms` raises `AttributeError` — so a helper there is not on the Stable
  Core surface and cannot appear in a user's ordinary script. Neither
  `append_multi_controlled_x` nor any of the three `apply_power_controlled` copies
  is reachable from `fq.*` today.

The three rejected alternatives share one property: each one turns one semantic
rule into two implementations. That is the defect the family exists to close.

## 6. What `control` and `power` have to satisfy

This section states acceptance criteria rather than committing to an
implementation. Each criterion is either measurable now or is stated as a
decision the implementing slice must settle in its own change; naming a detail
here that the implementation has not measured would be a second source of truth
for it.

### 6.1 `Circuit.control`

Proposed shape, following the plan's `N1-4`:

```python
Circuit.control(n_controls: int, ctrl_qubits: Sequence[int]) -> "Circuit"
```

Semantics that must hold, whatever the mechanism:

- it returns a **new** `Circuit` and does not modify the receiver, so
  `Circuit.control` differs from `Circuit.compose`, which its own contract records
  as `mutates_receiver = true`. The reason is that `control` widens the program:
  a `Circuit(1).x(0).control(1, ctrl_qubits=(1,))` must report width 2. Growing the
  receiver in place would silently change the width of an object the caller may
  have shared or already serialized. Whichever way `N1-4` settles this, the
  contract must state it as explicitly as `[placement]` states it for `compose`;
- the result's width covers every qubit the operation names, and
  `to_ir().n_wires` reports it;
- every emitted instruction has an opcode in the 35-opcode registry, and
  `to_dict()["version"]` is still `"1.0"`;
- applying the controlled program to a basis state reproduces the receiver's
  action exactly when all control qubits are set, and the identity when any is
  not. This is the acceptance test, and it is measured against a statevector
  simulation rather than against an instruction listing, because an instruction
  listing cannot distinguish a correct ladder from a wrong one.

Refusals that must be contracted, in the shape `N1-8` established (exception class
plus a frozen message phrase, with a reachability flag):

| Condition | Class | Why it must refuse rather than approximate |
|---|---|---|
| a channel instruction | `CapabilityError` | a channel has no controlled form the IR can express |
| a dynamic operation | `CapabilityError` | not a unitary, so there is nothing to condition |
| a classically conditioned instruction | `CapabilityError` | two condition sources on one instruction is a different semantics |
| an opcode with no controlled form the IR can express | `CapabilityError` | this is the boundary Section 2 draws |
| fewer than one control | `ValidationError` | a zero-controlled operation is the operation itself, which has a name |
| a control qubit that is also a receiver qubit | `ValidationError` | the block would be conditioned on a qubit it writes |
| a repeated control qubit | `ValidationError` | silently collapses to fewer controls |
| a non-integer or out-of-range control qubit | `ValidationError` | the same reading rule `compose` already applies |

Decisions `N1-4` owns and this proposal does not make: whether the ancilla ladder
precedent in `append_multi_controlled_x` is reused or a per-opcode controlled form
is emitted; whether an ancilla argument is part of the signature or is allocated
from the circuit's free qubits; and what the exact refusal phrase is for each row.
An ancilla argument, if it exists, is a qubit sequence and is spelled with
`qubit`.

**Settled by `N1-4`**, recorded in
[`FQ-CIRCUIT-CONTROL-20261022.md`](../api-changes/FQ-CIRCUIT-CONTROL-20261022.md), and
now the reading of this section:

| Decision | Settled reading | Evidence |
|---|---|---|
| mechanism | each opcode declares its controlled form in `flagquantum/core/operator_schema.py` as a rule name, and `flagquantum/core/controlled.py` emits the resulting instruction sequence | 31 of the 35 registered opcodes declare a rule; the four that do not are exactly the channels |
| ancilla ladder precedent | **not** reused. `flagquantum/algorithms/primitives/oracle.py` is outside `core` and `append_multi_controlled_x` returns a wrong answer on a dirty ancilla, which was measured: 8 of 32 operands at three controls, 32 of 128 at four, with the ancilla itself restored | the emitter is ancilla-free, and the contract records `ancilla_qubits = 0` |
| ancilla argument | **declined**, because the chosen mechanism needs none | no `ancillas=` keyword exists; the declined candidate is recorded in the record's Decision Candidates |
| exact refusal phrases | 13 rows, all reachable, frozen in `[[refusals]]` of `contracts/circuit-composition-contract.toml` and read back by `tools/check_circuit_composition_contract.py` | the contract holds 30 refusal rows, of which 29 are reachable |
| the identity opcode | `i` under any number of controls emits **no** instruction, which is a rule about that one opcode and not a no-op path for the operation: `n_controls` below one is still refused, and a control label that is not outside the receiver is still refused | Section 7's prohibition is about the operation, not about the one gate whose controlled form is empty |

The acceptance test in the fourth bullet is measured twice: once as a simulated operator
comparison over all 31 unitary opcodes at one to four controls — 124 cases, which is where
the diagonal ladder, the basis change, the target's own arity and the phase correction all
have to be right at once — and once as an instruction-for-instruction comparison against a
hand-built ladder. The second is what makes the first load-bearing, because an operator
comparison alone cannot separate a correct expansion from a differently-written one that
happens to agree at the probe angle.

One cost this section did not anticipate, and which `N1-4` had to publish rather than
absorb: an ancilla-free ladder is **exponential** in the control count. A `w`-qubit block
under `k` controls needs a ladder of level `k + w - 1`, which emits
`4 * 3 ** (level - 1) - 3` instructions at its minimum — 1, 9, 33, 105, 321, 969, 2913,
8745, 26241, 78729 at levels one through ten. `MAX_LADDER_LEVEL` is 10 and a request past
it is refused by name. The narrow reading of the third bullet would have been cheaper and
wrong: it would refuse a control count the framework can express exactly.

### 6.2 `Circuit.power`

Proposed shape, following the plan's `N1-5`:

```python
Circuit.power(k: int) -> "Circuit"
```

Semantics the plan already fixed, and which this proposal adopts:

- `k == 0` returns the identity program of the same width;
- `k < 0` is `adjoint().power(-k)`, so the inverse rule has one implementation
  rather than two;
- for a program whose instructions are angle-parameterised, the result is exact,
  and the acceptance test is numerical: the powered program's action equals the
  receiver's action applied `k` times, to within machine precision.

The measured census that this has to be honest about: of the 35 registered
opcodes, **14 declare parameters** (`rx ry rz phase u1 u2 u3 crx cry crz cphase
rxx ryy rzz`, the same 14 the adjoint record identifies as the differentiable
set) and **17 are unitary with no declared parameters** (`i x y z h s sdg t tdg
sx sxdg cx cy cz swap ccx cswap`). The plan's rule — exact for the
angle-parameterised case, fail-closed otherwise — is therefore a real restriction
and not a formality, and it is the decision this proposal is least able to
justify from measurement alone.

That is recorded as an open question rather than settled here. Two readings are
defensible and they differ observably: `power` is "the single-instruction form
with a scaled angle", which refuses `h.power(2)` even though `h; h` is exact, or
`power` is "the program applied `k` times", which accepts every unitary and
refuses only the non-integer case. The first is the plan's reading; the second is
a larger operation than the plan scoped. **`N1-5` must choose one and measure it**;
this proposal records that the choice is open, because a proposal that picked the
narrower reading silently would make the wider one look like a missing feature
rather than a deferred decision.

### 6.3 Both

Neither method may add a root export, a default on an existing signature, a
result field, or a serialized key, and neither may add an opcode. `fq.__all__`
stays 36 names and `IR_VERSION` stays `"1.0"`.

## 7. Prohibited practices

- **Do not make composition a second IR.** No nesting node, no wrapper circuit
  type, no power field on `Instruction`, no `cu` opcode. Section 2 is the
  measurement that keeps this honest, and it is re-run per operation.
- **Do not implement `power` by falling back to an approximation.** A fractional
  exponent that the IR cannot express is a refusal, not a decomposition to within
  some tolerance. A silent approximation would make a training result wrong
  without making it fail.
- **Do not give `control` an identity or no-op path.** An uncontrolled operation
  is the operation itself; a control qubit outside the circuit is an error, not a
  promotion.
- **Do not repurpose the contract's `not_provided` text ahead of the
  implementation.** It was accurate while `control` and `power` were both absent,
  and it is the document that predicted this one. `N1-4` retired the `control`
  half of it in the same slice that added the method, which is the order this
  bullet asks for; the `power` half stays until `N1-5` lands.
- **Do not add a fourth spelling for a control qubit.** `ctrl_qubits` is the name
  fixed by the alignment plan's `N1-4` row and by its `wires=` → `qubits=` naming
  map (`ctrl_wires=` → `ctrl_qubits=`, `wire_map=` → `qubit_map=`, `work_wire=` →
  `work_qubit=`). `work_wires` is not an admissible spelling, and `N1-5`'s release
  note is not a place to introduce one.

## 8. Owner and approvals

Owner: FlagQuantum core maintainers (team `core` in `team-ownership.toml`) for
`flagquantum/circuit.py` and `flagquantum/core/**`; `integration` for `docs/**`
and `contracts/**`, which are shared paths.

**What this document does and does not approve.** It records the decision that
`control` and `power` belong to the composition family, on the evidence in
Section 5, and it records the acceptance criteria and refusal classes in Section
6 as the basis for review. It does not itself grant API-owner approval: under
`PUBLIC_API_PROTECTION.md` item 5 that is a review act, and `N1-4`/`N1-5` may not
treat this document as a substitute for it. The composition contract's
`not_provided_reason` said "no approved API change proposal", which was true of
both operations at `45a85cc3`; the `control` half of it is now false, and the slice
that made it false is the slice that edited it — `N1-4`, whose own record is
[`FQ-CIRCUIT-CONTROL-20261022.md`](../api-changes/FQ-CIRCUIT-CONTROL-20261022.md).
That record carries the authorization this proposal could not, because the Stable
Core surface it changes (`Circuit.control`'s signature, and the new public keyword
`ctrl_qubits=`) is a rule-8 change that only the user can authorize.

Not approved here: any change to `IR_VERSION`, to a root export, to the 35-opcode
registry, or to the compiler passes named by `N4-2`.

## 9. Verification

Every number in Sections 2, 4, 5, and 6 is produced by a command on this checkout;
none is transcribed by hand.

The Section 2 measurement, re-run after `N1-4` with the one assertion that `N1-4`
changed renamed rather than deleted:

```bash
python -c "
import flagquantum as fq
from flagquantum.core.ir import IR_VERSION
from flagquantum.core.operator_schema import _SCHEMAS
from flagquantum.ecosystem.conformance import semantic_fingerprint
c = fq.Circuit(5).h(0); block = fq.Circuit(3).h(0).cx(0, 1).rz(2, theta=0.3)
composed = c.compose(block, qubits=(1, 2, 3))
hand = fq.Circuit(5).h(0)
for i in block.to_ir().instructions:
    hand.gate(i.name, tuple(q + 1 for q in i.wires), params=i.params)
assert composed.to_ir().to_dict() == hand.to_ir().to_dict()
assert semantic_fingerprint(composed) == semantic_fingerprint(hand)
assert IR_VERSION == '1.0' and len(fq.__all__) == 36
assert len(_SCHEMAS) == 35 and not ({'compose','adjoint','control'} & {s.opcode for s in _SCHEMAS})
assert not hasattr(fq.Circuit(2), 'power')
print('composition is not a second IR; power is absent on this revision')
"
```

`control` is a method now and not an opcode, so it moved from the second clause of
the absence assertion into the first, which is what that assertion was always
testing for: an operation the IR does not describe. `power` remains absent here
and is provided by the separate `N1-5` change.

```bash
python -m pytest tests/unit/test_circuit_compose.py tests/unit/test_circuit_adjoint.py \
    tests/unit/test_circuit_composition_contract.py -q
# 193 passed

python -m pytest tests/unit/test_circuit_control.py -q
# 57 passed

python tools/check_circuit_composition_contract.py
# Circuit composition contract passed

python tools/check_docs_links.py
# Checked 551 markdown files; all local links and anchors resolve.
```

The composition contract's own verifier is the reason the shipped members
cannot drift from this document: it refuses a contract that names a test which is
not a test, and it refuses an operation that has no expansion test named.

## 10. Open questions

1. **Is the `power` restriction the plan describes the right one?** Section 6.2
   records the two readings. `N1-5` chooses.
2. **Does `control` mutate or copy?** Section 6.1 proposes a new `Circuit` and
   explains why that differs from `compose`, whose contract records
   `mutates_receiver = true`. The two members disagreeing is defensible only
   because widening in place is a different hazard from rewriting labels in place,
   and `N1-4` should confirm that reading rather than inherit it.
   **Settled by `N1-4`: it copies.** The contract records
   `receiver_is_mutated = false` and `result_width` as a separate key from the
   receiver's width, and the reading is measured rather than asserted — the
   contract verifier takes a two-qubit receiver, records its instruction list,
   controls it onto a fifth qubit, and checks that the receiver's instruction list
   is byte-for-byte what it was, that the receiver still reports the same batch
   size, device and dtype, and that the result reports width 5.
3. **Does `control` need an ancilla argument?** The ladder precedent in
   `algorithms/primitives/oracle.py:72` consumes `len(controls) - 2` ancillas for
   three or more controls and requires each to enter in `|0>`. Whether that
   requirement becomes part of `Circuit.control`'s signature, or whether the
   operation is restricted to control counts the IR can express without ancillas,
   is `N1-4`'s decision and it changes the user's call.
   **Settled by `N1-4`: no ancilla argument, and the precedent is not reused.**
   The third reading — restrict control counts to what fits without ancillas —
   was rejected too: the framework can express every control count exactly, so
   refusing one would be refusing a solvable request. The emitter is ancilla-free
   and pays for it in depth, which the contract publishes as
   `ladder_depth_growth` and `ladder_instruction_bound`. The measurement that
   decided it is in
   [`FQ-CIRCUIT-CONTROL-20261022.md`](../api-changes/FQ-CIRCUIT-CONTROL-20261022.md):
   `append_multi_controlled_x` restores its ancillas but returns a wrong answer on
   an ancilla that entered dirty — 8 of 32 basis operands at three controls, 32 of
   128 at four — and `Circuit` cannot certify that any qubit entered in `|0>`. An
   ancilla argument was therefore a hazard rather than a facility, and adding one
   would have needed its own rule-8 authorization.
4. **Should `control` and `power` be methods on `Circuit` or on the instruction
   sequence?** They are methods here because `Circuit` owns the instruction list,
   but `power` in particular could be read as a property of each instruction.
   That reading would put a `power` operation on `Instruction` and is rejected in
   Section 7; recorded here so the rejection is visible rather than implicit.
5. **Does the 66/65 collision recur?** Two numbers are issued here in dependency
   order rather than plan order, and one forward reference was already wrong. A
   gate that checks every `API_CHANGE_PROPOSAL_0NN` reference resolves to a file
   with a matching title would catch the next one; none exists today, and this
   document does not add one.
6. **An ancilla-free ladder is exponential. Is a linear construction wanted
   later?** `N1-4` publishes the depth instead of hiding it, which is the right
   default, but a caller who already knows a qubit is clean may reasonably want the
   cheaper route. That is a new public argument and a new authorization, so it is
   recorded here as an open question rather than left to the next slice to
   rediscover.
