# API Change Proposal 066: Construction-time circuit composition

## Status

**Proposed.** This proposal authorizes **two new Stable Core capabilities**,
`Circuit.control` and `Circuit.power`, and records them as members of one
construction-time composition family together with the two members that had
already shipped, `Circuit.compose` and `Circuit.adjoint`. The two earlier members
are re-examined here as the mechanism evidence for the family, not re-authorized;
Section 1 states which document owns which member. **Both of the two have since
landed**: `Circuit.power` shipped on `N1-5` and `Circuit.control` on `N1-4`, each
with its own authorization record, `FQ-CIRCUIT-POWER-20261021.md` and
`FQ-CIRCUIT-CONTROL-20261022.md`. This document therefore remains the *family*
argument it was written as; the per-member authority lives in those dated records.

It adds no root export, no result field, no default on an existing signature, and
no serialized schema change. `IR_VERSION` stays `"1.0"`. Measured on this checkout:
`fq.__all__` is 37 names — the extra one over the 36 this document was written
against is `fq.density_matrix`, an unrelated addition — and none of the four
operations is among them, because all four are methods on a type that is already a
stable export.

The authorization records it answers to are
[`FQ-CIRCUIT-COMPOSITION-20261002.md`](../api-changes/FQ-CIRCUIT-COMPOSITION-20261002.md)
(placement) and
[`FQ-CIRCUIT-ADJOINT-20261003.md`](../api-changes/FQ-CIRCUIT-ADJOINT-20261003.md)
(inversion). The machine-checked half is
[`contracts/circuit-composition-contract.toml`](../../contracts/circuit-composition-contract.toml),
whose `[scope]` section named the two missing operations and stated why they were
missing:

```toml
not_provided = ["Circuit.control", "Circuit.power"]
not_provided_reason = "no approved API change proposal; control and power are unplanned, not partial"
```

That block is quoted as the state this proposal was written against, and it is
history now. `N1-5` moved `Circuit.power` into `provided` in the change that
implemented it, and `N1-4` did the same for `Circuit.control`; `not_provided` is
empty, and the reason reads that the construction-time composition family is
complete because every one of its members has an approved API change proposal. The
live contract, not this quotation, is the source of truth.

This document is that proposal. It is written under non-negotiable rule 8 of
`AGENTS.md` ("Treat the Stable Core public API as protected"), and it answers the
four questions the alignment plan asked of `N1-11`.

## 1. What already exists, and what is actually being authorized

The family has four members and all four have shipped. Stating that plainly
matters, because a proposal that reads as if it had authorized all four would be
claiming authority over history it does not have.

| Method | State on this checkout | Authority |
|---|---|---|
| `Circuit.compose` | implemented, contracted, tested | `FQ-CIRCUIT-COMPOSITION-20261002.md` |
| `Circuit.adjoint` | implemented, contracted, tested | `FQ-CIRCUIT-ADJOINT-20261003.md` |
| `Circuit.power` | **implemented by `N1-5`**, contracted, tested | this proposal, recorded by `FQ-CIRCUIT-POWER-20261021.md` |
| `Circuit.control` | **implemented by `N1-4`**, contracted, tested | this proposal, recorded by `FQ-CIRCUIT-CONTROL-20261022.md` |

```
>>> fq.Circuit(2).h(0).control(1, ctrl_qubits=(2,))
Circuit(n_qubits=3, instructions=3)
>>> fq.Circuit(2).h(0).power(2)
Circuit(n_qubits=2, instructions=2)
```

The two earlier methods are re-examined here rather than merely cited, because
the question this proposal had to answer first was not "should `control` exist" but
"is the *family* one mechanism". If the family is one mechanism then the shipped
members are its evidence, and the two proposed members were its continuation. If
it is not, then `control` and `power` need their own proposals and their own
justification. Section 2 answers that with measurement, and the two members it
proposed are now the last two rows of the table above.

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
| Construction adds no opcode | composed opcodes vs the 35 registered opcodes | `{'h', 'cx', 'rz'} ⊆ registry`; `{'compose', 'adjoint', 'power', 'control'} ∩ registry = {}` |

The fingerprint row is the load-bearing one. `semantic_fingerprint` hashes
`ir_version`, `n_wires`, `instructions`, `observables`, and `measurements` and
excludes transport provenance, so two programs collide under it exactly when they
are the same executable program. Composition collides with the hand-built
program, which is the statement "composition is a way to build a program, not a
way to represent one".

`control` and `power` were bound by the same evidence requirement, and that
binding is the substantive decision in this document: **neither may introduce a new
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
mechanism, and both shipped members honour it: `control` conditions operations
whose controlled form the IR can already express and refuses the rest, and `power`
refuses a fractional exponent rather than approximating a matrix root. Widening
either one later is a separate proposal, because widening it means adding an
opcode.

## 3. Question 2 — the relationship to `AGENTS.md` rules 6 and 8

### 3.1 Rule 6 (keep the user API simple)

Rule 6 requires normal usage to stay centred on `fq.Circuit`, `fq.Module`, `run`,
`plan`, training loops, and deployment packages. All four operations are methods
on `fq.Circuit`, so they add no entry point, no namespace, and no root export:
`fq.__all__` moved by none of them. A user who never composes a program never
meets any of the four.

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
- All four exist, each with a recorded authorization (Section 1).

What rule 8 does cover, and what this proposal is therefore for, is that
`docs/development/PUBLIC_API_PROTECTION.md` lists seven requirements for an
additive stable API. Recorded honestly, with the owner of each:

| # | Requirement | State |
|---|---|---|
| 1 | a concrete user journey | Section 5, four journeys |
| 2 | a reason it belongs in Stable Core rather than a namespace or experimental | Section 5.1 |
| 3 | typing and documentation | both landed: `N1-4`/`N1-5` added the `Circuit.control` and `Circuit.power` sections to `docs/reference/API.md`, the "Repeating a program" and "Controlling a block" sections to `docs/guides/CIRCUIT_COMPOSITION.md`, and the `Examples:` path on each method |
| 4 | executable behavior contracts | `[scope]`, `[placement]`, `[adjoint]`, `[power]`, `[control]`, and the 32 `[[refusals]]` rows of the composition contract; extended in the same change as each method |
| 5 | API-owner approval | **not given by this document** — see Section 8 |
| 6 | release-note entry | both landed in `docs/reference/RELEASE_NOTES.md` under `## Unreleased` |
| 7 | updated machine-readable contract after approval | the same change that implements each method: `FQ-CIRCUIT-POWER-20261021.md` for `power` and `FQ-CIRCUIT-CONTROL-20261022.md` for `control` |

Rule 8 also forbids the inverse: never update a contract or snapshot merely to
make tests pass. The composition contract's `not_provided_reason` is **not** edited
by this proposal. It is edited by the slice that actually implements each method,
and that edit is legitimate precisely because an approved proposal exists at that
point — which is what the text quoted in Section 1 said was missing. Changing it
before a method existed would have been deleting an accurate statement to make a
plan look further along. `N1-5` did exactly that for `power`: `[scope]
not_provided` went from `["Circuit.control", "Circuit.power"]` to
`["Circuit.control"]`, `provided` gained `"Circuit.power"`, and the reason was
rewritten to say that `control` was unplanned rather than partial. `N1-4` then did
the same for `control`: `not_provided` went to `[]`, `provided` gained
`"Circuit.control"`, and the reason now reads that the four-member family is
complete because every member has an approved proposal.

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

1. **Place a reusable block on chosen qubits.** `compose` — one of the two members
   that shipped before this proposal (Section 1). A five-qubit program that wants a
   three-qubit block at qubits 1..3 has no method before it and must rewrite
   instruction labels by hand.
2. **Undo a block.** `adjoint` — the other earlier member. Training against an
   inverse, a Loschmidt echo, and undoing a state-preparation block all need "the
   same gates, backwards, each one inverted". The per-opcode rule was declared in
   three places before this family existed.
3. **Apply a block conditioned on other qubits.** `control`. When a controlled
   operation is not one of the named gates, the user was stuck before this method
   existed: `Circuit` had no attribute containing `contr` at all (measured: the list
   is empty), so the only routes were the named `ccx` and `cswap` opcodes, or a
   private helper reached by full module path:

   ```python
   from flagquantum.algorithms.primitives.oracle import append_multi_controlled_x
   append_multi_controlled_x(circuit, controls=[0, 1], target=2)   # -> ccx
   ```

   That helper is a ladder for one specific opcode, and it is in `algorithms/`,
   which is not an attribute of `fq` — so a user of the Stable Core surface cannot
   reach it or the two others like it without knowing an internal path. `control`
   is the Core answer: it conditions any operation whose controlled form the
   registry can express, without naming a private module.

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

   **`N1-5` landed the Core half and not the replacement.** What shipped is
   `Circuit.power`, whose repeating path emits exactly the instruction sequence
   those three `for _ in range(power)` loops emit. The three call sites are in
   `flagquantum/algorithms/**`, which `team-ownership.toml` assigns to a different
   team, so they were not rewritten in the same change; `power` is also the floor
   rather than the whole operation those loops perform, because each one is
   *controlled* as well, which is `Circuit.control`'s half of the family. This is
   the honest state: the capability that makes the replacement possible now exists
   and is contracted, and the replacement itself is still owed. It is recorded as
   an open item rather than presented as closed, per rule 5 of `AGENTS.md`.

### 5.1 Why Stable Core rather than a namespace or experimental

Because the operation is a property of `Circuit`, which owns the instruction list.
Three candidate alternatives were considered and rejected in the composition
record for `compose`, and the same reasoning applied to `control` and `power`:

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

This section stated acceptance criteria rather than committing to an
implementation. Each criterion was either measurable at the time or was stated as
a decision the implementing slice had to settle in its own change; naming a detail
here that the implementation had not measured would have been a second source of
truth for it. Both slices have since landed, so each criterion below is followed by
the reading its own record settled on.

### 6.1 `Circuit.control`

The shape this proposal proposed, following the plan's `N1-4`, and which
`N1-4` shipped:

```python
Circuit.control(n_controls: int, ctrl_qubits: Sequence[int]) -> "Circuit"
```

Semantics that had to hold, whatever the mechanism:

- it returns a **new** `Circuit` and does not modify the receiver, so
  `Circuit.control` differs from `Circuit.compose`, which its own contract records
  as `mutates_receiver = true`. The reason is that `control` widens the program:
  a `Circuit(1).x(0).control(1, ctrl_qubits=(1,))` reports width 2. Growing the
  receiver in place would silently change the width of an object the caller may
  have shared or already serialized. The contract states it as explicitly as
  `[placement]` states it for `compose`, as `receiver_is_mutated = false`;
- the result's width covers every qubit the operation names, and
  `to_ir().n_wires` reports it;
- every emitted instruction has an opcode in the 35-opcode registry, and
  `to_dict()["version"]` is still `"1.0"`;
- applying the controlled program to a basis state reproduces the receiver's
  action exactly when all control qubits are set, and the identity when any is
  not. This is the acceptance test, and it is measured against a statevector
  simulation rather than against an instruction listing, because an instruction
  listing cannot distinguish a correct ladder from a wrong one.

Refusals that had to be contracted, in the shape `N1-8` established (exception
class plus a frozen message phrase, with a reachability flag):

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

Decisions this proposal left to `N1-4` rather than making itself: whether the
ancilla ladder precedent in `append_multi_controlled_x` is reused or a per-opcode
controlled form is emitted; whether an ancilla argument is part of the signature or
is allocated from the circuit's free qubits; and what the exact refusal phrase is
for each row. An ancilla argument, had there been one, would have been a qubit
sequence and would have been spelled with `qubit`.

**Settled by `N1-4`**, recorded in
[`FQ-CIRCUIT-CONTROL-20261022.md`](../api-changes/FQ-CIRCUIT-CONTROL-20261022.md), and
now the reading of this section:

| Decision | Settled reading | Evidence |
|---|---|---|
| mechanism | each opcode declares its controlled form in `flagquantum/core/operator_schema.py` as a rule name, and `flagquantum/core/controlled.py` emits the resulting instruction sequence | 31 of the 35 registered opcodes declare a rule; the four that do not are exactly the channels |
| ancilla ladder precedent | **not** reused. `flagquantum/algorithms/primitives/oracle.py` is outside `core` and `append_multi_controlled_x` returns a wrong answer on a dirty ancilla, which was measured: 8 of 32 operands at three controls, 32 of 128 at four, with the ancilla itself restored | the emitter is ancilla-free, and the contract records `ancilla_qubits = 0` |
| ancilla argument | **declined**, because the chosen mechanism needs none | no `ancillas=` keyword exists; the declined candidate is recorded in the record's Decision Candidates |
| exact refusal phrases | 13 rows, all reachable, frozen in `[[refusals]]` of `contracts/circuit-composition-contract.toml` and read back by `tools/check_circuit_composition_contract.py` | the contract holds 32 refusal rows across the four members, of which 31 are reachable |
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

The shape this proposal proposed, following the plan's `N1-5`:

```python
Circuit.power(k: int) -> "Circuit"
```

Semantics the plan already fixed, and which this proposal adopted:

- `k == 0` returns the identity program of the same width;
- `k < 0` is `adjoint().power(-k)`, so the inverse rule has one implementation
  rather than two;
- for a program whose instructions are angle-parameterised, the result is exact,
  and the acceptance test is numerical: the powered program's action equals the
  receiver's action applied `k` times, to within machine precision.

As shipped, the parameter is named `exponent` rather than `k`, and the signature is
`Circuit.power(self, exponent: int) -> "Circuit"`. `k` is a bound variable in this
document's prose, not a user-facing name; `exponent` is the domain noun, per rule 9.
The method adds no keyword that names a qubit at all. Renaming the argument would
be a rule-8 change to a shipped signature, so the contract freezes `exponent` and
the gate refuses a contract that renames it.

The measured census this had to be honest about: of the 35 registered
opcodes, **14 declare parameters** (`rx ry rz phase u1 u2 u3 crx cry crz cphase
rxx ryy rzz`, the same 14 the adjoint record identifies as the differentiable
set) and **17 are unitary with no declared parameters** (`i x y z h s sdg t tdg
sx sxdg cx cy cz swap ccx cswap`). The plan's rule — exact for the
angle-parameterised case, fail-closed otherwise — was therefore a real restriction
and not a formality, and it was the decision this proposal was least able to
justify from measurement alone.

That was recorded here as a choice rather than settled: two readings were
defensible and they differ observably — `power` is "the single-instruction form
with a scaled angle", which refuses `h.power(2)` even though `h; h` is exact, or
`power` is "the program applied `k` times", which accepts every unitary and
refuses only the non-integer case. The first is the plan's reading; the second is
a larger operation than the plan scoped. **`N1-5` had to choose one and measure
it**; this proposal recorded the choice as open rather than picking the narrower
reading silently, which would have made the wider one look like a missing feature
rather than a deferred decision. Section 6.2.1 is the answer `N1-5` gave.

#### 6.2.1 The choice `N1-5` made, recorded here

`N1-5` chose the **wider** reading and measured it. `Circuit.power(k)` is "the
program applied `k` times", for every instruction, and the single-gate scaling is
an internal rewrite rather than the definition. In full:

- `k == 0` is the empty program with the receiver's `n_qubits`, `bsz`, `device`,
  and `dtype`;
- `k == 1` is a copy, and it deliberately does **not** scale the angles by one,
  because `theta * 1` is a different expression from `theta` for a symbolic
  parameter and a different tensor for a trainable one;
- `k < 0` is `adjoint().power(-k)`;
- `k >= 2` appends the receiver's program `k` times in the receiver's own order;
- the single rewrite fires only when the receiver is exactly one instruction that
  carries no matrix and whose opcode declares exactly one parameter. Then, and
  only then, the emitted program is that one instruction with the parameter
  multiplied by `k`.

The decision is declared rather than special-cased: `flagquantum/core/operator_schema.py`
gives every opcode a `power_rule`, either `scale_single_parameter` ("a unitary that
declares exactly one parameter", which is exactly `U(theta)^k == U(k*theta)`) or
`repeat_instruction` ("everything else"). Measured, that split is **12 opcodes in
closed form and 23 repetitions**, and the contract gate
`tools/check_circuit_composition_contract.py` compares the two dense operators for
every one of the 12, so a gate whose angle does not in fact scale is refused by
the gate rather than mis-announced by the field.

The narrower reading was rejected for a measured reason, not a preference: it
would refuse `hadamard.power(2)` even though two Hadamards are exactly the
identity, and it would refuse a custom operation carrying its own matrix even
though that matrix squares exactly. The wider reading is exact for every
instruction, so there is nothing to be fail-closed about at a positive exponent;
the only remaining refusals are the non-integer exponent (the IR has no way to
express a matrix square root) and a negative exponent on a channel (which is
`adjoint`'s existing refusal, reached rather than duplicated).

The census in Section 6.2 above counts 14 parameter-declaring opcodes; the
closed-form set is the 12 of those that declare **exactly one** parameter.
`u2` and `u3` declare several, and scaling all of them is a measurably different
operator. Measured as `max |U^2 - U(2 * params)|` on the dense operator, at
`u2(phi=0.9, lbd=1.3)` the residual is `4.135343356e-01` and at
`u3(theta=0.4, phi=0.9, lbd=1.3)` it is `2.191289381e-01`; both are asserted as the
same number, to a `1e-9` tolerance, in `tests/unit/test_circuit_power.py`, so the
counterexample is maintained rather than quoted. That is why those two repeat.

Fractional exponents are the gap this leaves open, and it is an owned gap rather
than a silence: `Circuit.power(0.5)` raises `TypeError`, and PennyLane 0.45.1
constructs `qml.Hadamard(0) ** 0.5` and runs it. Closing that gap needs an
`Instruction` form that can express a matrix square root, which is an IR question
and not a composition question. It is recorded in
[`docs/reference/KNOWN_LIMITATIONS.md`](../reference/KNOWN_LIMITATIONS.md) with
the same framing.

### 6.3 Both

Neither method adds a root export, a default on an existing signature, a result
field, or a serialized key, and neither adds an opcode. `fq.__all__` moved by
neither of them — it reads 37 because `fq.density_matrix` joined it in an unrelated
change — and `IR_VERSION` stays `"1.0"`.

### 6.4 Where the per-instruction rewrites live

`adjoint`, `power`, and `control` each apply their member one instruction at a time,
and each has to answer the same question about every instruction it meets: is there a
rewrite the IR can express? Three methods answering that separately is what put
`flagquantum/circuit.py` over the `default_module_line_ceiling = 1250` in
`architecture.toml` once `control` (1196 lines on `main`) and `power` (1156 on this
branch) shared a file that neither of them exceeded alone.

The overage was resolved by moving the answer, not by raising the ceiling. The four
per-instruction rewrites — invert one instruction, repeat or scale one instruction,
copy one instruction, expand one instruction under added controls — now live in
`flagquantum/core/_composition.py`, next to `flagquantum/core/controlled.py`, which
owns the controlled expansion itself and is where `control`'s reasoning already
lived. The module is private to Core, adds no exported name, and leaves
`flagquantum/circuit.py` at 1201 lines. Nothing in Section 5's Stable Core statement
changes: the three methods are still the public form of this behaviour, and the new
module is explicitly forbidden from becoming a second one.

This is the response the repository already chose for the same gate when
`statevector/reverse_adjoint_sweep.py` was three lines over it — remove the
duplication rather than widen the limit — and it is the one Section 7 needs, because
the alternative is a `legacy_exceptions` entry that would make the ceiling advisory
for the file that currently defines the composition family.

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
  and it is the contract that predicted this proposal. Each half was retired by
  the slice that implemented it and not before: `N1-5` moved `Circuit.power` from
  `not_provided` to `provided`, and `N1-4` did the same for `Circuit.control`. That
  is the order this bullet asks for, and it is why `not_provided` is now empty
  rather than empty ahead of time.
- **Do not add a fourth spelling for a control qubit.** `ctrl_qubits` is the name
  fixed by the alignment plan's `N1-4` row and by its `wires=` → `qubits=` naming
  map (`ctrl_wires=` → `ctrl_qubits=`, `wire_map=` → `qubit_map=`, `work_wire=` →
  `work_qubit=`). `work_wires` is not an admissible spelling, and `N1-5`'s release
  note introduced no such spelling: `power` adds one method whose single parameter
  is `exponent`, and no keyword naming a qubit at all.

## 8. Owner and approvals

Owner: FlagQuantum core maintainers (team `core` in `team-ownership.toml`) for
`flagquantum/circuit.py` and `flagquantum/core/**`; `integration` for `docs/**`
and `contracts/**`, which are shared paths.

**What this document does and does not approve.** It records the decision that
`control` and `power` belong to the composition family, on the evidence in
Section 5, and it records the acceptance criteria and refusal classes in Section
6 as the basis for review. It does not itself grant API-owner approval: under
`PUBLIC_API_PROTECTION.md` item 5 that is a review act.

The composition contract's `not_provided_reason` said "no approved API change
proposal", which was true of both operations at `45a85cc3`. Each half of that
statement was retired by the slice that implemented the method it named, against
that slice's own authorization record rather than against this document:

- `N1-5` retired the `power` half against
  [`FQ-CIRCUIT-POWER-20261021.md`](../api-changes/FQ-CIRCUIT-POWER-20261021.md);
- `N1-4` retired the `control` half against
  [`FQ-CIRCUIT-CONTROL-20261022.md`](../api-changes/FQ-CIRCUIT-CONTROL-20261022.md).

Both records are in the same series as the composition and adjoint records, and
both carry the authorization this proposal could not, because the Stable Core
surface each changes (`Circuit.power`'s signature, `Circuit.control`'s signature,
and the new public keyword `ctrl_qubits=`) is a rule-8 change that only the user can
authorize.

This document therefore remains the *family* argument — why the four members are
one mechanism, what their shared evidence requirement is, and what each of the two
proposed members had to satisfy — while the per-member authority lives in the dated
records. That division is deliberate: it keeps this proposal from reading as an
approval it never received.

Not approved here: any change to `IR_VERSION`, to a root export, to the 35-opcode
registry, or to the compiler passes named by `N4-2`.

## 9. Verification

Every number in Sections 2, 4, 5, and 6 is produced by a command on this checkout;
none is transcribed by hand.

The Section 2 measurement, re-run after both slices landed. The absence assertion
kept its shape and its subject: what it tests is that construction names nothing
the IR does not already describe, so the four method names are checked against the
opcode registry rather than against a shrinking list of not-yet-implemented names.

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
assert IR_VERSION == '1.0' and len(fq.__all__) == 37
assert len(_SCHEMAS) == 35
assert not ({'compose','adjoint','power','control'} & {s.opcode for s in _SCHEMAS})
assert all(hasattr(fq.Circuit(2), name) for name in ('compose','adjoint','power','control'))
print('composition is not a second IR; all four members are methods and none is an opcode')
"
# composition is not a second IR; all four members are methods and none is an opcode
```

`fq.__all__` reads 37 rather than the 36 this document was written against because
`fq.density_matrix` joined the root export set in an unrelated change; none of the
four members is a root export, and none of them moved that number.

```bash
python -m pytest tests/unit/test_circuit_compose.py tests/unit/test_circuit_adjoint.py \
    tests/unit/test_circuit_composition_contract.py tests/unit/test_circuit_power.py \
    tests/unit/test_circuit_control.py -q
# 556 passed

python tools/check_circuit_composition_contract.py
# Circuit composition contract passed

python tools/check_architecture.py
# architecture boundaries passed

python tools/check_docs_links.py
# Checked 562 markdown files; all local links and anchors resolve.
```

The composition contract's own verifier is the reason the shipped members
cannot drift from this document: it refuses a contract that names a test which is
not a test, and it refuses an operation that has no expansion test named.

## 10. Open questions

1. **Is the `power` restriction the plan describes the right one?** Section 6.2
   records the two readings, and `N1-5` chose the wider one — see Section 6.2.1,
   which is the record of that decision and of the measurements behind it. What
   survives as open is the narrower question underneath: whether a *fractional*
   exponent should be expressible, which needs an `Instruction` form for a matrix
   root and is therefore an IR decision rather than a composition one.
2. **Does `control` mutate or copy?** Section 6.1 proposed a new `Circuit` and
   explained why that differs from `compose`, whose contract records
   `mutates_receiver = true`. The two members disagreeing is defensible only
   because widening in place is a different hazard from rewriting labels in place,
   and `N1-4` had to confirm that reading rather than inherit it. `power` settled
   the same question for itself the other way — it returns a new circuit and leaves
   the receiver byte-identical — so `control` was the last member whose answer was
   still open.
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
   requirement became part of `Circuit.control`'s signature, or whether the
   operation was restricted to control counts the IR can express without ancillas,
   was `N1-4`'s decision and it changes the user's call.
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
