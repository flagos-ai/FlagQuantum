# Circuit adjoint: `Circuit.adjoint`

## Decision and authorization

Status: **implemented on this branch, pending review.** This proposal authorizes one
new method, `Circuit.adjoint`, and one new reading of the existing
`OperatorSchema.adjoint` field. It adds no stable root export, changes no serialized
schema version, and leaves `IR_VERSION` at `1.0`.

It is written under non-negotiable rule 8 of `AGENTS.md` ("Treat the Stable Core
public API as protected"). This document **is** the API change proposal for the
method: the numbered `docs/development/API_CHANGE_PROPOSAL_0NN_*.md` series
records the older, runtime-route proposals (`063` backend execution admission,
`064` stabilizer execution mode) and this change adds no runtime route, so there
is no second document to keep in step with this one. The sibling change
`Circuit.compose` has its own proposal,
[`FQ-CIRCUIT-COMPOSITION-20261002.md`](FQ-CIRCUIT-COMPOSITION-20261002.md). It owns
placement, this one owns inversion, and nothing here approves `control` or
`power`.

The authorization question was whether this needs one at all. `AGENTS.md`
non-negotiable rule 8 protects *stable exports, signatures, defaults, result
fields, enum/Literal values, documented exception behavior, and serialized public
schemas*. Measured against this checkout:

- `docs/public_api_v1.json` freezes **34 root exports** and does not enumerate
  methods, so `Circuit.adjoint` adds no frozen name.
- `Circuit` itself is already a stable export, and `docs/generated/STABLE_API.md`
  is generated from that same root list.
- `OperatorSchema.adjoint` already exists, is already serialized into
  `docs/operator_manifest.json`, and already has zero consumers.

So no new public name is introduced and no schema field is added. What this
proposal does need to justify is the *change in meaning* of an existing serialized
field, which is why it is written down rather than assumed: the 20 operators whose
`adjoint` reads `matrix_adjoint` today begin reading a concrete rule.

Scope of the affected surface: `flagquantum/core/operator_schema.py` (owned by
team `core`), `flagquantum/circuit.py` (owned by team `core`),
`docs/operator_manifest.json` (generated), `docs/reference/API.md`, and tests.

## Problem and affected user journey

### The inverse of a program could not be expressed

FlagQuantum could take a circuit apart but not invert one. Measured on
`origin/main` at `9dec7649`:

```python
>>> import flagquantum as fq
>>> fq.Circuit(2).h(0).cx(0, 1).adjoint()
AttributeError: 'Circuit' object has no attribute 'adjoint'
```

The gap is real for a user and not cosmetic. Training a variational program
against an inverse, building a Loschmidt echo, and undoing a state-preparation
block all need "the same gates, backwards, each one inverted". Every one of those
users had to reimplement the reversal by hand and get the per-opcode rule right
themselves.

### The rule was already declared twice, and read nowhere

`OperatorSchema` has carried an `adjoint` field that is written by every
constructor and read by no implementation. Measured over the 35 registered
opcodes:

| Declaration | Opcodes before this change |
| --- | --- |
| `self_inverse` | 11 |
| `matrix_adjoint` | 20 |
| `not_applicable` | 4 |

`matrix_adjoint` was the default for every parameterized gate, so `rx`, `cphase`,
and `u3` all declared the same thing while having three different inverse rules.
The field recorded *that* an inverse exists and not *what* it is.

Meanwhile `compiler/pipeline.py` maintained a second answer to the same question:

```python
_SELF_INVERSE = {"x", "y", "z", "h", "cx", "cy", "cz", "swap", "ccx", "cswap"}
_ROTATION_PARAM = ("rx", "ry", "rz", "phase", "u1", "rxx", "ryy", "rzz", "crx", "cry", "crz", "cphase")
```

and `algorithms/primitives/qft.py` a third: `append_qft(circuit, wires, inverse=True)`
reverses its own sequence and negates its controlled-phase angles by hand. Three
places, two of them hand-maintained, answering one semantic question, is the
defect this proposal closes for the construction path.

### The parameterized gates do not share one inverse rule

The obvious rule — negate every angle — is wrong for two opcodes. Measured with
the gate matrices this repository executes (`simulation/matrices.py`), for
`theta = 0.4, phi = 0.9, lbd = 1.3`, against `I`:

| Opcode | Candidate inverse | `max abs(candidate @ forward - I)` |
| --- | --- | --- |
| `rx ry rz phase u1 crx cry crz cphase rxx ryy rzz` | negate the angle | `1.11e-16` |
| `u3` | negate all three, same order | `7.74e-02` |
| `u3` | `theta -> -theta, phi -> -lbd, lbd -> -phi` | `2.22e-16` |
| `u3` | `theta -> -lbd, phi -> -phi, lbd -> -theta` | `7.72e-01` |
| `u2` | negate both phases | `9.80e-01` |
| `u2` | swap the phases, then negate | `1.00e+00` |
| `u2` | `phi -> -lbd - pi, lbd -> -phi - pi` | `8.07e-16` |

A "negate the angle" implementation would therefore have been silently wrong for
`u2` and `u3`, which are among the 14 differentiable opcodes and are emitted by
`Circuit.u`. This is the evidence that the declaration has to name a rule and not
a boolean.

### The two rules agree with an independent implementation

The rules are not this repository's own spelling of a convention. PennyLane 0.45.1
resolves the same two adjoints lazily, through `Adjoint(...).simplify()`, into a
different but equivalent spelling:

```python
>>> import pennylane as qml
>>> qml.adjoint(qml.U3)(0.4, 0.9, 1.3, wires=0).simplify()
U3(0.4, 1.841592653589793, 2.241592653589793, wires=[0])   # theta, pi - lbd, pi - phi
>>> qml.adjoint(qml.U2)(0.9, 1.3, wires=0).simplify()
U2(1.841592653589793, 2.241592653589793, wires=[0])        # pi - lbd, pi - phi
```

`u3(-theta, -lbd, -phi)` equals `U3(theta, pi - lbd, pi - phi)` elementwise, and
`u2(phi, lbd) = (-lbd - pi, -phi - pi)` differs from `U2(pi - lbd, pi - phi)` by
exactly `2 pi`. Measured at `complex128`, against the conjugate transpose of the
forward gate:

| Spelling | `max abs(candidate - forward^H)` | `max abs(candidate @ forward - I)` |
| --- | --- | --- |
| PennyLane `U3(theta, pi - lbd, pi - phi)` | `2.48e-16` | `2.22e-16` |
| this repository `u3(-theta, -lbd, -phi)` | **`0.0`** | `2.22e-16` |
| PennyLane `U2(pi - lbd, pi - phi)` | `1.67e-16` | `2.69e-16` |
| this repository `u2(-lbd - pi, -phi - pi)` | — | `8.07e-16` |

Both spellings are inside machine precision; the negative-angle spelling simply
avoids the two `pi -` subtractions, which is why it lands on `0.0`. What the
comparison establishes is weaker and more useful than a ranking: two independent
implementations resolve `u2` and `u3` to the same matrix, so the rule declared in
`_SCHEMAS` is the measured one and not a local choice.

### Affected user journeys

1. **Undo a reusable block.** Build a state-preparation circuit once, then append
   it and its inverse to measure a return probability.
2. **Train against an inverse.** A `Module` whose forward pass ends in the inverse
   of its input encoding, differentiated with respect to the encoding.
3. **Reuse an existing inverse.** `qft(n).adjoint()` should be the same program as
   `qft(n, inverse=True)`, so that the hand-written inverse in
   `algorithms/primitives/qft.py` stops being a second source of truth.

## Evidence

Repository state: `origin/main` at `9dec7649`, with this proposal applied. The
measurements below were taken there. The branch has since merged `origin/main` at
`ba018d42`, which carries the parameter-frequency declaration (#373), inline channels
(#371), and the sibling `Circuit.compose` (#375). None of the three reads or changes
`adjoint`, and the manifest difference against that revision is still the same 20
lines, so nothing measured here moved. Merging #375 is also what lets this document
link its sibling proposal above rather than name it in prose.

| Claim | Command | Result |
| --- | --- | --- |
| Every invertible unitary opcode is undone by its declared rule | `pytest tests/unit/test_circuit_adjoint.py -k restores_the_initial_state` | 31 passed; worst `max abs(psi_after - psi_initial)` = `3.25e-16` over `complex128` |
| The naive rule is far from the identity for `u2` and `u3` | `pytest tests/unit/test_circuit_adjoint.py -k obvious_one_is_wrong` | `u3` residual `7.74e-02` > `5e-02`; `u2` residual `9.80e-01` > `5e-01` |
| The declared rule is complete | `pytest tests/unit/test_circuit_adjoint.py -k every_unitary_opcode_declares_an_invertible_rule` | every `semantic_kind="unitary"` schema names a rule other than `matrix_adjoint` and `not_applicable`; every channel names `not_applicable` |
| Partner declarations are symmetric | `pytest tests/unit/test_circuit_adjoint.py -k declaration_is_defined_for_every_opcode` | `s <-> sdg`, `t <-> tdg`, `sx <-> sxdg` pair both ways |
| The hand-written QFT inverse is reproduced | `pytest tests/unit/test_circuit_adjoint.py -k matches_the_recorded_inverse_transform` | identical instruction sequences for `n = 1..5` (1, 4, 7, 12, 17 instructions) |
| Non-invertible operations fail closed | `pytest tests/unit/test_circuit_adjoint.py -k refuses_an_operation_without_a_unitary_inverse` | `CapabilityError` naming the opcode and its qubits for a channel, a dynamic operation, a conditioned gate, and a matrix with no conjugate transpose |
| The compiler's self-inverse list does not contradict the schema | `pytest tests/unit/test_circuit_adjoint.py -k compiler_self_inverse_transform_agrees` | `_SELF_INVERSE` is a subset of the declared set; the only difference is `i`, which `remove_identity_gates` already owns |
| The serialized field changed consistently | `python tools/operator_manifest.py --check` after regeneration | 20 changed lines, one per operator that stopped saying `matrix_adjoint` |
| The `u2` / `u3` rules match an independent implementation | `python -c "import pennylane as qml; print(qml.adjoint(qml.U3)(0.4, 0.9, 1.3, wires=0).simplify())"` with `pennylane==0.45.1` | `u3(-theta, -lbd, -phi)` equals `U3(theta, pi - lbd, pi - phi)` to `2.48e-16`; `u2(phi, lbd)` equals `U2(pi - lbd, pi - phi)` modulo `2 pi` |

## Decision Candidates

**A. Add `Circuit.adjoint()` that returns a new circuit (chosen).** The inverse of
a program is a different program. `Circuit.compose` and the generated gate methods
mutate and return `self` because they *extend* this program; `bind_parameters`,
`copy`, and `adjoint` produce another one. Returning a new circuit keeps the
forward block usable, which is what all three user journeys above need.

**B. Add `Circuit.adjoint()` that mutates in place.** Rejected. It makes
`circuit.adjoint()` destroy the circuit, and every user journey above would have to
take a copy first. It also has no natural `__iadjoint__` spelling.

**C. Leave the opcode rule in `circuit.py` and keep `OperatorSchema.adjoint` as a
label.** Rejected. That is the state the evidence above measures: the declaration
and the implementation disagree, and `compiler/pipeline.py` becomes the second
implementation.

**D. Compute the inverse from the gate matrix for every opcode.** Rejected. It
would give the right answer numerically and the wrong answer structurally: an
inverse expressed as an arbitrary matrix cannot be compiled, lowered, drawn, or
recognized by `merge_self_inverse`, and it would make an angle a matrix. The
matrix route is kept only for custom operations, where the matrix is what executes.

**E. Fail closed for `u2` instead of naming its angle rule.** Rejected. `u2` is a
legal public gate with a closed-form inverse, measured above. Refusing it would
report a limitation this repository does not have, which `AGENTS.md` clause 5
forbids.

## Prohibited Practices

1. Do not add a second inverse table. Any new caller that needs an opcode's
   inverse reads `OperatorSchema.adjoint` through `core.inverse_operator`.
2. Do not resolve an unrepresentable inverse by copying the instruction forward
   unchanged. A silent identity is a wrong program.
3. Do not widen the exception type to `ValueError` or `NotImplementedError`.
   `CapabilityError` is the class the errors-module boundary reserves for a
   capability that is absent rather than a value that is wrong.
4. Do not make `adjoint` an IR node. It is a construction-time rewrite that emits
   instructions the IR already describes.
5. Do not update `docs/operator_manifest.json` by hand. It is generated.

## Compatibility

- `IR_VERSION` stays `1.0`. `adjoint` rewrites the instruction sequence; it does
  not add an instruction kind.
- No root export is added, removed, or renamed. `docs/public_api_v1.json` and
  `docs/generated/STABLE_API.md` are unchanged, and `tools/public_api_snapshot.py`
  passes.
- `Circuit.adjoint` is a new method on an existing stable class. It does not
  change any existing signature, default, or documented exception.
- `OperatorSchema.adjoint` keeps the three existing values and adds three
  (`negate_parameters`, `adjoint_u2_angles`, `adjoint_u3_angles`) plus the
  existing convention of naming a partner opcode. `self_inverse`,
  `matrix_adjoint`, and `not_applicable` remain valid and remain handled.
- `docs/operator_manifest.json` changes: 20 operators that declared
  `matrix_adjoint` now declare their actual rule. The manifest's
  `schema_version` stays `flagquantum_operator_manifest_v1`; a consumer that
  matched the literal string `matrix_adjoint` must now compare against
  `ADJOINT_RULES` or call `inverse_operator`. No in-repository consumer did.
- Known duplication, deliberately not removed here: `compiler/pipeline.py`
  `_SELF_INVERSE` and `_ROTATION_PARAM` stay in place. Retiring them is `N4-2`,
  which owns the compiler passes that read them. This proposal adds the test that
  makes a disagreement fail.

## Acceptance Tests

- `tests/unit/test_circuit_adjoint.py` — 65 tests. Per-opcode state restoration
  across all 31 unitaries, the measured error of the naive rule for `u2` and `u3`,
  symbolic and per-batch parameters, the custom-matrix route, every refusal path,
  order and identity of the returned circuit, and the QFT conformance check.
- `tests/unit/test_operator_schema.py` — the schema contract, unchanged and
  passing.
- `tools/operator_manifest.py --check` — the regenerated manifest is current.
- `tools/public_api_snapshot.py` — the stable surface is unchanged.
- `tools/check_team_scope.py --team core` — the changed paths are owned.

## Open Questions

1. Should `Circuit.adjoint()` refuse a circuit that carries a channel, or invert
   the unitary part and leave the channel? Today it refuses the whole circuit,
   which is the fail-closed reading of principle 9. The alternative is defensible
   only if the result reports what it dropped.
2. Should `adjoint()` have an in-place spelling (`circuit.adjoint_()`), for a
   caller that wants to save the copy on a large program?
3. `N1-8` proposes a machine-checked `issue_code` vocabulary for the circuit
   expression contract. This method reports its refusals as `CapabilityError`
   messages because no such vocabulary exists yet; when it lands, these four
   refusals should carry codes.
4. Should `Circuit.control` and `Circuit.power` share this declaration, or does
   each need its own? `power` in particular has no single-gate rule for a
   non-integer exponent.

## Owner and approvals

Owner: FlagQuantum core maintainers (team `core` in `team-ownership.toml`).

Approval recorded by this document: the method name `Circuit.adjoint`, the four
rule names in `ADJOINT_RULES`, the partner-opcode convention, and
`CapabilityError` as the refusal class. Not approved here: any change to
`IR_VERSION`, to a root export, or to the compiler passes named in `N4-2`.
