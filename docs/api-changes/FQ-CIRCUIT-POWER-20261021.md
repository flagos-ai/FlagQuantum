# Circuit power: `Circuit.power`

## Decision and authorization

Status: **implemented on this branch, pending review.** This proposal authorizes one new
method, `Circuit.power`, and one new reading of the existing `OperatorSchema` declaration:
a `power_rule` property derived from the fields the schema already carries. It adds no
stable root export, changes no serialized schema version, and leaves `IR_VERSION` at
`1.0`.

It is written under non-negotiable rule 8 of `AGENTS.md` ("Treat the Stable Core public
API as protected"). This document records the *decision*; the numbered half is
[`API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md`](../development/API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md),
which records `compose`, `adjoint`, `control`, and `power` as one family. That proposal
left §6.2 open on purpose: it named two readings of `power` — a narrow one that accepts
only the gates whose angle scales, and a wide one that accepts every instruction — and
required `N1-5` to choose one and measure it. **This document is that choice.**

The authorization question was whether this needs one at all. `AGENTS.md` non-negotiable
rule 8 protects *stable exports, signatures, defaults, result fields, enum/Literal
values, documented exception behavior, and serialized public schemas*. Measured against
this checkout:

- `docs/public_api_v1.json` freezes **36 root exports** and does not enumerate methods, so
  `Circuit.power` adds no frozen name. `fq.__all__` stays at **36**.
- `Circuit` itself is already a stable export, and `docs/generated/STABLE_API.md` is
  generated from that same root list.
- `OperatorSchema` already exists and is already serialized into
  `docs/operator_manifest.json`. `power_rule` is a **property**, not a field, so it adds
  no constructor argument; `operator_manifest()` gains one key per operator, which is the
  same shape of change `adjoint` made in
  [`FQ-CIRCUIT-ADJOINT-20261003.md`](FQ-CIRCUIT-ADJOINT-20261003.md).

So no new public name is introduced and no schema *field* is added. What this proposal
does need to justify is why the contract's `not_provided` entry for `Circuit.power` is
being retired, which is why the contract change is listed below rather than assumed.

Scope of the affected surface: `flagquantum/core/operator_schema.py` (owned by team
`core`), `flagquantum/core/__init__.py`, `flagquantum/circuit.py` (owned by team `core`),
`contracts/circuit-composition-contract.toml` and its gate,
`docs/operator_manifest.json` (generated), `docs/reference/API.md`, and tests.

## Problem and affected user journey

### Repeating a program could not be expressed

FlagQuantum could place a program and invert one, but not repeat one. Measured on
`origin/main` at `98bbcb715`:

```python
>>> import flagquantum as fq
>>> fq.Circuit(1).rx(0, 0.3).power(2)
AttributeError: 'Circuit' object has no attribute 'power'
>>> fq.Circuit(1).rx(0, 0.3) ** 2
TypeError: unsupported operand type(s) for ** or pow(): 'Circuit' and 'int'
```

The gap is real for a user. Trotterized time evolution is literally `exp(-i H t)` written
as the same block applied `n` times; a Loschmidt echo pairs a block with its repetition; a
depth study sweeps `n`. Every one of those users had to write the loop by hand, and the
hand-written loop then loses the one fact the framework knows — that `rx(a)` applied
twice is exactly `rx(2a)`, so a two-instruction program collapses into a one-instruction
program that stays differentiable.

### The rule was already predictable from the declaration, and read nowhere

`OperatorSchema` carries `arity`, `parameters`, `semantic_kind`, `adjoint`, and
`parameter_frequencies`. The distance between those fields and the power rule is zero:

| Declaration | Derived rule | Opcodes |
| --- | --- | --- |
| `semantic_kind="unitary"` with exactly one parameter | `scale_single_parameter` | 12 |
| everything else | `repeat_instruction` | 23 |

The middle column is not stored. It is computed by `OperatorSchema.power_rule` from the
two fields next to it, so "this gate's angle scales" and "the powered gate is one
instruction" cannot drift apart — the same reasoning that made `OperatorSchema
.differentiable` a property of `parameter_frequencies` rather than a second flag.

### The prediction is a claim about the gates, so it is measured

`scale_single_parameter` asserts `U(theta)^k == U(k * theta)`. That is a claim about the
gate matrices, not about the field that predicts it, so
`tools/check_circuit_composition_contract.py::_power_measurement_errors` settles it by
comparing the two dense operators through `simulation/gate_matrix.py` — the same matrix
the simulator executes. Measured at `complex128`, over the 12 opcodes this rule names
times three angles `(0.3, 1.1, -0.7)` times three exponents `(2, 3, 5)`:

```text
closed-form measurement worst |U(theta)^k - U(k*theta)|: 5.551e-16
```

and the rule is not vacuous, which is what the same measurement establishes for the
opcodes it excludes:

| Opcode | Candidate form | `max abs(repeated - candidate)` |
| --- | --- | --- |
| `rx ry rz phase u1 crx cry crz cphase rxx ryy rzz` | scale the one angle | `5.55e-16` |
| `u3(theta=0.4, phi=0.9, lbd=1.3)`, `k = 2` | scale all three angles | `2.191289381e-01` |
| `u2(phi=0.9, lbd=1.3)`, `k = 2` | scale both parameters | `4.135343356e-01` |

Scaling a two- or three-angle gate is off by `1e-1`, so a "scale every parameter by `k`"
implementation would have been silently wrong for `u2` and `u3`, which are among the 14
differentiable opcodes and are emitted by `Circuit.u`. This is the evidence that the rule
is per-opcode and not global. The second and third rows name their angles so the
measurement can be re-run as stated; `u2`'s residual was first written here as
`5.90e-01`, which no convention reproduces, and `4.135343356e-01` is the measured value
that `tests/unit/test_circuit_power.py` now asserts.

### The chosen reading, and why the other one is worse

`066` §6.2 named two readings. Measured against the two of them:

| Reading | `fq.Circuit(1).h(0).power(2)` | Exact? |
| --- | --- | --- |
| narrow — only angle-scaling gates are accepted | `CapabilityError` | the refusal is false: `h;h` is a program and is exactly `I` |
| wide — every instruction is accepted | `['h', 'h']` | exact, and equal to the identity on the state |

**The wide reading is chosen.** The narrow one refuses an operation the framework can
express exactly, which is the failure mode
[`FQ-CIRCUIT-ADJOINT-20261003.md`](FQ-CIRCUIT-ADJOINT-20261003.md) already ruled out for
its own sibling case, in its decision candidate E: "Refusing it would report a limitation
this repository does not have, which `AGENTS.md` clause 5 forbids." Repetition needs no
property of the gate beyond being executable, so the wide reading fails closed exactly
where the IR cannot express the answer — which, for an integer exponent, is nowhere.

### The one place it does fail closed, and the one it does not

Two powers are **not** a repetition, and both are handled by refusing rather than
approximating:

- A **fractional** exponent is the matrix power of a gate. `H**0.5` is a real matrix but
  no instruction describes it, so `power(2.5)` is a `TypeError`. This differs from
  PennyLane, which accepts it; see the measured comparison below and the owned gap in
  `docs/reference/KNOWN_LIMITATIONS.md`.
- A **negative** exponent is the inverse, so it routes through `Circuit.adjoint` and
  inherits its refusals: `fq.Circuit(1).depolarizing(0, 0.1).power(-1)` is a
  `CapabilityError`, because a channel has no inverse.

The second is deliberate and is worth stating as a decision, because the obvious
alternative — refuse a channel at *any* exponent, for symmetry — is worse. `power(2)` of a
channel is a program: two depolarizing events in sequence. Refusing it would report a
limitation this repository does not have, in the same way the narrow reading does. The
sign of the exponent is what decides, and the contract says so.

### The rules agree with an independent implementation

FlagQuantum has no `__pow__` on `Circuit`, so the comparison is against PennyLane 0.45.1's
operator-level `**`, which is the alignment target named by this programme. On 0.45.1 `**`
returns an unevaluated power operator, so both readings are quoted below: the raw operator
is what `**` builds, and `qml.simplify` is what evaluates it. Measured with
`pennylane==0.45.1`:

```python
>>> import pennylane as qml
>>> qml.RX(0.3, wires=0) ** 2
RX(0.3, wires=[0])**2
>>> qml.simplify(qml.RX(0.3, wires=0) ** 2)
RX(0.6, wires=[0])
>>> qml.simplify(qml.RX(0.3, wires=0) ** 0)
I(0)
>>> qml.simplify(qml.RX(0.3, wires=0) ** -1)
RX(12.266370614359172, wires=[0])
>>> qml.simplify(qml.Hadamard(0) ** 2)
I(0)
>>> qml.simplify(qml.Hadamard(0) ** 3)
H(0)
>>> qml.simplify(qml.CNOT(wires=[0, 1]) ** 2)
I([0, 1])
>>> qml.simplify(qml.U3(0.1, 0.2, 0.3, wires=0) ** 2)
U3(0.1, 0.2, 0.3, wires=[0]) @ U3(0.1, 0.2, 0.3, wires=[0])
>>> qml.simplify(qml.S(wires=0) ** 2)
Z(0)
```

Four of those agree with this implementation exactly, and the two that look different agree
too:

- `RX(0.3) ** -1` is `RX(0.3 - 2*pi)`, which is the same operator as `RX(-0.3)`; FlagQuantum
  writes the latter. Deviation between the two spellings: `6.11e-16` at `complex128` and
  `0.0` at `complex64`.
- `U3 ** 2` is *not* simplified by PennyLane, and the scaled form is far from it: measured
  above at `2.19e-01`. So PennyLane and this implementation agree on which gate has a
  one-instruction square, and both leave `u3` as two instructions.
- `qml.QFT(wires=[0, 1]) ** 2` executes, and is the same state as `QFT; QFT` — PennyLane
  applies a power to a *template*, i.e. to a multi-instruction program. This implementation
  reaches the same program by repeating, and reaches the same state.

What the comparison does **not** establish is where PennyLane's fractional powers land, and
that is recorded as a gap rather than glossed: `qml.Hadamard(0) ** 0.5` constructs and runs
on `default.qubit` (it is `H^0.5`, a real unitary), while `fq.Circuit(1).h(0).power(0.5)`
raises `TypeError`. PennyLane also accepts a power of a *channel* object
(`qml.DepolarizingChannel(0.1, 0) ** 2` constructs) and then refuses it at execution on
`default.qubit` with `DeviceError ... does not provide a decomposition`; FlagQuantum accepts
`power(2)` of a channel and executes it, and refuses only `power(-1)`. Neither framework
makes an approximation.

### Affected user journeys

1. **Trotterized time evolution.** `block.power(steps)` is the propagator, and a
   single-angle block collapses to one instruction, so the depth of the training circuit
   does not grow with `steps`.
2. **Depth scaling.** A variational `Module` whose ansatz is `layer.power(depth)` for a
   swept `depth`, with the same qubit count, batch size, and dtype at every `depth`.
3. **Echo and return-probability experiments.** `block.adjoint().power(n)` and
   `block.power(n)` are the two halves; the inverse rule is not restated here, it is read
   from `Circuit.adjoint`.

## Evidence

Repository state: `origin/main` at `98bbcb715`, with this proposal applied. Every number
below was measured on this branch, not transcribed from the plan.

| Claim | Command | Result |
| --- | --- | --- |
| A power is the program applied that many times | `pytest tests/unit/test_circuit_power.py -k matches_the_repeated_program` | 155 tests; worst `max abs(U^power(k) - U applied k times)` = `3.51e-16` over the 31 unitary opcodes times `k in {0,1,2,3,5}` at `complex128` |
| A negative power is the matrix power of the program | `pytest tests/unit/test_circuit_power.py -k "negative_power_matches_the_inverse_applied or equals_the_matrix_power_of_the_program"` | 101 tests: worst `1.45e-15` over the 31 unitary opcodes times `k in {-1,-2,-3}` (93 of them), and `7.77e-16` for `h(0).rz(1,0.7).cnot(0,1)` against `torch.linalg.matrix_power` (8 of them) |
| The single-gate rewrite is exact for the gates it names | `python -m tools.check_circuit_composition_contract` (`_power_measurement_errors`) | worst `5.551e-16` over 12 opcodes times 3 angles times 3 exponents |
| The rewrite is not vacuously true | same gate, `u2` and `u3` rows | `u3(theta=0.4, phi=0.9, lbd=1.3)` `2.191289381e-01`, `u2(phi=0.9, lbd=1.3)` `4.135343356e-01` — both above the `1e-9` tolerance |
| The rewrite happens only for a one-instruction program | `pytest tests/unit/test_circuit_power.py -k rewrite_is_not_applied_to_a_longer_program` | `rx(0.3).cnot(0,1)` at `k=2` is `['rx', 'cx', 'rx', 'cx']`, not `rx(0.6).cnot.cnot` |
| An angle scaling is differentiable | `pytest tests/unit/test_circuit_power.py -k gradient_flows_through_a_scaled_angle` | `theta.grad = -0.29552020666133955` through `power(2)`, byte-identical to the gradient of `rx(0, 2.0 * theta)` written by hand and to `-sin(0.3)` |
| `power(1)` does not rewrite the parameters | `pytest tests/unit/test_circuit_power.py -k power_one_is_the_same_program` | a symbolic parameter comes back as `Parameter(name='t')`, not `t * 1`; a trainable angle comes back as the same tensor object |
| Non-integers fail closed | `pytest tests/unit/test_circuit_power.py -k non_integer_exponent_is_refused` | 8 tests; `TypeError` for `2.5`, `0.5`, `'2'`, `None`, `True`, `False` and `[2]`; `power` is not built on `int()` |
| The instruction bound fails closed | `pytest tests/unit/test_circuit_power.py -k instruction_bound_is_measured` | `power(MAX_POWER_REPEATS)` is accepted and `power(MAX_POWER_REPEATS + 1)` raises `ValueError` naming the emitted count, with `MAX_POWER_REPEATS = 4096` |
| The serialized declaration changed consistently | `python -m tools.operator_manifest --check` after regeneration | one new `"power"` key per operator, 35 lines changed |
| Rule agreement with PennyLane | `python -c "import pennylane as qml; print(qml.simplify(qml.RX(0.3, wires=0) ** 2))"` with `pennylane==0.45.1` | `RX(0.6, wires=[0])`; the same operator as FlagQuantum's `rx(0.6)`, deviation `0.0` on the dense operator |
| The `-1` spellings are the same operator | same interpreter, `qml.simplify(qml.RX(0.3, wires=0) ** -1)` | `RX(12.266370614359172, wires=[0])` vs FlagQuantum `rx(-0.3)`; deviation `6.11e-16` at `complex128` and `0.0` at `complex64` |

## Decision Candidates

**A. `Circuit.power(k)` accepting every instruction for an integer `k` (chosen).**
Repetition is exact for every instruction, so nothing has to be refused except the two
cases the IR cannot express: a fractional exponent, and a negative exponent of something
with no inverse. The single-gate rewrite is an optimization over that, not a precondition
for it, so no gate is excluded for lack of a rewrite.

**B. The narrow reading — accept only the gates whose angle scales.** Rejected. It refuses
`h.power(2)`, which is exactly `I` and which PennyLane answers with `I(0)` under
`qml.simplify`. Per the
sibling record's candidate E, refusing an operation the framework can express reports a
limitation this repository does not have.

**C. Support fractional exponents for the 12 angle-scaling opcodes.** Deferred, not
rejected. `RX(0.3).power(0.5) == RX(0.15)` is exact and PennyLane agrees, so this is a real
and reachable alignment win. It is left out of this slice because it makes the method's
contract two contracts — "applied `k` times" and "exactly the matrix power of one gate" —
and this slice's authorized surface is the integer one. It is recorded as an owned gap.

**D. Refuse a channel at any exponent.** Rejected. `power(2)` of a channel is two noise
events in order, which is a program; only `power(-1)` has no answer. A method whose
acceptance does not depend on the sign of its argument is a smaller contract.

**E. Compute the power from the gate matrix.** Rejected, for the reason the sibling record
gives for `adjoint`: an arbitrary matrix cannot be compiled, lowered, drawn, or recognized
by a peephole pass, and it would turn a differentiable angle into a constant matrix. The
matrix route is kept only where the matrix is what executes — a custom operation.

**F. Add `Circuit.__pow__` as a second spelling.** Rejected here. `**` binds tighter than
attribute access in a way that makes `c ** 2` on a `Circuit` read as arithmetic on an
object that is not a number, and the method form is the one `066` authorized. A `__pow__`
that delegates to `power` is a candidate for a later slice if user evidence asks for it.

## Prohibited Practices

1. Do not add a second power table. Any caller that needs an opcode's power rule reads
   `OperatorSchema.power_rule`.
2. Do not resolve an unrepresentable power by copying the instruction forward unchanged or
   by dropping it. A silent identity and a silent omission are both wrong programs.
3. Do not widen the instruction-bound refusal to `CapabilityError`. The program is
   expressible; the request is out of the supported size, which is a `ValueError`.
4. Do not make `power` an IR node. It is a construction-time rewrite that emits
   instructions the IR already describes.
5. Do not update `docs/operator_manifest.json` by hand. It is generated.
6. Do not scale every declared parameter by the exponent. That is wrong for `u2` and `u3`,
   measured above at `1e-1`.

## Compatibility

- `IR_VERSION` stays `1.0`. `power` rewrites the instruction sequence; it does not add an
  instruction kind.
- No root export is added, removed, or renamed. `fq.__all__` stays at **36**,
  `docs/public_api_v1.json` and `docs/generated/STABLE_API.md` are unchanged, and
  `tools/public_api_snapshot.py` passes.
- `Circuit.power` is a new method on an existing stable class. It does not change any
  existing signature, default, or documented exception.
- `OperatorSchema` gains one property and no constructor argument, so every existing
  `OperatorSchema(...)` call site is unaffected.
- `docs/operator_manifest.json` changes: every operator gains one `"power"` key. The
  manifest's `schema_version` stays `flagquantum_operator_manifest_v1`, and the same
  consumer note as the `adjoint` change applies: compare against `POWER_RULES` rather than
  against a literal rule name. No in-repository consumer reads the new key except this
  contract's gate.
- `contracts/circuit-composition-contract.toml` retires one `not_provided` entry
  (`Circuit.power`) and adds a `[power]` table plus two refusal rows. The contract's
  `schema` string is unchanged, because the surface is the same one it always described.
- Known limitation, deliberately not closed here: a fractional exponent is refused, and it
  is recorded as an owned gap in `docs/reference/KNOWN_LIMITATIONS.md` with the measured
  PennyLane contrast.

## Acceptance Tests

- `tests/unit/test_circuit_power.py` — the semantics: the repetition identity against the
  dense program operator, the matrix-power identity for negative exponents, the
  single-gate rewrite and the `u2`/`u3` counterexamples, `power(0)` and `power(1)`, the
  receiver being left alone, the preserved width/batch/dtype, symbolic and per-batch
  angles, a custom matrix, every refusal, and the instruction bound.
- `tests/unit/test_circuit_composition_contract.py` — the contract rows for `power`.
- `tools/check_circuit_composition_contract.py` — the closed-form measurement, the rule
  census, the argument list, and the bound against `MAX_POWER_REPEATS`.
- `tools/operator_manifest.py --check` — the regenerated manifest is current.
- `tools/public_api_snapshot.py` — the stable surface is unchanged.
- `tools/check_team_scope.py --team core` — the changed paths are owned.

## Open Questions

1. Does the fractional exponent deserve its own slice for the 12 angle-scaling opcodes?
   The measurement says it is exact and PennyLane agrees; the cost is a second contract on
   one method. Decision candidate C records both halves.
2. Should `Circuit.control` fold into this family next, or does it need a surface of its
   own? Its rewrite is not a repetition — it adds a control ladder — so it is not clear
   that `OperatorSchema` can predict it the way it predicts `adjoint` and `power`.
3. `N1-8` proposes a machine-checked `issue_code` vocabulary for the circuit expression
   contract. This method reports its two refusals as `TypeError` and `ValueError` messages
   because no such vocabulary exists yet; when it lands, both should carry codes.

## Owner and approvals

Owner: FlagQuantum core maintainers (team `core` in `team-ownership.toml`).

Approval recorded by this document: the method name `Circuit.power`, its single argument
`exponent`, the **wide** reading of `066` §6.2, the two rule names in `POWER_RULES`, the
sign-of-the-exponent rule for a channel, the integer-only exponent, and
`MAX_POWER_REPEATS = 4096` as the size bound. Not approved here: any change to
`IR_VERSION`, to a root export, or a `__pow__` spelling.
