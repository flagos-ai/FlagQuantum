# Circuit control: `Circuit.control`

## Decision and authorization

Status: **implemented on this branch, pending review.** This proposal authorizes one
new method, `Circuit.control(n_controls, ctrl_qubits)`, one new keyword argument
name, `ctrl_qubits`, one new reading of `OperatorSchema`, and one new serialized
key per operator in `docs/operator_manifest.json`. It adds no stable root export,
introduces no new opcode, and leaves `IR_VERSION` at `1.0`.

It is written under non-negotiable rule 8 of `AGENTS.md` ("Treat the Stable Core
public API as protected"). The numbered half of the family decision is
[`API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md`](../development/API_CHANGE_PROPOSAL_066_CIRCUIT_COMPOSITION.md),
which records `compose`, `adjoint`, `control`, and `power` as one family and fixes
the *shape* of `control`: its signature, the new-circuit/no-mutation rule, the
width rule, the statevector-measured acceptance test, and the refusal table.
`066` §6.1 deliberately left three things to this document — the mechanism, the
ancilla story, and the exact refusal phrases — and this document settles them.
`066` is not replaced or amended by this record; §6.1 is what it always said, and
the three items it deferred are now decided.

The sibling changes have their own records:
[`FQ-CIRCUIT-COMPOSITION-20261002.md`](FQ-CIRCUIT-COMPOSITION-20261002.md) owns
placement, [`FQ-CIRCUIT-ADJOINT-20261003.md`](FQ-CIRCUIT-ADJOINT-20261003.md) owns
inversion, and `FQ-CIRCUIT-POWER-20261021.md` owns repetition — that last record is
proposed on a separate branch and is not part of this change, so it is named here rather
than linked. Nothing here approves `power`.

The authorization question was whether this needs one at all, and the answer is
that it needs two. Measured against this checkout:

- `docs/public_api_v1.json` freezes **36 stable exports** in `stable_exports` and
  **36** in `verification`, and `fq.__all__` is **36** after this change as well,
  so no frozen root name is added, removed, or renamed.
- `Circuit` itself is already a stable export, so `Circuit.control` is a new
  method on an existing stable class and it does not change any existing
  signature, default, or documented exception.
- `OperatorSchema` gains a field, and `docs/operator_manifest.json` gains one key
  per operator. That is a **serialized public schema** change, which rule 8
  protects, and it is the first of the two authorizations this document records.

The second is the new public keyword name. `066` §6.1 fixes the argument as a
qubit sequence spelled with `qubit`, and this document fixes it as
`ctrl_qubits`. A second new keyword — an ancilla argument — was considered and
**declined**; the reasoning is in Decision Candidates below, and the decline is
recorded because a declined argument that is not written down is an argument
someone will add later.

Scope of the affected surface: `flagquantum/core/operator_schema.py` and
`flagquantum/core/controlled.py` (both owned by team `core`),
`flagquantum/core/__init__.py`, `flagquantum/circuit.py` (owned by team `core`),
`docs/operator_manifest.json` (generated), and tests.

## Problem and affected user journey

### A controlled program could not be expressed

FlagQuantum could take a circuit apart, place it, and invert it, but not make it
conditional. Measured on `origin/main` at `fc836f80`:

```python
>>> import flagquantum as fq
>>> fq.Circuit(1).x(0).control(1, ctrl_qubits=(1,))
AttributeError: 'Circuit' object has no attribute 'control'
```

The gap is real for a user and not cosmetic. Grover's diffusion step, the
multi-controlled rotation in every phase-estimation circuit, and the fan-in of a
Toffoli network all need "run this program only when these qubits are set".
Every one of those users had to build the ladder by hand from `h` and `cphase`,
get the recursive angle halving right, and get the per-opcode basis change right.

### The declaration existed for one opcode at a time, and read nowhere

`OperatorSchema` has carried an `adjoint` field that this family now reads, and
the same field set had no controlled counterpart at all. The repository instead
held the knowledge in two hand-maintained places with two different mechanisms:
`flagquantum/algorithms/primitives/oracle.py::append_multi_controlled_x`, which
needs caller-supplied ancillas, and the `apply_power_controlled` protocol copied
into `algorithms/svd.py` and `algorithms/pca.py`.

The first of those is the precedent this proposal measured and rejected. It
recurses on the ancilla count and, at three controls, requires one caller-supplied
ancilla that must start in `|0>`. Measured over every computational-basis operand
of a five-qubit register at three controls (32 operands, one ancilla) and a
six-qubit register at four controls (128 operands, two ancillas), applying the
precedent and reading back the resulting basis state:

| Controls | Operands measured | Wrong result | Ancilla left dirty |
| --- | --- | --- | --- |
| 3 | 32 | **8** | 0 |
| 4 | 128 | **32** | 0 |

The ancilla is restored and the *answer* is wrong. The eight failing three-control
operands are exactly the ones where the ancilla starts in `|1>` and the other two
controls are both set: the target is flipped when it should be left alone, and
left alone when it should be flipped. That is the honesty trap. A convenience
wrapper around a construction whose precondition `Circuit` cannot check and whose
violation is silent is not a capability; it is a wrong answer with a friendly
name. The measurement is why the mechanism below needs no ancillas at all.

### The rule is predictable from the matrix, and the prediction is measured

A controlled unitary is not a rule about opcode names; it is a rule about
matrices. The ladder below is derived from the gate matrices this repository
executes (`simulation/matrices.py`) and then *re*-derived from an independent
reference implementation, so the two derivations are not the same code read
twice. Measured at `complex128` over all **31 unitary opcodes** and control
counts `k = 1..4`, that is **124 probes**:

```text
$ PYTHONPATH=. python /tmp/n19/n14_verify_core.py     # instrument, not shipped
failures=0 worst_maxdiff=1.355e-13
```

and end to end, through `Circuit.control` and `Circuit.state()`, over the same 31
opcodes and `k = 1..4`:

```text
$ PYTHONPATH=. python /tmp/n19/n14_e2e.py             # instrument, not shipped
statevector acceptance: failures=0 worst_maxdiff=1.409e-13
```

Both numbers are at the rounding floor of the reference, not a tuned tolerance.

### The ladder, stated once

Only one primitive is needed. For controls `c_1..c_k`, a target `t`, and a phase
angle `phi`, `CtrlPhase(k, controls, target, phi)` applies `P(phi)` to `t` when
every control is set and the identity otherwise. It is defined recursively with no
ancilla:

```text
CtrlPhase(1, [c], t, phi) = CPhase(c, t, phi)
CtrlPhase(k, [c_1..c_k], t, phi) =
    H(t)
    CPhase(c_1, t, pi)
    H(t)
    CtrlPhase(k-1, [c_2..c_k], t, -phi/2)
    H(t)
    CPhase(c_1, t, pi)
    H(t)
    CtrlPhase(k-1, [c_2..c_k], t, +phi/2)
    CtrlPhase(k-1, [c_1, c_3..c_k], t, +phi/2)
```

The emitted-instruction count `T(k)` of that recursion satisfies
`T(k) = 3 * T(k-1) + 6` with `T(1) = 1`, so

```text
T(k) = 4 * 3**(k-1) - 3      1, 9, 33, 105, 321, 969, 2913, 8745, 26241, 78729
```

and the level a gate needs is not its control count but

```text
level = n_controls + arity - 1
```

because a multi-qubit target consumes ladder rungs too: the two-qubit target is
itself resolved by one more level of the same recursion. The gate checks
`control_ladder_level(arity, n_controls) == n_controls + arity - 1` over the grid
arity `1..3` by control count `1..3`, reading the formula off the implementation
rather than restating it here.

Every opcode then reaches the primitive through an exact factorization, and the
factorization is where the global phase lives. `C^k(V)` for a conjugated gate is
**not** `V_t * CtrlPhase(...) * V_t^dagger`: the two `V_t` factors are not
controlled, so a global phase `e^{i*gamma}` carried by the target's factorization
turns into a *relative* phase and the naive spelling is wrong by a controlled
phase. A separate `Corr(k, controls, gamma)` term pays it back, and three
factorizations need it — `rz`, `ry`, and the `u` family — which is exactly the
set whose declared rules are `rz_ladder`, `ry_ladder`, and `u_angle_ladder`.

### The cost is real, and it is published rather than hidden

`T(level)` is exponential in the level. Measured emitted-instruction counts, in
native mode, for control counts `k = 1..6`:

| Opcode | arity | `k=1` | `k=2` | `k=3` | `k=4` | `k=5` | `k=6` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `i` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `z` `s` `sdg` `t` `tdg` `phase` `u1` | 1 | 1 | 9 | 33 | 105 | 321 | 969 |
| `x` | 1 | 1 | 11 | 35 | 107 | 323 | 971 |
| `h` `sx` `sxdg` | 1 | 3 | 11 | 35 | 107 | 323 | 971 |
| `y` | 1 | 1 | 13 | 37 | 109 | 325 | 973 |
| `rz` | 1 | 1 | 10 | 42 | 138 | 426 | 1290 |
| `rx` | 1 | 1 | 12 | 44 | 140 | 428 | 1292 |
| `ry` | 1 | 1 | 14 | 46 | 142 | 430 | 1294 |
| `u2` `u3` | 1 | 8 | 32 | 112 | 352 | 1072 | 3232 |
| `cz` `cphase` | 2 | 9 | 33 | 105 | 321 | 969 | 2913 |
| `crz` | 2 | 10 | 42 | 138 | 426 | 1290 | 3882 |
| `crx` | 2 | 12 | 44 | 140 | 428 | 1292 | 3884 |
| `cry` | 2 | 14 | 46 | 142 | 430 | 1294 | 3886 |
| `rzz` | 2 | 24 | 80 | 256 | 784 | 2368 | 7120 |
| `rxx` | 2 | 28 | 84 | 260 | 788 | 2372 | 7124 |
| `ryy` | 2 | 32 | 88 | 264 | 792 | 2376 | 7128 |
| `cx` | 2 | 1 | 35 | 107 | 323 | 971 | 2915 |
| `cy` | 2 | 13 | 37 | 109 | 325 | 973 | 2917 |
| `swap` | 2 | 1 | 105 | 321 | 969 | 2913 | 8745 |
| `ccx` | 3 | 35 | 107 | 323 | 971 | 2915 | 8747 |
| `cswap` | 3 | 105 | 321 | 969 | 2913 | 8745 | 26241 |

The table is not decoration. It is the evidence for two contract claims: that
`k=1` on a gate with a registered one-control partner emits **one** instruction
(`x -> cx`, `cnot -> ccx`, `swap -> cswap`, `rz -> crz`) and not a ladder, and
that the ladder itself is the recursion's cost and not an artifact of how the
emitter is written. A contract check re-derives the bound
`4 * 3 ** (level - 1) - 3` for the opcodes whose declared rule is
`diagonal_ladder` — `z`, `t`, `phase`, `s`, `sdg`, `cz`, `cphase`, at levels 1
through 6 — because those are the opcodes with no wrapper around the primitive,
so the identity is measurable directly instead of through a correction term.

The ceiling follows from the same number. `MAX_LADDER_LEVEL = 10` is enforced,
and the widest route that fits is a three-qubit target at 8 controls, which is
level 10 and emits **236193** instructions. Beyond that the emitter refuses with
a `ValueError` naming the level it would need and the ceiling, rather than
running until the machine gives up.

## Evidence

Repository state: `origin/main` at `fc836f80`, with this proposal applied. The
measurements below were taken there. The sibling `N1-2` (qubit vocabulary, PR
#522) is still open and does not touch `core/operator_schema.py`'s field set, so
nothing measured here moves with it.

| Claim | Command | Result |
| --- | --- | --- |
| The emitted program is the reference ladder, instruction by instruction | `python /tmp/n19/n14_verify_core.py` (the scratch reference; the same comparison is now a test, next row) | 124 probes (31 unitaries at `k=1..4`), `failures=0` |
| The emitted program computes the right operator | `pytest tests/unit/test_circuit_control.py -k runs_only_when_every_control_is_set` | 31 opcodes x 4 control counts, operator against the definition, `atol=1e-12`; the scratch route's residual was `worst_maxdiff = 1.355e-13` |
| End to end through `Circuit.control` and `Circuit.state()` | `python /tmp/n19/n14_e2e.py`; the in-repo half is `pytest tests/unit/test_circuit_control.py -k multi_controlled_rotation_inside_a_wider_program` | `failures=0`, `worst_maxdiff = 1.409e-13` |
| The receiver is not mutated | `pytest tests/unit/test_circuit_composition_contract.py -k control_flags` | the receiver's instruction list is identical before and after |
| The result width is one past the greatest control | same test | `fq.Circuit(3).h(0).control(2, ctrl_qubits=(7, 9)).n_qubits == 10` |
| The receiver's input state is carried | same test | `[0.6, 0.8]` on one qubit becomes `[0.6, 0, 0.8, 0]` on two, at `1e-14` |
| Zero angles are emitted, not elided | same test | `rz(0, 0.0)` and `rz(0, 0.37)` emit the same count |
| Every emitted opcode is registered | same test | no opcode in the expansion is absent from `OPERATOR_SCHEMAS` |
| The one-control shortcut is the registered partner | `pytest tests/unit/test_circuit_composition_contract.py -k expands_to_the_hand_built_ladder` | `x -> cx`, `cnot -> ccx`, `swap -> cswap`, `rz -> crz`, one instruction each |
| The ladder is written out by hand and matched | same test | `t` at 2 controls equals the nine-instruction `h`/`cphase` program |
| The ladder computes the gate | same test | four prepared registers against the definition's basis and phase |
| The census is load-bearing | `python tools/check_circuit_composition_contract.py` | 31 of 35 opcodes declare a form; the 4 refusals are exactly the channels |
| The ceiling is enforced | same run | level 11 refuses with the level it needs and `MAX_LADDER_LEVEL` |
| The serialized field changed consistently | `python tools/operator_manifest.py --check` after regeneration | 35 added lines, one `control` key per operator |

The two `/tmp/n19/` instruments are **scratch probes, not repository files**, and they
were written before the emitter existed so that the recursion could be checked against a
torch-free `float64` reference rather than against itself. Both were re-run unchanged at
the committed head; the durable half of what they measured was ported into
`tests/unit/test_circuit_control.py`, which is why the first three rows name a selector as
well. A reader who has only the repository can reproduce every claim in the table from the
tests; a reader who also has the probes can reproduce the residuals to the digit shown.

## Decision Candidates

**A. Emit the ancilla-free phase ladder directly from a new `core` module
(chosen).** `flagquantum/core/controlled.py` owns the recursion, the
factorizations, and the phase correction. It emits only registered opcodes — `h`,
`s`, `sdg`, `phase`, `cphase`, `ry`, and the native partners — so the result is a
program the IR already describes, the compiler already lowers, and the drawer
already draws. It needs no ancillas, so it has no precondition the caller can
violate.

**B. Reuse `algorithms/primitives/oracle.py::append_multi_controlled_x`.
Rejected, three times over.** It is a layer violation: `flagquantum/core/AGENTS.md`
forbids `core` from importing any implementation layer, and `algorithms` is one.
It needs `len(controls) - 2` caller-supplied ancillas, and `Circuit` has no way to
name a free qubit that is guaranteed clean. And when the precondition is violated
it returns a wrong answer rather than an error — measured above at 8 of 32
operands with three controls and 32 of 128 with four.

**C. Add a second new keyword, `ancillas=`, with a linear construction.
Declined.** With ancillas the ladder is linear rather than exponential, which is a
real improvement, and for a user who *does* have clean qubits it is the right
construction. But it needs its own rule-8 authorization, and `066` §6.1 commits
the family to spelling an ancilla argument with `qubit`, so the name would have to
be settled too. More importantly, the confirmed ladder needs no ancillas, so the
argument would be dead weight on every call that does not use it, for a
construction whose precondition `Circuit` cannot check. This is recorded as
declined rather than forgotten; a later change that wants the linear construction
owns the argument and the precondition check.

**D. Emit a raw matrix for each controlled gate.** Rejected. It would give the
right operator and the wrong artifact: `066` §6.1 requires every emitted
instruction to carry an opcode in the 35-opcode registry, precisely so that a
controlled program is compilable, lowerable, discoverable by `merge_self_inverse`
and friends, and drawable. A matrix also evades the `OperatorSchema.control`
declaration, which would leave the declaration unread again.

**E. Declare `not_available` for everything and refuse.** Rejected. It would
report a limitation this repository does not have, which `AGENTS.md` clause 5
forbids, and the measurements above show 31 of 35 opcodes have a closed form.

**F. Resolve the global phase by dropping it.** Rejected. `C^k(V P(phi) V^H)` is
not `V_t C^k(P(phi)) V_t^H`, and the difference is a controlled phase, not a
global one — dropping it changes the operator. This is why `Corr` exists and why
`rz_ladder`, `ry_ladder`, and `u_angle_ladder` are separate rules rather than
sharing `diagonal_ladder`.

## Prohibited Practices

1. Do not add a second controlled-form table. Any caller that needs an opcode's
   controlled form reads `OperatorSchema.control` through
   `core.controlled_instructions`.
2. Do not import an implementation layer into `core` to reuse a construction.
   `flagquantum/core/AGENTS.md` forbids it, and the measured precedent above is
   what the prohibition is protecting against.
3. Do not accept an ancilla argument without a check that the named qubits start
   in `|0>`. An unchecked ancilla argument is Decision Candidate B with a nicer
   signature.
4. Do not elide a zero angle. A `Parameter`, a per-batch list, and a bound tensor
   cannot all be compared to a float, and a tolerance that dropped a small
   non-zero angle would change the program by more than round-off. The emitted
   count is a function of shape alone.
5. Do not re-angle a gate the caller wrote. The ladder angles must carry the
   instruction's own parameter objects (`p * 0.5`, `-p`), so a trainable rotation
   stays in the autograd graph. This is the standing rule of
   `flagquantum/compiler/basis_translation.py`.
6. Do not silently drop metadata. An instruction carrying metadata other than
   `is_channel`, `is_dynamic`, or `conditions` is refused rather than copied
   forward, because a controlled form is a different program and metadata that
   records how the original was placed does not describe the replacement.
7. Do not widen a refusal to `ValueError` where `CapabilityError` is the class the
   errors module reserves for an absent capability.
8. Do not update `docs/operator_manifest.json` by hand. It is generated.

## Compatibility

- `IR_VERSION` stays `1.0`. `control` rewrites the instruction sequence; it does
  not add an instruction kind, and every emitted opcode is one the registry
  already declares.
- No opcode is added. The registry remains 35 opcodes, and the four channels
  remain the only ones that declare no controlled form.
- No root export is added, removed, or renamed. `docs/public_api_v1.json`'s
  `stable_exports` and `verification` stay at **36**, and `fq.__all__` stays at
  **36**. There is deliberately no module-level `fq.control`: the family is
  methods on `Circuit`, and a module-level spelling would be a second way to say
  the same thing.
- `Circuit.control` is a new method on an existing stable class. It changes no
  existing signature, default, or documented exception.
- `OperatorSchema` gains one field, `control`, defaulting to `"not_available"`,
  and `CONTROL_RULES` names the 15 values it may take. The default is the
  fail-closed value: a schema that does not declare a form is refused rather than
  guessed at. `docs/operator_manifest.json` gains one `control` key per operator;
  its `schema_version` stays `flagquantum_operator_manifest_v1`. A consumer that
  iterated the operator dicts tolerantly is unaffected; a consumer that asserted
  an exact key set must now expect `control`.
- Known duplication, deliberately not removed here: the three
  `apply_power_controlled` copies in `flagquantum/algorithms/**`, and
  `append_multi_controlled_x` itself. Replacing them is an `algorithms`-owned
  change — `N1-10` — because `core` may not reach into that layer and that layer's
  callers need the ancilla-saving construction this proposal declined.

## Acceptance Tests

- `tests/unit/test_circuit_composition_contract.py` — the `[control]` table of
  `contracts/circuit-composition-contract.toml`, its 13 refusal rows, the
  30-row refusal vocabulary, the opcode census, the expansion test that the gate
  requires by name, and the eleven mutations the gate must reject.
- `tests/unit/test_circuit_control.py` — per-opcode statevector acceptance against
  the definition of a controlled gate, the ladder written out by hand, the
  one-control shortcut, the parameter-object identity of every emitted angle, the
  refusal paths, and the ceiling.
- `tools/check_circuit_composition_contract.py` — re-derives the level formula,
  the ladder bound, the census, the ceiling, the signature, and the width and
  input-state claims from the implementation rather than from the contract text.
- `tools/operator_manifest.py --check` — the regenerated manifest is current.
- `tools/public_api_snapshot.py` — the stable surface is unchanged.
- `tools/check_team_scope.py --team core` — the changed paths are owned.

## Open Questions

1. Should `Circuit.control` accept a `control_values` argument, so that a control
   can be satisfied by `|0>` rather than `|1>`? PennyLane's `qml.ctrl` has one. It
   is a second new keyword and needs its own authorization, and the current
   signature can express the same programs by conjugating with `x` on the control
   qubit, so it is not a capability gap — it is an ergonomics question.
2. Should the linear-with-ancillas construction be added later, behind an explicit
   `ancillas=` argument whose named qubits `Circuit` verifies start in `|0>`? The
   decline above is a decline of the *unchecked* form. The checked form is a real
   improvement for `k >= 5`, where the exponential count first becomes painful, and
   it is the shape the three `apply_power_controlled` copies want.
3. `MAX_LADDER_LEVEL = 10` is a ceiling on emitted instructions, not on controls,
   and it was chosen so that the widest reachable route stays under a quarter of a
   million instructions. Should it be a configuration value rather than a module
   constant? It is a constant today because a limit that a user can raise is a
   limit that will be raised by someone who has not read the table above.
4. `N1-8` proposes a machine-checked `issue_code` vocabulary for the circuit
   expression contract. These refusals are `CapabilityError` and `ValidationError`
   messages with contracted phrases because no such vocabulary exists yet; when it
   lands, the 13 rows should carry codes.
5. Should `control` and `power` compose in one call — `circuit.control(1,
   ctrl_qubits=(2,)).power(3)` is legal today and is not the same program as
   `circuit.power(3).control(1, ctrl_qubits=(2,))`. The difference is real and
   correct, but it is the kind of thing a user will get wrong once, so a documented
   golden path in `docs/guides/CIRCUIT_COMPOSITION.md` may be worth more than an
   API change.

## Owner and approvals

Owner: FlagQuantum core maintainers (team `core` in `team-ownership.toml`).

Approval recorded by this document: the method name `Circuit.control`, the keyword
name `ctrl_qubits`, the `OperatorSchema.control` field and the 15 values in
`CONTROL_RULES`, the `"not_available"` default, the `control` key in
`docs/operator_manifest.json`, and `CapabilityError`/`ValidationError` as the
refusal classes. Recorded as declined: an `ancillas=` argument, and a
`control_values=` argument. Not approved here: any change to `IR_VERSION`, to a
root export, to the opcode registry, or to the three `apply_power_controlled`
copies and `append_multi_controlled_x`, which `N1-10` owns.
