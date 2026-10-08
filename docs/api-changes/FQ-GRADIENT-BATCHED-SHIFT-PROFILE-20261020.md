# The batch parameter-shift profile reads the opcode declaration

- **Status:** implemented
- **Owner:** core (`flagquantum/gradients.py`)
- **Date:** 2026-10-20
- **Related:** [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md),
  [Circuit composition](FQ-CIRCUIT-COMPOSITION-20261002.md),
  [`fq.gradient` and the reported gradient method](FQ-GRADIENT-API-20261002.md)

## Problem

`batched_parameter_shift_gradient` evaluates a gradient through one remote batch
request. It sends one positively shifted and one negatively shifted circuit per
input parameter, combines the returned losses, and hands back a detached tensor.
Two evaluations per parameter is a **two-term** rule, and a two-term rule is
exact only for a gate parameter whose generator has **one frequency**.

Before this change the profile did not read that fact from anywhere. It held it
as three hardcoded sets:

```python
_BATCH_PARAMETER_SHIFT_GATES = frozenset({"h", "x", "rx", "ry", "rz", "cx"})
_PARAMETER_SHIFT_GATES = frozenset({"rx", "ry", "rz"})
```

and the two-term arithmetic itself was written out as
`0.5 * (losses[plus] - losses[minus])` with the displacement guarded by an
equality check against `pi / 2`.

Three things were wrong with that, and they were wrong in different ways.

1. **The lists were narrower than the truth.** `flagquantum/core/operator_schema.py`
   declares one frequency for nine more gates the profile never admitted:
   `U1`, `U2`, `U3`, `PHASE`, `CPHASE`, `RXX`, `RYY`, `RZZ`. A user who wrote a
   remote-safe VQE with `rzz` entanglers -- an ordinary thing to write -- was
   refused by the profile even though the rule it implements is exactly the rule
   `rzz` needs.
2. **The wider rules were answered rather than refused.** `CRX`, `CRY`, and `CRZ`
   declare `{0.5, 1.0}`, which needs two evaluation pairs. They were excluded from
   `_BATCH_PARAMETER_SHIFT_GATES` by absence rather than by a decision about the
   rule, so the only reason a truncated answer was not produced is that nobody had
   yet typed the name into the set.
3. **The constant gates were mixed into the shiftable ones.** `h`, `x`, and `cx`
   declare no parameter. They are in the profile because the remote-safe profile
   has to permit *some* circuit around the differentiated gate, which is a
   statement about which circuits have been exercised -- not a derivative rule.
   Putting them in the same set as `rx`/`ry`/`rz` made the set look like a rule
   when half of it was a scope decision.

The generic path in the same module had already been migrated the round before
(see [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md)):
`parameter_shift_gradient` reads `OperatorSchema.shift_rule` and refuses what it
cannot answer. That record closed with Open Question 3 asking whether
`batched_parameter_shift_gradient` needed the same treatment. It did: the residue
was a third copy of a rule whose owner is the declaration.

## Decision

The profile derives its parameterized half from the declaration, and names only
the half the declaration has no opinion on.

```python
# The constant half carries no parameter, so it states a protocol scope rather
# than a derivative rule; the parameterized half is derived below.
_BATCH_PROFILE_CONSTANT_GATES = frozenset({"h", "x", "cx"})


def _single_pair_shift_opcodes(schemas):
    """Opcodes that differentiate with one positive and one negative evaluation."""
    admitted = set()
    for opcode, schema in schemas.items():
        if not schema.differentiable:
            continue
        if all(len(schema.shift_rule(name)) == 2 for name in schema.parameters):
            admitted.add(opcode)
    return frozenset(admitted)


_BATCH_PROFILE_GATES = _BATCH_PROFILE_CONSTANT_GATES | _single_pair_shift_opcodes(
    OPERATOR_SCHEMAS
)
```

`_BATCH_PARAMETER_SHIFT_GATES` and `_PARAMETER_SHIFT_GATES` are deleted. The
two-term arithmetic is now the declared rule's own coefficient, read per
occurrence by `_batch_occurrence_coefficient`, and the `pi / 2` guard is gone:
`_validated_parameters` only requires the displacement to be a positive finite
number, and a displacement that is not the rule's own now produces a message that
names the displacement the rule prescribes.

Which opcodes qualify is a property of the declaration, so the profile asks and
does not restate. The consequence is the intended one: registering a frequency
set with more terms removes an opcode from this profile without anyone editing
`gradients.py`, and registering a one-frequency gate adds it.

The **right** way to widen a protocol is to make it read the thing that owns the
fact. It is not to add a name to a list next to the one that was already there.

## Scope

| | Before | After |
| --- | --- | --- |
| Admitted, one frequency | `rx`, `ry`, `rz` | `cphase`, `phase`, `rx`, `rxx`, `ry`, `ryy`, `rz`, `rzz`, `u1`, `u2`, `u3` |
| Admitted, no parameter | `h`, `x`, `cx` | `h`, `x`, `cx` (unchanged) |
| Refused, wider rule | by absence | `crx`, `cry`, `crz`, naming the pair count |
| Refused, channel | by absence | `bit_flip`, `phase_flip`, `depolarizing`, `amplitude_damping` |
| Refused, constant outside scope | by absence | every other parameterless opcode, by name |

Everything the profile admitted before it still admits, and every circuit it
refused before it still refuses. This is additive: no user-visible behavior that
was correct changed, and refusals became diagnosable.

### Explicitly out of scope

The five hardcoded gate lists that express a **kernel capability** are untouched:
`flagquantum/simulation/statevector/split_real_imag.py`,
`flagquantum/simulation/statevector/fixed_layer_cpu.py`,
`flagquantum/simulation/statevector/double_single_device_gates.py`, and
`flagquantum/runtime/executors/statevector/split_real_imag.py`. They answer "which
gates can this kernel split or apply natively", which is not the question this
change answers. `SPLIT_REAL_IMAG_PARAMETER_SHIFT_GATES` is additionally read by
`tools/check_split_real_imag_p1_contract.py`, so it is a gated contract surface
rather than an oversight. Open Question 1 of the previous record is therefore
**expired, superseded by measurement**: those lists are not underived copies of
this rule, they are a different rule. `P4_PARAMETER_GATES` in particular is a
subset of what the derivation now admits, so no kernel is being asked to do
something the profile cannot phrase.

## Prohibited practices

- Do not re-add a gate list to `flagquantum/gradients.py`. The gate census fails
  on any opcode name written in code outside `_BATCH_PROFILE_CONSTANT_GATES`.
- Do not widen `_BATCH_PROFILE_CONSTANT_GATES` to admit a parameterless gate
  without recording why that circuit shape is in the exercised profile.
- Do not answer a wider rule with a truncated two-term number. It is a refusal.
- Do not derive the profile from any source other than `OPERATOR_SCHEMAS`. A
  second source of truth for "which gates are differentiable" is the defect this
  change removes.
- Do not edit `contracts/parameter-shift-coverage-contract.toml` to make the gate
  pass. The contract records a measurement; a disagreement is a finding.

## Compatibility

- **No stable API changes.** `batched_parameter_shift_gradient` is not in
  `docs/public_api_v1.json` (`stable_exports` is 36 and contains no `batched`
  name), so this is not an API-owner change and no change proposal is required.
  `fq.__all__` is unchanged.
- **No capability row moves.** `capability-maturity.toml` records gradient support
  for the training entry points (`fq.train`, `fq.Module`, autograd);
  `batched_parameter_shift_gradient` is an auxiliary helper outside that surface.
  This fixes a support boundary that was narrower than the rule it implements
  rather than moving a claimed capability.
- **`docs/guides/JIUDING.md` is corrected.** Its prose stated the profile's old
  vocabulary ("supports H, X, RX, RY, RZ, and CX"). The guide is not executed by
  any test, so it is corrected by hand and the correction is part of this change.
- **`docs/api-changes/FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md` is amended,
  not rewritten.** Its "batched_parameter_shift_gradient is unchanged" sentence
  and its Open Question 3 were true when written and are expiring assertions; a
  `**Delivered:**` block records which of them this change fulfilled.
- **`docs/api-changes/FQ-GRADIENT-API-20261002.md` is amended the same way.** Its
  "kernels are not modified" sentence described the `fq.gradient` change of that
  round and is corrected by a forward pointer rather than edited.

## Verification

`tools/check_parameter_shift_coverage_contract.py` re-derives the profile from
`OPERATOR_SCHEMAS`, censuses `flagquantum/gradients.py` for opcode names written
in code, drives the implementation for every registered opcode, builds every
refusal row, and compares all of it against
`contracts/parameter-shift-coverage-contract.toml`. It runs in the `quality` job
of `.github/workflows/ci.yml` as a **step** and in `tools/pre_push.py`. It is a
step rather than a job deliberately:
`tools/validate_required_checks.py` pins six externally configured required
checks across four jobs, so adding a job would change the required-check contract
and adding a step does not.

Measured on this checkout:

```console
$ python tools/check_parameter_shift_coverage_contract.py
Parameter-shift coverage contract passed: 11 opcodes differentiate from one evaluation pair, 24 do not
```

The gate is negative-tested, so a zero finding is evidence rather than silence.
Each of the following was applied to the checkout, the gate was run, and the
checkout was restored:

| Mutation | Gate output |
| --- | --- |
| Contract row claims `rzz` is refused | `rzz: contract admitted=False, measured True` |
| `frozenset({"rx", "ry", "rz"})` re-added to `gradients.py` | `the implementation names opcodes in code outside the declared scope: measured ['cx', 'h', 'rx', 'ry', 'rz', 'x'], declared ['cx', 'h', 'x']` |
| Contract declares `crz` frequencies as `[[1.0]]` | `crz: contract declared_frequencies is [[1.0]], measured [[0.5, 1.0]]` |
| Tolerance tightened to `1e-30` | 12 rows report `disagrees with autograd by <e> relative, above the declared 1.000e-30 rule-arithmetic bound` |
| Vacuity floor raised to `1.0` | 14 rows report `the autograd reference is <v>, below the declared floor 1.000e+00, so the comparison would be vacuous` |

The exactness column is not decorative. Each admitted parameter is compared with
PyTorch autograd on a loss that is **non-degenerate for it** -- a bare
differentiated gate followed by a Z measurement sends several of these
derivatives to zero, which would make the comparison vacuous rather than passing,
so the circuit is prepared with `H` then `RZ(0.3 + 0.1 * qubit)` on every qubit,
the differentiated gate follows, and the measurement basis is turned by `RY(0.4)`
then `RZ(0.13)` on qubit 0. The reference magnitudes are asserted against a
declared floor as well as the relative errors against a declared tolerance. Worst
relative disagreement is `1.287e-15` against a `1e-12` bound.

Full-record agreement, measured earlier at `float64` batch losses with one
request per gradient: a four-parameter circuit
`u3(theta=v0) + ry(v1) + rzz(v2) + phase(v3)` gives
`[-0.7410419654151496, 0.14853236160475733, 0.1825409683340126, 0.0]` against
autograd's `[-0.7410419654151498, 0.14853236160475702, 0.18254096833401268, 0.0]`
-- a maximum absolute error of `3.0531133177191805e-16` -- in one
`batch_loss_fn` call of 8 circuits, with the result's shape and dtype preserved.

A refusal that is never built is a refusal nobody has seen work, so the contract
carries eight refusal rows and the gate builds every one of them: a multi-pair
opcode, one input controlling two occurrences, a scaled angle, a channel, a
constant outside the protocol scope, a caller-supplied wrong displacement, a
custom matrix, and a circuit whose structure follows its parameter. Each row
asserts the message fragment, so a refusal that stops firing is a failure rather
than a silent acceptance.

`tests/unit/test_parameter_shift_coverage_contract.py` owns the contract and the
gate: it asks whether the gate still reads the clauses it claims to, by mutating
six of them and requiring the gate to name each one, and it asserts that the gate
is invoked by both CI and `tools/pre_push.py`. `tests/test_native_circuit.py`
owns the semantics and gained one row per admitted parameter plus four new
refusal rows.

```console
$ python -m pytest tests/unit/test_parameter_shift_coverage_contract.py -q
35 passed
$ python -m pytest tests/unit/test_native_circuit.py -k batched_parameter_shift -q
15 passed, 154 deselected
```

## Open Questions

1. **Which eigenvalues does `CRX`'s generator actually have?** Inherited unchanged
   from the previous record. The declared `{0.5, 1.0}` is verified against
   autograd, not derived, and it is now load-bearing for a *refusal* as well as
   for an answer: if the declaration were narrower than the truth the profile
   would admit `crx` and silently truncate it. A generator-spectrum test would
   make the declaration self-checking.
2. **Should the profile take the displacement from the rule instead of validating
   the caller's?** `shift` is now checked against the declared rule and the
   mismatch message names the prescribed value. Accepting `shift` at all is a
   caller convenience inherited from the previous signature; a rule-read
   displacement would remove the possibility of asking for the wrong one.
3. **Does `examples/remote/` need a multi-opcode sample?** The two remote examples
   use `rx`/`ry`/`rz`. Nothing about them is wrong, but neither shows an
   entangler that this change newly admits, so the wider profile has no executable
   worked example outside the tests.

## Owner and approvals

- **Implementation owner:** core (`flagquantum/gradients.py`,
  `flagquantum/core/operator_schema.py` as the declaration it reads).
- **Integration surfaces touched:** `contracts/` (one new contract),
  `.github/workflows/ci.yml` (one step), `tools/` (one gate, one pre-push entry).
  These are protected paths, so the change is an integration change and carries
  its own contract, gate, tests, and record.
- **Authorization:** `docs/api-changes/FQ-GRADIENT-BATCHED-SHIFT-PROFILE-20261020.md`
  (this document), listed in the contract's `authorization` field.
- **No API-owner approval required.** The change touches no stable export.
