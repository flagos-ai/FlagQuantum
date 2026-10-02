# Gradient parameter frequencies

## Decision and authorization

Status: **proposed, not approved.** This document records a measured defect in
`flagquantum.gradients.parameter_shift_gradient`, the contract that fixes it, and the
evidence for both. It contains no claim of approval.

The defect is not an unimplemented feature. `parameter_shift_gradient` is a shipped,
documented entry point whose result is wrong for three registered gates:
**`crx`, `cry`, and `crz`**. The gate has a four-term shift rule; the function
applies a two-term rule to every gate regardless of identity. In the probe sweep
below the returned derivative is between **0.80x and 3.17x** the true value,
depending only on how the gate is embedded.

Scope of the affected surface: `flagquantum/core/operator_schema.py`,
`flagquantum/core/__init__.py`, `flagquantum/gradients.py`, and the generated
`docs/operator_manifest.json`. No public stable name is added or removed: the
gradient entry points are not part of `docs/public_api_v1.json`, and
`OperatorSchema` is not exported from `flagquantum/__init__.py`.

## Problem and affected user journey

### The shift rule is keyed on the shift, not on the gate

`flagquantum/gradients.py` computes every partial derivative as one fixed
combination of two evaluations:

```python
plus_loss = loss_fn(circuit_builder(plus))
minus_loss = loss_fn(circuit_builder(minus))
grads.append(0.5 * (plus_loss - minus_loss))
```

`0.5 * (f(t + pi/2) - f(t - pi/2))` is the exact derivative of a gate whose
generator has a **single** frequency of 1 — `RX`, `RY`, `RZ`, and the rest of the
one-frequency family. It is not the derivative of a gate with two frequencies.

`CRX(theta)` on control `c` and target `t` is
`exp(-i theta/2 * |1><1|_c (x) X_t)`, whose generator has eigenvalues `0` and `-1/2`,
so its frequencies are `{0.5, 1.0}` and its derivative needs **four** evaluations.
The module records the one-frequency assumption three more times as hardcoded gate
sets, all three disagreeing with the schema:

| Location | Set | Size |
| --- | --- | --- |
| `core/operator_schema.py` `OperatorSchema.differentiable` | `cphase crx cry crz phase rx rxx ry ryy rz rzz u1 u2 u3` | 14 |
| `gradients.py:17` `_BATCH_PARAMETER_SHIFT_GATES` | `h x rx ry rz cx` | 6 |
| `gradients.py:18` `_PARAMETER_SHIFT_GATES` | `rx ry rz` | 3 |
| `runtime/executors/statevector/split_real_imag.py:30` | `rx ry rz rxx ryy rzz` | 6 |
| `simulation/statevector/fixed_layer_cpu.py:27` | `rx ry rz` | 3 |
| `simulation/statevector/double_single_device_gates.py:19` | `rx ry rz rxx ryy rzz` | 6 |

`_PARAMETER_SHIFT_GATES` is declared and never read: the generic loop does not
consult gate identity at all. `OperatorSchema.differentiable` was read only by
`gate_info()` and `operator_manifest()`, so it advertised a capability that the
gradient path did not consume and could not have honoured.

### Measured: the documented rule does not reproduce the derivative

Circuit: `H(0); RY(1, prep); CR*(0, 1, theta=0.37)`, loss = the sum of all six
single-qubit Pauli expectations, `complex128`, compared against
`torch.autograd` and against a finite difference of the same loss.

| Gate | Prep `RY(1, p)` | autograd | current 2-term | ratio | declared 4-term |
| --- | --- | --- | --- | --- | --- |
| `crx` | 0.0 | -0.7389446555 | -0.7770412300 | 1.0516 | -0.7389446555 |
| `crx` | 0.31 | -0.8580324352 | -0.9582306392 | 1.1168 | -0.8580324352 |
| `crx` | 1.1 | -0.8234369899 | -1.0429591091 | 1.2666 | -0.8234369899 |
| `cry` | 0.0 | 0.1933826901 | 0.1552861156 | 0.8030 | 0.1933826901 |
| `cry` | 0.31 | -0.0175834194 | -0.0556799939 | 3.1666 | -0.0175834194 |
| `cry` | 1.1 | -0.5391225750 | -0.5772191494 | 1.0707 | -0.5391225750 |
| `crz` | 0.0 | -0.5834413921 | -0.8251107295 | 1.4142 | -0.5834413921 |
| `crz` | 0.31 | -0.4729645619 | -0.7049303125 | 1.4905 | -0.4729645619 |
| `crz` | 1.1 | -0.0605899733 | -0.1910263634 | 3.1528 | -0.0605899733 |

The error is not a scale factor that a user could absorb: it changes sign
relation, and it is largest exactly where the two-frequency term is largest.
The other eleven differentiable opcodes (`rx ry rz phase u1 u2 u3 cphase rxx ryy
rzz`) agree with autograd to `4.4e-16`, which is why the defect survived — the
one-frequency majority is correct.

### Why this is the schema's job, not the gradient function's

The framework already knows each gate's generator spectrum: the schema holds
`parameters` and `semantic_kind` for all 35 opcodes. The rule for a parameter is
a function of that gate's declared frequencies, so the derivative can be derived
rather than enumerated. Keeping a second, hardcoded list is what produced six
disagreeing answers to "which gates can be differentiated".

## Decision Candidates

**Candidate 1 — Declare frequencies in the schema and derive the rule. (chosen)**
Add `parameter_frequencies: tuple[tuple[float, ...], ...]` to `OperatorSchema`,
one entry per declared parameter. Add a module-level `parameter_shift_rule` that
turns one frequency set into `(coefficient, shift)` pairs, and
`OperatorSchema.shift_rule(parameter)` that resolves a parameter name to its rule.
Make `differentiable` a *derived* property of the declared frequencies instead of a
stored boolean, so the capability flag and the rule cannot disagree. Point
`parameter_shift_gradient` at `shift_rule`.

**Candidate 2 — Fix `crx/cry/crz` with a special case in `gradients.py`.**
Smallest possible diff, and it leaves the six lists in place. It also leaves the
next gate with two frequencies needing another special case, and leaves
`differentiable` as a stored boolean that no execution path reads.

**Candidate 3 — Move the differentiation rule to the executors.** Each executor
returns the shift structure it can support. This is where the other five hardcoded
lists live, so the duplication is closest to its cause, but it makes the rule a
property of the backend rather than of the gate, and it cannot be declared without
an executor.

**Candidate 4 — Derive frequencies numerically from the generator matrix.** Build
each gate's generator, diagonalize it, and read the spectrum. It removes the
hand-declaration entirely, but it puts a `torch.linalg.eigvalsh` call on the import
path and makes the schema's numbers depend on a floating-point decomposition.

## Prohibited Practices

1. **Special-casing a gate name inside a differentiation loop.** The `crx` defect
   is what a name-keyed assumption in a generic loop produces. A fourth
   `if opcode == ...` branch is the same defect with a different trigger.
2. **Storing a rule and its precondition in two places.** `differentiable` and
   `parameter_frequencies` must not both be writable independently; the derived
   property in candidate 1 is the mechanism, and re-adding a stored flag
   reintroduces the drift this document records.
3. **Silently differentiating an opcode whose rule is unknown.** A shift rule that
   does not exist must raise, naming the opcode and the parameter, and must not
   fall back to the one-frequency formula.
4. **Approximating a non-equidistant frequency set with the equidistant closed
   form.** No built-in opcode declares one; the helper must reject it rather than
   return coefficients that are merely close.
5. **Changing the shift value a caller can request without a stated contract.**
   `parameter_shift_gradient`'s `shift` keyword is removed by candidate 1 because
   per-gate rules make a single global shift meaningless. The removal is recorded
   here; a caller that passed a non-default value has to say what it meant.

## Compatibility

- **`parameter_shift_gradient(..., shift=...)`.** The keyword is removed. It was
  only ever meaningful for the one-frequency family, where the rule already
  hardcodes `pi/2`; passing anything else silently computed the wrong derivative.
  Callers that passed `shift=pi/2` are unaffected; any other value was a bug in the
  caller.
- **A parameter that scales or offsets its gate now raises.** The function always
  shifted the *user* parameter by the rule's shift, which is correct only when the
  user parameter is the gate angle. `rx(0, theta=2 * scale)` used to return a
  plausible number that was not the derivative; it now raises with a message naming
  the gate, the parameter, and the movement the probe caused.
- **`OperatorSchema` gains two members** (`parameter_frequencies`, `shift_rule`)
  and **loses one stored field** (`differentiable` becomes a property). The name
  still exists and still reads the same for all 35 built-in opcodes, so
  `gate_info()`, `operator_manifest()`, and the generated
  `docs/operator_manifest.json` keep their meaning. `differentiable` is no longer
  a constructor argument; `tests/unit/test_operator_schema.py` constructed a
  `custom` schema with `differentiable=False` and no parameters, which now reads
  `False` by derivation.
- **`operator_manifest.json` gains a key.** The generated file is a documented
  artifact with `schema_version`, so the regeneration is part of this change and
  `tools/operator_manifest.py --check` fails until it is regenerated.
- **`parameter_shift_rule` is added to `flagquantum.core`'s namespace**, not to
  `flagquantum`'s. `fq.__all__` is unchanged.
- **`batched_parameter_shift_gradient` is unchanged.** It still supports only
  `H, X, RX, RY, RZ, CX` and still says so; the five hardcoded executor lists are
  untouched by this proposal and are recorded as open questions.
- **No capability row moves.** `capability-maturity.toml` records gradient support
  for the training entry points (`fq.train`, `fq.Module`, autograd), and
  `parameter_shift_gradient` is an auxiliary helper outside that surface: it is not
  in `docs/public_api_v1.json` and no capability row names it. The
  `Split real/imag FP32 observable and parameter-shift P1` row describes the
  split-precision executor's own occurrence-wise rule, which this change does not
  touch. The support boundary is therefore unchanged, and
  `docs/reference/KNOWN_LIMITATIONS.md` needs no edit: this fixes a wrong number
  rather than widening or narrowing what is supported.

## Acceptance Tests

1. **Every differentiable opcode reproduces autograd.** For all fourteen opcodes,
   including the two- and three-parameter ones, `parameter_shift_gradient` matches
   `torch.autograd` on an all-Pauli loss to `atol=rtol=1e-9`. This test fails on
   the current code for `crx`, `cry`, and `crz`.
2. **The rule is a function of the declared frequencies.**
   `parameter_shift_rule((1.0,))` is a two-term rule with coefficients
   `(+0.5, -0.5)`; `parameter_shift_rule((0.5, 1.0))` is a four-term rule whose
   coefficients sum to zero and whose shifts are `pi/2` and `3pi/2` in magnitude.
3. **`differentiable` is derived.** Every parameterized unitary reports
   `differentiable`, no parameterless gate and no channel does, and each schema's
   frequency-set count equals its parameter count.
4. **A declaration that cannot be used is rejected at import time.**
   A frequency set that is empty, non-positive, repeated, or non-equidistant
   raises when the schema is constructed, not when a gradient is requested.
5. **An unknown rule raises instead of approximating.**
   `shift_rule` on a channel or on an undeclared parameter name raises `ValueError`
   naming the opcode.
6. **A parameter that does not control exactly one gate parameter is rejected.**
   A parameter shared by two gates raises rather than silently double-counting.
7. **A parameter that scales its gate is rejected rather than mis-differentiated.**
   `rx(0, theta=2 * scale)` raises, because the rule shifts the gate's angle and a
   shifted user parameter is only an angle when it *is* that angle. The guard reads
   the movement the probe caused, so a sign flip (`theta=-scale`) stays legal.

## Open Questions

1. **Should the five remaining hardcoded gate lists be derived too?** Each encodes
   a backend capability rather than a gate rule, so the derivation is not the same
   one; `split_real_imag.py` and `double_single_device_gates.py` may need to answer
   "which gates can this kernel split" instead of "which gates are differentiable".
2. **Which eigenvalues does `CRX`'s generator actually have?** The declared
   `{0.5, 1.0}` is verified against autograd for three preparations, not derived.
   A generator-spectrum test would make the declaration self-checking, and that is
   the natural follow-up to candidate 4.
3. **Does `batched_parameter_shift_gradient` need the same treatment?** It refuses
   `RXX`/`RYY`/`RZZ` today, which are one-frequency gates the generic path already
   handles. The restriction may be about the batched kernel, not the rule.
4. **Should `shift_rule` be public?** It is reachable as
   `fq.core.OPERATOR_SCHEMAS["crx"].shift_rule("theta")`. Exporting it from
   `flagquantum.core` is done; promoting it into `fq.__all__` would make it a
   stable name and needs the API-owner path.

## Owner and approvals

- Owning domain: `core` for the schema declaration, with `verification` for the
  conformance test that compares every opcode against autograd.
  `python tools/check_team_scope.py --team core --files
  flagquantum/core/operator_schema.py flagquantum/core/__init__.py
  flagquantum/gradients.py tests/unit/test_operator_schema.py
  tests/test_native_circuit.py docs/operator_manifest.json` passes.
- Required approvals before implementation: **core domain owner** (the schema field
  and the derived `differentiable`), **API owner** only if `shift_rule` is promoted
  into `fq.__all__`, and **integration owner** for the generated manifest.
- Sequencing: this document, then the schema field, then `gradients.py`, then the
  conformance test. The conformance test is the gate: it must fail on `main` for
  three opcodes before the fix and pass for all fourteen after.
