# The metric tensor differentiates the state and reads no rule

- **Status:** implemented
- **Owner:** core (`flagquantum/gradients.py`)
- **Date:** 2026-10-31
- **Related:** [`fq.gradient` and the reported gradient method](FQ-GRADIENT-API-20261002.md),
  [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md), and
  the parameter-shift Hessian record from the immediately preceding slice
  (`FQ-PARAMETER-SHIFT-HESSIAN-20261030.md`), which is a sibling change record on
  its own branch and is therefore named rather than linked here

## Problem

FlagQuantum could differentiate a loss and, after the previous slice, a loss
twice. It could not answer a question about the **state**: how far the state
moves when a parameter moves, in the coordinates the parameters already define.
That quantity is the Fubini-Study metric tensor, and its absence blocked three
things at once -- a quantum natural gradient, an expressivity or barren-plateau
measurement, and any curvature-aware analysis of a circuit that has no loss yet.

The obvious implementation is the one already in the repository. `fq.gradient`
reads `OperatorSchema.shift_rule` to move a parameter, and the previous slice
reused that rule a second time for the second derivative. Applying the same rule
to the **state** instead of to an expectation value looks like the same trick
again, and it is wrong. Measured on the declaration itself, in
`contracts/metric-tensor-contract.toml`:

| Reading | Ratio to the exact derivative | Deviation from it |
| --- | --- | --- |
| `ry` at `theta = 0.4`, per state component | `1.414213562` | `2.029784e-01` |
| `ry` at `theta = 1.1`, per state component | `1.414213562` | `1.765636e-01` |
| `phase` at `0.4` after `h`, per component | `1.000000000` | `5.860885e-11` |
| `phase` at `1.1` after `h`, per component | `1.000000000` | `5.052384e-11` |
| the `product` witness's metric, diagonal | `2.000000000` | `2.500000e-01` |

`OperatorSchema.parameter_frequencies` is a statement about an **expectation
value**, and an observable's shift rule is a different object from the derivative
of a state. The two coincide for a phase gate and differ by `sqrt(2)` per
component for a rotation, so no constant rescaling repairs the reuse: on a
`phase`-only program the declared rule is right to `5.9e-11`, and on `ry` it is
wrong by `2.03e-01` on a component of size `1.4e-01`. The shape of the rule
depends on the generator's spectrum, and `OperatorSchema` records no spectrum --
there is no `generator`, `matrix`, or state-frequency field to read. Correcting
the declaration in the route would mean a hand-maintained second rule table,
which AGENTS.md principle 6 forbids.

## Decision

Add one module-level entry point:

```python
metric_tensor(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    *,
    options: ExecutionOptions | None = None,
    step: float | None = None,
) -> torch.Tensor
```

It differentiates the **state the execution delivers**, with a central difference,
and reads no rule at all:

```python
derivatives = central_difference_gradient(deliver, base, step=step).reshape(width, -1)
overlaps = derivatives @ derivatives.conj().T
projections = derivatives @ reference.conj()
metric = (overlaps - torch.outer(projections, projections.conj())).real
```

`deliver` is `_state_of(run(circuit_builder(values), options=options))`, so the
route is one execution per displaced point and nothing else.

### The convention is the Fubini-Study metric, and the factor is four

Cell `[i, j]` is `Re[<d_i|d_j> - <d_i|psi><psi|d_j>]`. The quantum Fisher
information is `Re Tr[rho L_i L_j]` with `L` the symmetric logarithmic derivative,
and `QFIM = 4 * metric_tensor(...)`. The factor is named in the docstring, written
into the contract as `fisher_factor = 4.0`, and **pinned by the gate** against a
hand-rolled definitional oracle that imports nothing from the package under test.

This is the convention `qml.metric_tensor` uses, so the two frameworks agree
numerically rather than up to a constant. The gate compares the implementation's
matrix times four to the definitional Fisher information and requires agreement
to `1e-09`; an implementation that returned the Fisher information would be wrong
by four in every cell and would pass every shape assertion in the test file. The
factor is a claim the user can check, not a footnote.

### The matrix is indexed by parameter, not by gate occurrence

For a program where one parameter drives two gates, PennyLane reports a
coordinate per gate occurrence -- three coordinates for a two-parameter program.
Folding those occurrences into the parameter coordinate is what makes the two
conventions comparable, and it is the only difference between them: measured
against PennyLane 0.45.1 on that program, the folded matrix agrees to `8.638e-12`.

### A mixed state is refused rather than reinterpreted

`2 Tr[d_i rho d_j rho]` is computable from a density matrix and is **not** the
Fisher information. On `depolarizing(0.4)` the two differ by `7.229115e-01`
across a matrix whose cells are near one, so a route that quietly returned the
cheaper expression would be wrong by most of its own value. Any rank-3 state read
is refused with `CapabilityError`, and the refusal names the symmetric logarithmic
derivative and says what the route does not compute.

### What is deliberately not here

- **No root export.** `metric_tensor` is reachable as
  `fq.gradients.metric_tensor` and is **absent** from `fq.__all__` and from the
  root namespace. The root surface stays at 37 exports, `docs/public_api_v1.json`
  is untouched, and this change needs no rule 8 authorization.
- **No `GradientResult` field.** The result object is not extended; the value is
  returned, not attached.
- **No quantum natural gradient.** The optimizer that consumes this matrix is a
  separate slice.
- **No batched program.** One state per matrix is required, so a batch is refused
  rather than averaged.
- **No second derivative and no parameter-dependent qubit count.** A program whose
  state changes shape between the base point and a displacement is refused.

## Scope

| | Before | After |
| --- | --- | --- |
| State-curvature quantity | absent | `fq.gradients.metric_tensor` |
| Root surface | 37 exports | 37 exports (unchanged) |
| `fq.gradient` methods | 5 | 5 (unchanged) |
| Serving modes | 4 | 4 (unchanged, `auto`/`statevector`/`mps`/`tensor_network`) |
| Differentiable opcodes reachable | 14 | 14 (unchanged, and not read by this route) |

No existing behaviour changes. `flagquantum/gradients.py` gains one public
function and one private helper, and `_GRADIENT_METHODS` is untouched: a metric
tensor is not a sixth way to compute a first derivative, and adding a name there
would turn the gradient contracts red for no benefit.

### Explicitly out of scope

- **`density_matrix` is not a serving mode.** It is the one exclusion this route
  owns: the state read comes back as a density operator, whose purity is not one.
- **`stabilizer` is not a serving mode.** It never carries a program to a state at
  all. `fq.run` refuses it without a sampling measurement
  (`ValidationError: mode='stabilizer' requires shots or a sampling measurement in
  the program`) and refuses it again once one is asked for, because the sampling
  backend either is an uninstalled optional dependency or will not approximate a
  non-Clifford gate. Both sentences are `CapabilityError` for the second case, and
  neither returns a number instead.
- **A state-producing mode asked for on a noisy program** is refused by the
  execution, before this route sees anything:
  `ValidationError: stable noisy execution supports mode='auto' or
  mode='density_matrix'`. The contract records that as a reading about the mode
  names rather than as a second exclusion, and the gate requires the exclusions
  and the serving list to be a partition of the six registered modes.

## Cost

One base state plus a central pair per parameter: `1 + 2 * n`. Measured by
counting builder calls:

| Parameters | Evaluations |
| --- | --- |
| 1 | 3 |
| 2 | 5 |
| 3 | 7 |

The derivative is computed once per parameter, so every matrix cell reuses that
pair and the cost is linear in the parameter count. A route that re-evaluated per
cell would cost the square, and the Hessian of the same width costs `4 * n**2 - n`
by comparison.

## Prohibited practices

- Do not read `OperatorSchema.shift_rule` or `parameter_frequencies` in this
  route. The gate's opcode census and declaration-symbol census are scoped to the
  route's own function bodies, so naming either is a failure.
- Do not name an opcode in the route's code. It must serve every opcode, including
  ones added later, and an explicit-matrix program that no opcode declares.
- Do not add `metric_tensor` to `_GRADIENT_METHODS`. It is not a gradient method.
- Do not export it from `fq`. The root surface stays at 37 exports and this slice
  deliberately needs no authorization.
- Do not return the Fisher information, and do not describe the returned value as
  "the quantum Fisher information up to a constant". `QFIM = 4 * metric_tensor`;
  the factor is measured, not implied.
- Do not reinterpret a mixed state as a pure one, or average a batch. Both are
  refusals with recorded sentences.
- Do not edit `contracts/metric-tensor-contract.toml` to make the gate pass. The
  contract records a measurement; a disagreement is a finding.

## Compatibility

- **No stable API changes.** `metric_tensor` is not in `docs/public_api_v1.json`,
  `fq.__all__` is still 37 entries, and no signature, default, result field, or
  serialized schema moves. No change proposal is required and no API-owner
  approval is needed.
- **No existing entry point changes behaviour.** `parameter_shift_gradient`,
  `batched_parameter_shift_gradient` and `parameter_shift_hessian` are untouched
  by this slice.
- **No capability row moves, and no limitation is relaxed.** The route adds a
  measured quantity that no capability row claimed or denied, so there is no row
  to promote. `docs/reference/KNOWN_LIMITATIONS.md` continues to hold: the
  mixed-state route is absent and is refused rather than approximated, and a
  batched program is not served.
- **The precision limit is documented rather than hidden.** The result carries the
  precision the *execution* ran in, not the precision of `parameters`. At the
  `complex64` default the matrix is good to about six digits and no further, and
  the contract records both readings rather than one absolute bound.

## Verification

`tools/check_metric_tensor_contract.py` re-derives the witnesses from
`contracts/metric-tensor-contract.toml`, drives the implementation, builds all
thirteen refusals, drives every mode, and compares all of it against the contract.
It runs in the `quality` job of `.github/workflows/ci.yml` as a **step** beside
the two gradient gates and in `tools/pre_push.py`. It is a step rather than a job
deliberately: `tools/validate_required_checks.py` pins six externally configured
required checks across four jobs, so adding a job would change the required-check
contract and adding a step does not.

The gate computes four routes and counts **three** of them as evidence:

| Route | Shares arithmetic with the implementation | Role |
| --- | --- | --- |
| central difference of the delivered state | -- | the artifact |
| reverse-mode autodiff through `fq.run` | no difference scheme at all | **the agreement claim** |
| a hand-rolled symmetric-logarithmic-derivative Fisher information | no package import | **the convention claim** |
| central differences of the oracle's own state | yes, the same scheme | deliberately **not** counted |

A fourth route that differenced the oracle's state with the implementation's own
central-difference scheme was written and is deliberately not counted: it is the
implementation's method with the oracle's constants, so its agreement would show
consistency rather than correctness. This is the same reasoning `N3-7` and `N3-11`
recorded.

Measured by the gate, with every witness at `complex128`:

| Witness | Parameters | Implementation vs recorded | Autodiff vs recorded | `4 x` implementation vs the definition |
| --- | --- | --- | --- | --- |
| `product` | 2 | `3.153672e-12` | `2.775558e-17` | `1.261424e-11` |
| `repeated` | 2 | `8.637646e-12` | `2.595452e-12` | `3.454970e-11` |
| `three` | 3 | `7.032042e-12` | `8.257839e-12` | `3.303124e-11` |

All three are inside the recorded `1e-09` bound on both routes, and the smallest
recorded cell magnitude is `0.235350974`, against a recorded floor of `1e-03`. A
matrix that came out at zero would make every comparison vacuous, so the floor is
asserted in the same direction the error can move.

- **All 13 refusals fire** with the recorded exception class and a message
  containing the recorded phrase.
- **The two exclusions and the four serving modes are a partition** of the six
  registered `ExecutionOptions` modes, and the three noisy-program mode refusals
  are driven through the call that owns them.
- **The matrix is symmetric to `8.645e-19`** on the widest witness, which is
  summation order and not zero.

### The bound is a claim about a declared precision

`fq.run` resolves an unset `precision` to `complex64`, so an unqualified
`fq.Circuit` delivers a `complex64` state and the matrix comes back `float32`.
Re-reading the `repeated` witness at the executor default gives
`5.692244e-06` against the pinned reference -- roughly five thousand times the
recorded bound, and outside it. The contract therefore records `program_dtype`,
the default dtype, the witness that reading uses, and its measured deviation; the
gate requires the default reading to stay **outside** the bound and the pinned
reading to stay inside it. Without that pinning the same gate would have measured
its own independent route at the wrong precision, which is exactly the
`1.7e-08`-against-`1e-09` failure this gate reported on its first run.

The public guidance follows the measurement: the docstring names the precision the
caller controls and says the matrix is good to about six digits at the default.

`tests/unit/test_metric_tensor_contract.py` (108 tests) owns the contract and the
gate. Its last section mutates one clause at a time -- a cost row, the agreement
bound, the reference floor, a recorded cell, a recorded deviation, the
default-precision reading, a dropped witness, a witness stripped of its
parameters, a refusal's exception class, a refusal's message, a dropped refusal, a
served mode that is also excluded, a misattributed exclusion, a forged exclusion,
a claimed root export, a claimed opcode dependence, a claimed declaration read, a
literal count, a forbidden symbol that does not exist, a claimed autograd graph, a
claimed mixed-state answer, a claimed explicit-matrix refusal, the convention
name, a blank convention entry, the falsification route, the recorded frequencies,
a component ratio, a coincident reading, a reference witness that does not exist,
a scope string, the contract test path, an expansion test path, a census string,
the implementation symbol, the evaluations per parameter, and the cost growth
string -- and requires the gate to name each one. Three separate crashes were
found this way -- a `KeyError` on a contract missing its vocabulary table, a
description that named the wrong field, and a witness dropped without breaking a
claim -- and each was fixed in the gate rather than worked around in the test. A
gate that crashes on a mutated contract is a broken gate.

Focused run:

```bash
python tools/check_metric_tensor_contract.py
python -m pytest tests/unit/test_metric_tensor_contract.py -q
python -m pytest tests/test_gradient_api.py tests/integration/test_gradient_modes.py -q
```

## Open Questions

1. **Should the second argument be `parameters` or a full parameter set object?**
   The route needs the same tensor `parameter_shift_gradient` takes and needs
   nothing else, so it takes the same tensor. A wider type would be a second
   parameter vocabulary for no measured capability.
2. **Should the step be exposed at all?** It is exposed because a caller comparing
   two precisions needs one number, and because the gate uses it to demonstrate
   that the default is derived from the delivered precision rather than fixed. If
   no caller uses it, removing it is a smaller surface, not a lost capability.
3. **What is the intended public home if a user asks for this by name?**
   PennyLane puts `qml.metric_tensor` at the root. FlagQuantum puts it under
   `fq.gradients` for the same reason as the Hessian: a root export is a rule 8
   authorization and a separate change.

## Owner and approvals

- **Implementation owner:** core (`flagquantum/gradients.py`).
- **Integration surfaces touched:** `contracts/` (one new contract),
  `.github/workflows/ci.yml` (one step), `tools/` (one gate, one pre-push entry),
  `docs/api-changes/README.md` (one index line). These are protected paths, so the
  change is an integration change and carries its own contract, gate, tests, and
  record.
- **Authorization:** `docs/api-changes/FQ-METRIC-TENSOR-20261031.md` (this
  document), listed in the contract's `authorization` field.
- **No API-owner approval required.** The change touches no stable export.
