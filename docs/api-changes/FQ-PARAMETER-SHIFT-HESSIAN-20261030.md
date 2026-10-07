# The parameter-shift Hessian reads the first-order rule twice

- **Status:** implemented
- **Owner:** core (`flagquantum/gradients.py`)
- **Date:** 2026-10-30
- **Related:** [`fq.gradient` and the reported gradient method](FQ-GRADIENT-API-20261002.md),
  [Gradient parameter frequencies](FQ-GRADIENT-PARAMETER-FREQUENCIES-20261002.md),
  [The batch parameter-shift profile reads the opcode declaration](FQ-GRADIENT-BATCHED-SHIFT-PROFILE-20261020.md)

## Problem

FlagQuantum could differentiate a loss once and not twice. A user who wanted a
quantum natural gradient, an error-bar estimate, or a curvature-aware
regularizer had to build a second derivative out of `fq.gradient` calls
themselves, and the honest ways to do that are all wrong in a way the framework
should not leave to the caller:

- Nesting two calls to `fq.gradient` is not one derivative. Each call returns a
  **detached** tensor, so the outer call has no graph to differentiate and the
  result is the derivative of a constant.
- Reading a finite difference of `fq.gradient` reintroduces a step size at the
  second order, where the truncation error grows with `1/h**2` rather than `h**2`.
- Wait for a metric tensor and a natural-gradient optimizer, as the plan
  originally sequenced, and the second derivative is still missing at the end.

The parameter-shift rule already answers this without a new declaration. A gate
whose generator has frequencies `{f_1, ..., f_n}` has an expectation value that
is a trigonometric polynomial in those frequencies. Its first derivative is a
fixed linear combination of evaluations at `base + s_i`; its second derivative is
the **same combination applied to itself**, evaluated at `base + s_i + s_j`. The
coefficients are the ones `OperatorSchema.shift_rule` already returns, so the
second order needs no table of its own.

That last point is the whole design constraint. The tempting implementation is a
`SECOND_ORDER_SHIFT_RULES` dict next to `OPERATOR_SCHEMAS`, or a per-opcode
special case in the gradient module. Either one is a second source of truth for a
value the declaration already owns, and the two copies then agree until the
declaration moves. AGENTS.md principle 6 forbids exactly that.

## Decision

Add one module-level entry point:

```python
parameter_shift_hessian(
    circuit_builder: CircuitBuilder,
    parameters: torch.Tensor,
    loss_fn: LossFunction,
) -> torch.Tensor
```

It composes `OperatorSchema.shift_rule` with itself. For each pair of flat
parameter positions it forms every product of one declared `(coefficient, shift)`
term from the left parameter and one from the right, evaluates the loss at the
sum of the two displacements, and sums `left_coefficient * right_coefficient *
loss` over the pairs. Nothing else is read.

```python
terms = [
    coefficient
    * _scalar_loss(
        loss_fn(circuit_builder((flat + shift).reshape_as(base))),
        source="loss_fn",
    )
    for coefficient, shift in _composed_shift_terms(
        rules[left], rules[right], parameters=(left, right), width=width, dtype=base.dtype
    )
]
cells.append(torch.stack(terms).sum())
```

Three consequences follow from that construction rather than from a decision
made about it:

- **The cost is the rule's width squared.** A parameter with a two-term rule
  contributes three evaluations to its own diagonal cell (the two shift sums that
  coincide are merged, and the zero displacement's coefficients cancel) and four
  to each off-diagonal cell. For `n` two-term parameters the total is `4 * n**2 -
  n`: `33` for the three-parameter reference, against the `6` a first-order
  gradient of the same program costs.
- **Symmetry is not asserted, it is inherited.** A twice-differentiable loss has
  equal mixed partials, so the two halves carry the same value. They are computed
  by independent evaluations and are therefore equal to floating-point summation
  order, not bit for bit.
- **Distinct displacements are merged.** Two term pairs can land on the same
  displaced circuit, and the merge is what makes the evaluation count a property
  of the rule rather than of the arithmetic that happens to cancel.

The merged count is exact, and it is measured rather than derived:

| Cell | Declared terms | Evaluations | Witness |
| --- | --- | --- | --- |
| diagonal | 2 and 2 | 3 | `rz` |
| diagonal | 4 and 4 | 7 | `crx` |
| off-diagonal | 2 and 2 | 4 | `rz`, `rz` |
| off-diagonal | 2 and 4 | 8 | `rz`, `crx` |
| off-diagonal | 4 and 4 | 16 | `crx`, `crx` |

### What is deliberately not here

- **No root export.** `parameter_shift_hessian` is reachable as
  `fq.gradients.parameter_shift_hessian` and is **absent** from `fq.__all__` and
  from the root namespace. `fq.__all__` stays at 37 (the count `origin/main`
  reaches after the authorized `density_matrix` export) and `docs/public_api_v1.json`
  is untouched, so this change needs no rule 8 authorization.
- **No fifth gradient method.** `hessian` is not added to `_GRADIENT_METHODS`, and
  `fq.gradient(..., method="hessian")` is still refused. A second derivative is
  not a fifth way to compute a first one, and adding the name would turn both
  `contracts/native-gradients-contract.toml` and
  `contracts/opcode-gradient-exactness-contract.toml` red to no benefit.
- **No second coefficient table.** The gate's opcode census is scoped to the two
  function bodies this change adds, so an opcode name appearing in the route's
  own code is a failure.
- **No metric tensor and no natural gradient.** That remains the next slice.

## Scope

| | Before | After |
| --- | --- | --- |
| Second derivative | absent | `fq.gradients.parameter_shift_hessian` |
| Differentiable opcodes reachable | 14 | 14 (unchanged, read from the declaration) |
| Serving modes | 5 | 5 (unchanged) |
| Root surface | 37 exports | 37 exports (unchanged) |
| `fq.gradient` methods | 5 | 5 (unchanged) |

The existing behaviour of the first order is unchanged except in one place, and
that place was a defect: `parameter_shift_gradient` now routes `loss_fn` through
the same scalar check the new entry point uses. Previously a vector-valued loss
made it return a **silently wrong shape** (`(2,)` where a scalar per parameter was
meant) while the new entry point raised a shape error from `torch.stack`. Both
now raise `ValidationError: loss_fn must return one scalar tensor`.

`parameter_shift_gradient` and `parameter_shift_hessian` also now share
`_validated_parameter_tensor`, which they previously did not. Measured on both
entry points:

| Input | Before | After |
| --- | --- | --- |
| `torch.tensor([])` | `IndexError: index 0 is out of bounds` | `ValidationError: parameters must not be empty` |
| complex or integral tensor | `TypeError` / `RuntimeError` from the gate | `ValidationError: parameters must be real floating-point values` |
| non-finite tensor | `IRValidationError` from the IR | `ValidationError: parameters must be finite` |
| a plain list | `AttributeError: 'list' object has no attribute 'detach'` | `TypeError: parameters must be a torch.Tensor` |

The first-order entry point had the `ValidationError` vocabulary available and did
not use it. Its refusals were the framework's internal errors leaking through a
public function, which is a diagnosability defect and not a behavior change worth
preserving.

### Explicitly out of scope

- **`stabilizer` is not a serving mode here.** It is the sixth
  `ExecutionOptions` mode and it cannot serve an expectation-valued observable at
  all: `fq.run` raises `CapabilityError: mode='stabilizer' samples measurement
  outcomes; it cannot serve measurement kind(s) expectation_ps`. That is the
  runtime refusing the measurement, not this route refusing a mode, so the
  contract records it in `excluded_mode` rather than in `serving_modes`, and the
  test asserts the refusal still fires.
- **Third and higher derivatives** are not generalised from this construction. The
  composition would work, but nothing asks for it yet and the cost grows as
  `n**k`.
- **The metric tensor** is a different object (`N3-8`): it is the Hessian of the
  state overlap rather than of a loss, so it needs its own construction.

## Prohibited practices

- Do not add a table of second-order coefficients anywhere. The declaration is the
  only source of `(coefficient, shift)`, and a second table disagrees with it the
  first time a frequency changes.
- Do not name an opcode in the second-order route's code. It must differentiate
  every opcode whose declaration admits it, including ones added later.
- Do not add `hessian` to `_GRADIENT_METHODS`. Both gradient contracts would go
  red, and the name would claim a second derivative is a first-order method.
- Do not export the function from `fq`. The root surface stays at 37 exports and this
  slice deliberately needs no authorization.
- Do not describe the matrix as symmetric "by construction" without saying what
  that means. It means equal mixed partials; it does not mean the two halves are
  bit-identical, and a test that asserted bit equality would fail on `u3`.
- Do not edit `contracts/parameter-shift-hessian-contract.toml` to make the gate
  pass. The contract records a measurement; a disagreement is a finding.

## Compatibility

- **No stable API changes.** `parameter_shift_hessian` is not in
  `docs/public_api_v1.json`, `fq.__all__` is still 37 entries, and no signature,
  default, result field, or serialized schema moves. No change proposal is
  required and no API-owner approval is needed.
- **`parameter_shift_gradient` gains refusals, not answers.** Every input it
  answered before it still answers with the same value; four malformed inputs it
  happened to survive now produce the documented `TypeError` / `ValidationError`
  pair. That is a fail-closed change to undocumented behavior on invalid input.
- **No capability row moves, and no limitation is relaxed.** The three
  higher-order boundaries in the repository are all about a *different* path and
  all remain true: the JAX bridge is `once_differentiable` and still rejects
  double backward, the Double-Single P5 lane is still first-order only, and
  `docs/reference/KNOWN_LIMITATIONS.md` still says so. The JAX section already
  scopes itself to the hybrid kernel ("It does not change the differentiation
  support of native PyTorch execution paths"), which is where this entry point
  lives, so it needs no edit. Nothing in `capability-maturity.toml` claims or
  denies a native second derivative, so there is no row to promote and no
  promotion is claimed here.
- **The sequencing rationale in the alignment plan is corrected, not followed.**
  The tracker predicted this slice would turn
  `tools/check_gradient_methods_contract.py` and
  `tools/check_opcode_gradient_exactness.py` red "until both matrices are
  revisited". Measured: both stay green, because the chosen surface touches
  neither `_GRADIENT_METHODS` nor the root exports. The prediction was written
  against a different candidate design (a `hessian` method value) and is recorded
  as falsified rather than quietly dropped.

## Verification

`tools/check_parameter_shift_hessian_contract.py` re-derives the composition from
`OPERATOR_SCHEMAS`, drives the implementation for every registered opcode, builds
every refusal, and compares all of it against
`contracts/parameter-shift-hessian-contract.toml`. It runs in the `quality` job of
`.github/workflows/ci.yml` as a **step** next to the batch profile's gate and in
`tools/pre_push.py`. It is a step rather than a job deliberately:
`tools/validate_required_checks.py` pins six externally configured required checks
across four jobs, so adding a job would change the required-check contract and
adding a step does not.

The gate computes three routes and counts only two of them as evidence:

| Route | Shares code with the implementation | Role |
| --- | --- | --- |
| the composed rule | no | gate's own composition; compared at `1e-12` to catch a divergence between the gate and the implementation |
| the implementation | -- | the artifact |
| Richardson central differences of `fq.run` | no | **the correctness claim** |

A second analytic application of the declared rule is deliberately **not** a
route. It and the composed rule are the same statement written twice, so their
agreement would establish self-consistency and not correctness. This is the same
reasoning `N3-11` recorded for the opcode exactness matrix.

Measured by the gate:

- **70 cells** (14 admitted opcodes `x` 5 serving modes) land inside
  `1e-09` of the difference route. Worst deviation
  `6.459544010795071e-11` at `ryy` in `mps`.
- **The reference is not degenerate.** The smallest cell magnitude over the whole
  sweep is `0.06059739937871045`, against a recorded floor of `1e-03`. A sweep
  that came out at zero would make the comparison vacuous, so the floor is
  asserted in the same direction the error can move.
- **The matrix is symmetric to `1.1102230246251565e-16`**, which is summation
  order and not zero.
- **Costs are `3`, `7`, `14`, `33`** for `rz`, `crx`, `u2`, `u3`, and the
  first-order route still charges `6` for the same three-parameter program.
- **All 11 refusals fire** with the recorded exception class and a message
  containing the recorded phrase.
- **The bound is a claim about a declared precision, and all of it is only true
  there.** `fq.run` resolves an unset `precision` to `complex64`, so an
  unqualified `fq.Circuit` returns `float32` expectations. This sweep is a
  second derivative assembled from those expectations, and it is three orders of
  magnitude finer than `float32` can carry: re-reading the `u2`/`mps` cell with
  the circuit left unqualified gives `1.7544981051331732e-07` against the pinned
  reference and `0.030165275765790034` when the reference route is taken at
  `float32` too. Neither is inside the bound. The witness program therefore pins
  `complex128`, the contract names the dtype it pins, and the gate refuses a
  contract whose recorded dtype the witness does not actually use. Without that
  pinning the sweep would still have passed and the bound would have described
  the wrong quantity.
- **Those two readings are recorded as decades, not as bits.** They are `float32`
  quantities, so their low bits belong to the platform's reduction order rather
  than to the composed rule. Read on x86_64 Linux the same cell gives
  `1.7544981051331732e-07` and `0.030165275765790034`; read on arm64 macOS it
  gives `1.828972331918699e-07` and `0.02846986220942621`, 4% away in both
  places. The first version of this contract recorded the x86_64 bits and
  asserted them to a relative `1e-06`, which is a claim about the machine and not
  about the code: it failed on any other platform before the implementation was
  ever consulted. The contract now records the decade each reading must fall in
  (`default_program_dtype_implementation_deviation_decade = -7`,
  `default_program_dtype_reference_deviation_decade = -2`), which is the part of
  the claim the platform does not own, and the gate additionally requires each
  decade to sit above `agreement_bound` -- the property the record exists to
  establish. Both measured readings and the platform each came from are kept in
  the contract's comments as provenance.

The same reading at the user's own scale, for the two-parameter program of the
docstring example, where the diagonal cell is exactly `-cos(0.4) * cos(0.9)`:

| Execution | Diagonal cell | Deviation from the exact value |
| --- | --- | --- |
| unqualified `fq.run` | `-0.5725407600402832` | `6.478e-08` |
| `precision="complex128"` | `-0.5725406952574803` | `3.331e-16` |

The public guidance follows the measurement: the docstring names the precision
the caller controls, and the contract records the deviation a caller who does not
set it should expect.

`tests/unit/test_parameter_shift_hessian_contract.py` (79 tests) owns the
contract and the gate. Its last section mutates one clause at a time -- a term
count, an admission, a cost, a cell cost, the agreement bound, the reference
floor, an opcode row, a refusal's exception class, a refusal's message, a dropped
refusal, a claimed root export, a claimed autograd graph, a claimed method value,
a scope string, the contract test path, a census string, the recorded program
dtype, the cell the default-precision reading names, each of the two
default-precision decades, and a decade that is above the bound but not the
decade the cell reads -- and requires the gate to name each one. An earlier draft of this gate **crashed** on a mutated
contract instead of reporting it, which is the failure mode those tests exist to
catch.

Focused run:

```bash
python tools/check_parameter_shift_hessian_contract.py
python -m pytest tests/unit/test_parameter_shift_hessian_contract.py -q
python -m pytest tests/test_gradient_api.py tests/integration/test_gradient_modes.py -q
```

The broader `python -m pytest -m "smoke or unit" -q` tier reports
`11 failed, 7719 passed, 271 skipped, 2629 deselected`. All eleven failures are
pre-existing: the identical ids fail on a pristine detached `origin/main`
worktree (`0ebc710b2`), measured with the same interpreter. Seven of them were
already failing at `8495a258d` before this branch merged `main` --
`test_qft_hadamard_phase_fusion_matches_graph_path_and_reduces_passes[dtype0]`
and `[dtype1]`, `test_product_state_qft_uses_hadamard_phase_fusion`,
`test_native_fused_rotation_layer_only_promotes_terminal_regions`,
`test_compact_cx_runtime_threshold_and_rollback`, and four chunking cases in
`tests/unit/test_statevector_batch_chunking.py`. The other two arrive with
`main` itself (merged by #552 and #560) and fail there:
`tests/unit/test_circuit_control.py::test_control_is_a_method_and_not_a_root_export`
and
`tests/unit/test_dependency_policy.py::test_collected_modules_guard_every_optional_reference`
(open PRs #569 and #578 are the fixes for those two). None of the eleven reads
`flagquantum/gradients.py`.

`python -m mypy flagquantum` reports the same **five** errors on this branch and
on a pristine `origin/main` with a cold cache -- two `sched_getaffinity` errors
in `flagquantum/benchmarking/socket_local_throughput.py`, one `ndarray.min`
overload error in `flagquantum/simulation/jax/mps/batched.py`, and two
`var-annotated` errors in `flagquantum/runtime/executors/jax/mps/pullbacks.py`.
`flagquantum/gradients.py` itself is clean, and the earlier
`Argument 1 to "stack" has incompatible type "list[Tensor | None]"` error this
change introduced was fixed rather than suppressed.

## Open Questions

1. **Should the route accept a `parameters` tensor that is not the builder's own
   input?** The composition reads the circuit profile at each displaced point, so
   a builder that ignores its input and reads a closure would be differentiated
   against the wrong parameter. Today that surfaces as "must control exactly one
   gate parameter; found 0" for the ignored position, which is diagnosable but
   describes the symptom. A declared builder contract would describe the cause.
2. **Should the diagonal be cheaper through a dedicated stencil?** The diagonal
   cell of a two-term rule costs three evaluations where a pure second difference
   would cost two. The third evaluation is the `f(0)` term that the two-term
   composition needs, so removing it changes the rule rather than the arithmetic.
3. **What is the intended public home if a user asks for this by name?** The
   chosen surface mirrors PennyLane, where `qml.jacobian`/`jvp`/`vjp` are top-level
   and the Hessian lives at `qml.gradients.param_shift_hessian` with no
   `qml.hessian`. If FlagQuantum later wants a root export, that is a rule 8
   authorization and a separate change.

## Owner and approvals

- **Implementation owner:** core (`flagquantum/gradients.py`,
  `flagquantum/core/operator_schema.py` as the declaration it reads).
- **Integration surfaces touched:** `contracts/` (one new contract),
  `.github/workflows/ci.yml` (one step), `tools/` (one gate, one pre-push entry),
  `docs/api-changes/README.md` (one index line). These are protected paths, so the
  change is an integration change and carries its own contract, gate, tests, and
  record.
- **Authorization:** `docs/api-changes/FQ-PARAMETER-SHIFT-HESSIAN-20261030.md`
  (this document), listed in the contract's `authorization` field.
- **No API-owner approval required.** The change touches no stable export.
