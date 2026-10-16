# API Change: `fq.jacobian`, `fq.jvp`, and `fq.vjp` for a program with several outputs

## Status

Implemented for review on 2026-10-26 with explicit API-owner authorization for
three additive Stable Core root exports, `fq.jacobian`, `fq.jvp`, and `fq.vjp`.
It adds three root names, changes no existing signature, changes no serialized
schema, and leaves `IR_VERSION` at `1.0`.

Under `docs/development/PUBLIC_API_PROTECTION.md` an additive stable API requires
a concrete user journey, a reason the names belong in Stable Core rather than a
namespace, typing and documentation, executable behavior contracts, API-owner
approval, a release-note entry, and the updated machine-readable contract. Each
is recorded below.

This record amends the gradient contract rather than creating a second one:
`contracts/gradient-api-v1-candidate.json` gains a `vector_derivatives` block and
three entries in `signatures`, and its `root_additions` grows from `["gradient"]`
to `["gradient", "jacobian", "jvp", "vjp"]`. Per Human Maintainability Guardrail
4, a new contract type is not introduced, because the existing authoritative
contract already carries the gradient entry points and their signatures.

## Problem

`fq.gradient` needs one scalar. `flagquantum/gradients.py` enforces that
deliberately: `_scalar_loss` refuses anything whose `numel()` is not `1`. A
program that returns a probability vector or one expectation value per qubit has
no single scalar gradient, so today a caller has three options, none of them
public:

| Route today | What it needs | What it reports |
| --- | --- | --- |
| Write a loop over `fq.gradient` with one scalar loss per output element | One program closure per element, each indexable by position | A list the caller assembles; the cost is one backward pass per element, and nothing says so |
| Sum or weight the outputs before differentiating | A scalar written by hand | The one combination the caller chose, never the derivative of the vector |
| `torch.autograd.grad` on an internal result tensor | Reaching past `fq.run` into whatever tensor the mode happens to keep | Nothing about FlagQuantum; the route is a property of an executor the caller had to know |

The second row is the one that costs real work. A training step that weights
several measured values — the shape every variational algorithm has — is exactly
`vjp`, and getting it by hand means summing weighted outputs and calling
`fq.gradient` on the sum. That works, and `fq.gradient` already serves it, but it
gives the caller no way to obtain the individual partial derivatives, no way to
apply a direction forward, and no statement about which route to use.

The PennyLane comparison makes the gap concrete. `qml.jacobian(fn)(p)` returns
the full derivative of a vector-valued QNode with no workaround, so a user
porting a variational loop reaches for a name that FlagQuantum does not have.
`qml.jvp` and `qml.vjp` also exist as names, but they are **not** usable the same
way; see "Scope of the support claim" below for the measured refusals.

## Decision

### Three new root exports

| Entry point | Returns | Shape |
| --- | --- | --- |
| `fq.jacobian(program, parameters)` | Every partial derivative | `(*program_output.shape, *parameters.shape)` |
| `fq.jvp(program, parameters, tangents)` | `J @ tangents` | `program_output.shape` |
| `fq.vjp(program, parameters, cotangents)` | `cotangents @ J` | `parameters.shape` |

`program` is a callable from a parameter tensor to a real tensor — the same shape
of callable `fq.gradient` accepts, minus the `loss`. Every result is detached and
carries the dtype and device of `parameters`.

The three names state their domain meaning directly (rule 9): a Jacobian is the
derivative matrix, and the Jacobian-vector and vector-Jacobian products are the
two ways to apply it. None of them is named for a route, a maturity, or an
implementation era.

### No `method`, and the reason is measured

`fq.gradient` takes `method=`, `step=`, `directions=`, and `generator=`. None of
the three new entry points takes any of them. This is a decision, not an
omission:

- The shift rule is a per-opcode scalar rule. `flagquantum/core/operator_schema.py`
  declares parameter frequencies for one parameter at a time, so there is no
  shift form for a vector output, and `method="parameter_shift"` would have to be
  refused anyway.
- `fq.jvp` and `fq.vjp` are exact by construction: they are the chain rule
  applied to the graph the executor already builds. There is no displacement to
  report, so `exact=True, step=None` would be the only possible outcome, which
  is a constant rather than a result field.
- `fq.jacobian` is one backward pass per output element, all through the same
  graph. It cannot be approximated, because a difference quotient would need a
  displacement the caller could not choose and whose effect is not readable from
  a matrix-shaped result.

A program whose output carries no PyTorch graph is therefore refused rather than
approximated, because the approximation route that `fq.gradient` falls back to
does not exist here. The refusal names the supported routes.

### Forward-mode autograd is unusable here, measured

The obvious implementation of `jvp` is `torch.autograd.forward_ad`. It does not
work in this tree, and the measurement is a single named line:

```console
$ python -c "import torch, flagquantum as fq; ..."   # dual-level probe over the reference circuit
NotImplementedError: Trying to use forward AD with mul_out that does not support it because it is an out= function
```

raised from `flagquantum/simulation/statevector/single_qubit_cpu.py:46`,
`torch.mul(zero, matrix[:, 0, 0].reshape(coefficient_shape), out=result_zero)`.
An out-of-place rewrite of that kernel is a separate change to the statevector
core with its own performance question; it is not a prerequisite for a
derivative entry point.

`fq.jvp` is therefore implemented as a double backward: seed the output with a
tracked zero tensor, take one reverse pass with `create_graph=True`, contract the
pulled-back gradient with the tangent, and differentiate that scalar with respect
to the seed. One implementation then serves all three differentiable modes.

The seed's dtype is the parameter dtype, not the output dtype. This is a measured
choice: the reverse pass carries the chain rule in the parameter dtype, and
seeding in the output dtype makes the result dtype depend on the seed rather than
on the parameters. Measured, with `p` in `float64`:

| Program output dtype | Tangent dtype | Seed dtype | Result dtype |
| --- | --- | --- | --- |
| `complex64` circuit | `float32` | `float64` | `float64` |
| `complex64` circuit | `float64` | `float32` | `float32` |
| `complex64` circuit | `float32` | `float32` | `float32` |

With the seed taken from `parameters.dtype`, all three entry points return the
parameter dtype for all six `circuit_dtype` × `parameter_dtype` combinations, in
all three differentiable modes — eighteen measured combinations, recorded in
`tests/test_vector_derivatives.py`.

### Shape rules are exact, not broadcast

A tangent must be shaped exactly like `parameters`, and a cotangent exactly like
the program output. A direction that merely broadcasts is refused:

```console
$ fq.jvp(program, parameters, torch.ones(1))     # parameters is (3,)
ValidationError: tangents must be shaped like the parameters: expected (3,), got (1,)
```

Broadcasting would differentiate along a different parameterization — a scalar
repeated three times is not the same direction as three independent values — and
the result would look plausible for the wrong reason. This mirrors the rule
`torch.autograd.grad` applies to `grad_outputs`, which is measured to reject a
flat `(2,)` cotangent against a `(2, 1)` output with
`RuntimeError: Mismatch in shape`. Refusing earlier, in domain terms, is the
fail-closed form of the same rule.

### Refusals are explicit, and there is no silent fallback

| Request | Outcome |
| --- | --- |
| a program whose output carries no autograd graph | `CapabilityError` naming `fq.gradient` and `fq.jacobian` |
| a complex program output | `ValidationError` naming the missing Wirtinger convention |
| a program that returns anything but a tensor | `ValidationError` |
| an empty program output | `ValidationError` |
| `fq.jvp` where the executor cannot build a graph of a graph | `CapabilityError` |
| a tangent or cotangent that is not a tensor | `TypeError` |
| a tangent or cotangent that is not real floating point | `ValidationError` |
| a tangent or cotangent that is not finite | `ValidationError` |
| a tangent or cotangent whose shape is not the paired tensor's | `ValidationError` naming both shapes |
| a parameter tensor that is not a tensor | `TypeError` |
| an empty, non-real, or non-finite parameter tensor | `ValidationError` |

The no-graph refusal is the one that matters most. Answering a program that
cannot be differentiated with a zero Jacobian would be the silent fallback
principle 9 forbids, and it is a plausible mistake: a program that scores itself
under `torch.no_grad()` returns a correct-looking value whose derivative is not
defined by any route here. It is refused, with the message
`the program's output does not depend on the supplied parameters through PyTorch
autograd, so it has no derivative with respect to them.`

The complex-output refusal is a scope statement rather than a missing feature.
No entry point in this tree defines a Wirtinger convention, so the derivative of
a complex output is not a single well-defined tensor. `torch.autograd.grad`
agrees: differentiating a `complex128` output raises
`RuntimeError: grad can be implicitly created only for real scalar outputs but
got torch.complex128`. The refusal says what to do instead: take a real value
such as an expectation.

### What is reused, and what is new

`flagquantum/gradients.py` gains 275 lines and changes none. The pre-existing
`gradient`, `parameter_shift_gradient`, and `batched_parameter_shift_gradient`
kernels are **not modified**; nor is any other route. What the new entry points
reuse is the same forward pass the scalar entry point uses:

| Concern | Reused |
| --- | --- |
| Parameter validation | `_validated_parameter_tensor` — the same non-tensor, empty, non-real, and non-finite refusals `fq.gradient` raises |
| The forward pass | `_vector_value` calls the program once with a tracked copy of the parameters, exactly as `fq.gradient`'s `auto` probe does |
| The reverse pass | `_pulled_back` wraps `torch.autograd.grad` with the same `allow_unused` handling the scalar route uses |
| Mode coverage | Nothing mode-specific: the derivative is taken through whatever executor the caller selected, so all three differentiable modes are served by one implementation |

`fq.jacobian` shares the forward pass across its rows rather than re-running the
program per output element, which the focused test pins by counting program
invocations. That is the difference between one forward pass and `numel()`
forward passes, and it is the reason the row loop uses `retain_graph=True`.

What this change **replaces** is the hand-written workaround. Before it, the
only way to obtain a derivative of a vector-valued program was to sum its
outputs into a scalar and differentiate that, which silently discards every
partial derivative the caller did not include. After it, the whole derivative is
reachable, and the two products are reachable in their natural shapes.

No new contract type, registry, manager, factory, or intermediate result object
is introduced. There is no `JacobianResult`: `GradientResult` exists because
`fq.gradient` has four facts to report (value, method, exactness, step), and the
three new entry points have exactly one value each, so a plain detached tensor is
the smallest object that carries the answer.

### Scope of the support claim

| Dimension | Boundary |
| --- | --- |
| Execution mode | `statevector`, `mps`, `tensor_network`, verified per mode |
| Hardware | CPU and single GPU, by reuse of the existing mode paths |
| Distribution | `single_device_fast_path`; no sharded-derivative claim is made |
| `stabilizer` mode | Refused with `CapabilityError`, because the mode samples measurement outcomes and cannot serve an expectation value |
| Route | Autograd only. No shift rule, no difference quotient, no SPSA route |
| Second derivative | Not provided; every result is detached. A second derivative needs an explicit route |
| `method=`, `step=`, `directions=`, `generator=` | Not accepted by any of the three |

Executing in MPS or tensor-network mode does not fall back to statevector.
Non-negotiable rule 3 is preserved because the derivative is taken through the
mode the caller selected.

### Relationship to PennyLane, measured

The comparison document's entry for this slice recorded PennyLane 0.45.1 as
providing all three. That is correct for the names and wrong for two of them.
Re-measured on 0.45.1, on the same two-qubit circuit:

| PennyLane call | Result |
| --- | --- |
| `qml.jacobian(qnode)(p)` | `ndarray`, shape `(2, 3)`, `float64` |
| `qml.jacobian(qnode)(p)` where the QNode returns a list | `ValueError: autograd can only differentiate with respect to arrays, not <class 'list'>` |
| `qml.jvp(qnode, (p,), (tangents,))` | `CompileError: PennyLane does not support the JVP function without QJIT.` |
| `qml.vjp(qnode, (p,), (cotangents,))` | `CompileError: PennyLane does not support the VJP function without QJIT.` |

So `qml.jvp` and `qml.vjp` exist as names and fail closed without a QJIT
compilation path, and `qml.jacobian` requires the QNode to return a stacked
array rather than a list. FlagQuantum's three entry points serve all three shapes
with no compilation step and accept either a stacked tensor or any real tensor
the program returns. The comparison document's `N3-6` block is corrected
accordingly; no parity-matrix entry claims a PennyLane behavior that 0.45.1 does
not exhibit.

## Compatibility

- Additive: three new root exports. No existing export is renamed, removed,
  reordered, or changed. `fq.gradient`'s signature is byte-identical to the base.
- `root_export_budget` moves from 36 to 39 in
  `contracts/public-api-v1-candidate.json`, and `stable_core.retain` follows,
  keeping the invariant that the budget equals the retained name count. This
  follows the `from_openqasm` and `gradient` precedents, each of which raised the
  budget with the count.
- `docs/public_api_v1.json` gains `jacobian`, `jvp`, and `vjp` in
  `stable_exports` with one verification node each, so the generated
  `docs/generated/STABLE_API.md` gains three rows.
- `capability-maturity.toml`'s `[capabilities.gradient_methods]` gains the three
  names in `public_apis` and the boundary paragraph quoted above. No new
  capability entry is created: these are the same capability, reached by
  programs with several outputs, and a second entry would be a second source of
  truth for the same support boundary.
- No serialized schema changes. `IR_VERSION` stays `1.0`. No exception type is
  added: the refusals reuse `CapabilityError`, `ValidationError`, and `TypeError`.
- `tools/public_api_snapshot.py`'s authorized-signature table gains the three
  names. The table is a list of the objects the contract's `signatures` block
  names; a name absent from it raised `KeyError` before the edit, which is the
  check failing closed rather than a check that had to be relaxed.

### Rollback

`flagquantum/gradients.py` and `flagquantum/_api.py` revert independently.
Reverting the three functions and their `_api.py` delegates leaves `fq.gradient`,
`parameter_shift_gradient`, and `batched_parameter_shift_gradient` untouched,
because this change adds to the module rather than modifying it. The contract,
manifest, capability entry, and documentation additions then revert with the same
commit.

## Verification

```console
$ python -m pytest tests/test_vector_derivatives.py -q
36 passed

$ python -m pytest tests/integration/test_gradient_modes.py -q
19 passed

$ python -m pytest tests/test_gradient_api.py -q
37 passed

$ python tools/public_api_snapshot.py
public API migration baseline passed

$ python tools/check_capability_maturity.py
capability maturity matrix passed

$ python tools/docs_source_of_truth.py --check
$ echo $?
0
```

The last command prints nothing on success and exits `0`; it lists the stale
generated files and exits non-zero when they drift, which is the state measured
before this change's regeneration.

The executable contract behind the entry points is measured on

```python
def build_circuit(parameters):
    return (fq.Circuit(2).ry(0, theta=parameters[0]).rx(0, theta=parameters[1])
            .ry(1, theta=parameters[2]).cx(1, 0))

def two_expectations(parameters):
    result = fq.run(build_circuit(parameters),
                    outputs=[fq.expectation(fq.Z(0)), fq.expectation(fq.Z(1))])
    return torch.stack(list(result.expectations))
```

with `parameters = (0.3, 0.7, -0.4)` in `float64`, and the circuit in the default
`complex64` amplitudes:

```console
$ python examples/gradient_methods/run.py
jacobian           shape=(2, 1, 3) matvec_error=5.582e-08
jvp/vjp            adjoint_error=5.960e-08
```

`fq.jacobian(two_expectations, parameters)` returns

| | `d/dp0` | `d/dp1` | `d/dp2` |
| --- | --- | --- | --- |
| `Z0` | `-2.081840634346e-01` | `-5.668621063232e-01` | `2.845408916473e-01` |
| `Z1` | `0.000000000000e+00` | `2.980232238770e-08` | `3.894183635712e-01` |

The two zeros in the `Z1` row are structural rather than numerical: qubit 1 is
rotated only by the third parameter, so its expectation cannot depend on the
first two, and the `2.98e-08` residual is the `complex64` amplitude dtype's
epsilon propagated through the backward pass. The analytic identity
`<Jv, c> == <v, J^T c>` holds to `5.960e-08` — the same epsilon — for the
tangent `(1.0, -2.0, 0.5)` and the cotangent `(0.25, -1.5)`. Widening the
circuit to `complex128` drops both residuals below `1e-15`, which is what
`tests/test_vector_derivatives.py` asserts, so the tolerance in the test is a
statement about the arithmetic and not a statement about the mode.

Measured across modes and dtypes, all eighteen combinations of
`circuit_dtype ∈ {default, complex64, complex128}` ×
`parameters ∈ {float32, float64}` ×
`mode ∈ {statevector, mps, tensor_network}` return a consistent `J`, and
`J.dtype == jvp.dtype == vjp.dtype == parameters.dtype` in every one.

## Documentation

- `docs/reference/API.md` gains a `### Differentiate several outputs at once`
  section and one row in the entry-point map.
- `docs/generated/STABLE_API.md` gains the three rows, generated from
  `docs/public_api_v1.json`.
- `docs/generated/CAPABILITIES.md` and `docs/reference/KNOWN_LIMITATIONS.md`
  gain the boundary paragraph, generated from `capability-maturity.toml`.
- `docs/reference/RELEASE_NOTES.md` gains one entry under `Unreleased`.
- `examples/gradient_methods/` — the existing ten-minute golden path — gains a
  `several outputs at once` section that runs the three entry points and prints
  the shape, the matrix-vector residual, and the adjoint residual.
- `docs/api-changes/README.md` indexes this document.

## Scope

This change does not add a second gradient implementation, a derivative registry,
or a per-mode derivative configuration. It does not claim sharded or distributed
derivatives: `distribution_semantics` is `single_device_fast_path`, and no
scalability claim is made or implied. It does not add `hessian`, `hvp`, or a
metric tensor; a second derivative needs an explicit route because every result
is detached. It does not change `fq.gradient`, `parameter_shift_gradient`,
`batched_parameter_shift_gradient`, or the `OperatorSchema.parameter_frequencies`
declarations the shift rule reads. It does not rewrite the statevector kernels to
support forward-mode autograd; that is a separate change to the statevector core
with its own performance question, and the double-backward route chosen here
serves the requirement without it.
