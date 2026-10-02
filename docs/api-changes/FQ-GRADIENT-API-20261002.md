# API Change: `fq.gradient` and the reported gradient method

## Status

Implemented for review on 2026-10-02 with explicit API-owner authorization for
two additive Stable Core root exports, `fq.from_openqasm` and `fq.gradient`; this
document covers the second of the two, whose sibling is the OpenQASM import
change authorized by the same directive. It adds one root name, changes no
existing signature, changes no serialized schema, and leaves `IR_VERSION` at
`1.0`.

Under `docs/development/PUBLIC_API_PROTECTION.md` an additive stable API requires
a concrete user journey, a reason the name belongs in Stable Core rather than a
namespace, typing and documentation, executable behavior contracts, API-owner
approval, a release-note entry, and the updated machine-readable contract. Each
is recorded below.

## Problem

FlagQuantum differentiates circuits today, but through three unrelated routes
that a user has to choose between before knowing which one applies:

```python
import flagquantum as fq
import torch

module = fq.Module(build_circuit, n_parameters=2)
loss = module(inputs).sum()
loss.backward()  # the autograd route

from flagquantum.gradients import parameter_shift_gradient

parameter_shift_gradient(program, parameters, loss)  # the shift route
```

The routes are not interchangeable, and none of them says which one it used:

| Route today | What it needs | What it reports |
| --- | --- | --- |
| `loss.backward()` on a result tensor | A live autograd graph and a `Module` or result that carries one | Nothing; the caller infers the route from having written it |
| `flagquantum.gradients.parameter_shift_gradient` | The circuit *and* a loss over it | Returns a plain tensor, so the caller cannot tell a refused program from a served one without reading the failure |
| `_finite_difference_gradient` | Nothing public | A private conformance helper inside `flagquantum/runtime/executors/statevector/split_real_imag_autograd_conformance.py`; not reachable as a user route |

Three consequences follow. A caller cannot write one program that works in
`statevector`, `mps`, and `tensor_network` mode without knowing which executor
each mode uses and which of them keeps a graph. A caller cannot tell an exact
derivative from an approximation, because the default displacement is computed
from the result dtype and never surfaced. And a caller who asks for a route that
does not exist for their program gets whatever the fallback chain happens to do
rather than a decision.

The PennyLane comparison makes the gap concrete: `qml.grad` returns a gradient
and the `diff_method` is a property of the device/tape, whereas here the two
concerns are tangled in whichever call the user reached for.

## Decision

### One new root export

`fq.gradient(program, parameters, loss=None, *, method="auto", step=None,
directions=1, generator=None)` returns a `GradientResult` with four fields:

| Field | Meaning |
| --- | --- |
| `gradient` | The detached derivative, with the shape and dtype of `parameters` |
| `method` | The route that actually ran: `autograd`, `parameter_shift`, `finite_difference`, or `spsa` |
| `exact` | `True` for the two exact routes, `False` for the two approximations |
| `step` | The displacement an approximation used; `None` for every exact route |

`gradient` is detached so the result can be handed to an optimizer or written to
a report without extending the autograd tape. `method` is never `"auto"`: `auto`
is an input, not a reported outcome.

### `auto` measures; it does not declare

`method="auto"` probes the program on a tracked copy of `parameters` and reads
the outcome:

1. if the probe builds an autograd graph, the route is `autograd`;
2. otherwise, if `loss=` was supplied, the route is `parameter_shift`, because
   the shift rule is declared per opcode and needs the circuit behind the loss;
3. otherwise the route is `finite_difference`.

Measured, on the reference circuit in `tests/test_gradient_api.py`:

| Program | `loss=` | Reported `method` | `exact` | `step` | Relative error vs. `autograd` |
| --- | --- | --- | --- | --- | --- |
| `fq.Circuit` builder returning an autograd tensor | the circuit loss | `autograd` | `True` | `None` | — |
| the same builder, with the loss detached | the detached loss | `parameter_shift` | `True` | `None` | `1.577e-07` |
| a closure that scores under `torch.no_grad()` | omitted | `finite_difference` | `False` | `4.92156660115185e-03` | `1.935e-05` |

The second row is why `auto` prefers the shift rule over a difference when
`loss=` is present: the circuit is available, so the exact route is reachable and
`auto` takes it rather than approximating.

The third row is the case the API exists for: a program that cannot carry a graph
still receives a derivative, and the result says the derivative is an
approximation and how large the displacement was. That displacement is
`float32 eps ** (1/3)`, chosen from the dtype the program actually returned.

### Refusals are explicit, and there is no silent fallback

| Request | Outcome |
| --- | --- |
| `method="adjoint"` | `CapabilityError` |
| `method="parameter_shift"` without `loss=` | `CapabilityError` |
| `step=` on an exact method | `ValidationError` |
| `directions=` or `generator=` outside `"spsa"` | `ValidationError` |
| an unknown `method` value | `ValidationError` naming every accepted value |
| a `step` that is not positive and finite | `ValidationError` |
| a `directions` value below `1` | `ValidationError` |
| a non-integer `directions` | `TypeError` |
| a parameter tensor that is not a `Tensor` | `ValidationError` |
| a `loss` that returns anything but one scalar tensor | `ValidationError` |

The `adjoint` refusal is a measured statement about the tree, not a placeholder.
FlagQuantum does implement a reversible adjoint sweep, as the backward pass of
`_ShardedStatevectorExpectation` in
`flagquantum/runtime/executors/statevector/reverse.py`, but it has no standalone
entry point: it is reachable only by calling `.backward()` on a result, and no
result reports whether backward replayed the circuit or accumulated a graph.
Exporting `"adjoint"` as a `method` value would therefore claim a user-visible
route that does not exist. The refusal names `"autograd"` as the supported
alternative instead.

`method="parameter_shift"` without `loss=` is refused for the same reason: the
shift rule is a per-opcode property and cannot be derived from an opaque
callable, so there is nothing to shift. The message names
`method="finite_difference"` as the route that does work without the circuit.

### `GradientResult` is documented but not exported

`GradientResult` lives at `flagquantum.gradients.GradientResult` and is named in
this document, the API reference, and the execution-options contract. It is not
added to `flagquantum.__all__`, so it does not spend a root-export slot and does
not become a stable name to freeze. A caller receives it and reads its four
fields; a caller who needs to name the type imports it from `flagquantum.gradients`.

### What is reused, and what is new

The diff of `flagquantum/gradients.py` against the base is 332 added lines and
one changed line: the pre-existing `parameter_shift_gradient` and
`batched_parameter_shift_gradient` kernels are **not modified**. Every existing
route keeps working exactly as before, and `fq.gradient` reuses them rather than
reimplementing them:

| Route | Kernel `fq.gradient` dispatches to | Kernel status |
| --- | --- | --- |
| `autograd` | `_autograd_gradient` over `_backward` | New here; the operator still publishes one reverse-mode graph |
| `parameter_shift` | the pre-existing `parameter_shift_gradient` | Unchanged; the exact per-opcode rule it already implemented |
| `finite_difference` | `_central_difference_gradient` | New here |
| `spsa` | `_spsa_gradient` | New here; a one- or many-direction difference, not a convergence claim |

What this change **replaces** is the user-facing choice between those routes.
Before it, a caller reached a route by importing it, and the route available
depended on which of three unrelated surfaces they had found:

| Route before | Reached by |
| --- | --- |
| autograd | `loss.backward()` on a `Module` or a result |
| `parameter_shift` | `from flagquantum.gradients import parameter_shift_gradient` |
| a numerical difference | nothing public; `_finite_difference_gradient` existed only inside `flagquantum/runtime/executors/statevector/split_real_imag_autograd_conformance.py`, as a private check that the statevector autograd implementation agrees with a difference quotient |

After it, all four routes are reached through one name, and the one that did not
exist for callers at all — a numerical difference — is reachable and reported.

`tests/test_gradient_api.py::test_the_public_entry_point_is_the_module_function`
pins the reuse: `fq.gradient` and `flagquantum.gradients.gradient` must return
the same reported method, exactness, step, and gradient for the same inputs, so a
second implementation appearing behind the public name fails the test.
`contracts/gradient-api-v1-candidate.json` records
`second_gradient_implementation: false` as a rule of the contract.

The 332 added lines are the dispatcher, the validation, the three new kernels,
and the docstring. No new contract type, registry, manager, or factory is
introduced: `GradientResult` is a plain frozen dataclass with four fields.

### Scope of the support claim

| Dimension | Boundary |
| --- | --- |
| Execution mode | `statevector`, `mps`, `tensor_network` |
| Hardware | CPU and single GPU, by reuse of the existing mode paths |
| Distribution | `single_device_fast_path`; no sharded-gradient claim is made |
| `stabilizer` mode | Refused with `CapabilityError`, because the mode samples measurement outcomes and cannot serve an expectation value |
| Exactness | `autograd` and `parameter_shift` exact; `finite_difference` and `spsa` declared approximations |
| Fidelity | FlagQuantum's `spsa` is a finite-difference estimator in a random direction, not a stochastic-approximation convergence claim; it is statistical at every `directions` value |

Executing in MPS or tensor-network mode does not fall back to statevector.
Non-negotiable rule 3 is preserved because the derivative is taken through the
mode the caller selected, and the mode is visible in the result's `runtime`
provenance.

## Compatibility

- Additive: one new root export, `fq.gradient`. No existing export is renamed,
  removed, reordered, or changed.
- `root_export_budget` moves from 34 to 35 in
  `contracts/public-api-v1-candidate.json`, and `stable_core.retain` follows,
  keeping the invariant that the budget equals the retained name count.
- `docs/public_api_v1.json` gains `gradient` in `stable_exports` with its
  verification test, so the generated `docs/generated/STABLE_API.md` gains one
  row.
- `capability-maturity.toml` gains `[capabilities.gradient_methods]` at
  `production_supported`, with the limitations quoted above.
- No serialized schema changes. `IR_VERSION` stays `1.0`. No exception type is
  added: the refusals reuse `CapabilityError`, `ValidationError`, and `TypeError`.
- The two pre-existing shift functions keep their public signatures in
  `flagquantum.gradients`; nothing in the tree is broken by the addition.

### Rollback

`flagquantum/gradients.py` and `flagquantum/_api.py` revert independently.
Reverting the `gradient` function and its `_api.py` delegate leaves the
pre-existing `parameter_shift_gradient` behaviour untouched, because this change
dispatches to it rather than modifying it. The contract, manifest, capability
entry, and documentation additions then revert with the same commit.

## Verification

```console
$ python -m pytest tests/test_gradient_api.py -q
37 passed

$ python -m pytest tests/integration/test_gradient_modes.py -q
11 passed

$ python tools/public_api_snapshot.py
public API migration baseline passed

$ python tools/check_capability_maturity.py
capability maturity matrix passed

$ python tools/docs_source_of_truth.py --check
documentation source-of-truth check passed
```

The executable contract behind the entry point, measured on
`fq.Circuit(2).ry(0, theta=p[0]).rx(0, theta=p[1]).ry(1, theta=p[2]).cx(1, 0)`
with `p = (0.3, 0.7, -0.4)` in `float64`, loss `fq.expectation(fq.Z(0))`:

| Route | `exact` | `step` | Value |
| --- | --- | --- | --- |
| `autograd` | `True` | `None` | `[-0.208184063, -0.566862106, 0.284540892]` |
| `autograd` in `mps` | `True` | `None` | identical to the row above |
| `autograd` in `tensor_network` | `True` | `None` | identical to the row above |
| `parameter_shift` | `True` | `None` | max relative error `1.577e-07` |
| `finite_difference` | `False` | `4.92156660115185e-03` | max relative error `1.935e-05` |
| `spsa`, `directions=512`, seed 0 | `False` | `4.92156660115185e-03` | max relative error `8.014e-02` |

The exact routes agree across all three modes to `rtol=1e-9`. A single-qubit
`Ry` closed form pins the absolute scale: `autograd` returns
`-0.29552021622657776` where `-sin(0.3)` is `-0.29552020666133955`, an absolute
error of `9.57e-09`. `fq.gradient(...).gradient.requires_grad` is `False`, and
`len(fq.__all__)` is `35`.

The 34-row gate is the reason this entry point has a focused test file at all.
Enumerating every declared parameter of every differentiable opcode and checking
that it moves the loss gives, under a bare basis-state probe, a vacuous result:
`rx.theta` and the diagonal gates have zero derivative there, and a global phase
is invisible without a coherent superposition. The committed test therefore uses
two probes, a bare basis state and the same circuit behind a leading Hadamard,
and asserts `compared == 34` with every row moving by more than `1e-6`. It runs
in under a second (`elapsed=0.4s`, `rows=34`, `bad=0`).

## Documentation

- `docs/reference/API.md` gains a `## Differentiate a program` section and one
  row in the entry-point map.
- `docs/generated/STABLE_API.md` gains the `fq.gradient` row, generated from
  `docs/public_api_v1.json`.
- `docs/generated/CAPABILITIES.md` gains the `gradient_methods` section,
  generated from `capability-maturity.toml`.
- `docs/reference/KNOWN_LIMITATIONS.md` gains the boundary row, generated from
  the same capability entry.
- `docs/reference/RELEASE_NOTES.md` gains one entry under `Unreleased`.
- `examples/gradient_methods/` is a new ten-minute golden path that runs every
  method on one circuit and prints what each reported.
- `docs/api-changes/README.md` indexes this document.

## Scope

This change does not add a second gradient implementation, a gradient registry,
or a per-mode gradient configuration. It does not claim sharded or distributed
gradients: `distribution_semantics` is `single_device_fast_path`, and no
scalability claim is made or implied. It does not add `adjoint` as a method
value; that remains an owned gap until a standalone adjoint entry point exists
that reports whether the sweep was replayed. It does not change
`parameter_shift_gradient`, `batched_parameter_shift_gradient`, or the
`OperatorSchema.parameter_frequencies` declarations that the shift rule reads.
