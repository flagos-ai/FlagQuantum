# API Change Proposal 080: Full-space matrix expectation requests

## Status and decision

**Proposed; implementation is not approved.** This is a bounded M2 developer
workflow step in the [shared-framework roadmap](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md).
The inspected baseline is main `4488cc5a17abb631ebc1c4d46d698bb0c408aa1e` on
2026-10-20. Proposals 073/078/079 already authorize their implemented dynamics
contracts; they do not authorize this change to the protected output boundary.

Approve or revise these five decisions together:

1. Widen the existing expectation factory and OutputRequest field to accept a
   full-space dense torch.Tensor, retaining the existing Pauli Observable form.
2. Consume that request on the existing CPU Lindblad path; reject it explicitly
   on circuit and remote paths that cannot represent it in CircuitIR 1.0.
3. Snapshot new matrix-request values at planning while preserving autograd, and
   reuse the existing numeric plan formats and result accessors.
4. Refuse collisions between explicit and generated Lindblad output names,
   instead of silently overwriting a requested output.
5. Define tensor-request equality by the tensor's object identity, preserving
   existing Pauli equality and hashes and avoiding tensor-valued comparisons.

Document review or merge does not approve implementation. The API owner must
explicitly approve this contract before production signatures, behavior or
protected snapshots change. No capability or parity maturity is promoted here.

## User problem and existing arithmetic

A three-level, spin-mode or explicitly labelled model can already evolve and
differentiate on the shared dynamics path. Its projectors are matrices, but
`fq.expectation(matrix)` currently raises
`TypeError: expectation output requires an Observable`. The root Observable
type specifically denotes a real Pauli sum; relabelling a three-level projector
as a Pauli term or padding it with fictitious qubits changes its meaning.

The lower-level dynamics API already evaluates full-space Hermitian matrices.
This current-API example is runnable before approval, with no hardware or
credentials. It uses the same leakage model as
`examples/mixed_dimension_dynamics.py`:

```python
import torch
from flagquantum.simulation.lindblad import evolve_density_matrix

g = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
transition = torch.zeros((3, 3), dtype=torch.complex128)
transition[1, 2] = transition[2, 1] = 1
projector = torch.diag(torch.tensor([0., 0., 1.], dtype=torch.complex128))
times = torch.linspace(0., 0.5, 51, dtype=torch.float64)
result = evolve_density_matrix(
    g * transition, [0., 1., 0.], None, times,
    observables={"leakage": projector}, dimensions=(3,), require_gradients=True,
)
value = result.expectation("leakage")[-1]
(gradient,) = torch.autograd.grad(value, g)
print(f"Leakage expectation: {value.item():.8f}")
print(f"Coupling derivative: {gradient.item():.8f}")
print("Matches population:", torch.equal(
    result.expectation("leakage"), result.populations[:, 2]
))
```

Observed Python 3.12 CPU output:

```text
Leakage expectation: 0.11757891
Coupling derivative: 0.32210884
Matches population: True
```

The independent reference is sin(gT)^2, with derivative T sin(2gT), at T=0.5.
All frequencies use the inverse of the grid's time unit. The proposal connects
this existing arithmetic to the common output request; it adds no numerical
core, training loop, result wrapper or model builder.

## Exact proposed surface

Existing signature:

```python
expectation(observable: Observable, *, name: str | None = None) -> OutputRequest
```

Proposed signature on the same root and observables-namespace function:

```python
expectation(
    observable: Observable | torch.Tensor, *, name: str | None = None
) -> OutputRequest
```

The existing frozen OutputRequest dataclass keeps its fields, order, defaults
and slots. Only the `observable` annotation and accepted expectation value widen:

```python
observable: Observable | torch.Tensor | None = None
```

No new root export, factory, output kind, numerical backend or matrix-observable
class is introduced. Observable itself remains the Pauli algebra. Type checking
may refer to torch without making Core IR depend on it. Keep normal package
imports lazy where they are lazy today.

With the model variables above, the proposed replacement is:

```python
# Proposed API; not executable on the unmodified baseline.
import flagquantum as fq
import flagquantum.lindblad as fql

plan = fql.plan(
    g * transition, [0., 1., 0.], times, dimensions=(3,),
    outputs=fq.expectation(projector, name="leakage"),
    options=fq.ExecutionOptions(require_gradients=True),
)
result = fql.run(plan)
value = result.expectation("leakage")[-1]
```

The expectation and its coupling derivative must match the current-API example
above, using a fresh gradient evaluation. The existing high-level plan/run
signatures do not change. In particular, this does not add dynamics dispatch to
root `fq.run` or root `fq.plan`.

## Matrix meaning and validation ownership

The matrix denotes one operator on the entire declared D-dimensional space.
Its rows and columns follow existing product order or the exact order of
`basis_labels`. A two-element explicit basis remains non-product. There is no
qubit inference, implicit tensor factor, embedding, projection or basis change.

The new input is an ordinary, non-quantized strided torch.Tensor with real,
complex or integral numeric dtype, excluding bool and meta tensors. It must be
rank two, nonempty and square. Refuse other object types and unsupported tensor
representations with TypeError; refuse invalid rank or square shape with
ValueError at request construction, when those facts are already known.
Array-like inputs can be made explicit with `torch.as_tensor`. Sparse tensors,
operator batches, interval-dependent matrices, local selectors, callables and
direct BosonOperator requests are outside this bounded addition. A boson model
can use its existing explicitly truncated `to_matrix` conversion.

Planning checks exact D-by-D compatibility, finiteness and Hermiticity using
the existing Simulation normalizers and their existing exception categories,
field context and **1e-6 absolute, zero-relative Hermiticity check**. Do not fork
those numerical rules or change their budget. The common constructor owns only
context-free representation checks; the declared-space normalizer owns numeric
interpretation and target dtype/device conversion. The supported execution
target remains CPU at the existing requested complex precision.

Only expectation requests accept the new tensor form. Samples/counts remain
Pauli-only, and density/probability/entropy rules stay unchanged. Direct
OutputRequest construction must enforce the same new expectation rules rather
than offer a bypass. A tensor observable on another output kind must raise
TypeError; currently a direct samples/counts constructor can instead reach an
AttributeError while looking for Pauli terms. For matrix expectations, a
nonempty `qubits` selection raises ValueError because the matrix is full-space.
Existing valid Pauli requests remain unchanged; the named collision correction
below is a separate proposed change to their behavior.

## Shared results, names and axes

Retain EvolutionResult and all its accessors. An expectation has shape `(T,)`
for an unbatched initial state or `(T, B)` for a declared batch. The matrix is
shared by all B states and all T times. An explicitly batched initial-state
tensor with B=1 keeps its existing axis; do not squeeze it, reshape an unbatched
input merely because batch_size=1, or introduce an operator-batch axis. The existing
`expectations` tuple preserves request order, and `expectation(index_or_name)`
selects the same trajectory. No result field or serialized result shape changes.

### Request equality and hashing

The existing frozen dataclass comparison cannot simply compare two independent
tensor fields: tensor equality returns an array, which can raise an ambiguous
boolean RuntimeError. For the new matrix form, two OutputRequests compare equal
only when they have the same concrete request class, matching other fields and
the **same tensor object**. A clone with equal entries is a different live
input and compares unequal. A matrix and a Pauli observable compare unequal.
Comparison must return a scalar boolean without reading tensor values,
synchronizing devices or affecting autograd. Retain existing same-class
comparison and field-value equality for all old Pauli/None requests.

Keep hashability. For matrix requests, hash the existing field tuple with the
matrix's object identity in the observable position; for old requests, retain
the exact existing field-tuple hash. Mutating matrix entries does not change
the descriptor's hash. This is live-input identity, not numeric plan equality:
two separately allocated matrices with equal entries and the same names may
produce equal numeric plan identities while their requests compare unequal.
No new public identity field, serialized key or hashing registry is added.

### Effective output names

For Lindblad requests, resolve every effective name first: use the explicit name
unchanged, otherwise `expectation_{index}` with the existing zero-based request
index. If any final keys collide, raise the existing EvolutionValidationError
with code `duplicate_output_name`, field `outputs`, before creating the mapping
or integrating. Do not rename an explicit request or silently choose a winner.
This applies to Pauli and matrix expectations on this namespace. It does not
change circuit output naming.

The measured baseline case is:

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql

result = fql.run(
    torch.zeros((2, 2)), [1., 0.], [0., 0.1],
    outputs=[fq.expectation(fq.Z(0), name="expectation_1"),
             fq.expectation(fq.X(0))],
)
print(len(result.expectations))  # Current behavior: 1; two were requested.
```

The approved replacement would refuse this ambiguity. Existing explicitly
duplicated names retain their current category. Users of a colliding pair can
migrate by supplying distinct explicit names; valid names and automatic names
do not change. No deprecation or removal of a valid distinct-output form is
proposed, but the API owner must approve this correction as a behavior change.

## Live tensors, sealed plans and replay

An OutputRequest may reference a caller tensor until planning, as PyTorch model
inputs do. At planning, the new matrix-request path takes a differentiable copy
of its normalized operator values before retaining them in the live plan.
Subsequent mutation of the source tensor or request must not change that plan's
values, recorded identity or result. Cloning must not detach the graph or turn
an output parameter into a constant. A new optimizer step requires a newly
built plan, just as current Hamiltonian training does.

This is a requirement for the newly accepted request form, not approval to
redesign tensor ownership throughout existing low-level or Hamiltonian paths.
Reuse the current named-matrix plan boundary and refuse semantic overrides in
`run(plan, ...)` exactly as today.

Keep `require_gradients`'s existing meaning: it requires a differentiable
trajectory. A trainable observable alone does not satisfy that flag when the
state evolution is constant; that case keeps the existing
`gradients_not_available` refusal. Without that trajectory requirement,
PyTorch may differentiate an expectation with respect to its live matrix, and
that connection must be tested. With a trainable Hamiltonian or initial state,
the existing flag and the new observable graph must both be preserved.

Lindblad plans already serialize ordered `{name, operator}` entries as complex
shape/value payloads. Reuse versions 1.3, 1.4, 1.5 and 1.6 according to the
existing state-space and schedule rules. **No 1.7 plan format is proposed.**
Old accepted requests retain exact payload bytes and identities. Equal named
matrix requests through low-level and common boundaries must encode the same
numeric content. Names, order, matrix entries, labels and schedule still affect
identity under the current hashing rules.

Numeric JSON replay retains values, names, axes and basis interpretation, not
an autograd graph or the source tensor's trainability. Detached replay must
continue to refuse a trajectory-gradient requirement; do not fake zero
gradients. Serialization of OutputRequest itself is not added. CircuitIR 1.0,
ExecutionPlan's circuit schema and deployment packages do not change.

## Unsupported paths and refusal stages

| Request | Required behavior |
| --- | --- |
| Supported matrix through fql.plan/run | Existing normalization, plan, generator, trajectory and result path |
| Wrong D, nonfinite or non-Hermitian matrix | Existing EvolutionValidationError before integration, with observable context |
| Invalid tensor representation/rank | Constructor TypeError/ValueError as specified above |
| Explicit/generated name collision | `duplicate_output_name`, field `outputs`, before mapping construction |
| Matrix through root circuit plan/run or remote circuit outputs | CapabilityError explaining that dense expectations currently require the Lindblad path, before numerical execution or job submission |
| Matrix on a non-expectation request, including direct samples/counts construction | TypeError for the new tensor form, without implicit diagonalization or sampling |
| Full-space matrix with a nonempty qubits selection | Constructor ValueError; no implicit subsystem expansion |
| Request override on a sealed plan | Existing TypeError; no replan or replacement of recorded outputs |

The common circuit `lower_outputs` boundary currently assumes `.terms` on
expectation observables. It must explicitly distinguish matrices before that
assumption. Audit its existing root, remote compute, receipt, emulation and
submission consumers; a matrix must not become AttributeError, a fabricated
Pauli MeasurementNode or a submitted unsupported job. Reuse the common refusal
where it already covers these paths rather than add a parallel dispatcher.

## Alternatives and subtraction review

Keeping only the low-level observables mapping leaves the measured developer
workflow gap. Adding `matrix_expectation`, DenseObservable or a second output
request type duplicates the existing boundary without a second responsibility.
Generalizing Pauli Observable would mix an algebra and a full-space coordinate
matrix. Automatic Pauli expansion would impose qubit semantics on non-product
models and invent scaling costs. A new plan version would describe no new
numeric data. None of these alternatives is selected.

The selected design adds no production dependency, abstract base class,
backend/observable registry, solver, trainer or alternate result representation.
Existing Core state spaces, scalar binding, Liouvillian, integrators, plans and
PyTorch optimization remain authoritative. Native device measurement,
accelerator execution, differentiable time boundaries and continuously varying
observables retain separate roadmap gates.

## Delivery and acceptance sequence

1. Obtain explicit API-owner approval of this document. The integration owner
   then lands a replaceable contract baseline and consumer assertions using the
   existing low-level matrix execution plus independent trace/analytic
   references. Do not implement a fake numerical solver.
2. Before production changes, capture exact old 1.3/1.4/1.5/1.6 numeric fixtures
   and the existing Pauli output behavior. Protect this narrow approved delta
   in the existing observable-output candidate/checker and API-change record;
   do not create another registry or regenerate unrelated contracts.
3. Implement the output boundary change, then replace the baseline adapter
   with the existing public request factory and plan/run consumer. Keep the
   consumer assertions and scientific budgets unchanged. Land owner-separated
   implementation changes if domain ownership requires them.
4. Complete the scenario and refusal matrix below, the normal default/runtime
   tiers, repository/API checks and latest full CI. Merge and queue entry each
   require the user's applicable authorization; proposal approval alone is not
   a blanket merge grant.

| Acceptance property | Required independent evidence |
| --- | --- |
| Same model, same shared path | Qutrit projector versus sin(gT)^2; spin-mode observable versus an independent trace; existing three-integrator coverage |
| Complex and non-product coordinates | Hermitian matrix with imaginary off-diagonal entries; consistent matrix/state/basis permutation preserves the expectation; mismatched permutation is detected by the reference |
| Results and batching | T and declared B axes, B=1, multiple output names/order and accessors; no output overwritten |
| Request comparison | Scalar equality and matching hashes for the same tensor object; unequal clones and Pauli/matrix pairs; distinct request classes and old Pauli hashes retain existing behavior; no device synchronization or tensor-value comparison |
| Gradients and training | Static and scheduled Hamiltonian derivatives, initial-state gradients, matrix-entry derivatives through a Hermitian parameterization, finite differences and an optimizer loss decrease |
| Value ownership | Source matrix mutation after planning leaves numeric replay and live values consistent; live gradients still reach the original parameters |
| Exact-plan and replay | Live/JSON numeric agreement, closed-plan override refusal, matrix/name/order identity sensitivity and fixed old payloads |
| Early refusal | Unsupported circuit/remote paths never reach execution/submission; malformed matrices and name collisions never reach interval integration |
| Existing behavior | Valid Pauli outputs, sampling/counts, qubit plans and all approved product/explicit-basis formats retain their assertions |

Discretization, iterative residual, truncation and gradient error budgets remain
independent. Neither a passing matrix example nor the final CI permits relaxing
them, certifying native photonic/topological instructions or claiming M2-M4
complete. API naming review, exact approved deltas, release notes and the
authorization record remain requirements of
[Public API Protection](PUBLIC_API_PROTECTION.md).
