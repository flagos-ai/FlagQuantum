# API Change Proposal 078: Mixed-dimension state spaces for existing dynamics

## Status

Approved by the repository owner on 2026-10-19. The owner reviewed this concrete
proposal and explicitly authorized the compatible `dimensions=` namespace
extension, preservation of CircuitIR 1.0 and existing qubit behavior/1.3 plan
identities, and the 1.4 mixed-dimension plan extension. The recorded working-session
decision was to approve the contract and proceed with implementation.
This authorizes implementation, not a claim that its acceptance gates have passed.

This is the contract-first part of M1 in
[`FLAGOS_ALIGNED_RELEASE_TRAIN.md`](../roadmap/FLAGOS_ALIGNED_RELEASE_TRAIN.md).
It authorizes no time-dependent controls, new numerical engine, public IR version,
or hardware package. The subsequent implementation must extend the existing
Lindblad path rather than introduce a second evolution API.

## User outcome and measured reuse

A researcher must be able to evolve a three-level system or a spin coupled to a
truncated motional mode without padding the state to a qubit register. Both
models must use the same validation, planning, integration, result, and gradient
path as existing qubit dynamics.

The existing code already supplies most of that path:

- `core/ir/` owns domain semantics; `CircuitIR` schema 1.0 is the public qubit
  circuit boundary and must retain its meaning.
- `simulation/lindblad_request.py` validates dense states and operators by matrix
  dimension. Its Pauli, bitstring, and named collapse forms assume qubits.
- `simulation/lindblad_generator.py` constructs `Liouvillian` from the existing
  `operators.SuperOperator` algebra using an explicit Hilbert dimension.
- `simulation/lindblad_integrators.py` owns all three existing integration
  methods. Their derivative interface does not require a qubit count.
- `lindblad/_plan.py` owns sealed plans, identities, live tensor bindings, and
  detached JSON restoration. It currently derives the dimension as `2**n_qubits`.
- `observables/boson.py` already creates truncated ladder matrices and orders the
  lowest degree on the most significant index. It is reused, not reimplemented.

A preparatory CPU experiment on the M0 branch used those existing generator,
integrator, and boson modules directly for dimensions `(3,)` and `(2, 3)`.
With coupling 0.7, final time 0.5, 50 intervals, and complex128, the maximum
population errors against `sin(coupling * time)**2` were approximately
`4.5e-12` (Runge-Kutta), `9.3e-7` (Crank-Nicolson), and `7.0e-12`
(Krylov exponential). The Runge-Kutta coupling-gradient error was approximately
`3.8e-11`. This is feasibility evidence about the existing arithmetic, not
evidence that a mixed-dimension user workflow has shipped.

## Narrow internal contract

The proposed internal `ProductStateSpace` belongs under the existing
`flagquantum/core/ir/` boundary. It is not re-exported from `flagquantum`,
`flagquantum.core.ir`, or a new state-space package. It describes a finite ordered
tensor product, not a universal physical system or a second program IR.

| Item | Meaning |
| --- | --- |
| `dimensions` | Nonempty immutable tuple of strictly positive Python integers; booleans are refused. A factor of one is valid, including a vacuum-only boson truncation. |
| `dimension` | Product of all subsystem dimensions; never rounded to a power of two. |
| Basis order | Subsystem 0 is the most significant index, matching existing qubits and boson matrices. |
| Basis index | For occupations `j[k]`, `index = sum(j[k] * product(dimensions[k+1:]))`; every occupation is in range. |
| Qubit compatibility | Every factor must equal two. A total dimension that happens to be a power of two is insufficient: `(4,)` is not `(2, 2)`. |
| Ownership | Core owns this mathematical description; Simulation consumes it and owns numerical validation and evolution. |

The versioned candidate contract and its conformance fake must land before the
dependent domain implementation. The fake lives only in tests. The same
consumer-level tests must then run against the real type without changing their
assertions. No second registry, pass manager, artifact envelope, solver selector,
or training loop is admitted.

The candidate is
[`product-state-space-internal-v1-candidate.json`](../../contracts/product-state-space-internal-v1-candidate.json),
read by
[`test_product_state_space_contract.py`](../../tests/unit/test_product_state_space_contract.py).

An independent contract is necessary because the existing internal IR contract
owns level conversion, def-use, and linearity, while `core/contracts.py` owns
runtime execution/evidence records. Neither currently describes a finite
tensor-product basis. This contract does not redeclare either vocabulary.

## Proposed namespace API

Add one optional keyword, `dimensions: Sequence[int] | None = None`, to the
existing `flagquantum.lindblad.plan`, `run`, and `adjoint_gradient` namespace
entry points, and propagate it to their current Simulation implementations.
No Stable Core root export or `fq.expectation` signature changes.

```python
import torch
import flagquantum.lindblad as fql

hamiltonian = torch.tensor(
    [[0.0, 0.0, 0.0], [0.0, 0.0, 0.35], [0.0, 0.35, 0.0]],
    dtype=torch.complex128,
)
plan = fql.plan(
    hamiltonian,
    [0.0, 1.0, 0.0],
    torch.linspace(0.0, 0.5, 51, dtype=torch.float64),
    dimensions=(3,),
    return_density_matrices=True,
)
result = fql.run(plan)
print(result.populations[-1])
```

This example specifies the approved API. It is not executable until the dependent
implementation lands. That implementation must add an executable ten-minute
example and measure the acceptance items below.

The existing low-level Simulation functions keep their positional `n_qubits`
argument for source compatibility. They may accept `None` there only when
explicit `dimensions` are supplied. There is no renamed or parallel low-level
entry point.

### Resolution and refusal

- With neither explicit `dimensions` nor `n_qubits`, preserve existing qubit
  inference and its rejection of non-power-of-two input.
- With `n_qubits` only, preserve existing validation, behavior, serialized plans,
  and plan identities exactly.
- With `dimensions` only, use their exact product. Do not infer subsystem
  structure from matrix size.
- With both arguments, accept only `dimensions == (2,) * n_qubits`; otherwise
  refuse the ambiguity at planning time.
- Normalize an explicitly all-qubit space to the existing qubit plan form. There
  must not be two identities for the same qubit request solely because its
  dimensions were spelled explicitly.
- For a non-qubit space, initially accept full dense Hamiltonian, collapse, and
  observable matrices of the declared size. Refuse Pauli descriptors, PauliSum,
  bitstrings, and named qubit collapse operators, including on `(4,)`. Do not
  reinterpret their wires as levels or pad them with zeros.
- Keep current initial-state batching rules: a square `(D, D)` input is one
  density matrix; a batch of `D` vectors must be represented unambiguously as
  density matrices. No new shape heuristic is introduced.
- `fq.expectation` remains qubit-only. Dense observables remain available through
  the existing Simulation mapping interface; the high-level M1 example may read
  populations and retained density matrices. Broadening output requests belongs
  to a separately reviewed shared workflow change.
- CPU complex64/complex128 and static generators remain the support boundary.
  Unsupported requests fail during planning without invoking an integrator.

### Plans, results, and gradients

Extend the existing plan representation with subsystem dimensions. Existing
qubit requests continue to write and read schema `flagquantum.lindblad_plan`
1.3 with unchanged content and identity. New non-qubit requests require a
versioned schema extension that records the exact ordered dimensions in both
request and decision and includes them in the existing identity calculation.
The reader must verify that these fields and the total dimension agree.

This proposal assigns 1.4 to mixed-dimension plans. Proposal 073 also proposes
1.4 but is unapproved and unimplemented; it must not be treated as an
implemented predecessor or combined into this change. Any subsequent schedule
implementation must rebase its proposed schema extension onto the accepted
mixed-dimension baseline rather than reuse that version for another schema.

Add a dimensions accessor to `LindbladPlan`. Its existing `n_qubits` accessor
continues to return a count for qubit spaces and raises `CapabilityError` for
a non-qubit space; it must never report a subsystem count as a qubit count.
The implementation must document and test that refusal rather than return a
rounded count or silently change the property's meaning.

`EvolutionResult` remains the result type. Populations have shape `(time, D)` or
`(time, batch, D)`; retained density matrices have shape `(time, D, D)` or
`(time, batch, D, D)`. The plan supplies the basis interpretation. No new result
type or density-matrix execution mode is introduced.

In-process plans retain differentiable tensors through the existing live request.
JSON stores numerical values, not an autograd graph. Restored plans with
`require_gradients=True` retain the current explicit refusal. The existing
adjoint route continues to return a cotangent for the normalized initial density
matrix, not a new promise of control gradients.

## Constrained-space review

Four Ising `sigma` anyons with fixed total charge `1` provide a concrete design
counterexample to treating every physical system as a tensor product. With
fusion rules `sigma x sigma = 1 + psi`, `sigma x psi = sigma`, and
`psi x psi = 1`, the sequential fusion paths are
`(1, sigma, 1)` and `(psi, sigma, 1)`. That sector has two basis states, not
four independent two-level subsystems.

It is numerically possible to evolve a two-by-two matrix in this basis, but
`ProductStateSpace((2,))` does not encode its total charge, fusion tree, allowed
paths, or basis transformations. The contract fake must refuse a constrained
fusion-space descriptor rather than silently flattening it into a product
space. This keeps the product-space name honest and leaves an explicit
extension boundary for M3. No production anyon type, braid opcode, or QEC
decoder is built in this slice.

## Staged acceptance

1. **Contract review.** Approve names, dimensions resolution, qubit refusals,
   basis order, plan evolution/backward reading, and the constrained-space
   boundary. Land the versioned candidate, fake, and conformance tests before
   dependent implementation. Public snapshots remain untouched.
2. **Core implementation.** Replace the contract fake behind the same internal
   consumer boundary. Prove `(3,)`, `(2, 3)`, `(3, 2)`, `(1, 3)`, and the
   `(4,)` versus `(2, 2)` distinction, including invalid inputs and basis
   indexing. Retain exact CircuitIR 1.0 round trips and existing qubit semantics.
3. **Static dynamics vertical slice.** Reuse the existing request normalizers,
   Liouvillian/SuperOperator, all three integrators, result type, and sealed plan.
   Verify qutrit Rabi oscillation, qutrit decay, and qubit-oscillator exchange
   using existing boson matrices and independent analytic references. Test
   batch axes, trace drift, precision, method convergence, explicit failure,
   and serialization tampering. No new solver is accepted.
4. **Differentiation and reuse.** Compare live coupling gradients with analytic
   derivatives and finite differences. Verify initial-density cotangents and
   JSON gradient refusal. Run the same solver and result path for both physical
   models. Record truncation error separately from solver error; a finite Fock
   example does not establish a photonic backend.
5. **Repository gates.** Run the existing architecture, dependency, public API,
   internal IR, documentation, capability/parity, ownership, smoke/unit, and local
   integration checks. Update authoritative support descriptions only when the
   corresponding production workflow has evidence; do not promote maturity from
   a contract fake or preparatory numerical experiment.

Time dependence follows proposal 073 after the static path is accepted. Domain
packages, acceleration, hardware execution, and distributed dynamics remain
outside this first slice.
