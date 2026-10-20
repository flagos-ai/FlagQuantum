# Continuous-Time Lindblad Evolution

This domain translates public requests, interprets ExecutionOptions, and owns
sealed LindbladPlan serialization. Simulation owns numeric input validation,
Liouvillian, all three integrators, the shared trajectory and initial-density
adjoint. Core owns parameters and mathematical basis semantics.

Public entries are plan, run, LindbladPlan, CollapseOperator, EvolutionResult,
amplitude_damping and adjoint_gradient. No distributed setup is needed for this
CPU path. A sealed plan is closed to semantic overrides; rebuild live plans after
optimizer updates. JSON retains values, not an autograd graph.

## Declare the state coordinates

Use dimensions=(2, 3) for an ordered finite tensor product. Subsystem 0 is most
significant. Qubit plans retain schema 1.3, non-qubit static product plans use 1.4,
and piecewise product controls use 1.5.

Use basis_labels for an ordered orthonormal basis without a tensor factorization:

```python
import torch
import flagquantum.lindblad as fql

labels = ((0, 1), (1, 0))
exchange = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
p = fql.plan(
    0.7 * exchange, [1., 0.], [0., 0.5], basis_labels=labels,
    method="krylov-exponential",
)
r = fql.run(p)
assert p.basis_labels is not None
print(f"Transfer: {r.populations[-1, p.basis_labels.index((1, 0))].item():.8f}")
```

Expected: `Transfer: 0.11757891`.

Labelled plans use schema 1.6 for static and piecewise requests. Their decision
has n_wires=None and no product dimensions. plan.n_qubits and plan.dimensions
raise CapabilityError with guidance to inspect basis_labels. Labels cannot be
combined with dimensions or n_qubits. A two-state labelled basis remains
non-product; label arity never determines a wire count.

The caller owns orthonormality, physical constraints, charge conventions and any
projection. Check the full generator preserves a proposed reduced space, including
collapse effects. Retain explicit loss states when a jump leaves a sector.
Literal labels do not define an anyon model, native braid or fusion measurement.

## Reuse the numerical and training path

- Select runge-kutta, crank-nicolson or krylov-exponential. Local Krylov targets
  are not global population or autograd-error bounds.
- Full numeric state/operator matrices use product or explicit bases. Pauli
  descriptors, bitstrings, named qubit collapse operators and local qubit
  selectors require a qubit register. Direct BosonOperator conversion requires
  explicit product dimensions; a labelled model supplies validated matrices.
- parameter_values supplies one complete Core scalar binding per output interval
  on fixed full-space terms. Initial-state batches share that control sequence.
  Callable/interpolated controls and time-dependent collapse rates are unsupported.
- Populations have axes (time, D) or (time, batch, D). A square (D, D) input is one
  density matrix; use rank-three density matrices to disambiguate square batches.
- Live tensors retain PyTorch graphs. Use run for control gradients and
  adjoint_gradient for a cotangent of the normalized initial density. Krylov
  autograd conditioning limits also apply to labelled models. The adjoint's
  discrete-map reference and autograd parity are tested separately.

Run `python -m examples.constrained_basis_reference` for a photon-sector gradient
and a model-owned Ising convention with retained plan labels. The example's final
line is `Plan retains declared basis labels: True`. This is matrix-space
execution, not a hardware or native anyon-program claim.

For a change, run test_explicit_basis_contract.py (fake/Core and encoded/labelled
replacement), test_lindblad_explicit_basis.py (public refusals, plan replay,
physical loss, gradients and adjoints), and the existing Lindblad suites in
`tests/unit/`. Broaden with the repository testing tiers. The authoritative
support boundary is continuous_time_lindblad in capability-maturity.toml.
