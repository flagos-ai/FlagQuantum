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
has a null n_wires field and no product dimensions. plan.n_qubits and plan.dimensions
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

## Native moving-atom programs

The experimental RydbergProgram and its typed Core ProgramRecord are accepted by
these same plan/run/adjoint entries. Start with
`python -m examples.rydberg_native_dynamics`. Its one/two-actor CPU profile derives
ordered (3, ...) g/r/lost coordinates, C6/r**6 interactions and two absorbing
loss jumps per atom. Native programs own their collapse operators and dimensions;
n_qubits, basis_labels and non-None collapse overrides are refused. Larger valid
rosters fail before exponential allocation. Low-level numerical APIs retain
mathematical inputs only; circuit/deployment/provider routes refuse this profile.

The output grid starts at zero and includes every move/pulse boundary. Positions
are sampled at interval midpoints of the commanded linear path, an explicit
approximation requiring its own grid-convergence check. The same three integrators
execute the resulting pre-bound intervals. Amplitude, detuning and phase accept
Core parameters/expressions and the existing per-interval parameter_values maps.
Live bindings preserve autograd; geometry, C6, loss rates and times are fixed.
Use run for control gradients and adjoint_gradient for the initial-density
cotangent, including the existing batch axes. Existing option/device refusals apply.

Every native sealed plan uses version 1.7, even a static program. Its seven outer
fields are schema, version, request, decision, program, bindings and identity.
The actual Core source and precision-normalized bindings are sealed alongside the
existing numerical request/decision. Decode validates source, boundaries, product
dimensions, interval matrices and jumps, including rehashed inconsistent payloads.
The reconstruction allowance is exactly 16 real-dtype epsilons, distinct from the
physical error budgets. run(plan) executes captured matrices without recompiling
or rebinding source. JSON stores values without a live graph; gradient-required
restored plans refuse execution. Old 1.3/1.4/1.5/1.6 formats remain unchanged.

For a numerical change, edit Simulation's rydberg_model.py and run
`tests/unit/test_rydberg_transport_contract.py` plus
`tests/unit/test_native_rydberg_dynamics.py`. For source encoding or consistency,
edit _native_plan.py and run those tests plus the existing plan compatibility
consumers. Core owns geometry and scalar codecs; do not reproduce them here.
Physical survival, unconditional RR and the detector no-flag effect are distinct
ordinary matrix outputs. This profile provides no sampled detector records,
feedback, heating, quantum motion, cloud submission or full M3 acceptance.
