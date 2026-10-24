# Simulation

Numerical algorithms for quantum state evolution, observables, noise, and
differentiation. The design goal is a shared numerical foundation whose
precision and approximation behavior remain explicit across execution paths.

Simulation consumes Core semantics. It does not choose devices, schedule ranks,
submit jobs, authorize fallbacks, or construct public execution evidence.
Runtime owns those decisions and calls the appropriate numerical primitives.

## Choose the numerical owner

| Work | Entry point |
| --- | --- |
| Dense gates, sampling, and adjoints | [statevector/](statevector/README.md) |
| MPS updates, factorization, and truncation | [mps/](mps/README.md) |
| Contraction, slicing, and pullbacks | [tensor_network/](tensor_network/README.md) |
| Exact noisy evolution | [density_matrix.py](density_matrix.py) |
| Continuous-time Lindblad evolution | [lindblad.py](lindblad.py) |
| Clifford sampling beyond an amplitude store | [stabilizer/](stabilizer/README.md) |
| Reusable arithmetic and precision | [numerics/](numerics/README.md) |
| Optional JAX kernels | [jax/](jax/README.md) |

For a typical local change, run the matching suite from the repository root:

```bash
python -m pytest tests/test_native_circuit.py -q
python -m pytest tests/test_mps.py -q
python -m pytest tests/test_tensor_network.py -q
python -m pytest tests/test_noise.py -q
python -m pytest tests/unit/test_density_matrix_execution_capability.py -q
python -m pytest tests/unit/test_dynamic_density_matrix.py -q
python -m pytest tests/team/simulation/test_stabilizer_engine.py -q
```

Choose the suite for the affected representation, then broaden by the
[testing policy](../../docs/development/TESTING.md). Verify values and gradients
against independent references; truncation and emulated precision need their
own error envelope. Local numerical changes must not introduce a dependency on
Runtime orchestration.

## One instruction, one dense operator

Every torch engine in this directory reaches
[`gate_matrix.py`](gate_matrix.py) for the dense operator of one instruction, so
that function is where an instruction with no dense operator is refused. It
refuses by category and names the repair, because the alternative is a dictionary
key: `barrier`, `measure`, `reset`, a misspelled gate, and a channel whose
declared parameters were never turned into Kraus operators all used to escape as
`KeyError: 'barrier'`, and a carried matrix of the wrong size escaped from
`reshape` naming no instruction at all.

| Instruction | Refusal |
| --- | --- |
| A unitary opcode the operator registry declares | The operator is returned |
| An instruction carrying its own dense operator | That operator is returned, whatever the opcode is called |
| A declared channel, with or without its Kraus operators | `CapabilityError` naming the operators it carries, or the declared parameters that still have to become operators |
| An opcode the registry does not declare | `CapabilityError` saying a misspelled gate and a directive both arrive here |
| A carried value that is not a dense operator of the instruction's size | `ValidationError`, or `TypeError` when it is not a matrix type at all |

Refusal is not new policy: [engineering decision principle 9](../../AGENTS.md)
requires unsupported capabilities to fail at the earliest knowable stage, and
`contracts/errors-module-boundary-v1-candidate.json` already declares these
categories. A directive is still not executed by any engine here — engine
selection, and the `barrier` directive in particular, is Runtime's decision, not
this directory's. What this directory owns is saying so clearly instead of
raising a key error.

## Exact density-matrix execution

`density_matrix_from_ir` executes a circuit as a dense `2**n` by `2**n` square
matrix, so a channel acts on the state exactly instead of being sampled. It is
the small-system correctness oracle: the representation is exact and the cost is
`4**n`, measured here up to 13 wires, where one `complex128` state is about
1.07 GB. Nothing above that is claimed, and readout has a lower ceiling than
storage — probabilities over more than 8 wires are refused until
`max_marginal_wires` is raised explicitly.

Wire 0 is the highest-order index, so `expand_operator` of a one-wire gate on
wire 0 of two wires is that gate tensored with the identity on the right, not the
identity on the left. `mode="density_matrix"` selects the route and `mode="auto"`
selects it exactly when the program carries a channel instruction;
`mode="statevector"` and `mode="mps"` refuse such a program by name instead of
approximating it. `Circuit.density_matrix()` reads a program with neither a
channel nor a dynamic instruction off its statevector as an outer product and
routes any other program through `density_matrix_from_ir`, because an outer
product is rank one and cannot be the mixed state a channel or an unread
measurement produces.

The engine is device-generic `torch`: it takes the operands' own device and
precision, and asks the kernel catalog once per product whether a fused batched
complex matmul is declared for that device and precision. On CPU, and at every
precision the catalog does not declare, the reference `torch.bmm` route runs and
the route taken is counted rather than assumed. The density matrix is not
sharded; no rank partitions it. `memory_limit_bytes` is recorded into the noisy
plan and is not enforced on this route, so a dense state larger than a declared
limit still executes.

A dynamic program — a mid-circuit measurement, a reset, or a classically
conditioned instruction — executes here too, and executes exactly. The route
keeps one dense density matrix per classical record. A measurement whose
recorded bit no instruction reads is the non-selective channel
`P0 . P0 + P1 . P1`, so it dephases in place and costs no second branch; a
measurement somebody reads splits the branch in two, one per outcome, and the
outcome is written into that branch's record. A condition is a gate on exactly
the branches its clause selects, and a `reset` is the trace-preserving channel
`{P0, P1 X}`, which returns the wire to `|0>` without inventing a record. The
returned matrix is the sum of the branches, which is what a reader who never sees
the classical register holds, and its trace is one. A branch whose trace is
exactly zero cannot occur and is dropped rather than carried as a zero.

This is the exact answer `fq.experimental.dynamic.run_dynamic(..., shots=...)` is
a sampled estimate of. It is reached through `DynamicCircuit.density_matrix()`;
`fq.run` and `DynamicCircuit.state()` still refuse a dynamic program by name,
because one state is not the mixture a dynamic program produces and substituting
one branch for the sum would be a silent answer.

The route carries an exact autograd graph. `d<Z(0)>/dtheta` for `RY(theta)` on
`|0>` is `-sin(theta)`: `complex128` reproduces it exactly and the default
`complex64` lands about `1e-8` away. `examples/density_matrix_execution.py` is
the ten-minute path through the gate and channel part of the above, and
`examples/dynamic_density_matrix.py` is the ten-minute path through the dynamic
part.

```bash
python -m examples.density_matrix_execution
python -m examples.dynamic_density_matrix
```

## Continuous-time open-system evolution

`flagquantum.simulation.evolve_density_matrix` integrates the Lindblad master
equation on an explicit, strictly increasing output grid. Collapse entries pair
an operator with a physical rate; the named `amplitude_damping` operator is
`sqrt(rate) * |0><1|`, so its rate is never confused with the dimensionless
probability accepted by the instruction-level noise channel.

The integrator is selectable with `method`, and the choice is validated rather
than silently replaced: `runge-kutta` (the default), `crank-nicolson` and
`krylov-exponential`. The two order-`p` schemes take two substeps per output
interval and report that order; the exponential scheme takes each interval whole
because the exponential of a time-independent generator is exact on the grid, so
it reports no order and `solve_tolerance` sets its accuracy instead. Results report the actual method, order, maximum internal
step, precision, basis populations, requested observables, maximum trace drift,
and whether populations stayed within the documented tolerance (`1e-9` for
complex128 and `2e-5` for complex64). Density matrices are returned only when
requested. `plan_density_matrix_evolution` validates the identical request and
reports dimensions and retained-trajectory memory without evolving it.

The issue #234 reference problem can be run directly:

```bash
python -m examples.lindblad_evolution
```

The minimal SDK call is:

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql

result = fql.run(
    0.5 * fq.X(0) + 0.1 * fq.Z(0),
    "1",
    torch.linspace(0.0, 8.0, 161),
    collapse_operators=[fql.amplitude_damping(rate=0.1, qubit=0)],
    outputs=fq.expectation(fq.Z(0), name="z"),
)

print(result.populations.shape)          # torch.Size([161, 2])
print(result.expectation("z")[-1])
print(result.maximum_trace_drift)
print(result.population_bounded)
```

For an inspectable or cross-process workflow, serialize the sealed plan and run
the restored request without semantic overrides:

```python
plan = fql.plan(
    0.5 * fq.X(0) + 0.1 * fq.Z(0),
    "1",
    torch.linspace(0.0, 8.0, 161),
    collapse_operators=[fql.amplitude_damping(rate=0.1, qubit=0)],
    outputs=fq.expectation(fq.Z(0), name="z"),
)
restored = fql.LindbladPlan.from_json(plan.to_json())
result = fql.run(restored)
assert result.plan is restored
```

Set `return_density_matrices=True` when the complete density trajectory is
needed. Otherwise it is omitted from the returned result.

### Finite mixed-dimension spaces

The same Lindblad path accepts explicit ordered subsystem dimensions. For
example, a three-level leakage model uses a three-component vector directly:

```python
import torch
import flagquantum.lindblad as fql

transition = torch.zeros((3, 3), dtype=torch.complex128)
transition[1, 2] = transition[2, 1] = 0.7
plan = fql.plan(
    transition, [0.0, 1.0, 0.0],
    torch.linspace(0.0, 0.5, 51, dtype=torch.float64),
    dimensions=(3,),
)
result = fql.run(plan)
print(plan.dimensions, result.populations.shape)  # (3,), torch.Size([51, 3])
```

Subsystem 0 is most significant: for `(2, 3)`, `|spin=1, occupation=0>` is
flat index 3. A four-level subsystem `(4,)` is distinct from two qubits `(2, 2)`;
the ordered dimensions enter the sealed plan identity. No padding occurs.
Non-qubit spaces require full numeric states and full operator matrices or
explicitly truncated boson operators as described below. Pauli
inputs, bitstrings, named qubit collapse operators, and `fq.expectation`'s
qubit-only observable forms are refused for these spaces. Dense observable
matrices can be supplied through the existing low-level
`evolve_density_matrix(..., n_qubits=None, dimensions=..., observables=...)`
interface. Supplying `n_qubits` together with `dimensions` is legal only when
every declared subsystem is a qubit and the counts agree.

Run both a three-level leakage model with a coupling gradient and a spin-mode
exchange model built from the existing boson matrices:

```bash
python -m examples.mixed_dimension_dynamics
```

All three integrators, initial-state batching, the existing `EvolutionResult`,
and initial-density adjoints are reused. Population axes are `(time, D)` or
`(time, batch, D)`; retained density axes append `(D, D)` instead. A square
`(D, D)` initial input still means one density matrix. The new non-qubit plans
use schema 1.4; qubit plans keep their 1.3 payload and identity, including when
their dimensions are explicitly supplied. `plan.n_qubits` raises
`CapabilityError` for a non-qubit space; inspect `plan.dimensions` instead.
Live plans preserve autograd; JSON contains values, so restored plans requesting
gradients retain the existing `gradients_not_available` refusal at execution.

This slice is static, CPU-only, and finite-dimensional. Fock truncation is an
explicit modeling approximation, distinct from integration error. The tests
separate time-grid refinement from truncation in a fixed excitation sector;
that sector does not establish convergence for an arbitrary oscillator state.
A dense matrix made with the boson algebra is not a photonic backend. Native
fusion models, continuously varying controls, native qudit circuit operations,
and GPU/distributed dynamics remain future work under
[proposal 078](../../docs/development/API_CHANGE_PROPOSAL_078_MIXED_DIMENSION_STATE_SPACES.md).

### Ordered explicit bases

The same plan/run and numerical core accept basis_labels instead of dimensions.
Labels name an ordered finite orthonormal matrix basis; they do not imply tensor
factors, even at dimensions two or four. The caller owns physical sectors,
embeddings and basis conventions. The internal ExplicitBasisSpace value lives
beside ProductStateSpace in Core and adds no program IR, registry or solver.

```python
import torch
import flagquantum.lindblad as fql

labels = ((0, 1), (1, 0))
exchange = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128)
p = fql.plan(0.7 * exchange, [1., 0.], [0., 0.5], basis_labels=labels,
             method="krylov-exponential")
r = fql.run(p)
assert p.basis_labels is not None
print(f"Transfer: {r.populations[-1, p.basis_labels.index((1, 0))].item():.8f}")
```

Expected: `Transfer: 0.11757891`. Run
`python -m examples.constrained_basis_reference` for a gradient and a declared
Ising convention through the same existing numerical path.

basis_labels is mutually exclusive with explicit dimensions or n_qubits.
Qubit-only inputs remain refused. Direct boson operators need product dimensions;
a labelled model explicitly materializes and validates its reduced matrices.
The existing low-level observables mapping accepts full numeric matrices;
fq.expectation remains qubit-only.

Labelled static and scheduled plans use 1.6, retain request labels and an exactly
revalidated decision copy, and report a null n_wires field without product dimensions.
plan.n_qubits and plan.dimensions raise CapabilityError for this path; inspect
plan.basis_labels and plan.dimension. Existing 1.3/1.4/1.5 requests, identities,
fields and numerical trajectories are preserved. JSON stores numerical values
and ordered labels; gradient-required restored plans refuse the missing graph.
EvolutionResult and its population/density/batch axes are unchanged. Retain the
plan JSON alongside a stored result dictionary to interpret its matrix axes.

The new tests independently compare a vacuum-plus-one-photon model with its
ambient two-mode space, retaining loss states. They also check all three
methods, label permutations, control gradients, batches, optimizer updates,
strict replay/refusals and initial-density adjoints. A sector projection must be
validated against the full generator, including dissipative terms, rather than
silently dropping leakage. No native fusion rules or braiding API is implied.

A default two-state dissipative example also reproduces the old Krylov graph
conditioning limit on both product and labelled inputs: complex128 autograd
versus a dense discrete-map reference differs by about 7.15e-9 (CN) or 3.82e-8
(exponential), while the initial-density adjoint matches that reference below
5e-16. Complex64 CN's graph difference is about 4.49e-5 on the same fixture.
These measurements are not portable accuracy guarantees. The test retains the
original independent-adjoint budgets, 1e-9 and 2e-5, and checks graph parity
against the existing product path separately. No numerical kernel or tolerance
was changed to certify a stronger autograd bound.

### Consuming existing boson operators

Pass a `BosonOperator` directly as the Hamiltonian or as the operator in an
existing `(operator, rate)` collapse pair. Explicit `dimensions=` is required,
even for two-level modes: mode `k` is subsystem `k`, and any untouched subsystem
is an identity factor. The Simulation boundary calls the existing
`BosonOperator.to_matrix(dict(enumerate(dimensions)))`; the algebra, normal
ordering, finite truncation semantics, and `DEFAULT_DENSE_MATRIX_BYTES` ceiling
are reused. Missing mode dimensions and non-Hermitian Hamiltonians are refused
before integration.

```python
import math
import torch
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate, create

hamiltonian = 0.7 * (create(0) * annihilate(1) + create(1) * annihilate(0))
initial = torch.zeros(9, dtype=torch.complex128)
initial[4] = 1  # |1, 1>, subsystem 0 most significant.
result = fql.run(
    hamiltonian, initial, [0.0, math.pi / (4 * 0.7)],
    dimensions=(3, 3), method="krylov-exponential",
)
print(result.populations[-1, [2, 6]])  # P(0, 2) = P(2, 0) = 0.5
```

Run `python -m examples.boson_interference` for the same two-mode interference
with independent loss on each mode, plan serialization, and unconditional output
probabilities. Three levels per mode contain the complete two-excitation sector;
two levels cannot represent bunching. Agreement at larger cutoffs checks this
sector only, not arbitrary driven oscillator convergence.

The low-level `evolve_density_matrix(..., observables={"n": number(0)},
dimensions=(3,))` input mapping also accepts boson operators. `fq.expectation`
continues to require its existing qubit observable type. The plan stores the
same normalized dense matrices as an explicitly materialized request, with the
same identity and existing schema 1.3/1.4. No new serializer or solver is used.
Initial-state batches, live initial-state autograd, and initial-density adjoints
reuse their existing paths. The algebra's coefficients remain numeric constants;
for a trainable coupling, multiply its materialized matrix by a live tensor as
in `examples/mixed_dimension_dynamics.py`. Direct symbolic/tensor coefficient
binding remains a separate M1 gate. This bounded CPU consumer is not a bosonic
or photonic backend, a Gaussian simulator, or a new circuit operation.

### Static operator coefficients

The existing Hamiltonian term list accepts static full-space operator entries:
`[{"operator": full_space_operator, "coefficient": bound_scalar}, ...]`.
Each operator is a full-space numeric matrix or existing `BosonOperator`;
qubit selectors and Pauli descriptors cannot be mixed into these entries. A
boson operator still requires explicit mode dimensions and the existing byte
ceiling. The total Hamiltonian, rather than each term separately, must be
Hermitian. Coefficients are finite numeric scalars or zero-dimensional PyTorch
tensors; live tensors must retain their graph through normalization and planning.

Bind a named coefficient with the existing `Parameter.bind(values)` or
`ParameterExpression.bind(values)` before passing the terms. No second binding
function, parameter registry, or symbolic algebra is introduced. Unbound names,
non-scalar coefficients, invalid matrices, and non-finite values fail during
planning. This is a static Hamiltonian; a coefficient vector is not a time grid.
The existing Pauli descriptor path remains matrix-free and keeps its numeric
coefficient behavior; this extension does not claim trainable Pauli-sum terms.

These entries reuse current dense normalization, sealed plans,
Liouvillian, integrators, and result/gradient paths. A bound term list and its
explicitly assembled dense Hamiltonian seal the same payload and identity.
Plan schemas stay 1.3 for qubits and 1.4 for non-qubit spaces. JSON restores
values rather than an autograd graph. No Stable Core export or signature changes,
and no widening of `BosonTerm.coefficient`, are required. `tests/unit/test_lindblad_operator_coefficients.py` covers independent gradients,
a real optimizer update, dense/plan equivalence, batching, complex conjugate
coefficient pairs, and unsupported inputs before integration.

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql
from flagquantum.observables.boson import annihilate, create

symbol = fq.Parameter("coupling")
coupling = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
exchange = create(0) * annihilate(1) + create(1) * annihilate(0)
terms = [{"operator": exchange, "coefficient": symbol.bind({"coupling": coupling})}]
result = fql.run(
    terms, [0., 0., 0., 1., 0., 0.], [0., 0.5],
    dimensions=(2, 3), method="krylov-exponential",
    options=fq.ExecutionOptions(require_gradients=True),
)
(derivative,) = torch.autograd.grad(result.populations[-1, 1], coupling)
print(f"Coupling derivative: {derivative.item():.8f}")  # 0.32210884
```

Run `python -m examples.train_boson_coupling` for an actual Adam optimization
using the same binding, plan/run, and autograd path. The algebra's canonical
coefficients stay numeric; trainable values belong to the existing Hamiltonian
request entries. Rebuild the bound terms and plan after each optimizer step.
A live plan keeps its current tensor graph, rather than acting as a reusable
optimizer closure.

### Piecewise-constant controls on the same dynamics path

Approved [proposal 073](../../docs/development/API_CHANGE_PROPOSAL_073_EVOLUTION_SCHEDULE.md)
adds `parameter_values=` to the existing plan, run and initial-density-adjoint
entry points. Supply one complete parameter mapping per interval of `times`.
Each operator stays fixed; Core binds its scalar coefficient on
`[times[k], times[k+1])`. Every initial-state batch member shares those controls.
All three integrators use the same trajectory loop and frozen interval generator.

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql

drive = fq.Parameter("drive")
omega = torch.tensor([1., 3., 7.], dtype=torch.float64, requires_grad=True)
transition = torch.zeros((3, 3), dtype=torch.complex128)
transition[1, 2] = transition[2, 1] = 1
plan = fql.plan(
    [{"operator": transition, "coefficient": .5 * drive}],
    [0., 1., 0.], [0., .1, .3, .35], dimensions=(3,),
    parameter_values=[{"drive": value} for value in omega.unbind()],
    method="krylov-exponential",
    options=fq.ExecutionOptions(require_gradients=True),
)
result = fql.run(plan)
(gradient,) = torch.autograd.grad(result.populations[-1, 2], omega)
print(f"Final level-2 probability: {result.populations[-1, 2].item():.8f}")
print([round(value, 8) for value in gradient.tolist()])
# Final level-2 probability: 0.25121448
# [0.04337116, 0.08674232, 0.02168558]
```

Run `python -m examples.train_piecewise_drive` for an actual optimizer update.
Rebind and rebuild after each update. Scheduled plans use version 1.5 and store
ordered numerical Hamiltonians; static qubit 1.3 and finite-product 1.4 plans
keep their contents and identities. JSON stores values rather than graphs, so
restored plans requesting gradients fail at execution as before.

The control grid is also the output grid and must have at least two times.
Missing, unused, duplicate-normalized or non-scalar bindings are refused before
integration, and every total Hamiltonian must be Hermitian. Bind conjugate pairs
explicitly for complex coefficients. The initial-density adjoint reverses the
matching generators and keeps no control graph; it does not return control
gradients. Use autograd through `run` for those derivatives.

Fixed dense/BosonOperator terms are supported. Callable/interpolated controls,
scheduled Pauli descriptors, time-dependent collapse rates, differentiable time
boundaries, moving atoms and hardware pulses are outside this contract. Fock
truncation and time-grid approximation errors are separate. Krylov's local
defect target is not a global state-error guarantee; the refinement test budgets
the projected-exponential numerical floor independently. Interval matrices cost
an additional `N*D*D*element_bytes`; `trajectory_bytes` counts density history only.

Exactly closed Krylov spaces need sensitivity in directions absent from their
forward basis. This includes a zero drive before a later nonzero pulse. GMRES
uses its own adjoint solve for the residual correction; the exponential uses
its own action on an augmented vector of size `n+m`, with a Torch VJP of the
existing linear action. Forward value bits are preserved. These closure
corrections support first-order reverse-mode differentiation; requesting higher
orders raises rather than returning an uncertified Hessian. Their backward
solve can also fail if the same Krylov budget cannot reach its tolerance.
The independently measured non-closed solve/gradient conditioning limits above
remain applicable. `tests/unit/test_matrix_free_closed_subspace_gradients.py`
compares the corrections to dense Torch derivatives without changing the
existing solver regressions.


### Differentiating the initial state without retaining the sweep

`fql.adjoint_gradient` returns the gradient of a scalar cost with respect to the
initial density matrix, propagated by the adjoint equation on the same grid with
the same scheme. Its reason to exist is retention rather than speed: the forward
sweep runs under `torch.no_grad()`, so the only thing the reverse pass holds is
the trajectory the forward pass already returned, and
`tests/unit/test_lindblad_adjoint_gradient.py` measures that as 0 retained
tensors against 180 for the recorded forward graph of the same fixture, at the
same cost and the same step. `torch.autograd` through `fql.run` remains the route
to use when the gradient is with respect to the Hamiltonian or a gate parameter.

```python
import torch
import flagquantum as fq
import flagquantum.lindblad as fql

gradient = fql.adjoint_gradient(
    0.5 * fq.X(0) + 0.1 * fq.Z(0),
    [[1.0, 0.0], [0.0, 0.0]],
    torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
    lambda trajectory: torch.real(trajectory[-1, 1, 1]),
    collapse_operators=[fql.amplitude_damping(rate=0.1, qubit=0)],
)
print(gradient.shape)                    # torch.Size([2, 2])
```

`cost` reads the whole trajectory -- one density matrix per time, stacked along a
leading axis -- and must return a scalar. `options.require_gradients` is refused,
because the route deliberately produces no differentiable trajectory to honour
it; use `run` when that is what is wanted.

The cotangent is the exact transpose of the discretized scheme the caller named,
which is verified against that scheme's dense transpose and by the pairing
identity between the generator and its Hilbert--Schmidt adjoint. One property of
the forward path is disclosed here rather than repaired: at the base point where
the two routes differ most, the adjoint route agrees with the dense transpose of
the scheme it was given to roundoff, while autograd's backward pass through the
matrix-free solve is `3.5e-06` (Crank--Nicolson) and `1.7e-05`
(krylov-exponential) from a five-point stencil of that same scheme. That is a
measured statement about the current matrix-free backward pass at one base
point, not a general accuracy claim about either route: the explicit scheme,
which takes no solve, is at roundoff on both, and at a second base point all
three agree to `2.5e-14`.

## Clifford sampling beyond an amplitude store

`flagquantum.simulation.stabilizer.sample_stabilizer` samples
computational-basis outcomes from a Clifford circuit through Pauli stabilizer
tracking, whose storage grows with the qubit count squared instead of
exponentially. A thousand-qubit GHZ chain is a normal input for it and an
unrepresentable one for every engine in this directory.

```bash
pip install 'flagquantum[stim]'
python -m examples.stabilizer_sampling
```

```python
import flagquantum as fq
from flagquantum.simulation.stabilizer import sample_stabilizer

samples = sample_stabilizer(fq.Circuit(2).h(0).cx(0, 1), shots=1000, seed=7)
print(samples.shape)     # torch.Size([1000, 2])
print(samples.dtype)     # torch.int64
```

The accepted gates are exactly the thirteen Clifford opcodes. A parameterized
rotation, `t`, `ccx`, `cswap`, or a noise channel is refused with
`CapabilityError` naming the gate set rather than being approximated, and a
missing engine raises `StabilizerDependencyError` naming the extra instead of
falling back to an amplitude path. Sampling is not differentiable and this
engine is not an execution route: no planner or executor selects it, so it
carries no plan, result, or scalability evidence. Read
[stabilizer/README.md](stabilizer/README.md) before changing it.

[Detailed source map](IMPLEMENTATION.md) locates shared gate primitives,
rank-local kernels, specialized precision paths, and numerical migration rules.

## Native Rydberg model assembly

rydberg_model.py consumes verified Core physical records and assembles bounded
one/two-actor CPU interval Hamiltonians and absorbing loss operators. It reuses
Core's exact path geometry and eight-ULP boundary semantics, scalar binding and
the existing request mapping validator. The high-level Lindblad facade consumes
these matrices through the existing private pre-bound interval path. It owns the
native-only 1.7 source capture; Simulation owns no second plan schema or binder.
The initial-density adjoint delegates to the same forward/reverse integrator loop
under no_grad; run retains live control graphs. Low-level mathematical entries
retain their existing input contract.

Run `python -m examples.rydberg_native_dynamics`; expected final survival is
0.81873075, unconditional RR is 0.40936538 and exact numeric replay is True.
The imposed linear trajectory uses interval midpoints: convergence of this
approximation is separate from integration accuracy. The unchanged physical
consumers in test_rydberg_transport_contract.py now use native fixtures;
test_native_rydberg_dynamics.py covers source-aware replay, controls and refusals.
Keep both sets when changing assembly. No quantum motion, heating, sampled events,
feedback, GPU/distributed native dynamics or hardware control is implemented.

## Fixed Ising braid and unread-charge actions

`ising_braid.py` consumes Core's approved four-sigma, total-vacuum source. Its
internal `evolve_ising_braid` entry applies each ordered F/R generator through
the existing `SuperOperator.left_right_multiply(...).apply(...)` algebra and
returns one CPU complex128 density matrix in the declared fusion coordinates.
It introduces no numerical time, Hamiltonian, integrator, backend or result
wrapper. Clockwise exchanges use the adjoints with their full complex phases.
The approved `ising.dephase_pair` operation uses the existing
`SuperOperator.from_kraus` with the two current-pair charge projectors. It sums
both branches without selection or renormalization, retaining zero-weight
branches as zero. Ordered continuation can therefore evolve a mixed density.

`tests/unit/test_ising_braid_actions.py` uses independent closed-form complex
generators and vector evolution to check both braid relations, distant
commutation, inverses and complete density matrices, all at atol=1e-12, rtol=0.
Global unitary phase and phase-invariant density action have separate checks.
The source is validated by Core before numerical work; other profiles are
refused. Braid-only native execution is already available through Runtime.
This change does not activate `fq.plan`/`fq.run` for source 1.2 or change
executable schemas. Runtime admission and replacement of the four proposal 089
baseline factories remain separate gates. `tests/unit/test_ising_unread_actions.py`
checks independent full-density and weighted conditional continuations, pure/
mixed echoes, trace, coherence, positivity and unchanged RNG at the same budget.
