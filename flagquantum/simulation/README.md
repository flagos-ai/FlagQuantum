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
python -m pytest tests/team/simulation/test_stabilizer_engine.py -q
```

Choose the suite for the affected representation, then broaden by the
[testing policy](../../docs/development/TESTING.md). Verify values and gradients
against independent references; truncation and emulated precision need their
own error envelope. Local numerical changes must not introduce a dependency on
Runtime orchestration.

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
approximating it. `Circuit.density_matrix()` reads a channel-free program off its
statevector as an outer product and routes a channel-carrying program through
`density_matrix_from_ir`, because an outer product is rank one and cannot be the
mixed state a channel produces.

The engine is device-generic `torch`: it takes the operands' own device and
precision, and asks the kernel catalog once per product whether a fused batched
complex matmul is declared for that device and precision. On CPU, and at every
precision the catalog does not declare, the reference `torch.bmm` route runs and
the route taken is counted rather than assumed. The density matrix is not
sharded; no rank partitions it. `memory_limit_bytes` is recorded into the noisy
plan and is not enforced on this route, so a dense state larger than a declared
limit still executes. The circuit IR carries no mid-circuit measurement, reset,
or classically conditioned instruction, so this route cannot express them;
mid-circuit measurement is reached through `flagquantum.dynamic.DynamicCircuit`
on a statevector trajectory instead.

The route carries an exact autograd graph. `d<Z(0)>/dtheta` for `RY(theta)` on
`|0>` is `-sin(theta)`: `complex128` reproduces it exactly and the default
`complex64` lands about `1e-8` away. `examples/density_matrix_execution.py` is
the ten-minute path through all of the above.

```bash
python -m examples.density_matrix_execution
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
tracking, whose storage grows with the wire count squared instead of
exponentially. A thousand-wire GHZ chain is a normal input for it and an
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
