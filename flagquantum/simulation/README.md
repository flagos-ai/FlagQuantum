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

## Continuous-time open-system evolution

`flagquantum.simulation.evolve_density_matrix` integrates the Lindblad master
equation on an explicit, strictly increasing output grid. Collapse entries pair
an operator with a physical rate; the named `amplitude_damping` operator is
`sqrt(rate) * |0><1|`, so its rate is never confused with the dimensionless
probability accepted by the instruction-level noise channel.

The deterministic CPU implementation uses two fourth-order Runge--Kutta steps
per output interval. Results report the actual method, order, maximum internal
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
