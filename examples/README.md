# FlagQuantum Examples

FlagQuantum examples are short, runnable scripts that demonstrate a concrete
workflow. They complement tutorials: tutorials explain concepts, while examples
provide reusable code templates.

The [capability catalog](../docs/generated/CAPABILITIES.md) is the complete
task-oriented entry point. It identifies the canonical quick example, maturity,
hardware, gradient support, and distribution semantics for every product
capability.

## Curated Paths

Start with the end-to-end quantum-classical hybrid AI example:

```bash
python examples/quick_start.py --steps 40
```

It combines a `torch.nn.Linear` encoder with an `fq.Module` quantum layer in
one ordinary PyTorch optimizer loop. The model learns a one-dimensional
analytical target and reports both classical and quantum training results.

| Path | Purpose | Start here |
| --- | --- | --- |
| `tutorials/` | Beginner notebooks for circuit basics, measurement, gradients, and QML | `tutorials/README.md` |
| `quick_start.py` | Minimal quantum-classical hybrid AI training | Run this first |
| `single_machine_quantum_ai/` | Official CPU/one-GPU quantum AI examples | `single_machine_quantum_ai/README.md` |
| `distributed_statevector_topologies/` | Sharded statevector topology and ownership examples | `distributed_statevector_topologies/README.md` |
| `distributed_mps/` | Rank-owned variable-bond MPS capacity examples | `distributed_mps/README.md` |
| `mps_hamiltonian_identification/` | Experimental 512–1024 qubit MPS system-identification research path | `mps_hamiltonian_identification/README.md` |
| `train_parameterized_circuit_then_deploy.py` | Train a parameterized circuit and package it for deployment | Run as a deployment bridge example |
| `hybrid_jax_torch_training.py` | Minimal PyTorch training interface with a JAX quantum kernel | Use for integration experiments |
| `quantum_transformer.py` | Larger application-style demo | Treat as a demo candidate, not a minimal example |

## Runtime Planning Example

Use `Circuit.runtime_plan(...)` or `fq.plan_runtime_selection(...)` when an
example needs to explain why it chose statevector, MPS, tensor network, JAX, or
distributed execution:

```python
import flagquantum as fq

circuit = fq.Circuit(4)
circuit.h(0).cx(0, 1).rzz(1, 2, theta=0.2)

plan = circuit.runtime_plan(prefer_jax=True, require_gradients=True)
print(plan.summary()["recommended_mode"])
```

Examples must print distribution semantics before making performance or
scalability statements.

## Recommended First Runs

```bash
python examples/single_machine_quantum_ai/00_local_fast_path_check.py
python examples/single_machine_quantum_ai/01_vqe_statevector.py --steps 2 --n-qubits 3
python examples/single_machine_quantum_ai/02_quantum_classifier.py --steps 2
python examples/single_machine_quantum_ai/03_mps_training.py --steps 2 --n-qubits 4 --max-bond 8
python examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py --steps 1 --bench-iters 1
python examples/single_machine_quantum_ai/05_mps_1000q_dimer_training.py --steps 2 --n-qubits 20
```

The curated single-machine examples intentionally avoid distributed backend
initialization. They are local fast-path examples and do not make distributed
scalability claims.

## Adding A New Example

Use a new example when the user should be able to copy the file, change a few
arguments, and run a task end to end.

Each curated example should include:

- A module docstring that states the task and runtime mode.
- A corresponding capability entry with maturity and a support boundary.
- `argparse` options for problem size, steps, device, and optional benchmarks.
- A short smoke command suitable for tests or documentation.
- Clear output keys for final metrics and backend/runtime summary.
- Honest wording about local, replicated, sliced, or sharded execution.

Prefer putting shared helpers in the nearest local `common.py` instead of adding
new framework abstractions from an example.
