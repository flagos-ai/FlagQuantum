# FlagQuantum Examples

Runnable, copy-ready workflows for building and training quantum AI programs
with the current FlagQuantum API.

All curated examples use:

- `import flagquantum as fq` as the public entry point;
- `fq.Circuit(n_qubits=...)` for circuit construction;
- PyTorch for parameters, gradients, and optimizers;
- explicit runtime and distribution semantics when making performance claims.

For exact support levels, consult the
[capability catalog](../docs/generated/CAPABILITIES.md).

## First CPU execution

Start with the complete local execution path:

```bash
python -m examples.cpu_statevector
```

The example builds a Bell-state circuit, creates an inspectable execution plan,
runs that exact plan on the PyTorch CPU statevector engine, and checks the
double-precision result against its analytical state. It disables backend
fallback so a successful run proves the reported CPU path was actually used.

To inspect target-independent compiler optimization separately:

```bash
python -m examples.compiler_optimize
```

This example removes redundant gates, verifies that optimization reaches a
fixed point, and compares the optimized program with the original numerical
result. It uses `compiler.optimize`; target-aware lowering and routing belong to
`compiler.compile`.

## Start in one minute

From an editable development installation:

```bash
python examples/quick_start.py --mode sv --steps 40
```

This trains a hybrid model with a `torch.nn.Linear` encoder and an `fq.Module`
quantum layer in one PyTorch optimizer loop. The example uses an analytical
target, so it reports correctness as well as training loss.

Switch the simulation representation without rewriting the model:

```bash
python examples/quick_start.py --mode mps --steps 40
python examples/quick_start.py --mode tn --steps 40
```

Statevector is the recommended first run. MPS and tensor-network support
boundaries are listed in the capability catalog.

## Choose a workflow

| Goal | Recommended entry | Scope |
| --- | --- | --- |
| Learn circuits, measurements, gradients, and QML | [Tutorials](tutorials/README.md) | Guided notebooks |
| Verify the local CPU or one-GPU path | [Single-machine quantum AI](single_machine_quantum_ai/README.md) | Supported local workflows |
| Train a local statevector VQE | [`01_vqe_statevector.py`](single_machine_quantum_ai/01_vqe_statevector.py) | Exact differentiable simulation |
| Train with MPS | [`03_mps_training.py`](single_machine_quantum_ai/03_mps_training.py) | Low-entanglement systems |
| Use a JAX kernel through PyTorch | [`04_jax_kernel_torch_layer.py`](single_machine_quantum_ai/04_jax_kernel_torch_layer.py) | Optional accelerator path |
| Inspect sharded statevector ownership | [Distributed statevector](distributed_statevector_topologies/README.md) | One logical statevector across ranks |
| Inspect rank-owned MPS execution | [Distributed MPS](distributed_mps/README.md) | Development evidence |
| Train and package a circuit | [`train_parameterized_circuit_then_deploy.py`](train_parameterized_circuit_then_deploy.py) | Deployment bridge |
| Build an extension | [`extensions/reference_extensions.py`](extensions/reference_extensions.py) | Experimental API |

Larger application and research examples are intentionally not presented as
minimal getting-started paths.

## Plan before execution

Use the runtime planner when you need to inspect representation choice,
gradient support, or blockers before running:

```python
import flagquantum as fq

circuit = (
    fq.Circuit(n_qubits=4)
    .h(0)
    .cx(0, 1)
    .rzz(1, 2, theta=0.2)
)

plan = circuit.runtime_plan(prefer_jax=True, require_gradients=True)
print(plan.summary())
```

A plan is an explanation of intended execution, not benchmark evidence.
Performance and scalability statements must use runtime-generated records and
report their `distribution_semantics`.

## Recommended smoke runs

These small commands are suitable for checking a development environment:

```bash
python examples/single_machine_quantum_ai/00_local_fast_path_check.py
python examples/single_machine_quantum_ai/01_vqe_statevector.py \
  --backend torch --steps 2 --n-qubits 3
python examples/single_machine_quantum_ai/02_quantum_classifier.py --steps 2
python examples/single_machine_quantum_ai/03_mps_training.py \
  --steps 2 --n-qubits 4 --max-bond 8
```

Optional JAX check:

```bash
python examples/single_machine_quantum_ai/04_jax_kernel_torch_layer.py \
  --steps 1 --bench-iters 1
```

The curated single-machine examples do not initialize distributed backends and
make no distributed scalability claim.

## Example quality contract

A curated example must:

1. state the task, runtime mode, and maturity boundary;
2. expose practical arguments for problem size, steps, and device;
3. run end to end from a documented installation;
4. report a correctness metric or an explicit reference value;
5. identify local, replicated, sliced, or sharded execution accurately;
6. use stable public API unless explicitly labeled experimental.

Shared helpers belong next to the examples that use them. Example-only
convenience code must not become a framework abstraction without a separate API
review.
