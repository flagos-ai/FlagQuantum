# FlagQuantum Hybrid Runtime Architecture

FlagQuantum's long-term runtime philosophy is:

- PyTorch is the primary training interface.
- JAX is an optional quantum-kernel accelerator.
- DLPack bridges tensor ownership without CPU round-trips.
- `torch.autograd.Function` owns cross-framework gradient boundaries.
- `torchrun`, DDP, and FSDP own distributed AI training.
- FlagQuantum IR owns circuit lowering across statevector, MPS, tensor-network, and cloud deployment.
- Trained parameterized circuits must remain deployable to quantum cloud and hardware targets.

## Runtime Layers

### PyTorch Training Surface

User code should remain ordinary PyTorch:

```python
import flagquantum as fq
import torch

def circuit_builder(parameters):
    return (
        fq.Circuit(n_qubits=2)
        .ry(0, parameters[0])
        .cx(0, 1)
        .ry(1, parameters[1])
    )

layer = fq.Module(
    circuit_builder,
    n_parameters=2,
    policy=fq.RuntimePolicy(
        execution_options=fq.ExecutionOptions(
            backend="jax",
            allow_backend_fallback=True,
        ),
        observable_wires=(1,),
    ),
)
optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)

optimizer.zero_grad()
loss = layer().mean()
loss.backward()
optimizer.step()
```

The user-facing contract is PyTorch tensors, PyTorch modules, PyTorch optimizers,
and PyTorch distributed launchers.

### JAX Quantum Kernel

JAX kernels are an acceleration backend, not a second user-facing framework.
`flagquantum.runtime.backends.jax.compile_quantum_kernel(..., backend="jax", interface="torch")` lowers a
FlagQuantum circuit builder to a JAX value-and-gradient kernel, then exposes it
as a PyTorch differentiable callable.

Current kernel support includes:

- statevector execution
- batched parameters via JAX `vmap`
- JAX `jit`
- common training gates: `rx`, `ry`, `rz`, `phase`, `u1`, `u2`, `u3`, `h`, `x`,
  `y`, `z`, `s`, `sdg`, `t`, `tdg`, `sx`, `sxdg`, `cx`, `cy`, `cz`, `swap`,
  `crx`, `cry`, `crz`, `cphase`, `rxx`, `ryy`, `rzz`
- `z`, `z_sum`, and Hamiltonian observables

### Tensor Bridge

The bridge uses DLPack:

```text
torch.Tensor -> DLPack -> JAX array -> JAX value_and_grad
JAX value/grad -> DLPack -> torch.Tensor
```

No `.cpu().numpy()` transfer should appear on the hot path.

### Gradient Boundary

PyTorch does not trace through JAX directly. FlagQuantum owns the boundary with
`torch.autograd.Function`:

```text
forward:
    torch params -> JAX params
    JAX value_and_grad(params)
    save quantum gradient for backward

backward:
    return upstream_gradient * saved_quantum_gradient
```

This makes the quantum kernel look like a normal PyTorch operation to optimizers,
DDP, FSDP, and larger neural networks.

### Distributed Training

The first supported distributed design is rank-local JAX kernels:

```text
torchrun launches ranks
each rank owns a local PyTorch process and local JAX kernel
quantum gradients return to PyTorch tensors
PyTorch distributed all-reduce aggregates gradients
```

This avoids competing JAX collectives and PyTorch NCCL collectives in the same
first-stage design. Later stages can add JAX sharding for large quantum kernels
behind the same FlagQuantum interface.

### FlagQuantum IR and Deployment

The same circuit builder should support:

- PyTorch-native simulation
- JAX accelerated quantum kernels
- MPS/TN/statevector lowering
- distributed execution
- export to quantum cloud/hardware deployment packages

The user should continue to write:

```python
import flagquantum as fq
```

Backend selection is an execution concern, not a different product identity.

## Near-Term Roadmap

1. Extend the JAX bridge from statevector to MPS and tensor-network kernels.
2. Add rank-local JAX quantum-kernel tests under `torchrun`.
3. Add benchmark reporting for PyTorch MPS, FlagQuantum JAX bridge, and external JAX baselines.
4. Extend cloud deployment export from trained `fq.Module` parameters.
5. Add production examples for classical PyTorch models producing batched quantum parameters.

## Verification Commands

Local hybrid tests:

```bash
python -m pytest tests/test_hybrid_jax.py -q
```

Rank-local JAX kernel under PyTorch distributed:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m pytest tests/distributed/test_hybrid_jax_runtime.py -q
```

On Windows PyTorch builds without libuv support, local `torchrun` may fail before
tests start with a C10d store error. The test is intended for the Linux/GPU
cluster path used for distributed validation.

Backend precision/speed benchmark:

```bash
python benchmarks/backend_compare.py \
  --n-wires 8 --layers 2 --observable z_sum \
  --iters 50 --warmup 10

python benchmarks/backend_compare.py \
  --n-wires 8 --layers 2 --batch-size 16 --observable ising \
  --iters 50 --warmup 10
```

The benchmark reports:

- FlagQuantum PyTorch statevector loss/gradient time
- FlagQuantum JAX quantum-kernel loss/gradient time
- loss absolute error
- gradient max absolute error
- JAX-over-PyTorch speedup

GPU-cluster backend gradient benchmark:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false

torchrun --standalone --nproc_per_node=8 \
  benchmarks/distributed_backend_compare.py \
  --device cuda --dist-backend nccl \
  --n-wires 8 --layers 2 --batch-size 16 --observable ising \
  --iters 100 --warmup 20 \
  --json-output gpu_backend_compare_8q_b16_ising.json
```

This benchmark runs one PyTorch/JAX process per GPU rank, compares loss and
parameter gradients on every rank, and reports rank-aggregated timing:

- PyTorch native statevector/autograd mean/min/max rank time
- JAX quantum-kernel mean/min/max rank time
- maximum loss absolute error across ranks
- maximum gradient absolute error across ranks
- JAX-over-PyTorch speedup by mean rank time and slowest-rank time

For multi-node launches, use the normal `torchrun` rendezvous arguments for the
cluster scheduler and keep the benchmark arguments unchanged.

PennyLane comparison benchmarks:

```bash
python benchmarks/pennylane_backend_compare.py \
  --device cpu --dist-backend none \
  --pennylane-device default.qubit --pennylane-interface torch \
  --pennylane-diff-method backprop \
  --n-wires 8 --layers 2 --batch-size 16 --observable ising \
  --iters 50 --warmup 10 \
  --json-output pennylane_default_qubit_cpu.json
```

GPU-cluster comparison with PennyLane Lightning:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false

torchrun --standalone --nproc_per_node=8 \
  benchmarks/pennylane_backend_compare.py \
  --device cuda --dist-backend nccl \
  --pennylane-device lightning.qubit --pennylane-interface torch \
  --pennylane-diff-method adjoint \
  --n-wires 8 --layers 2 --batch-size 16 --observable ising \
  --iters 100 --warmup 20 \
  --json-output pennylane_lightning_qubit_gpu_cluster.json
```

If the cluster has PennyLane's GPU Lightning plugin installed, the same script
also supports `--pennylane-device lightning.gpu`. The output reports whether the
requested PennyLane device is unavailable, so missing optional plugins are
visible in the benchmark JSON instead of being hidden by the run harness.

The default PennyLane interface in this benchmark is `torch`, which is the fair
comparison against FlagQuantum's PyTorch training surface. A JAX-interface
reference run can still be produced explicitly:

```bash
python benchmarks/pennylane_backend_compare.py \
  --device cpu --dist-backend none \
  --pennylane-device default.qubit --pennylane-interface jax \
  --pennylane-diff-method backprop \
  --n-wires 8 --layers 2 --batch-size 16 --observable ising \
  --iters 50 --warmup 10 \
  --json-output pennylane_default_qubit_jax_reference_cpu.json
```
