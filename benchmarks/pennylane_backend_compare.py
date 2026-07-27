"""Compare FlagQuantum JAX kernels with PennyLane simulator backends.

The benchmark keeps the training interface and circuit workload aligned:

* FlagQuantum uses the JAX quantum-kernel backend exposed through PyTorch.
* PennyLane defaults to the PyTorch interface for fair training-surface comparison.
* The same parameters, ansatz, observable, warmup, and iteration counts are used.

Examples
--------
CPU comparison with PennyLane default.qubit:

    python benchmarks/pennylane_backend_compare.py \
        --device cpu --dist-backend none \
        --pennylane-device default.qubit --pennylane-interface torch \
        --pennylane-diff-method backprop \
        --n-wires 8 --layers 2 --batch-size 16 --observable ising \
        --iters 50 --warmup 10 \
        --json-output pennylane_default_qubit_cpu.json

GPU-cluster comparison with PennyLane lightning.qubit:

    torchrun --standalone --nproc_per_node=8 benchmarks/pennylane_backend_compare.py \
        --device cuda --dist-backend nccl \
        --pennylane-device lightning.qubit --pennylane-interface torch \
        --pennylane-diff-method adjoint \
        --n-wires 8 --layers 2 --batch-size 16 --observable ising \
        --iters 100 --warmup 20 \
        --json-output pennylane_lightning_qubit_gpu_cluster.json

PennyLane's GPU-specific device is commonly named lightning.gpu. This script
also accepts --pennylane-device lightning.gpu when that plugin is installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
import torch.distributed as dist


REPRO_COMMANDS = """
CPU default.qubit vs FlagQuantum JAX statevector:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device default.qubit --pennylane-interface torch --pennylane-diff-method backprop --flagquantum-mode statevector --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_pennylane_default_vs_fq_jax_statevector.json

CPU default.qubit vs FlagQuantum JAX MPS:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device default.qubit --pennylane-interface torch --pennylane-diff-method backprop --flagquantum-mode mps --flagquantum-max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_pennylane_default_vs_fq_jax_mps.json

CPU default.qubit vs FlagQuantum JAX tensor network:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device default.qubit --pennylane-interface torch --pennylane-diff-method backprop --flagquantum-mode tensor_network --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 50 --warmup 10 --json-output cpu_pennylane_default_vs_fq_jax_tn.json

CPU default.tensor MPS vs FlagQuantum JAX MPS:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device default.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method mps --pennylane-max-bond-dim 32 --flagquantum-mode mps --flagquantum-max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 3 --warmup 1 --pennylane-early-stop-speedup 30000 --pennylane-run-timeout-seconds 600 --json-output cpu_pennylane_default_tensor_mps_vs_fq_jax_mps.json
 
CPU default.tensor TN vs FlagQuantum JAX tensor network:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device default.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method tn --flagquantum-mode tensor_network --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 3 --warmup 1 --pennylane-early-stop-speedup 30000 --pennylane-run-timeout-seconds 600 --json-output cpu_pennylane_default_tensor_tn_vs_fq_jax_tn.json

CPU default.tensor TN vs FlagQuantum JAX tensor network high precision:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device default.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method tn --flagquantum-mode tensor_network --flagquantum-jax-compute-dtype complex128 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 3 --warmup 1 --pennylane-early-stop-speedup 30000 --pennylane-run-timeout-seconds 600 --json-output cpu_pennylane_default_tensor_tn_vs_fq_jax_tn_fp64.json

GPU lightning.qubit vs FlagQuantum JAX MPS:
  torchrun --standalone --nproc_per_node=8 benchmarks/pennylane_backend_compare.py --device cuda --dist-backend nccl --pennylane-device lightning.qubit --pennylane-interface torch --pennylane-diff-method adjoint --flagquantum-mode mps --flagquantum-max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --json-output gpu_pennylane_lightning_qubit_vs_fq_jax_mps.json

GPU lightning.qubit vs FlagQuantum JAX MPS high precision:
  torchrun --standalone --nproc_per_node=8 benchmarks/pennylane_backend_compare.py --device cuda --dist-backend nccl --pennylane-device lightning.qubit --pennylane-interface torch --pennylane-diff-method adjoint --flagquantum-mode mps --flagquantum-max-bond 32 --flagquantum-jax-compute-dtype complex128 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --json-output gpu_pennylane_lightning_qubit_vs_fq_jax_mps_fp64.json

GPU lightning.tensor MPS vs FlagQuantum JAX MPS, when installed:
  torchrun --standalone --nproc_per_node=8 benchmarks/pennylane_backend_compare.py --device cuda --dist-backend nccl --pennylane-device lightning.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method mps --pennylane-max-bond-dim 32 --flagquantum-mode mps --flagquantum-max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 20 --warmup 5 --pennylane-early-stop-speedup 30000 --pennylane-run-timeout-seconds 600 --json-output gpu_pennylane_lightning_tensor_mps_vs_fq_jax_mps.json

GPU lightning.tensor MPS vs FlagQuantum JAX MPS high precision, when installed:
  torchrun --standalone --nproc_per_node=8 benchmarks/pennylane_backend_compare.py --device cuda --dist-backend nccl --pennylane-device lightning.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method mps --pennylane-max-bond-dim 32 --flagquantum-mode mps --flagquantum-max-bond 32 --flagquantum-jax-compute-dtype complex128 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 20 --warmup 5 --pennylane-early-stop-speedup 30000 --pennylane-run-timeout-seconds 600 --json-output gpu_pennylane_lightning_tensor_mps_vs_fq_jax_mps_fp64.json

GPU lightning.gpu vs FlagQuantum JAX statevector, when installed:
  torchrun --standalone --nproc_per_node=8 benchmarks/pennylane_backend_compare.py --device cuda --dist-backend nccl --pennylane-device lightning.gpu --pennylane-interface torch --pennylane-diff-method adjoint --flagquantum-mode statevector --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --json-output gpu_pennylane_lightning_gpu_vs_fq_jax_statevector.json

PennyLane lightning.tensor TN vs FlagQuantum JAX tensor network, when installed:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device lightning.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method tn --flagquantum-mode tensor_network --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 3 --warmup 1 --pennylane-early-stop-speedup 30000 --pennylane-run-timeout-seconds 600 --json-output cpu_pennylane_lightning_tensor_tn_vs_fq_jax_tn.json

GPU lightning.tensor TN vs FlagQuantum JAX tensor network, when installed:
  torchrun --standalone --nproc_per_node=8 benchmarks/pennylane_backend_compare.py --device cuda --dist-backend nccl --pennylane-device lightning.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method tn --flagquantum-mode tensor_network --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 20 --warmup 5 --pennylane-early-stop-speedup 30000 --pennylane-run-timeout-seconds 600 --json-output gpu_pennylane_lightning_tensor_tn_vs_fq_jax_tn.json

Tiny tensor-device smoke:
  python benchmarks/pennylane_backend_compare.py --device cpu --dist-backend none --pennylane-device default.tensor --pennylane-interface torch --pennylane-diff-method parameter-shift --pennylane-tensor-method mps --pennylane-max-bond-dim 8 --flagquantum-mode mps --flagquantum-max-bond 8 --n-wires 3 --layers 1 --batch-size 1 --observable z_sum --iters 1 --warmup 0 --json-output cpu_pennylane_default_tensor_smoke.json
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402


def _configure_torch_precision(mode: str) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(mode)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = mode != "highest"
        torch.backends.cudnn.allow_tf32 = mode != "highest"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def _resolve_device(requested: str, local_rank: int) -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false.")
        if ":" in requested:
            torch.cuda.set_device(torch.device(requested))
            return requested
        torch.cuda.set_device(int(local_rank))
        return f"cuda:{int(local_rank)}"
    return requested


def _resolve_dist_backend(requested: str, device: str) -> str | None:
    if requested == "auto":
        return "nccl" if device.startswith("cuda") else "gloo"
    if requested == "none":
        return None
    return requested


def _init_distributed(args: argparse.Namespace) -> tuple[Any, int, int, int, str, str | None]:
    env_rank = _env_int("RANK", 0)
    env_world_size = _env_int("WORLD_SIZE", 1)
    env_local_rank = _env_int("LOCAL_RANK", env_rank)
    device = _resolve_device(args.device, env_local_rank)
    backend = _resolve_dist_backend(args.dist_backend, device)
    context = None
    if backend is not None and (env_world_size > 1 or "RANK" in os.environ):
        context = fq.init_torch_distributed(
            backend=backend,
            world_size=env_world_size,
            rank=env_rank,
            local_rank=env_local_rank,
            device=device,
            force_initialize=True,
        )
        rank = int(context.rank)
        world_size = int(context.world_size)
        local_rank = int(context.local_rank)
        device = str(context.device)
        backend = str(context.backend)
    else:
        rank = env_rank
        world_size = env_world_size
        local_rank = env_local_rank
    return context, rank, world_size, local_rank, device, backend


def _sync_torch(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _block_jax(value: Any) -> None:
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    elif isinstance(value, (tuple, list)):
        for item in value:
            _block_jax(item)
    elif isinstance(value, dict):
        for item in value.values():
            _block_jax(item)


def _torch_to_jax(parameters: torch.Tensor) -> Any:
    import jax.dlpack

    return jax.dlpack.from_dlpack(parameters.detach().contiguous())


def _init_params(
    n_wires: int,
    layers: int,
    *,
    batch_size: int,
    device: str,
    rank: int,
    rank_offset: bool,
) -> torch.Tensor:
    total = int(layers) * int(n_wires) * 3
    base = torch.linspace(-0.37, 0.41, steps=total, dtype=torch.float32, device=device).reshape(
        int(layers),
        int(n_wires),
        3,
    )
    if rank_offset:
        base = base + float(rank) * 0.003
    if int(batch_size) <= 1:
        return base
    offsets = torch.linspace(-0.09, 0.09, steps=int(batch_size), dtype=torch.float32, device=device)
    return base.unsqueeze(0) + offsets.reshape(int(batch_size), 1, 1, 1)


def _build_circuit(params: torch.Tensor, *, device: str) -> fq.Circuit:
    layers, n_wires, _ = params.shape
    circuit = fq.Circuit(int(n_wires), device=device)
    for layer in range(int(layers)):
        for wire in range(int(n_wires)):
            circuit.rx(wire, theta=params[layer, wire, 0])
            circuit.ry(wire, theta=params[layer, wire, 1])
            circuit.rz(wire, theta=params[layer, wire, 2])
        for wire in range(int(n_wires) - 1):
            circuit.cx(wire, wire + 1)
        if int(n_wires) > 2:
            circuit.rxx(0, int(n_wires) - 1, theta=params[layer, 0, 0] * 0.25)
    return circuit


def _ising_hamiltonian(n_wires: int) -> fq.Hamiltonian:
    terms = []
    for wire in range(int(n_wires) - 1):
        terms.append(fq.pauli_term(0.7, "ZZ", (wire, wire + 1)))
    for wire in range(int(n_wires)):
        terms.append(fq.pauli_term(-0.2, "X", (wire,)))
        terms.append(fq.pauli_term(0.05, "Z", (wire,)))
    return fq.Hamiltonian(terms)


def _resolve_pennylane_device_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    device_name = str(args.pennylane_device).lower()
    tensor_method = str(args.pennylane_tensor_method).lower()
    if tensor_method == "auto":
        if "tensor" in device_name:
            tensor_method = "tn" if args.flagquantum_mode in {"tensor_network", "tn"} else "mps"
        else:
            tensor_method = "none"
    if tensor_method != "none":
        kwargs["method"] = tensor_method
    if args.pennylane_max_bond_dim is not None:
        kwargs["max_bond_dim"] = int(args.pennylane_max_bond_dim)
    if args.pennylane_cutoff is not None:
        kwargs["cutoff"] = float(args.pennylane_cutoff)
    return kwargs


def _is_pennylane_tensor_workload(args: argparse.Namespace) -> bool:
    device_name = str(args.pennylane_device).lower()
    tensor_method = str(args.pennylane_tensor_method).lower()
    return "tensor" in device_name or tensor_method in {"mps", "tn"}


def _effective_pennylane_early_stop_speedup(args: argparse.Namespace) -> float | None:
    if args.pennylane_early_stop_speedup is not None:
        return float(args.pennylane_early_stop_speedup)
    if _is_pennylane_tensor_workload(args):
        return 30000.0
    return None


def _flagquantum_value_and_grad(kernel: Any, params_seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    values = kernel(params)
    loss = values.sum()
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _time_flagquantum(fn: Any, *, warmup: int, iters: int, device: str) -> dict[str, Any]:
    for _ in range(int(warmup)):
        fn()
    _sync_torch(device)
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        loss, grad = fn()
    _sync_torch(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
    }


def _build_pennylane_value_and_grad(
    *,
    n_wires: int,
    parameter_shape: tuple[int, ...],
    observable: str,
    pennylane_device: str,
    pennylane_device_kwargs: dict[str, Any],
    diff_method: str,
    jit: bool,
) -> Any:
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    qdev = qml.device(pennylane_device, wires=int(n_wires), **pennylane_device_kwargs)

    if observable == "ising":
        coeffs = []
        ops = []
        for wire in range(int(n_wires) - 1):
            coeffs.append(0.7)
            ops.append(qml.PauliZ(wire) @ qml.PauliZ(wire + 1))
        for wire in range(int(n_wires)):
            coeffs.append(-0.2)
            ops.append(qml.PauliX(wire))
            coeffs.append(0.05)
            ops.append(qml.PauliZ(wire))
        hamiltonian = qml.Hamiltonian(coeffs, ops)
    else:
        hamiltonian = None

    @qml.qnode(qdev, interface="jax", diff_method=diff_method)
    def qnode(values: Any) -> Any:
        for layer in range(int(values.shape[0])):
            for wire in range(int(n_wires)):
                qml.RX(values[layer, wire, 0], wires=wire)
                qml.RY(values[layer, wire, 1], wires=wire)
                qml.RZ(values[layer, wire, 2], wires=wire)
            for wire in range(int(n_wires) - 1):
                qml.CNOT(wires=(wire, wire + 1))
            if int(n_wires) > 2:
                qml.IsingXX(values[layer, 0, 0] * 0.25, wires=(0, int(n_wires) - 1))
        if observable == "z_sum":
            return [qml.expval(qml.PauliZ(wire)) for wire in range(int(n_wires))]
        assert hamiltonian is not None
        return qml.expval(hamiltonian)

    def loss_single(values: Any) -> Any:
        out = qnode(values)
        if observable == "z_sum":
            return jnp.sum(jnp.stack(out))
        return out

    base_value_and_grad = jax.value_and_grad(loss_single)

    def value_and_grad(values: Any) -> Any:
        if tuple(values.shape) == parameter_shape:
            return base_value_and_grad(values)
        batch_shape = values.shape[: len(values.shape) - len(parameter_shape)]
        flat = values.reshape((-1,) + parameter_shape)
        losses, grads = jax.vmap(base_value_and_grad)(flat)
        return jnp.sum(losses), grads.reshape(values.shape)

    return jax.jit(value_and_grad) if jit else value_and_grad


def _pennylane_jax_value_and_grad(value_and_grad: Any, params_seed: torch.Tensor) -> tuple[Any, Any]:
    params = _torch_to_jax(params_seed)
    loss, grad = value_and_grad(params)
    _block_jax((loss, grad))
    return loss, grad


def _time_pennylane_jax(
    fn: Any,
    *,
    warmup: int,
    iters: int,
    reference_avg_seconds: float | None = None,
    early_stop_speedup: float | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    for _ in range(int(warmup)):
        _block_jax(fn())
    start = time.perf_counter()
    loss = None
    grad = None
    completed = 0
    early_stop_reason = None
    for _ in range(int(iters)):
        iter_start = time.perf_counter()
        loss, grad = fn()
        _block_jax((loss, grad))
        completed += 1
        iter_seconds = time.perf_counter() - iter_start
        elapsed_so_far = time.perf_counter() - start
        if (
            reference_avg_seconds is not None
            and early_stop_speedup is not None
            and reference_avg_seconds > 0
            and iter_seconds / reference_avg_seconds >= float(early_stop_speedup)
        ):
            early_stop_reason = "speedup_threshold_reached"
            break
        if max_seconds is not None and elapsed_so_far >= float(max_seconds):
            early_stop_reason = "max_seconds_reached"
            break
    _block_jax((loss, grad))
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    avg_seconds = elapsed / max(1, int(completed))
    speedup_lower_bound = (
        avg_seconds / reference_avg_seconds
        if reference_avg_seconds is not None and reference_avg_seconds > 0
        else None
    )
    return {
        "avg_seconds": avg_seconds,
        "loss": float(torch.utils.dlpack.from_dlpack(loss).detach().cpu()),
        "grad": torch.utils.dlpack.from_dlpack(grad).detach().cpu(),
        "completed_iters": int(completed),
        "requested_iters": int(iters),
        "early_stopped": early_stop_reason is not None,
        "early_stop_reason": early_stop_reason,
        "speedup_lower_bound_over_reference": speedup_lower_bound,
    }


def _build_pennylane_torch_loss(
    *,
    n_wires: int,
    observable: str,
    pennylane_device: str,
    pennylane_device_kwargs: dict[str, Any],
    diff_method: str,
) -> Any:
    import pennylane as qml

    qdev = qml.device(pennylane_device, wires=int(n_wires), **pennylane_device_kwargs)

    if observable == "ising":
        coeffs = []
        ops = []
        for wire in range(int(n_wires) - 1):
            coeffs.append(0.7)
            ops.append(qml.PauliZ(wire) @ qml.PauliZ(wire + 1))
        for wire in range(int(n_wires)):
            coeffs.append(-0.2)
            ops.append(qml.PauliX(wire))
            coeffs.append(0.05)
            ops.append(qml.PauliZ(wire))
        hamiltonian = qml.Hamiltonian(coeffs, ops)
    else:
        hamiltonian = None

    @qml.qnode(qdev, interface="torch", diff_method=diff_method)
    def qnode(values: torch.Tensor) -> Any:
        for layer in range(int(values.shape[0])):
            for wire in range(int(n_wires)):
                qml.RX(values[layer, wire, 0], wires=wire)
                qml.RY(values[layer, wire, 1], wires=wire)
                qml.RZ(values[layer, wire, 2], wires=wire)
            for wire in range(int(n_wires) - 1):
                qml.CNOT(wires=(wire, wire + 1))
            if int(n_wires) > 2:
                qml.IsingXX(values[layer, 0, 0] * 0.25, wires=(0, int(n_wires) - 1))
        if observable == "z_sum":
            return [qml.expval(qml.PauliZ(wire)) for wire in range(int(n_wires))]
        assert hamiltonian is not None
        return qml.expval(hamiltonian)

    def loss_single(values: torch.Tensor) -> torch.Tensor:
        out = qnode(values)
        if observable == "z_sum":
            return torch.stack(tuple(out)).sum()
        return out

    def loss(values: torch.Tensor) -> torch.Tensor:
        if values.ndim == 3:
            return loss_single(values)
        return torch.stack([loss_single(row) for row in values]).sum()

    return loss


def _pennylane_torch_value_and_grad(loss_fn: Any, params_seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    loss = loss_fn(params)
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _time_pennylane_torch(
    fn: Any,
    *,
    warmup: int,
    iters: int,
    device: str,
    reference_avg_seconds: float | None = None,
    early_stop_speedup: float | None = None,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    for _ in range(int(warmup)):
        fn()
    _sync_torch(device)
    start = time.perf_counter()
    loss = None
    grad = None
    completed = 0
    early_stop_reason = None
    for _ in range(int(iters)):
        iter_start = time.perf_counter()
        loss, grad = fn()
        _sync_torch(device)
        completed += 1
        iter_seconds = time.perf_counter() - iter_start
        elapsed_so_far = time.perf_counter() - start
        if (
            reference_avg_seconds is not None
            and early_stop_speedup is not None
            and reference_avg_seconds > 0
            and iter_seconds / reference_avg_seconds >= float(early_stop_speedup)
        ):
            early_stop_reason = "speedup_threshold_reached"
            break
        if max_seconds is not None and elapsed_so_far >= float(max_seconds):
            early_stop_reason = "max_seconds_reached"
            break
    _sync_torch(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    avg_seconds = elapsed / max(1, int(completed))
    speedup_lower_bound = (
        avg_seconds / reference_avg_seconds
        if reference_avg_seconds is not None and reference_avg_seconds > 0
        else None
    )
    return {
        "avg_seconds": avg_seconds,
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
        "completed_iters": int(completed),
        "requested_iters": int(iters),
        "early_stopped": early_stop_reason is not None,
        "early_stop_reason": early_stop_reason,
        "speedup_lower_bound_over_reference": speedup_lower_bound,
    }


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left.detach().cpu() - right.detach().cpu())).item())


def _comm_device(device: str, backend: str | None) -> torch.device:
    if backend == "nccl":
        return torch.device(device)
    return torch.device("cpu")


def _reduce_scalar(value: float, *, op: Any, device: torch.device, initialized: bool) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    if initialized:
        dist.all_reduce(tensor, op=op)
    return float(tensor.detach().cpu())


def _aggregate_time(value: float, *, world_size: int, device: torch.device, initialized: bool) -> dict[str, float]:
    total = _reduce_scalar(value, op=dist.ReduceOp.SUM, device=device, initialized=initialized)
    min_value = _reduce_scalar(value, op=dist.ReduceOp.MIN, device=device, initialized=initialized)
    max_value = _reduce_scalar(value, op=dist.ReduceOp.MAX, device=device, initialized=initialized)
    return {
        "rank0_seconds": float(value),
        "mean_seconds": total / max(1, int(world_size)),
        "min_seconds": min_value,
        "max_seconds": max_value,
    }


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.write_text(text + "\n", encoding="utf-8")


def _kernel_summary(kernel: Any) -> dict[str, Any]:
    summary = kernel.summary()
    terms = summary.get("hamiltonian_terms", ())
    if len(terms) > 12:
        summary = dict(summary)
        summary["hamiltonian_terms"] = list(terms[:12])
        summary["hamiltonian_term_count"] = len(terms)
    return summary


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "unknown"))


def _pennylane_worker(config: dict[str, Any], queue: Any) -> None:
    try:
        _configure_torch_precision(str(config["torch_matmul_precision"]))
        device = str(config["device"])
        params_seed = _init_params(
            int(config["n_wires"]),
            int(config["layers"]),
            batch_size=int(config["batch_size"]),
            device=device,
            rank=int(config["rank"]),
            rank_offset=bool(config["rank_offset_parameters"]),
        )
        example_params = params_seed[0].detach() if params_seed.ndim == 4 else params_seed.detach()
        if str(config["pennylane_interface"]) == "jax":
            value_and_grad = _build_pennylane_value_and_grad(
                n_wires=int(config["n_wires"]),
                parameter_shape=tuple(int(dim) for dim in example_params.shape),
                observable=str(config["observable"]),
                pennylane_device=str(config["pennylane_device"]),
                pennylane_device_kwargs=dict(config["pennylane_device_kwargs"]),
                diff_method=str(config["pennylane_diff_method"]),
                jit=not bool(config["no_pennylane_jit"]),
            )
            result = _time_pennylane_jax(
                lambda: _pennylane_jax_value_and_grad(value_and_grad, params_seed),
                warmup=int(config["pennylane_effective_warmup"]),
                iters=int(config["iters"]),
                reference_avg_seconds=float(config["reference_avg_seconds"]),
                early_stop_speedup=config["pennylane_effective_early_stop_speedup"],
                max_seconds=config["pennylane_max_seconds"],
            )
        else:
            loss = _build_pennylane_torch_loss(
                n_wires=int(config["n_wires"]),
                observable=str(config["observable"]),
                pennylane_device=str(config["pennylane_device"]),
                pennylane_device_kwargs=dict(config["pennylane_device_kwargs"]),
                diff_method=str(config["pennylane_diff_method"]),
            )
            result = _time_pennylane_torch(
                lambda: _pennylane_torch_value_and_grad(loss, params_seed),
                warmup=int(config["pennylane_effective_warmup"]),
                iters=int(config["iters"]),
                device=device,
                reference_avg_seconds=float(config["reference_avg_seconds"]),
                early_stop_speedup=config["pennylane_effective_early_stop_speedup"],
                max_seconds=config["pennylane_max_seconds"],
            )
        result["grad_shape"] = list(result["grad"].shape)
        result["grad"] = result["grad"].detach().cpu().reshape(-1).tolist()
        queue.put({"ok": True, "result": result})
    except Exception as exc:
        queue.put({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}})


def _time_pennylane_in_subprocess(
    config: dict[str, Any],
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    context = mp.get_context("spawn")
    queue: Any = context.Queue()
    process = context.Process(target=_pennylane_worker, args=(config, queue))
    process.start()
    process.join(float(timeout_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5)
        return None, {
            "type": "TimeoutError",
            "message": f"PennyLane subprocess exceeded {float(timeout_seconds):.3f} seconds.",
            "timeout_seconds": float(timeout_seconds),
        }
    if queue.empty():
        return None, {
            "type": "RuntimeError",
            "message": f"PennyLane subprocess exited with code {process.exitcode} without returning a result.",
            "exitcode": process.exitcode,
        }
    payload = queue.get()
    if not payload.get("ok"):
        return None, payload.get("error")
    result = payload["result"]
    result["grad"] = torch.tensor(result["grad"], dtype=torch.float32).reshape(tuple(result["grad_shape"]))
    del result["grad_shape"]
    return result, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--observable", choices=("z_sum", "ising"), default="z_sum")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or an explicit torch device.")
    parser.add_argument("--dist-backend", default="auto", choices=("auto", "nccl", "gloo", "none"))
    parser.add_argument("--pennylane-device", default="default.qubit")
    parser.add_argument("--pennylane-interface", default="torch", choices=("torch", "jax"))
    parser.add_argument("--pennylane-diff-method", default="best")
    parser.add_argument("--pennylane-tensor-method", default="auto", choices=("auto", "mps", "tn", "none"))
    parser.add_argument("--pennylane-max-bond-dim", type=int, default=None)
    parser.add_argument("--pennylane-cutoff", type=float, default=None)
    parser.add_argument(
        "--flagquantum-mode",
        choices=("statevector", "mps", "tensor_network", "tn"),
        default="statevector",
    )
    parser.add_argument("--no-flagquantum-jit", action="store_true")
    parser.add_argument("--flagquantum-jax-matmul-precision", default="highest")
    parser.add_argument("--flagquantum-jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--flagquantum-max-bond", type=int, default=None)
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--no-pennylane-jit", action="store_true")
    parser.add_argument("--rank-offset-parameters", action="store_true")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--pennylane-max-seconds", type=float, default=None)
    parser.add_argument("--pennylane-run-timeout-seconds", type=float, default=None)
    parser.add_argument("--pennylane-early-stop-speedup", type=float, default=None)
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    _configure_torch_precision(args.torch_matmul_precision)

    context, rank, world_size, local_rank, device, backend = _init_distributed(args)
    initialized = bool(context is not None and context.initialized)
    comm_device = _comm_device(device, backend)

    params_seed = _init_params(
        args.n_wires,
        args.layers,
        batch_size=args.batch_size,
        device=device,
        rank=rank,
        rank_offset=args.rank_offset_parameters,
    )
    example_params = params_seed[0].detach() if params_seed.ndim == 4 else params_seed.detach()
    hamiltonian = _ising_hamiltonian(args.n_wires) if args.observable == "ising" else None
    pennylane_device_kwargs = _resolve_pennylane_device_kwargs(args)
    pennylane_effective_early_stop_speedup = _effective_pennylane_early_stop_speedup(args)
    pennylane_effective_warmup = 0 if pennylane_effective_early_stop_speedup is not None else int(args.warmup)

    flagquantum_kernel = fq.compile_quantum_kernel(
        lambda values: _build_circuit(values, device=device),
        example_params,
        backend="jax",
        interface="torch",
        mode="tensor_network" if args.flagquantum_mode == "tn" else args.flagquantum_mode,
        n_wires=args.n_wires,
        observable="z_sum" if args.observable == "z_sum" else "hamiltonian",
        hamiltonian=hamiltonian,
        jit=not args.no_flagquantum_jit,
        matmul_precision=None
        if args.flagquantum_jax_matmul_precision == "default"
        else args.flagquantum_jax_matmul_precision,
        compute_dtype=args.flagquantum_jax_compute_dtype,
        max_bond=args.flagquantum_max_bond,
    )
    flagquantum_result = _time_flagquantum(
        lambda: _flagquantum_value_and_grad(flagquantum_kernel, params_seed),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    pennylane_result = None
    pennylane_error = None
    if importlib.util.find_spec("pennylane") is None:
        pennylane_error = {
            "type": "ModuleNotFoundError",
            "message": "pennylane is not installed",
        }
    else:
        try:
            if args.pennylane_run_timeout_seconds is not None:
                pennylane_result, pennylane_error = _time_pennylane_in_subprocess(
                    {
                        "n_wires": args.n_wires,
                        "layers": args.layers,
                        "batch_size": args.batch_size,
                        "observable": args.observable,
                        "device": device,
                        "rank": rank,
                        "rank_offset_parameters": args.rank_offset_parameters,
                        "pennylane_device": args.pennylane_device,
                        "pennylane_device_kwargs": pennylane_device_kwargs,
                        "pennylane_interface": args.pennylane_interface,
                        "pennylane_diff_method": args.pennylane_diff_method,
                        "no_pennylane_jit": args.no_pennylane_jit,
                        "iters": args.iters,
                        "pennylane_effective_warmup": pennylane_effective_warmup,
                        "torch_matmul_precision": args.torch_matmul_precision,
                        "reference_avg_seconds": flagquantum_result["avg_seconds"],
                        "pennylane_effective_early_stop_speedup": pennylane_effective_early_stop_speedup,
                        "pennylane_max_seconds": args.pennylane_max_seconds,
                    },
                    timeout_seconds=float(args.pennylane_run_timeout_seconds),
                )
            elif args.pennylane_interface == "jax":
                pennylane_value_and_grad = _build_pennylane_value_and_grad(
                    n_wires=args.n_wires,
                    parameter_shape=tuple(int(dim) for dim in example_params.shape),
                    observable=args.observable,
                    pennylane_device=args.pennylane_device,
                    pennylane_device_kwargs=pennylane_device_kwargs,
                    diff_method=args.pennylane_diff_method,
                    jit=not args.no_pennylane_jit,
                )
                pennylane_result = _time_pennylane_jax(
                    lambda: _pennylane_jax_value_and_grad(pennylane_value_and_grad, params_seed),
                    warmup=pennylane_effective_warmup,
                    iters=args.iters,
                    reference_avg_seconds=flagquantum_result["avg_seconds"],
                    early_stop_speedup=pennylane_effective_early_stop_speedup,
                    max_seconds=args.pennylane_max_seconds,
                )
            else:
                pennylane_loss = _build_pennylane_torch_loss(
                    n_wires=args.n_wires,
                    observable=args.observable,
                    pennylane_device=args.pennylane_device,
                    pennylane_device_kwargs=pennylane_device_kwargs,
                    diff_method=args.pennylane_diff_method,
                )
                pennylane_result = _time_pennylane_torch(
                    lambda: _pennylane_torch_value_and_grad(pennylane_loss, params_seed),
                    warmup=pennylane_effective_warmup,
                    iters=args.iters,
                    device=device,
                    reference_avg_seconds=flagquantum_result["avg_seconds"],
                    early_stop_speedup=pennylane_effective_early_stop_speedup,
                    max_seconds=args.pennylane_max_seconds,
                )
        except Exception as exc:  # pragma: no cover - depends on optional plugin availability.
            pennylane_error = {
                "type": type(exc).__name__,
                "message": str(exc),
            }

    aggregated: dict[str, Any] = {
        "flagquantum_seconds": _aggregate_time(
            flagquantum_result["avg_seconds"],
            world_size=world_size,
            device=comm_device,
            initialized=initialized,
        )
    }
    comparison: dict[str, Any]
    if pennylane_result is not None:
        loss_abs_error = abs(float(flagquantum_result["loss"]) - float(pennylane_result["loss"]))
        grad_max_abs_error = _max_abs_diff(flagquantum_result["grad"], pennylane_result["grad"])
        aggregated.update(
            {
                "pennylane_seconds": _aggregate_time(
                    pennylane_result["avg_seconds"],
                    world_size=world_size,
                    device=comm_device,
                    initialized=initialized,
                ),
                "loss_abs_error_max": _reduce_scalar(
                    loss_abs_error,
                    op=dist.ReduceOp.MAX,
                    device=comm_device,
                    initialized=initialized,
                ),
                "grad_max_abs_error_max": _reduce_scalar(
                    grad_max_abs_error,
                    op=dist.ReduceOp.MAX,
                    device=comm_device,
                    initialized=initialized,
                ),
            }
        )
        status = (
            "ok"
            if aggregated["loss_abs_error_max"] <= float(args.loss_atol)
            and aggregated["grad_max_abs_error_max"] <= float(args.grad_atol)
            else "precision_mismatch"
        )
        comparison = {
            "status": status,
            "loss_abs_error_rank0": loss_abs_error,
            "grad_max_abs_error_rank0": grad_max_abs_error,
            "loss_abs_error_max": aggregated["loss_abs_error_max"],
            "grad_max_abs_error_max": aggregated["grad_max_abs_error_max"],
            "loss_atol": float(args.loss_atol),
            "grad_atol": float(args.grad_atol),
            "pennylane_effective_warmup": int(pennylane_effective_warmup),
            "pennylane_early_stop_speedup": pennylane_effective_early_stop_speedup,
            "pennylane_early_stopped": bool(pennylane_result.get("early_stopped", False)),
            "pennylane_early_stop_reason": pennylane_result.get("early_stop_reason"),
            "pennylane_completed_iters": pennylane_result.get("completed_iters"),
            "pennylane_requested_iters": pennylane_result.get("requested_iters"),
            "speedup_flagquantum_over_pennylane_mean_rank_time": (
                aggregated["pennylane_seconds"]["mean_seconds"] / aggregated["flagquantum_seconds"]["mean_seconds"]
                if aggregated["flagquantum_seconds"]["mean_seconds"] > 0
                else None
            ),
            "speedup_flagquantum_over_pennylane_slowest_rank_time": (
                aggregated["pennylane_seconds"]["max_seconds"] / aggregated["flagquantum_seconds"]["max_seconds"]
                if aggregated["flagquantum_seconds"]["max_seconds"] > 0
                else None
            ),
        }
    else:
        timeout_lower_bound = None
        timeout_lower_bound_min = None
        if pennylane_error and pennylane_error.get("type") == "TimeoutError" and flagquantum_result["avg_seconds"] > 0:
            timeout_lower_bound = float(pennylane_error["timeout_seconds"]) / float(flagquantum_result["avg_seconds"])
            timeout_lower_bound_min = _reduce_scalar(
                timeout_lower_bound,
                op=dist.ReduceOp.MIN,
                device=comm_device,
                initialized=initialized,
            )
        comparison = {
            "status": "timeout" if timeout_lower_bound is not None else "unavailable",
            "reason": pennylane_error,
            "pennylane_effective_warmup": int(pennylane_effective_warmup),
            "pennylane_early_stop_speedup": pennylane_effective_early_stop_speedup,
            "pennylane_run_timeout_seconds": args.pennylane_run_timeout_seconds,
            "speedup_lower_bound_flagquantum_over_pennylane_rank0": timeout_lower_bound,
            "speedup_lower_bound_flagquantum_over_pennylane_min_rank": timeout_lower_bound_min,
        }

    if rank == 0:
        payload: dict[str, Any] = {
            "benchmark": "pennylane_backend_compare",
            "n_wires": args.n_wires,
            "layers": args.layers,
            "batch_size": args.batch_size,
            "observable": args.observable,
            "device": device,
            "dist_backend": backend,
            "world_size": world_size,
            "rank": rank,
            "local_rank": local_rank,
            "iters": args.iters,
            "warmup": args.warmup,
            "pennylane_effective_warmup": int(pennylane_effective_warmup),
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "jax": _module_version("jax"),
                "pennylane": _module_version("pennylane"),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "platform": platform.platform(),
                "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
                "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
                "torch_float32_matmul_precision": args.torch_matmul_precision,
                "torch_cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32
                if torch.cuda.is_available()
                else None,
                "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
            },
            "flagquantum_jax": {
                "backend": "jax",
                "interface": "torch",
                "jit": not args.no_flagquantum_jit,
                "requested_matmul_precision": args.flagquantum_jax_matmul_precision,
                "requested_compute_dtype": args.flagquantum_jax_compute_dtype,
                "mode": args.flagquantum_mode,
                "rank0_avg_seconds": flagquantum_result["avg_seconds"],
                "distributed_seconds": aggregated["flagquantum_seconds"],
                "loss_rank0": flagquantum_result["loss"],
                "kernel": _kernel_summary(flagquantum_kernel),
            },
            "pennylane": {
                "backend": args.pennylane_device,
                "device_kwargs": pennylane_device_kwargs,
                "interface": args.pennylane_interface,
                "diff_method": args.pennylane_diff_method,
                "jit": (not args.no_pennylane_jit) if args.pennylane_interface == "jax" else False,
                "status": "ok" if pennylane_result is not None else comparison["status"],
                "rank0_avg_seconds": None if pennylane_result is None else pennylane_result["avg_seconds"],
                "completed_iters": None if pennylane_result is None else pennylane_result.get("completed_iters"),
                "requested_iters": None if pennylane_result is None else pennylane_result.get("requested_iters"),
                "early_stopped": None if pennylane_result is None else pennylane_result.get("early_stopped"),
                "early_stop_reason": None if pennylane_result is None else pennylane_result.get("early_stop_reason"),
                "early_stop_speedup": pennylane_effective_early_stop_speedup,
                "effective_warmup": int(pennylane_effective_warmup),
                "run_timeout_seconds": args.pennylane_run_timeout_seconds,
                "speedup_lower_bound_over_flagquantum": comparison.get(
                    "speedup_lower_bound_flagquantum_over_pennylane_rank0"
                )
                if pennylane_result is None
                else pennylane_result.get("speedup_lower_bound_over_reference"),
                "distributed_seconds": None if pennylane_result is None else aggregated["pennylane_seconds"],
                "loss_rank0": None if pennylane_result is None else pennylane_result["loss"],
                "error": pennylane_error,
            },
            "comparison": comparison,
        }
        if comparison["status"] == "timeout":
            lower_bound = comparison.get("speedup_lower_bound_flagquantum_over_pennylane_min_rank")
            payload["conclusion"] = {
                "headline": "PennyLane did not finish within the configured timeout for this workload.",
                "recommended_claim": (
                    f"Against PennyLane {args.pennylane_device}, on {world_size} rank(s), device={device}, "
                    f"pennylane_interface={args.pennylane_interface}, pennylane_device_kwargs={pennylane_device_kwargs}, "
                    f"flagquantum_mode={args.flagquantum_mode}, {args.n_wires} wires, {args.layers} layers, "
                    f"batch_size={args.batch_size}, observable={args.observable}, FlagQuantum is at least "
                    f"{float(lower_bound):.2f}x faster before the PennyLane timeout."
                    if lower_bound is not None
                    else None
                ),
            }
        if comparison["status"] in {"ok", "precision_mismatch"}:
            speedup = comparison["speedup_flagquantum_over_pennylane_mean_rank_time"]
            payload["conclusion"] = {
                "headline": (
                    "FlagQuantum JAX matches PennyLane gradient precision for this workload."
                    if comparison["status"] == "ok"
                    else "FlagQuantum JAX and PennyLane did not meet the requested precision thresholds."
                ),
                "recommended_claim": (
                    f"Against PennyLane {args.pennylane_device}, on {world_size} rank(s), device={device}, "
                    f"pennylane_interface={args.pennylane_interface}, pennylane_device_kwargs={pennylane_device_kwargs}, "
                    f"flagquantum_mode={args.flagquantum_mode}, {args.n_wires} wires, {args.layers} layers, batch_size={args.batch_size}, "
                    f"observable={args.observable}, FlagQuantum is {float(speedup):.2f}x faster by mean rank "
                    f"time with max loss_abs_error={comparison['loss_abs_error_max']:.3e} and "
                    f"max grad_abs_error={comparison['grad_max_abs_error_max']:.3e}."
                    if speedup is not None
                    else None
                ),
            }

        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)

    if initialized:
        dist.barrier()
        fq.destroy_torch_distributed()


if __name__ == "__main__":
    main()
