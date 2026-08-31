"""Distributed GPU benchmark for FlagQuantum PyTorch and JAX backends.

This script is intended for torchrun on GPU clusters. It compares the native
PyTorch statevector/autograd path with FlagQuantum's JAX quantum kernel exposed
through the PyTorch interface, then aggregates precision and speed across ranks.

Examples
--------
Single node, 8 GPUs:

    torchrun --standalone --nproc_per_node=8 benchmarks/distributed_backend_compare.py \
        --device cuda --dist-backend nccl \
        --n-wires 8 --layers 2 --batch-size 16 --observable ising \
        --iters 100 --warmup 20 \
        --json-output gpu_backend_compare_8q_b16_ising.json

The same script also runs as a single-process smoke test without torchrun.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
import torch.distributed as dist


REPRO_COMMANDS = """
Single-node GPU statevector:
  torchrun --standalone --nproc_per_node=8 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --mode statevector --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --jax-matmul-precision highest --torch-matmul-precision highest --json-output gpu_jax_statevector_compare_8q_b16_ising.json

Single-node GPU MPS:
  torchrun --standalone --nproc_per_node=8 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --mode mps --max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --jax-matmul-precision highest --torch-matmul-precision highest --json-output gpu_jax_mps_compare_8q_b16_ising.json

Single-node GPU MPS high precision:
  torchrun --standalone --nproc_per_node=8 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --mode mps --max-bond 32 --jax-compute-dtype complex128 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --jax-matmul-precision highest --torch-matmul-precision highest --json-output gpu_jax_mps_compare_8q_b16_ising_fp64.json

Single-node GPU tensor network:
  torchrun --standalone --nproc_per_node=8 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --mode tensor_network --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --jax-matmul-precision highest --torch-matmul-precision highest --json-output gpu_jax_tn_compare_8q_b16_ising.json

Single-node GPU tensor network high precision:
  torchrun --standalone --nproc_per_node=8 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --mode tensor_network --jax-compute-dtype complex128 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --jax-matmul-precision highest --torch-matmul-precision highest --json-output gpu_jax_tn_compare_8q_b16_ising_fp64.json

Two-node heterogeneous MPS smoke, run once on each node with node_rank=0/1:
  NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1 XLA_PYTHON_CLIENT_PREALLOCATE=false torchrun --nnodes=2 --nproc_per_node=8 --node_rank=<0-or-1> --master_addr=<MASTER_IP> --master_port=29500 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --mode mps --max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 2 --warmup 1 --dist-timeout-seconds 600 --jax-matmul-precision highest --torch-matmul-precision highest --json-output multinode_a100_a800_mps_smoke.json

Two-node diagnostics-only preflight, run once on each node with node_rank=0/1:
  NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1 XLA_PYTHON_CLIENT_PREALLOCATE=false torchrun --nnodes=2 --nproc_per_node=8 --node_rank=<0-or-1> --master_addr=<MASTER_IP> --master_port=29500 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --diagnostics-only --dist-timeout-seconds 300 --json-output multinode_a100_a800_preflight.json

Two-node heterogeneous MPS benchmark, run once on each node with node_rank=0/1:
  NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_BLOCKING_WAIT=1 XLA_PYTHON_CLIENT_PREALLOCATE=false torchrun --nnodes=2 --nproc_per_node=8 --node_rank=<0-or-1> --master_addr=<MASTER_IP> --master_port=29500 benchmarks/distributed_backend_compare.py --device cuda --dist-backend nccl --mode mps --max-bond 32 --n-wires 8 --layers 2 --batch-size 16 --observable ising --iters 100 --warmup 20 --dist-timeout-seconds 900 --jax-matmul-precision highest --torch-matmul-precision highest --json-output multinode_a100_a800_mps_8q_b16_ising.json

CPU smoke:
  python benchmarks/distributed_backend_compare.py --device cpu --dist-backend none --mode mps --max-bond 8 --n-wires 4 --layers 1 --batch-size 2 --observable z_sum --iters 2 --warmup 1
"""


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402
import flagquantum.backends as fqb  # noqa: E402


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


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _log_rank(message: str, *, rank: int | None = None, enabled: bool = True) -> None:
    if not enabled:
        return
    prefix = f"[rank {rank}] " if rank is not None else ""
    print(f"{prefix}{message}", file=sys.stderr, flush=True)


def _init_distributed(args: argparse.Namespace) -> tuple[Any, int, int, int, str, str | None]:
    env_rank = _env_int("RANK", 0)
    env_world_size = _env_int("WORLD_SIZE", 1)
    env_local_rank = _env_int("LOCAL_RANK", env_rank)
    _log_rank(
        (
            "startup "
            f"hostname={socket.gethostname()} pid={os.getpid()} "
            f"env_rank={env_rank} env_world_size={env_world_size} env_local_rank={env_local_rank}"
        ),
        rank=env_rank,
        enabled=not args.quiet_rank_status,
    )
    device = _resolve_device(args.device, env_local_rank)
    backend = _resolve_dist_backend(args.dist_backend, device)
    context = None
    if backend is not None and (env_world_size > 1 or "RANK" in os.environ):
        _log_rank(
            (
                "init_process_group begin "
                f"backend={backend} device={device} timeout_seconds={args.dist_timeout_seconds}"
            ),
            rank=env_rank,
            enabled=not args.quiet_rank_status,
        )
        context = fq.init_torch_distributed(
            backend=backend,
            world_size=env_world_size,
            rank=env_rank,
            local_rank=env_local_rank,
            device=device,
            force_initialize=True,
            timeout_seconds=args.dist_timeout_seconds,
        )
        rank = int(context.rank)
        world_size = int(context.world_size)
        local_rank = int(context.local_rank)
        device = str(context.device)
        backend = str(context.backend)
        _log_rank(
            (
                "init_process_group done "
                f"backend={backend} world_size={world_size} local_rank={local_rank} device={device}"
            ),
            rank=rank,
            enabled=not args.quiet_rank_status,
        )
    else:
        rank = env_rank
        world_size = env_world_size
        local_rank = env_local_rank
    return context, rank, world_size, local_rank, device, backend


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


def _pauli_dense_matrix(
    ops: tuple[tuple[int, str], ...],
    n_wires: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    op_map = {int(wire): str(name).lower() for wire, name in ops}
    complex_dtype = dtype if dtype.is_complex else torch.complex64
    identity = torch.eye(2, dtype=complex_dtype, device=device)
    matrices = {
        "i": identity,
        "x": torch.tensor([[0, 1], [1, 0]], dtype=complex_dtype, device=device),
        "y": torch.tensor([[0, -1j], [1j, 0]], dtype=complex_dtype, device=device),
        "z": torch.tensor([[1, 0], [0, -1]], dtype=complex_dtype, device=device),
    }
    out = matrices[op_map.get(0, "i")]
    for wire in range(1, int(n_wires)):
        out = torch.kron(out, matrices[op_map.get(wire, "i")])
    return out


def _dense_hamiltonian_loss(state_or_circuit: Any, hamiltonian: fq.Hamiltonian) -> torch.Tensor:
    state = state_or_circuit.state()
    if state.ndim == 1:
        state = state.reshape(1, -1)
    n_wires = int(round(torch.log2(torch.tensor(state.shape[-1], dtype=torch.float64)).item()))
    total = None
    for term in hamiltonian.terms:
        ops = tuple((int(wire), str(name).lower()) for wire, name in term.ops if str(name).lower() != "i")
        if not ops:
            value = torch.ones(state.shape[0], dtype=torch.float32, device=state.device)
        else:
            matrix = _pauli_dense_matrix(ops, n_wires, device=state.device, dtype=state.dtype)
            transformed = state @ matrix.transpose(-1, -2)
            value = torch.real(torch.sum(torch.conj(state) * transformed, dim=-1))
        contribution = torch.real(term.coefficient * value)
        total = contribution if total is None else total + contribution
    assert total is not None
    return total.sum()


def _loss_from_state(
    state_or_circuit: Any,
    *,
    observable: str,
    hamiltonian: fq.Hamiltonian | None,
) -> torch.Tensor:
    if observable == "z_sum":
        if hasattr(state_or_circuit, "expectation_z_sum"):
            return state_or_circuit.expectation_z_sum().sum()
        return state_or_circuit.expectation_z().sum()
    if observable == "ising":
        assert hamiltonian is not None
        return _dense_hamiltonian_loss(state_or_circuit, hamiltonian)
    raise ValueError(f"Unsupported observable {observable!r}.")


def _pytorch_value_and_grad(
    params_seed: torch.Tensor,
    *,
    device: str,
    mode: str,
    max_bond: int | None,
    observable: str,
    hamiltonian: fq.Hamiltonian | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    def loss_for(row: torch.Tensor) -> torch.Tensor:
        circuit = _build_circuit(row, device=device)
        if mode == "statevector":
            return _loss_from_state(circuit, observable=observable, hamiltonian=hamiltonian)
        if mode == "mps":
            return _loss_from_state(
                fqb.run_mps(circuit, max_bond=max_bond),
                observable=observable,
                hamiltonian=hamiltonian,
            )
        if mode in {"tensor_network", "tn"}:
            return _loss_from_state(
                fqb.run_tensor_network(circuit),
                observable=observable,
                hamiltonian=hamiltonian,
            )
        raise ValueError("mode must be 'statevector', 'mps', or 'tensor_network'.")

    if params.ndim == 3:
        loss = loss_for(params)
    else:
        losses = [loss_for(row) for row in params]
        loss = torch.stack(losses).sum()
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _jax_value_and_grad(kernel: Any, params_seed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    params = params_seed.detach().clone().requires_grad_(True)
    values = kernel(params)
    loss = values.sum()
    loss.backward()
    grad = params.grad if params.grad is not None else torch.zeros_like(params)
    return loss.detach(), grad.detach()


def _time_value_and_grad(fn: Any, *, warmup: int, iters: int, device: str) -> dict[str, Any]:
    for _ in range(int(warmup)):
        fn()
    _sync(device)
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        loss, grad = fn()
    _sync(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "loss": float(loss.detach().cpu()),
        "grad": grad.detach().cpu(),
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


def _gather_float_list(value: float, *, world_size: int, device: torch.device, initialized: bool) -> list[float]:
    if not initialized:
        return [float(value)]
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(int(world_size))]
    dist.all_gather(gathered, tensor)
    return [float(item.detach().cpu()) for item in gathered]


def _imbalance_max_over_min(values: list[float]) -> float | None:
    if not values:
        return None
    min_value = min(values)
    max_value = max(values)
    if min_value <= 0:
        return None
    return float(max_value / min_value)


def _aggregate_time(value: float, *, world_size: int, device: torch.device, initialized: bool) -> dict[str, Any]:
    total = _reduce_scalar(value, op=dist.ReduceOp.SUM, device=device, initialized=initialized)
    min_value = _reduce_scalar(value, op=dist.ReduceOp.MIN, device=device, initialized=initialized)
    max_value = _reduce_scalar(value, op=dist.ReduceOp.MAX, device=device, initialized=initialized)
    rank_seconds = _gather_float_list(value, world_size=world_size, device=device, initialized=initialized)
    return {
        "rank0_seconds": float(value),
        "mean_seconds": total / max(1, int(world_size)),
        "min_seconds": min_value,
        "max_seconds": max_value,
        "rank_seconds": rank_seconds,
        "imbalance_max_over_min": _imbalance_max_over_min(rank_seconds),
    }


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.write_text(text + "\n", encoding="utf-8")


_DIAGNOSTIC_ENV_KEYS = (
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "NCCL_DEBUG",
    "NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT",
    "NCCL_SOCKET_IFNAME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "CUDA_VISIBLE_DEVICES",
    "USE_LIBUV",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
)


def _cuda_device_diagnostics(device: str) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {"available": torch.cuda.is_available()}
    cuda_device = torch.device(device)
    index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    free_bytes = None
    total_visible_bytes = None
    try:
        free_bytes, total_visible_bytes = torch.cuda.mem_get_info(index)
    except Exception:
        pass
    return {
        "available": True,
        "index": int(index),
        "name": torch.cuda.get_device_name(index),
        "capability": [int(props.major), int(props.minor)],
        "total_memory_bytes": int(props.total_memory),
        "mem_get_info_free_bytes": None if free_bytes is None else int(free_bytes),
        "mem_get_info_total_bytes": None if total_visible_bytes is None else int(total_visible_bytes),
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
    }


def _rank_diagnostics(*, rank: int, world_size: int, local_rank: int, device: str, backend: str | None) -> dict[str, Any]:
    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "local_rank": int(local_rank),
        "hostname": socket.gethostname(),
        "pid": int(os.getpid()),
        "device": device,
        "backend": backend,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": _cuda_device_diagnostics(device),
        "env": {key: os.environ.get(key) for key in _DIAGNOSTIC_ENV_KEYS if os.environ.get(key) is not None},
    }


def _gather_objects(value: Any, *, initialized: bool) -> list[Any]:
    if not initialized:
        return [value]
    output: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(output, value)
    return output


def _distributed_probe(*, rank: int, world_size: int, device: torch.device, initialized: bool) -> dict[str, Any]:
    if not initialized:
        return {
            "initialized": False,
            "barrier_seconds": None,
            "all_reduce_seconds": None,
            "all_reduce_sum": float(rank + 1),
            "all_reduce_expected_sum": float(rank + 1),
            "all_reduce_ok": True,
            "all_gather_object_seconds": None,
        }
    start = time.perf_counter()
    dist.barrier()
    barrier_seconds = time.perf_counter() - start

    tensor = torch.tensor(float(rank + 1), dtype=torch.float64, device=device)
    start = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    all_reduce_seconds = time.perf_counter() - start
    expected_sum = float(int(world_size) * (int(world_size) + 1) / 2)

    start = time.perf_counter()
    gathered = _gather_objects({"rank": int(rank), "hostname": socket.gethostname()}, initialized=True)
    all_gather_object_seconds = time.perf_counter() - start

    return {
        "initialized": True,
        "barrier_seconds": float(barrier_seconds),
        "all_reduce_seconds": float(all_reduce_seconds),
        "all_reduce_sum": float(tensor.detach().cpu()),
        "all_reduce_expected_sum": expected_sum,
        "all_reduce_ok": abs(float(tensor.detach().cpu()) - expected_sum) < 1e-9,
        "all_gather_object_seconds": float(all_gather_object_seconds),
        "all_gather_object_count": len(gathered),
    }


def _summarize_rank_diagnostics(rank_diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    hostnames = [str(item.get("hostname")) for item in rank_diagnostics]
    gpu_names = [
        str((item.get("cuda") or {}).get("name"))
        for item in rank_diagnostics
        if (item.get("cuda") or {}).get("name") is not None
    ]
    ranks_by_host: dict[str, int] = {}
    for hostname in hostnames:
        ranks_by_host[hostname] = ranks_by_host.get(hostname, 0) + 1
    gpu_counts: dict[str, int] = {}
    for name in gpu_names:
        gpu_counts[name] = gpu_counts.get(name, 0) + 1
    warnings = []
    if len(ranks_by_host) > 1 and len(set(gpu_names)) > 1:
        warnings.append("heterogeneous_gpu_names_detected")
    if len(set(hostnames)) > 1 and len(set(ranks_by_host.values())) > 1:
        warnings.append("uneven_ranks_per_host")
    local_pairs = {(item.get("hostname"), item.get("local_rank")) for item in rank_diagnostics}
    if len(local_pairs) != len(rank_diagnostics):
        warnings.append("duplicate_hostname_local_rank_mapping")
    return {
        "host_count": len(ranks_by_host),
        "ranks_by_host": ranks_by_host,
        "gpu_counts": gpu_counts,
        "gpu_names": sorted(set(gpu_names)),
        "warnings": warnings,
    }


def _build_diagnostics_payload(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: str,
    backend: str | None,
    initialized: bool,
    comm_device: torch.device,
) -> dict[str, Any]:
    _log_rank("distributed preflight probe begin", rank=rank, enabled=not args.quiet_rank_status)
    probe = _distributed_probe(rank=rank, world_size=world_size, device=comm_device, initialized=initialized)
    _log_rank("distributed preflight probe done", rank=rank, enabled=not args.quiet_rank_status)
    rank_diag = _rank_diagnostics(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )
    all_rank_diagnostics = _gather_objects(rank_diag, initialized=initialized)
    all_probes = _gather_objects(probe, initialized=initialized)
    return {
        "benchmark": "distributed_backend_preflight",
        "device": device,
        "dist_backend": backend,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "dist_timeout_seconds": args.dist_timeout_seconds,
        "status": "ok" if all(item.get("all_reduce_ok", False) for item in all_probes) else "communication_mismatch",
        "rank_summary": _summarize_rank_diagnostics(all_rank_diagnostics),
        "rank_diagnostics": all_rank_diagnostics,
        "communication_probes": all_probes,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "platform": platform.platform(),
            "diagnostic_env": {
                key: os.environ.get(key) for key in _DIAGNOSTIC_ENV_KEYS if os.environ.get(key) is not None
            },
        },
    }


def _kernel_summary(kernel: Any) -> dict[str, Any]:
    summary = kernel.summary()
    terms = summary.get("hamiltonian_terms", ())
    if len(terms) > 12:
        summary = dict(summary)
        summary["hamiltonian_terms"] = list(terms[:12])
        summary["hamiltonian_term_count"] = len(terms)
    return summary


def _build_payload(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
    local_rank: int,
    device: str,
    backend: str | None,
    pytorch_result: dict[str, Any],
    jax_result: dict[str, Any] | None,
    kernel: Any | None,
    loss_abs_error: float | None,
    grad_max_abs_error: float | None,
    aggregated: dict[str, Any],
    rank_diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "benchmark": "distributed_backend_compare",
        "distribution_semantics": "rank_local_replicated_kernel",
        "scalability_claim_allowed": False,
        "scalability_note": (
            "This benchmark launches one PyTorch/JAX quantum kernel per torchrun rank and aggregates "
            "timing/precision with torch.distributed collectives. It does not shard one quantum "
            "statevector/MPS/TN workload across ranks."
        ),
        "n_wires": args.n_wires,
        "layers": args.layers,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "observable": args.observable,
        "device": device,
        "dist_backend": backend,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "iters": args.iters,
        "warmup": args.warmup,
        "rank_offset_parameters": bool(args.rank_offset_parameters),
        "dist_timeout_seconds": args.dist_timeout_seconds,
        "rank_summary": _summarize_rank_diagnostics(rank_diagnostics),
        "rank_diagnostics": rank_diagnostics,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
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
            "diagnostic_env": {
                key: os.environ.get(key) for key in _DIAGNOSTIC_ENV_KEYS if os.environ.get(key) is not None
            },
        },
        "flagquantum_pytorch": {
            "backend": "pytorch",
            "mode": args.mode,
            "rank0_avg_seconds": pytorch_result["avg_seconds"],
            "loss_rank0": pytorch_result["loss"],
            "distributed_seconds": aggregated["pytorch_seconds"],
        },
    }
    if jax_result is None:
        payload["flagquantum_jax"] = {
            "backend": "jax",
            "status": "unavailable",
            "reason": "jax is not installed",
        }
        payload["comparison"] = {
            "status": "unavailable",
            "reason": "jax is not installed",
        }
        return fq.attach_distributed_scalability_audit(payload)

    assert loss_abs_error is not None and grad_max_abs_error is not None
    status = (
        "ok"
        if aggregated["loss_abs_error_max"] <= float(args.loss_atol)
        and aggregated["grad_max_abs_error_max"] <= float(args.grad_atol)
        else "precision_mismatch"
    )
    speedup_mean = (
        aggregated["pytorch_seconds"]["mean_seconds"] / aggregated["jax_seconds"]["mean_seconds"]
        if aggregated["jax_seconds"]["mean_seconds"] > 0
        else None
    )
    speedup_by_slowest_rank = (
        aggregated["pytorch_seconds"]["max_seconds"] / aggregated["jax_seconds"]["max_seconds"]
        if aggregated["jax_seconds"]["max_seconds"] > 0
        else None
    )
    payload["flagquantum_jax"] = {
        "backend": "jax",
        "interface": "torch",
        "mode": args.mode,
        "jit": not args.no_jax_jit,
        "status": status,
        "rank0_avg_seconds": jax_result["avg_seconds"],
        "loss_rank0": jax_result["loss"],
        "distributed_seconds": aggregated["jax_seconds"],
        "kernel": _kernel_summary(kernel),
    }
    payload["comparison"] = {
        "status": status,
        "loss_abs_error_rank0": loss_abs_error,
        "grad_max_abs_error_rank0": grad_max_abs_error,
        "loss_abs_error_max": aggregated["loss_abs_error_max"],
        "grad_max_abs_error_max": aggregated["grad_max_abs_error_max"],
        "loss_atol": float(args.loss_atol),
        "grad_atol": float(args.grad_atol),
        "speedup_jax_over_pytorch_mean_rank_time": speedup_mean,
        "speedup_jax_over_pytorch_slowest_rank_time": speedup_by_slowest_rank,
        "slowdown_jax_vs_pytorch_mean_rank_time": (
            aggregated["jax_seconds"]["mean_seconds"] / aggregated["pytorch_seconds"]["mean_seconds"]
            if aggregated["pytorch_seconds"]["mean_seconds"] > 0
            else None
        ),
    }
    payload["conclusion"] = {
        "headline": (
            "FlagQuantum JAX quantum-kernel backend matches PyTorch gradients on the GPU cluster."
            if status == "ok"
            else "FlagQuantum JAX quantum-kernel backend did not meet the requested GPU-cluster precision thresholds."
        ),
        "recommended_claim": (
            f"On {world_size} rank(s), device={device}, {args.n_wires} wires, {args.layers} layers, "
            f"mode={args.mode}, batch_size={args.batch_size}, observable={args.observable}, FlagQuantum JAX is "
            f"{float(speedup_mean):.2f}x faster than FlagQuantum PyTorch by mean rank time, with "
            f"max loss_abs_error={aggregated['loss_abs_error_max']:.3e} and "
            f"max grad_abs_error={aggregated['grad_max_abs_error_max']:.3e}."
            if speedup_mean is not None
            else None
        ),
    }
    return fq.attach_distributed_scalability_audit(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=("statevector", "mps", "tensor_network", "tn"), default="statevector")
    parser.add_argument("--observable", choices=("z_sum", "ising"), default="z_sum")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', or an explicit torch device.")
    parser.add_argument("--dist-backend", default="auto", choices=("auto", "nccl", "gloo", "none"))
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--no-jax-jit", action="store_true")
    parser.add_argument("--jax-matmul-precision", default="highest")
    parser.add_argument("--jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--max-bond", type=int, default=None)
    parser.add_argument("--rank-offset-parameters", action="store_true")
    parser.add_argument("--loss-atol", type=float, default=1e-4)
    parser.add_argument("--grad-atol", type=float, default=1e-4)
    parser.add_argument("--dist-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--diagnostics-only", action="store_true")
    parser.add_argument("--quiet-rank-status", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    _configure_torch_precision(args.torch_matmul_precision)

    context, rank, world_size, local_rank, device, backend = _init_distributed(args)
    initialized = bool(context is not None and context.initialized)
    comm_device = _comm_device(device, backend)
    if args.diagnostics_only:
        payload = _build_diagnostics_payload(
            args,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=backend,
            initialized=initialized,
            comm_device=comm_device,
        )
        if rank == 0:
            text = json.dumps(payload, indent=2, sort_keys=True)
            print(text)
            if args.json_output:
                _write_json_output(args.json_output, text)
        if initialized:
            _log_rank("diagnostics final barrier begin", rank=rank, enabled=not args.quiet_rank_status)
            dist.barrier()
            _log_rank("diagnostics final barrier done", rank=rank, enabled=not args.quiet_rank_status)
            fq.destroy_torch_distributed()
        return

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

    _log_rank("pytorch reference timing begin", rank=rank, enabled=not args.quiet_rank_status)
    pytorch_result = _time_value_and_grad(
        lambda: _pytorch_value_and_grad(
            params_seed,
            device=device,
            mode=args.mode,
            max_bond=args.max_bond,
            observable=args.observable,
            hamiltonian=hamiltonian,
        ),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    _log_rank(
        f"pytorch reference timing done avg_seconds={pytorch_result['avg_seconds']:.6g}",
        rank=rank,
        enabled=not args.quiet_rank_status,
    )

    kernel = None
    jax_result = None
    loss_abs_error = None
    grad_max_abs_error = None
    jax_available = importlib.util.find_spec("jax") is not None
    if jax_available:
        _log_rank("jax kernel compile/setup begin", rank=rank, enabled=not args.quiet_rank_status)
        kernel = fq.compile_quantum_kernel(
            lambda values: _build_circuit(values, device=device),
            example_params,
            backend="jax",
            interface="torch",
            mode="tensor_network" if args.mode == "tn" else args.mode,
            n_wires=args.n_wires,
            observable="z_sum" if args.observable == "z_sum" else "hamiltonian",
            hamiltonian=hamiltonian,
            jit=not args.no_jax_jit,
            matmul_precision=None if args.jax_matmul_precision == "default" else args.jax_matmul_precision,
            compute_dtype=args.jax_compute_dtype,
            max_bond=args.max_bond,
        )
        _log_rank("jax kernel timing begin", rank=rank, enabled=not args.quiet_rank_status)
        jax_result = _time_value_and_grad(
            lambda: _jax_value_and_grad(kernel, params_seed),
            warmup=args.warmup,
            iters=args.iters,
            device=device,
        )
        _log_rank(
            f"jax kernel timing done avg_seconds={jax_result['avg_seconds']:.6g}",
            rank=rank,
            enabled=not args.quiet_rank_status,
        )
        loss_abs_error = abs(float(pytorch_result["loss"]) - float(jax_result["loss"]))
        grad_max_abs_error = _max_abs_diff(pytorch_result["grad"], jax_result["grad"])

    aggregated = {
        "pytorch_seconds": _aggregate_time(
            pytorch_result["avg_seconds"],
            world_size=world_size,
            device=comm_device,
            initialized=initialized,
        )
    }
    if jax_result is not None and loss_abs_error is not None and grad_max_abs_error is not None:
        aggregated.update(
            {
                "jax_seconds": _aggregate_time(
                    jax_result["avg_seconds"],
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

    _log_rank("rank diagnostics gather begin", rank=rank, enabled=not args.quiet_rank_status)
    all_rank_diagnostics = _gather_objects(
        _rank_diagnostics(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=backend,
        ),
        initialized=initialized,
    )
    _log_rank("rank diagnostics gather done", rank=rank, enabled=not args.quiet_rank_status)

    if rank == 0:
        payload = _build_payload(
            args,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            device=device,
            backend=backend,
            pytorch_result=pytorch_result,
            jax_result=jax_result,
            kernel=kernel,
            loss_abs_error=loss_abs_error,
            grad_max_abs_error=grad_max_abs_error,
            aggregated=aggregated,
            rank_diagnostics=all_rank_diagnostics,
        )
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)

    if initialized:
        _log_rank("final barrier begin", rank=rank, enabled=not args.quiet_rank_status)
        dist.barrier()
        _log_rank("final barrier done", rank=rank, enabled=not args.quiet_rank_status)
        fq.destroy_torch_distributed()


if __name__ == "__main__":
    main()
