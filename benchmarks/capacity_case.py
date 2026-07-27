"""Single replicated per-rank capacity case for FlagQuantum MPS/TN on torchrun.

This file is normally launched by ``benchmarks/capacity_sweep.py``. It runs one
problem size on every torchrun rank, reports per-rank latency and memory, and
writes a JSON result from rank 0.

Important: every rank runs the same full problem. This file measures per-GPU
capacity and cluster health under torchrun. It does not measure single-problem
multi-GPU expansion for cases that do not fit on one GPU.

The default ``--engine native`` is the conservative path for maximum-qubit
capacity limits because it measures FlagQuantum's native MPS/TN runtime without
JAX compilation overhead. ``--engine jax`` measures the JAX quantum-kernel path.
For ``z_sum`` observables, JAX MPS/TN kernels use native MPS/TN contractions and
must not materialize dense statevectors.

Example
-------
  torchrun --standalone --nproc_per_node=8 benchmarks/capacity_case.py \
    --device cuda --dist-backend nccl --engine native --mode mps --max-bond 32 \
    --n-wires 64 --layers 2 --batch-size 1 --observable z_sum \
    --iters 3 --warmup 1 --json-output capacity_mps_64q.json

  torchrun --standalone --nproc_per_node=8 benchmarks/capacity_case.py \
    --device cuda --dist-backend nccl --engine jax --mode mps --max-bond 32 \
    --n-wires 64 --layers 2 --batch-size 1 --observable z_sum \
    --iters 20 --warmup 5 --json-output capacity_jax_mps_64q.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flagquantum as fq  # noqa: E402


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def _configure_torch_precision(mode: str) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(mode)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = mode != "highest"
        torch.backends.cudnn.allow_tf32 = mode != "highest"


def _resolve_device(requested: str, local_rank: int) -> str:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
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
            timeout_seconds=args.dist_timeout_seconds,
        )
        return context, int(context.rank), int(context.world_size), int(context.local_rank), str(context.device), str(context.backend)
    return context, env_rank, env_world_size, env_local_rank, device, backend


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _write_json_output(path_text: str, text: str) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = Path("benchmarks") / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _init_params(n_wires: int, layers: int, *, batch_size: int, device: str) -> torch.Tensor:
    total = int(layers) * int(n_wires) * 3
    base = torch.linspace(-0.37, 0.41, steps=total, dtype=torch.float32, device=device).reshape(
        int(layers),
        int(n_wires),
        3,
    )
    if int(batch_size) <= 1:
        return base
    offsets = torch.linspace(-0.09, 0.09, steps=int(batch_size), dtype=torch.float32, device=device)
    return base.unsqueeze(0) + offsets.reshape(int(batch_size), 1, 1, 1)


def _build_circuit(params: torch.Tensor, *, device: str, entangler: str) -> fq.Circuit:
    layers, n_wires, _ = params.shape
    circuit = fq.Circuit(int(n_wires), device=device)
    for layer in range(int(layers)):
        for wire in range(int(n_wires)):
            circuit.rx(wire, theta=params[layer, wire, 0])
            circuit.ry(wire, theta=params[layer, wire, 1])
            circuit.rz(wire, theta=params[layer, wire, 2])
        if entangler in {"chain", "brickwork"}:
            start = layer % 2 if entangler == "brickwork" else 0
            for wire in range(start, int(n_wires) - 1, 2 if entangler == "brickwork" else 1):
                circuit.cx(wire, wire + 1)
        elif entangler == "ring":
            for wire in range(int(n_wires) - 1):
                circuit.cx(wire, wire + 1)
            if int(n_wires) > 2:
                circuit.cx(int(n_wires) - 1, 0)
        elif entangler == "none":
            pass
        else:
            raise ValueError(f"Unsupported entangler {entangler!r}.")
    return circuit


def _nvidia_smi_memory(local_rank: int) -> dict[str, Any] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={int(local_rank)}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=3,
        )
    except Exception:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        used, total = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return {"used_mb": used, "total_mb": total, "used_fraction": used / total if total > 0 else None}


def _torch_memory(device: str) -> dict[str, Any]:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    dev = torch.device(device)
    return {
        "torch_allocated_mb": torch.cuda.memory_allocated(dev) / 1024**2,
        "torch_reserved_mb": torch.cuda.memory_reserved(dev) / 1024**2,
        "torch_max_allocated_mb": torch.cuda.max_memory_allocated(dev) / 1024**2,
        "torch_max_reserved_mb": torch.cuda.max_memory_reserved(dev) / 1024**2,
    }


def _time_kernel(
    kernel: Any,
    params_seed: torch.Tensor,
    *,
    warmup: int,
    iters: int,
    device: str,
    forward_only: bool,
) -> dict[str, Any]:
    def run_once() -> tuple[torch.Tensor, torch.Tensor]:
        params = params_seed.detach().clone().requires_grad_(not forward_only)
        out = kernel(params)
        loss = out.sum()
        if forward_only:
            grad = torch.zeros_like(params)
        else:
            loss.backward()
            grad = params.grad if params.grad is not None else torch.zeros_like(params)
        return loss.detach(), grad.detach()

    warmup_start = time.perf_counter()
    for _ in range(int(warmup)):
        run_once()
    _sync(device)
    warmup_elapsed = time.perf_counter() - warmup_start
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        loss, grad = run_once()
    _sync(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "measured_seconds": elapsed,
        "warmup_seconds": warmup_elapsed,
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad.detach().cpu()).item()) if not forward_only else None,
    }


def _native_loss(
    params: torch.Tensor,
    *,
    mode: str,
    max_bond: int | None,
    device: str,
    entangler: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    circuit = _build_circuit(params, device=device, entangler=entangler)
    if mode == "mps":
        state = fq.run_mps(circuit, max_bond=max_bond)
        loss = state.expectation_z_sum().sum()
        return loss, {"state_summary": state.summary()}
    if mode == "tensor_network":
        state = fq.run_tensor_network(circuit)
        loss = state.expectation_z().sum()
        return loss, {"state_summary": state.summary()}
    raise ValueError(f"Unsupported native capacity mode {mode!r}.")


def _time_native(
    params_seed: torch.Tensor,
    *,
    mode: str,
    max_bond: int | None,
    warmup: int,
    iters: int,
    device: str,
    entangler: str,
    forward_only: bool,
) -> dict[str, Any]:
    last_extra: dict[str, Any] = {}

    def run_once() -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal last_extra
        params = params_seed.detach().clone().requires_grad_(not forward_only)
        if params.ndim == 3:
            loss, last_extra = _native_loss(
                params,
                mode=mode,
                max_bond=max_bond,
                device=device,
                entangler=entangler,
            )
        else:
            losses = []
            extras = []
            for row in params:
                row_loss, row_extra = _native_loss(
                    row,
                    mode=mode,
                    max_bond=max_bond,
                    device=device,
                    entangler=entangler,
                )
                losses.append(row_loss)
                extras.append(row_extra)
            loss = torch.stack(losses).sum()
            last_extra = {"batch_state_summaries": [item.get("state_summary") for item in extras]}
        if forward_only:
            grad = torch.zeros_like(params)
        else:
            loss.backward()
            grad = params.grad if params.grad is not None else torch.zeros_like(params)
        return loss.detach(), grad.detach()

    warmup_start = time.perf_counter()
    for _ in range(int(warmup)):
        run_once()
    _sync(device)
    warmup_elapsed = time.perf_counter() - warmup_start
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    start = time.perf_counter()
    loss = None
    grad = None
    for _ in range(int(iters)):
        loss, grad = run_once()
    _sync(device)
    elapsed = time.perf_counter() - start
    assert loss is not None and grad is not None
    return {
        "avg_seconds": elapsed / max(1, int(iters)),
        "measured_seconds": elapsed,
        "warmup_seconds": warmup_elapsed,
        "loss": float(loss.detach().cpu()),
        "grad_norm": float(torch.linalg.vector_norm(grad.detach().cpu()).item()) if not forward_only else None,
        **last_extra,
    }


def _aggregate_rank_objects(obj: dict[str, Any], initialized: bool) -> list[dict[str, Any]]:
    if not initialized:
        return [obj]
    out: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return list(out)


def _stats(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(clean), "max": max(clean), "mean": sum(clean) / len(clean)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-wires", type=int, required=True)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--engine", choices=("native", "jax"), default="native")
    parser.add_argument("--mode", choices=("mps", "tensor_network", "tn"), default="mps")
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--observable", choices=("z_sum",), default="z_sum")
    parser.add_argument("--entangler", choices=("chain", "brickwork", "ring", "none"), default="chain")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dist-backend", default="auto", choices=("auto", "nccl", "gloo", "none"))
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-jax-jit", action="store_true")
    parser.add_argument("--jax-compute-dtype", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--jax-matmul-precision", default="highest")
    parser.add_argument("--torch-matmul-precision", default="highest")
    parser.add_argument("--dist-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    _configure_torch_precision(args.torch_matmul_precision)
    context, rank, world_size, local_rank, device, backend = _init_distributed(args)
    initialized = bool(context is not None and context.initialized)
    mode = "tensor_network" if args.mode == "tn" else args.mode
    status = "ok"
    error = None
    rank_payload: dict[str, Any]
    try:
        params_seed = _init_params(args.n_wires, args.layers, batch_size=args.batch_size, device=device)
        kernel_build_seconds = None
        if args.engine == "native":
            result = _time_native(
                params_seed,
                mode=mode,
                max_bond=args.max_bond,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
                entangler=args.entangler,
                forward_only=bool(args.forward_only),
            )
        else:
            example_params = params_seed[0].detach() if params_seed.ndim == 4 else params_seed.detach()
            kernel_build_start = time.perf_counter()
            kernel = fq.compile_quantum_kernel(
                lambda values: _build_circuit(values, device=device, entangler=args.entangler),
                example_params,
                backend="jax",
                interface="torch",
                mode=mode,
                n_wires=args.n_wires,
                observable="z_sum",
                jit=not args.no_jax_jit,
                max_bond=args.max_bond,
                compute_dtype=args.jax_compute_dtype,
                matmul_precision=None if args.jax_matmul_precision == "default" else args.jax_matmul_precision,
            )
            kernel_build_seconds = time.perf_counter() - kernel_build_start
            result = _time_kernel(
                kernel,
                params_seed,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
                forward_only=bool(args.forward_only),
            )
        rank_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "device": device,
            "status": status,
            "avg_seconds": result["avg_seconds"],
            "measured_seconds": result.get("measured_seconds"),
            "warmup_seconds": result.get("warmup_seconds"),
            "kernel_build_seconds": kernel_build_seconds,
            "loss": result["loss"],
            "grad_norm": result["grad_norm"],
            "state_summary": result.get("state_summary"),
            "batch_state_summaries": result.get("batch_state_summaries"),
            "torch_memory": _torch_memory(device),
            "nvidia_smi_memory": _nvidia_smi_memory(local_rank),
        }
    except Exception as exc:
        status = "error"
        error = {"type": type(exc).__name__, "message": str(exc)}
        rank_payload = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "device": device,
            "status": status,
            "error": error,
            "torch_memory": _torch_memory(device),
            "nvidia_smi_memory": _nvidia_smi_memory(local_rank),
        }

    rank_results = _aggregate_rank_objects(rank_payload, initialized)
    if rank == 0:
        statuses = [item.get("status") for item in rank_results]
        successful = [item for item in rank_results if item.get("status") == "ok"]
        payload = {
            "benchmark": "capacity_case",
            "distribution_semantics": "replicated_per_rank",
            "scalability_claim_allowed": False,
            "scalability_note": (
                "Each rank runs the same complete case independently. This result must not be used "
                "as evidence that one large MPS/TN problem is sharded across GPUs."
            ),
            "status": "ok" if len(successful) == len(rank_results) else "error",
            "n_wires": int(args.n_wires),
            "layers": int(args.layers),
            "batch_size": int(args.batch_size),
            "mode": mode,
            "engine": args.engine,
            "path_semantics": {
                "native": "FlagQuantum native PyTorch MPS/TN runtime; best for maximum-qubit capacity limits.",
                "jax": "FlagQuantum JAX quantum-kernel runtime through the torch interface; z_sum on MPS/TN uses direct MPS/TN observable contractions.",
                "materializes_statevector_for_mps_tn_z_sum": False if args.engine == "jax" else None,
            },
            "max_bond": int(args.max_bond) if mode == "mps" and args.max_bond is not None else args.max_bond,
            "observable": args.observable,
            "entangler": args.entangler,
            "forward_only": bool(args.forward_only),
            "iters": int(args.iters),
            "warmup": int(args.warmup),
            "world_size": int(world_size),
            "backend": backend,
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "platform": platform.platform(),
                "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
            },
            "rank_statuses": statuses,
            "rank_results": rank_results,
            "summary": {
                "avg_seconds": _stats([item.get("avg_seconds") for item in successful]),
                "torch_max_allocated_mb": _stats(
                    [
                        (item.get("torch_memory") or {}).get("torch_max_allocated_mb")
                        for item in successful
                    ]
                ),
                "nvidia_smi_used_mb": _stats(
                    [
                        (item.get("nvidia_smi_memory") or {}).get("used_mb")
                        for item in successful
                    ]
                ),
            },
        }
        payload = fq.attach_distributed_scalability_audit(payload)
        text = json.dumps(payload, indent=2, sort_keys=True)
        print(text)
        if args.json_output:
            _write_json_output(args.json_output, text)

    if initialized:
        dist.barrier()
        fq.destroy_torch_distributed()


if __name__ == "__main__":
    main()
