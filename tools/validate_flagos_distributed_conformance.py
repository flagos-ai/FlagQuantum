#!/usr/bin/env python
"""Run single-node FlagOS ProcessGroup and sharded-statevector conformance.

Launch this tool with ``torchrun``. Torch-FL is imported before PyTorch because
some provider distributions require that order while registering ``flagos``.
The emitted payload proves only the public FlagOS boundary; it never infers the
provider's inner collective route.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from flagquantum.runtime.distributed import init_torch_distributed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError(
            "FlagOS distributed conformance requires an explicit Torch-FL install"
        ) from exc
    torch = importlib.import_module("torch")
    fq = importlib.import_module("flagquantum")
    if not hasattr(torch, "flagos"):
        raise RuntimeError("Torch-FL imported without registering torch.flagos")
    return torch_fl, torch, fq


def _log(rank: int, phase: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[flagos-conformance rank={rank}] phase={phase}{suffix}", flush=True)


def _source_revision() -> str:
    declared = os.environ.get("FLAGQUANTUM_SOURCE_REVISION")
    if declared:
        return declared
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _synchronize(torch: Any) -> None:
    torch.flagos.synchronize()


def _global_max(torch: Any, value: float, *, device: Any) -> float:
    metric = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.MAX)
    return float(metric.item())


def _measure(
    torch: Any,
    operation: Callable[[], tuple[Any, Any]],
    *,
    device: Any,
) -> tuple[float, float, int, str]:
    torch.distributed.barrier()
    _synchronize(torch)
    started = time.perf_counter()
    actual, expected = operation()
    _synchronize(torch)
    elapsed = time.perf_counter() - started
    if actual.device.type != "flagos":
        raise AssertionError(f"collective payload escaped flagos: {actual.device}")
    local_error = float(
        torch.max(torch.abs(actual.detach() - expected.detach())).item()
    )
    maximum_error = _global_max(torch, local_error, device=device)
    maximum_elapsed = _global_max(torch, elapsed, device=device)
    payload_bytes = int(actual.numel()) * int(actual.element_size())
    return maximum_error, maximum_elapsed, payload_bytes, actual.device.type


def _collective_operation(
    torch: Any,
    *,
    primitive: str,
    dtype: Any,
    device: Any,
    rank: int,
    world_size: int,
) -> Callable[[], tuple[Any, Any]]:
    dist = torch.distributed

    if primitive == "broadcast":

        def broadcast() -> tuple[Any, Any]:
            actual = torch.tensor(
                [complex(rank + 1, -rank)], dtype=dtype, device=device
            )
            dist.broadcast(actual, src=0)
            expected = torch.tensor([1.0 + 0.0j], dtype=dtype, device=device)
            return actual, expected

        return broadcast

    if primitive == "all_reduce":

        def all_reduce() -> tuple[Any, Any]:
            actual = torch.tensor(
                [complex(rank + 1, rank + 0.5)], dtype=dtype, device=device
            )
            dist.all_reduce(actual, op=dist.ReduceOp.SUM)
            expected = torch.tensor(
                [
                    complex(
                        world_size * (world_size + 1) / 2,
                        world_size * world_size / 2,
                    )
                ],
                dtype=dtype,
                device=device,
            )
            return actual, expected

        return all_reduce

    if primitive == "all_gather_into_tensor":

        def all_gather_into_tensor() -> tuple[Any, Any]:
            local = torch.tensor(
                [complex(rank + 1, -(rank + 1))], dtype=dtype, device=device
            )
            actual = torch.empty(world_size, dtype=dtype, device=device)
            dist.all_gather_into_tensor(actual, local)
            expected = torch.tensor(
                [complex(index + 1, -(index + 1)) for index in range(world_size)],
                dtype=dtype,
                device=device,
            )
            return actual, expected

        return all_gather_into_tensor

    if primitive == "reduce_scatter_tensor":

        def reduce_scatter_tensor() -> tuple[Any, Any]:
            local = torch.tensor(
                [
                    complex((rank + 1) * 10 + chunk, rank + chunk)
                    for chunk in range(world_size)
                ],
                dtype=dtype,
                device=device,
            )
            actual = torch.empty(1, dtype=dtype, device=device)
            dist.reduce_scatter_tensor(actual, local, op=dist.ReduceOp.SUM)
            expected = torch.tensor(
                [
                    complex(
                        10 * world_size * (world_size + 1) / 2 + world_size * rank,
                        world_size * (world_size - 1) / 2 + world_size * rank,
                    )
                ],
                dtype=dtype,
                device=device,
            )
            return actual, expected

        return reduce_scatter_tensor

    if primitive == "isend_irecv":

        def isend_irecv() -> tuple[Any, Any]:
            send_peer = (rank + 1) % world_size
            receive_peer = (rank - 1) % world_size
            send = torch.tensor(
                [complex(rank + 1, rank + 0.25)], dtype=dtype, device=device
            )
            actual = torch.empty_like(send)
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, send, send_peer),
                    dist.P2POp(dist.irecv, actual, receive_peer),
                ]
            )
            for request in requests:
                request.wait()
            expected = torch.tensor(
                [complex(receive_peer + 1, receive_peer + 0.25)],
                dtype=dtype,
                device=device,
            )
            return actual, expected

        return isend_irecv

    raise ValueError(f"unsupported conformance primitive {primitive!r}")


def _reference_circuit(fq: Any, *, dtype: Any) -> Any:
    return (
        fq.Circuit(3, dtype=dtype)
        .h(0)
        .rx(2, theta=0.2)
        .cx(0, 2)
        .rz(1, theta=-0.3)
        .cx(2, 1)
    )


def _global_indices(torch: Any, result: Any) -> Any:
    shard_state = result.shard_state
    if shard_state.global_indices.numel():
        return shard_state.global_indices.detach().cpu().to(torch.long)
    local = torch.arange(shard_state.amplitudes.shape[-1], dtype=torch.long)
    if result.plan.distribution == "qubit_address_sharded":
        return (local << len(result.plan.sharded_wires)) | shard_state.rank
    return local + shard_state.shard.amplitude_start


def _statevector_check(
    torch: Any,
    fq: Any,
    *,
    dtype_name: str,
    device: Any,
    rank: int,
) -> Any:
    from flagquantum.runtime.backends.statevector import (
        execute_torch_distributed_statevector,
    )
    from flagquantum.runtime.distributed.conformance import (
        FlagOSStatevectorConformanceCheck,
    )

    dtype = getattr(torch, dtype_name)
    tolerance = 2e-5 if dtype_name == "complex64" else 1e-11
    circuit = _reference_circuit(fq, dtype=dtype)
    _log(rank, "statevector_reference_start", f"dtype={dtype_name}")
    reference = circuit.state().detach().cpu().to(torch.complex128).reshape(-1)
    _log(rank, "statevector_reference_ready", f"dtype={dtype_name}")
    _log(rank, "statevector_executor_start", f"dtype={dtype_name}")
    result = execute_torch_distributed_statevector(
        circuit,
        device=device,
        dtype=dtype,
        persistent_wire_layout=False,
    )
    _log(rank, "statevector_executor_ready", f"dtype={dtype_name}")
    if result.shard_state.amplitudes.device.type != "flagos":
        raise AssertionError("statevector shard escaped the logical flagos device")
    indices = _global_indices(torch, result)
    expected = reference.index_select(0, indices)
    actual = (
        result.shard_state.amplitudes.detach().cpu().to(torch.complex128).reshape(-1)
    )
    local_error = float(torch.max(torch.abs(actual - expected)).item())
    maximum_error = _global_max(torch, local_error, device=device)
    passed = bool(
        maximum_error <= tolerance
        and result.plan.world_size > 1
        and result.distributed_gate_count > 0
        and result.communication_count > 0
        and result.shard_state.amplitudes.numel() < result.plan.total_amplitudes
    )
    _log(
        rank,
        "statevector",
        f"dtype={dtype_name} max_abs_error={maximum_error:.3e} "
        f"communications={result.communication_count}",
    )
    return FlagOSStatevectorConformanceCheck(
        dtype=dtype_name,
        passed=passed,
        max_abs_error=maximum_error,
        tolerance=tolerance,
        local_amplitudes=int(result.shard_state.amplitudes.numel()),
        total_amplitudes=int(result.plan.total_amplitudes),
        distributed_gate_count=int(result.distributed_gate_count),
        communication_count=int(result.communication_count),
        communication_bytes=int(result.communication_bytes),
        device_type=result.shard_state.amplitudes.device.type,
    )


def _rank_placement(torch: Any, context: Any) -> tuple[dict[str, Any], ...]:
    local = torch.tensor(
        [context.rank, context.local_rank, context.device.index],
        dtype=torch.int64,
        device=context.device,
    )
    gathered = torch.empty(
        context.world_size * local.numel(), dtype=local.dtype, device=context.device
    )
    torch.distributed.all_gather_into_tensor(gathered, local)
    rows = gathered.detach().cpu().reshape(context.world_size, local.numel()).tolist()
    return tuple(
        {
            "rank": int(rank),
            "local_rank": int(local_rank),
            "device_index": int(device_index),
            "device": f"flagos:{int(device_index)}",
        }
        for rank, local_rank, device_index in rows
    )


def _forbid_object_collectives(torch: Any) -> dict[str, Any]:
    originals = {}

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("FlagOS conformance used a Python object collective")

    for name in (
        "all_gather_object",
        "broadcast_object_list",
        "gather_object",
        "scatter_object_list",
    ):
        if hasattr(torch.distributed, name):
            originals[name] = getattr(torch.distributed, name)
            setattr(torch.distributed, name, forbidden)
    return originals


def _restore_object_collectives(torch: Any, originals: dict[str, Any]) -> None:
    for name, value in originals.items():
        setattr(torch.distributed, name, value)


def run(*, timeout_seconds: float) -> Any:
    torch_fl, torch, fq = _load_runtime()
    from flagquantum.runtime.distributed.conformance import (
        REQUIRED_FLAGOS_COLLECTIVES,
        REQUIRED_FLAGOS_DTYPES,
        FlagOSCollectiveConformanceCheck,
        FlagOSDistributedConformanceReport,
    )

    context = init_torch_distributed(
        backend="flagos",
        device="flagos",
        timeout_seconds=timeout_seconds,
    )
    rank = context.rank
    if context.world_size < 2:
        raise RuntimeError("FlagOS distributed conformance requires at least two ranks")
    if context.node_count != 1 or context.local_world_size != context.world_size:
        raise RuntimeError("FlagOS F1 conformance is restricted to one complete node")
    if context.identity is None:
        raise RuntimeError("FlagOS process group did not emit DistributedIdentity")
    _log(
        rank, "initialized", f"device={context.device} world_size={context.world_size}"
    )

    originals = _forbid_object_collectives(torch)
    try:
        collective_checks = []
        for dtype_name in REQUIRED_FLAGOS_DTYPES:
            dtype = getattr(torch, dtype_name)
            for primitive in REQUIRED_FLAGOS_COLLECTIVES:
                operation = _collective_operation(
                    torch,
                    primitive=primitive,
                    dtype=dtype,
                    device=context.device,
                    rank=rank,
                    world_size=context.world_size,
                )
                tolerance = 1e-5 if dtype_name == "complex64" else 1e-12
                try:
                    error, elapsed, payload_bytes, device_type = _measure(
                        torch, operation, device=context.device
                    )
                    failure = None
                except Exception as exc:  # noqa: BLE001 - evidence must retain failures
                    error = elapsed = 0.0
                    payload_bytes = 0
                    device_type = context.device.type
                    failure = f"{type(exc).__name__}: {exc}"
                collective_checks.append(
                    FlagOSCollectiveConformanceCheck(
                        primitive=primitive,
                        dtype=dtype_name,
                        passed=failure is None and error <= tolerance,
                        max_abs_error=error,
                        elapsed_seconds=elapsed,
                        payload_bytes=payload_bytes,
                        device_type=device_type,
                        error=failure,
                    )
                )
                outcome = (
                    f"error={failure}"
                    if failure is not None
                    else f"max_abs_error={error:.3e}"
                )
                _log(
                    rank,
                    "collective",
                    f"primitive={primitive} dtype={dtype_name} {outcome}",
                )
        statevector_checks = tuple(
            _statevector_check(
                torch,
                fq,
                dtype_name=dtype_name,
                device=context.device,
                rank=rank,
            )
            for dtype_name in REQUIRED_FLAGOS_DTYPES
        )
        placement = _rank_placement(torch, context)
    finally:
        _restore_object_collectives(torch, originals)

    device_name_getter = getattr(torch.flagos, "get_device_name", None)
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_fl": str(getattr(torch_fl, "__version__", "unknown")),
        "torch_cuda_runtime": str(torch.version.cuda),
        "hostname": socket.gethostname(),
        "device_name": (
            str(device_name_getter(context.device.index))
            if device_name_getter is not None
            else "unknown"
        ),
        "source_revision": _source_revision(),
        "torch_fl_source_revision": os.environ.get(
            "TORCH_FL_SOURCE_REVISION", "unavailable"
        ),
        "payload_tensor_residency_verified": True,
        "object_collectives_forbidden": True,
        "reference_materialization": "per_rank_cpu_small_case_only",
    }
    report = FlagOSDistributedConformanceReport(
        identity=context.identity,
        collective_checks=tuple(collective_checks),
        statevector_checks=statevector_checks,
        rank_placement=placement,
        environment=environment,
        world_size=context.world_size,
        local_world_size=context.local_world_size,
        node_count=context.node_count,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    try:
        report = run(timeout_seconds=args.timeout_seconds)
        rank = report.identity.rank
        torch = importlib.import_module("torch")
        if rank == 0:
            payload = report.to_dict()
            encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.output.with_suffix(args.output.suffix + ".tmp")
                temporary.write_text(encoded, encoding="utf-8")
                temporary.replace(args.output)
            print(json.dumps(payload, sort_keys=True), flush=True)
        torch.distributed.barrier()
        return 0 if report.mechanical_conformance_accepted else 1
    finally:
        try:
            torch = importlib.import_module("torch")
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                importlib.import_module("flagquantum").destroy_torch_distributed()
        except (ImportError, RuntimeError):
            # Preserve the original validation error; process exit is the final fallback.
            pass


if __name__ == "__main__":
    raise SystemExit(main())
