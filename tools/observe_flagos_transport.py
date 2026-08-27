#!/usr/bin/env python
"""Observe FlagOS collective transport without inferring its inner provider route."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.runtime.distributed.transport_observability import (  # noqa: E402
    TRANSPORT_DTYPES,
    TRANSPORT_PRIMITIVES,
    TRANSPORT_RANK_SCHEMA,
    TRANSPORT_RUN_SCHEMA,
    TRANSPORT_WORLD_SIZES,
    FlagOSTransportEvidenceError,
    FlagOSTransportObservation,
    TransportProfilerEvent,
    build_flagos_transport_observability_profile,
    classify_explicit_host_transfer,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _source_revision() -> str:
    declared = os.environ.get("FLAGQUANTUM_SOURCE_REVISION")
    if declared:
        return declared
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError("F6 requires an explicit Torch-FL installation") from exc
    torch = importlib.import_module("torch")
    fq = importlib.import_module("flagquantum")
    if not hasattr(torch, "flagos"):
        raise RuntimeError("Torch-FL did not register torch.flagos")
    return torch_fl, torch, fq


def _log(rank: int, phase: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[flagos-transport rank={rank}] phase={phase}{suffix}", flush=True)


def _rank_path(prefix: Path, rank: int) -> Path:
    return prefix.parent / f"{prefix.name}.rank-{rank}.json"


def _synchronize(torch: Any) -> None:
    torch.flagos.synchronize()


def _prepared_operation(
    torch: Any,
    *,
    primitive: str,
    dtype: Any,
    device: Any,
    rank: int,
    world_size: int,
) -> tuple[Any, Callable[[], tuple[Any, Any]]]:
    dist = torch.distributed
    if primitive == "broadcast":
        value = complex(1.0 if rank == 0 else -1.0, 0.5 if rank == 0 else -0.5)
        input_tensor = torch.tensor([value], dtype=dtype, device=device)
        expected = torch.tensor([1.0 + 0.5j], dtype=dtype, device=device)

        def operation() -> tuple[Any, Any]:
            dist.broadcast(input_tensor, src=0)
            return input_tensor, expected

        return input_tensor, operation

    if primitive == "all_reduce":
        input_tensor = torch.tensor(
            [complex(rank + 1, rank + 0.5)], dtype=dtype, device=device
        )
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

        def operation() -> tuple[Any, Any]:
            dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM)
            return input_tensor, expected

        return input_tensor, operation

    if primitive == "all_gather_into_tensor":
        input_tensor = torch.tensor(
            [complex(rank + 1, -(rank + 1))], dtype=dtype, device=device
        )
        output = torch.empty(world_size, dtype=dtype, device=device)
        expected = torch.tensor(
            [complex(index + 1, -(index + 1)) for index in range(world_size)],
            dtype=dtype,
            device=device,
        )

        def operation() -> tuple[Any, Any]:
            dist.all_gather_into_tensor(output, input_tensor)
            return output, expected

        return input_tensor, operation

    if primitive == "isend_irecv":
        send_peer = (rank + 1) % world_size
        receive_peer = (rank - 1) % world_size
        input_tensor = torch.tensor(
            [complex(rank + 1, rank + 0.25)], dtype=dtype, device=device
        )
        output = torch.empty_like(input_tensor)
        expected = torch.tensor(
            [complex(receive_peer + 1, receive_peer + 0.25)],
            dtype=dtype,
            device=device,
        )

        def operation() -> tuple[Any, Any]:
            requests = dist.batch_isend_irecv(
                [
                    dist.P2POp(dist.isend, input_tensor, send_peer),
                    dist.P2POp(dist.irecv, output, receive_peer),
                ]
            )
            for request in requests:
                request.wait()
            return output, expected

        return input_tensor, operation

    raise ValueError(f"unsupported F6 primitive {primitive!r}")


def _event_time(event: Any, primary: str, fallback: str) -> float:
    value = getattr(event, primary, None)
    if value is None:
        value = getattr(event, fallback, 0.0)
    return max(0.0, float(value or 0.0))


def _profiler_events(profiler: Any) -> tuple[TransportProfilerEvent, ...]:
    events = []
    for event in profiler.key_averages():
        name = str(event.key)
        events.append(
            TransportProfilerEvent(
                name=name,
                count=max(1, int(event.count)),
                self_cpu_time_us=_event_time(
                    event, "self_cpu_time_total", "cpu_time_total"
                ),
                self_device_time_us=_event_time(
                    event, "self_device_time_total", "self_cuda_time_total"
                ),
                host_transfer_direction=classify_explicit_host_transfer(name),
            )
        )
    return tuple(sorted(events, key=lambda event: event.name))


def _observe(
    torch: Any,
    *,
    primitive: str,
    dtype_name: str,
    device: Any,
    rank: int,
    world_size: int,
) -> FlagOSTransportObservation:
    dtype = getattr(torch, dtype_name)
    input_tensor, operation = _prepared_operation(
        torch,
        primitive=primitive,
        dtype=dtype,
        device=device,
        rank=rank,
        world_size=world_size,
    )
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.version.cuda is not None and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    activity_names = tuple(str(activity.name) for activity in activities)
    profiler_error = None
    profiler = None
    torch.distributed.barrier()
    _synchronize(torch)
    started = time.perf_counter()
    try:
        with torch.profiler.profile(activities=activities) as profiler:
            actual, expected = operation()
            _synchronize(torch)
    except Exception as exc:  # noqa: BLE001 - the artifact retains profiler failure
        profiler_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(
            f"profiler failed for {primitive}/{dtype_name}: {profiler_error}"
        ) from exc
    elapsed = time.perf_counter() - started
    if actual.device.type != "flagos" or input_tensor.device.type != "flagos":
        raise AssertionError("collective payload escaped the logical FlagOS device")
    error = float(torch.max(torch.abs(actual.detach() - expected.detach())).item())
    tolerance = 1e-5 if dtype_name == "complex64" else 1e-12
    events = _profiler_events(profiler)
    return FlagOSTransportObservation(
        primitive=primitive,
        dtype=dtype_name,
        passed=error <= tolerance,
        max_abs_error=error,
        elapsed_seconds=elapsed,
        payload_bytes=int(actual.numel() * actual.element_size()),
        input_device_type=input_tensor.device.type,
        output_device_type=actual.device.type,
        profiler_available=True,
        profiler_activities=activity_names,
        profiler_events=events,
        profiler_error=profiler_error,
    )


def _worker(*, output_prefix: Path, timeout_seconds: float) -> int:
    torch_fl, torch, fq = _load_runtime()
    context = fq.init_torch_distributed(
        backend="flagos", device="flagos", timeout_seconds=timeout_seconds
    )
    try:
        if context.node_count != 1 or context.local_world_size != context.world_size:
            raise RuntimeError("F6 is restricted to one complete node")
        rank = context.rank
        _log(rank, "initialized", f"world_size={context.world_size}")
        observations = []
        for dtype_name in TRANSPORT_DTYPES:
            for primitive in TRANSPORT_PRIMITIVES:
                _log(rank, "collective_start", f"{primitive}/{dtype_name}")
                observation = _observe(
                    torch,
                    primitive=primitive,
                    dtype_name=dtype_name,
                    device=context.device,
                    rank=rank,
                    world_size=context.world_size,
                )
                observations.append(observation.to_dict())
                _log(
                    rank,
                    "collective_done",
                    f"{primitive}/{dtype_name} error={observation.max_abs_error:.3e}",
                )
        device_name_getter = getattr(torch.flagos, "get_device_name", None)
        payload = {
            "schema": TRANSPORT_RANK_SCHEMA,
            "rank": rank,
            "local_rank": context.local_rank,
            "world_size": context.world_size,
            "local_world_size": context.local_world_size,
            "node_count": context.node_count,
            "device": str(context.device),
            "device_type": context.device.type,
            "physical_device_name": (
                str(device_name_getter(context.device.index))
                if device_name_getter is not None
                else "unknown"
            ),
            "observations": observations,
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torch_fl": str(getattr(torch_fl, "__version__", "unknown")),
                "torch_cuda_runtime": str(torch.version.cuda),
                "hostname": socket.gethostname(),
            },
        }
        _write_json(_rank_path(output_prefix, rank), payload)
        return 0 if all(item["passed"] for item in observations) else 2
    finally:
        fq.destroy_torch_distributed()


def _run_payload(
    *, world_size: int, ranks: list[dict[str, Any]], source_revision: str
) -> dict[str, Any]:
    complete = len(ranks) == world_size
    passed = complete and all(
        all(observation.get("passed") for observation in rank["observations"])
        for rank in ranks
    )
    explicit_transfers = [
        {
            "rank": rank["rank"],
            "primitive": observation["primitive"],
            "dtype": observation["dtype"],
            **event,
        }
        for rank in ranks
        for observation in rank["observations"]
        for event in observation["explicit_host_transfer_events"]
    ]
    return {
        "schema": TRANSPORT_RUN_SCHEMA,
        "status": "passed" if passed else "failed",
        "world_size": world_size,
        "local_world_size": world_size,
        "node_count": 1,
        "outer_backend": "flagos",
        "inner_communication_route": "unattributed",
        "flagcx_route_verified": False,
        "host_staging_observed": True if explicit_transfers else None,
        "no_host_staging_certified": False,
        "communication_claim_allowed": False,
        "scalability_claim_allowed": False,
        "production_support_claim_allowed": False,
        "release_gate_allowed": False,
        "ranks": ranks,
        "explicit_host_transfer_events": explicit_transfers,
        "source_revision": source_revision,
        "torch_fl_source_revision": os.environ.get(
            "TORCH_FL_SOURCE_REVISION", "unavailable"
        ),
    }


def _controller(*, output: Path, timeout_seconds: float) -> int:
    _load_runtime()
    source_revision = _source_revision()
    runs = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="flagquantum-flagos-transport-") as temp:
        temp_path = Path(temp)
        for world_size in TRANSPORT_WORLD_SIZES:
            prefix = temp_path / f"world-{world_size}"
            command = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={world_size}",
                str(Path(__file__).resolve()),
                "--worker",
                "--output",
                str(prefix),
                "--timeout-seconds",
                str(timeout_seconds),
            ]
            print(
                f"[flagos-transport] phase=world_size_start value={world_size}",
                flush=True,
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=False,
                    timeout=timeout_seconds * 4,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"world_size_{world_size}_timeout")
                continue
            ranks = []
            for rank in range(world_size):
                path = _rank_path(prefix, rank)
                if path.exists():
                    ranks.append(json.loads(path.read_text(encoding="utf-8")))
            if completed.returncode != 0:
                failures.append(f"world_size_{world_size}_exit_{completed.returncode}")
            runs.append(
                _run_payload(
                    world_size=world_size,
                    ranks=ranks,
                    source_revision=source_revision,
                )
            )
            print(
                f"[flagos-transport] phase=world_size_done value={world_size}",
                flush=True,
            )
    if failures:
        _write_json(
            output,
            {
                "schema": "flagquantum_flagos_transport_observability_f6_v1",
                "status": "failed",
                "runs": runs,
                "worker_failures": failures,
                "release_gate_allowed": False,
            },
        )
        return 2
    try:
        profile = build_flagos_transport_observability_profile(runs)
    except FlagOSTransportEvidenceError as exc:
        _write_json(
            output,
            {
                "schema": "flagquantum_flagos_transport_observability_f6_v1",
                "status": "failed",
                "runs": runs,
                "validation_error": str(exc),
                "release_gate_allowed": False,
            },
        )
        return 2
    payload = profile.to_dict()
    _write_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return _worker(
            output_prefix=args.output,
            timeout_seconds=args.timeout_seconds,
        )
    return _controller(output=args.output, timeout_seconds=args.timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
