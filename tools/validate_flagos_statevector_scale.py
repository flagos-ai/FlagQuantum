#!/usr/bin/env python
"""Measure a fail-closed 2/4/8-card FlagOS statevector scale ladder.

The default controller launches a fresh ``torchrun`` for each requested world
size.  ``--worker`` is internal and must run under torchrun.  The profile is
development hardware evidence only: it does not infer FlagCX, provider-native
buffer residency, production support, or release-grade scalability.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from flagquantum.runtime.distributed import (
    destroy_torch_distributed,
    init_torch_distributed,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _log(rank: int | str, phase: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[flagos-scale rank={rank}] phase={phase}{suffix}", flush=True)


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError("FlagOS scale validation requires Torch-FL") from exc
    torch = importlib.import_module("torch")
    fq = importlib.import_module("flagquantum")
    if not hasattr(torch, "flagos"):
        raise RuntimeError("Torch-FL imported without registering torch.flagos")
    return torch_fl, torch, fq


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _global_max(torch: Any, value: float, device: Any) -> float:
    metric = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.MAX)
    return float(metric.item())


def _global_norm_error(torch: Any, local_norm: Any, device: Any) -> float:
    """Sum one FP64 norm scalar per rank without provider SUM narrowing.

    The gathered payload is at most eight scalars.  It is not a statevector
    gather and deliberately isolates state accuracy from an unrelated provider
    implementation of real-valued all-reduce SUM.
    """

    metric = local_norm.reshape(1).to(device=device, dtype=torch.float64)
    gathered = torch.empty(
        torch.distributed.get_world_size(), dtype=torch.float64, device=device
    )
    torch.distributed.all_gather_into_tensor(gathered, metric)
    return abs(float(gathered.detach().cpu().sum(dtype=torch.float64).item()) - 1.0)


def _gather_ints(
    torch: Any, value: int, device: Any, world_size: int
) -> tuple[int, ...]:
    local = torch.tensor([value], dtype=torch.int64, device=device)
    gathered = torch.empty(world_size, dtype=torch.int64, device=device)
    torch.distributed.all_gather_into_tensor(gathered, local)
    return tuple(int(item) for item in gathered.detach().cpu().tolist())


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
            "physical_device_name": _physical_device_identity(torch, int(device_index))[
                0
            ],
            "physical_device_uuid": _physical_device_identity(torch, int(device_index))[
                1
            ],
            "ownership": "distinct_amplitude_shard",
        }
        for rank, local_rank, device_index in rows
    )


def _physical_device_identity(torch: Any, index: int) -> tuple[str, str | None]:
    getter = getattr(torch.flagos, "get_device_name", None)
    if getter is not None:
        try:
            return str(getter(index)), None
        except (RuntimeError, TypeError):
            pass
    if torch.version.cuda is not None and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(index)
        uuid = getattr(properties, "uuid", None)
        return str(properties.name), str(uuid) if uuid is not None else None
    return "unknown", None


def _global_indices(torch: Any, result: Any) -> Any:
    shard = result.shard_state
    if shard.global_indices.numel():
        return shard.global_indices.detach().cpu().to(torch.long)
    local = torch.arange(shard.amplitudes.shape[-1], dtype=torch.long)
    if result.plan.distribution == "qubit_address_sharded":
        return (local << len(result.plan.sharded_wires)) | shard.rank
    return local + shard.shard.amplitude_start


def _physical_to_logical_indices(torch: Any, result: Any) -> Any:
    physical_indices = _global_indices(torch, result)
    mapping = tuple(int(item) for item in result.logical_to_physical_wires)
    if mapping == tuple(range(result.plan.n_wires)):
        return physical_indices
    logical_indices = torch.zeros_like(physical_indices)
    for logical_wire, physical_wire in enumerate(mapping):
        source_shift = result.plan.n_wires - physical_wire - 1
        target_shift = result.plan.n_wires - logical_wire - 1
        logical_indices |= ((physical_indices >> source_shift) & 1) << target_shift
    return logical_indices


def _circuit(fq: Any, *, n_wires: int, rank_bits: int, dtype: Any) -> Any:
    circuit = fq.Circuit(n_wires, dtype=dtype).h(0).ry(1, theta=-0.173)
    for offset in range(rank_bits):
        wire = n_wires - 1 - offset
        circuit.rx(wire, theta=0.11 * (offset + 1))
        circuit.cx(0, wire)
        circuit.rz(wire, theta=-0.07 * (offset + 1))
        circuit.cx(wire, 1)
    return circuit.h(2).cx(2, n_wires - 1)


def _device_peak_bytes(torch: Any, result: Any) -> tuple[int, str]:
    getter = getattr(torch.flagos, "max_memory_allocated", None)
    if getter is not None:
        try:
            measured = int(getter(result.shard_state.amplitudes.device.index))
            if measured > 0:
                return measured, "provider_peak_allocator_bytes"
        except (RuntimeError, TypeError):
            pass
    local_state = int(
        result.shard_state.amplitudes.numel()
        * result.shard_state.amplitudes.element_size()
    )
    return (
        local_state
        + int(result.peak_scratch_bytes)
        + int(result.exchange_workspace_reserved_bytes),
        "runtime_accounted_state_scratch_workspace",
    )


def _run_case(
    torch: Any,
    fq: Any,
    *,
    name: str,
    dtype_name: str,
    n_wires: int,
    context: Any,
) -> Any:
    from flagquantum.runtime.backends.statevector import (
        execute_torch_distributed_statevector,
    )
    from flagquantum.runtime.distributed.scale_profile import (
        FlagOSStatevectorScaleCase,
    )

    dtype = getattr(torch, dtype_name)
    tolerance = 3e-5 if dtype_name == "complex64" else 2e-11
    persistent = name == "persistent_layout_reference"
    rank_bits = int(math.log2(context.world_size))
    circuit = _circuit(fq, n_wires=n_wires, rank_bits=rank_bits, dtype=dtype)
    reference = None
    reference_scope = "rank_local_invariant_only"
    if name != "capacity_invariant":
        reference = circuit.state().detach().cpu().to(torch.complex128).reshape(-1)
        reference_scope = "bounded_per_rank_cpu_complex128_reference"

    reset_peak = getattr(torch.flagos, "reset_peak_memory_stats", None)
    if reset_peak is not None:
        try:
            reset_peak(context.device.index)
        except (RuntimeError, TypeError):
            pass
    torch.distributed.barrier()
    torch.flagos.synchronize()
    started = time.perf_counter()
    first = execute_torch_distributed_statevector(
        circuit,
        device=context.device,
        dtype=dtype,
        persistent_wire_layout=persistent,
    )
    torch.flagos.synchronize()
    elapsed = _global_max(torch, time.perf_counter() - started, context.device)
    if first.shard_state.amplitudes.device.type != "flagos":
        raise AssertionError("scale case shard escaped the logical flagos device")

    actual = first.shard_state.amplitudes.detach()
    # Avoid provider implementations of complex abs that may narrow the real
    # reduction even when the state is complex128.  Accumulate the explicit
    # real-valued squared norm in float64 on the logical FlagOS device.
    squared_norm = actual.real * actual.real + actual.imag * actual.imag
    local_norm = torch.sum(squared_norm, dtype=torch.float64)
    norm_error = _global_norm_error(torch, local_norm, context.device)
    max_abs_error = None
    if reference is not None:
        indices = _physical_to_logical_indices(torch, first)
        expected = reference.index_select(0, indices)
        local_error = float(
            torch.max(
                torch.abs(actual.cpu().to(torch.complex128).reshape(-1) - expected)
            ).item()
        )
        max_abs_error = _global_max(torch, local_error, context.device)

    # A second execution checks deterministic rank-local ownership and values.
    second = execute_torch_distributed_statevector(
        circuit,
        device=context.device,
        dtype=dtype,
        persistent_wire_layout=persistent,
    )
    local_determinism = float(
        torch.max(torch.abs(second.shard_state.amplitudes - actual)).item()
    )
    determinism_error = _global_max(torch, local_determinism, context.device)
    local_state_bytes = int(actual.numel() * actual.element_size())
    first_memory, first_memory_method = _device_peak_bytes(torch, first)
    second_memory, second_memory_method = _device_peak_bytes(torch, second)
    local_memory = max(first_memory, second_memory)
    memory_measurement = (
        "provider_peak_allocator_bytes"
        if first_memory_method
        == second_memory_method
        == "provider_peak_allocator_bytes"
        else "runtime_accounted_state_scratch_workspace"
    )
    memory_by_rank = _gather_ints(
        torch, local_memory, context.device, context.world_size
    )
    passed = bool(
        first.plan.world_size == context.world_size
        and actual.numel() * context.world_size == first.plan.total_amplitudes
        and (
            first.distributed_gate_count > 0
            or (persistent and first.wire_layout == "persistent")
        )
        and first.communication_count > 0
        and first.communication_bytes > 0
        and norm_error <= tolerance
        and determinism_error <= tolerance
        and (max_abs_error is None or max_abs_error <= tolerance)
    )
    _log(
        context.rank,
        "case_complete",
        f"name={name} dtype={dtype_name} wires={n_wires} "
        f"reference_error={max_abs_error} norm_error={norm_error:.3e} "
        f"determinism_error={determinism_error:.3e}",
    )
    return FlagOSStatevectorScaleCase(
        name=name,
        dtype=dtype_name,
        passed=passed,
        n_wires=n_wires,
        local_amplitudes=int(actual.numel()),
        total_amplitudes=int(first.plan.total_amplitudes),
        local_state_bytes=local_state_bytes,
        local_memory_bytes_by_rank=memory_by_rank,
        memory_measurement=memory_measurement,
        communication_count=int(first.communication_count),
        communication_bytes=int(first.communication_bytes),
        peak_scratch_bytes=int(first.peak_scratch_bytes),
        distributed_gate_count=int(first.distributed_gate_count),
        elapsed_seconds=elapsed,
        max_abs_error=max_abs_error,
        norm_error=norm_error,
        determinism_error=determinism_error,
        tolerance=tolerance,
        persistent_wire_layout=persistent,
        device_type=actual.device.type,
        reference_scope=reference_scope,
    )


def _worker(*, output: Path, timeout_seconds: float, capacity_wires: int) -> int:
    torch_fl, torch, fq = _load_runtime()
    from flagquantum.runtime.distributed.scale_profile import (
        FLAGOS_SCALE_DTYPES,
        FlagOSStatevectorScaleRun,
    )

    context = init_torch_distributed(
        backend="flagos", device="flagos", timeout_seconds=timeout_seconds
    )
    try:
        if context.world_size not in {2, 4, 8}:
            raise RuntimeError("F2 worker world size must be 2, 4, or 8")
        if context.node_count != 1 or context.local_world_size != context.world_size:
            raise RuntimeError("F2 is restricted to one complete node")
        _log(context.rank, "initialized", f"world_size={context.world_size}")
        rank_bits = int(math.log2(context.world_size))
        reference_wires = max(6, rank_bits + 4)
        cases = []
        for dtype_name in FLAGOS_SCALE_DTYPES:
            for name, n_wires in (
                ("cross_shard_reference", reference_wires),
                ("persistent_layout_reference", reference_wires),
                ("capacity_invariant", capacity_wires),
            ):
                _log(
                    context.rank,
                    "case_start",
                    f"name={name} dtype={dtype_name} wires={n_wires}",
                )
                cases.append(
                    _run_case(
                        torch,
                        fq,
                        name=name,
                        dtype_name=dtype_name,
                        n_wires=n_wires,
                        context=context,
                    )
                )
        placement = _rank_placement(torch, context)
        physical_device_name, _ = _physical_device_identity(torch, context.device.index)
        environment = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_fl": str(getattr(torch_fl, "__version__", "unknown")),
            "torch_cuda_runtime": str(torch.version.cuda),
            "hostname": socket.gethostname(),
            "device_name": physical_device_name,
            "device_identity_source": (
                "torch_cuda_reference_fallback"
                if physical_device_name != "unknown" and torch.version.cuda is not None
                else "torch_flagos_or_unavailable"
            ),
            "source_revision": _source_revision(),
            "torch_fl_source_revision": os.environ.get(
                "TORCH_FL_SOURCE_REVISION", "unavailable"
            ),
            "reference_materialization": "bounded_per_rank_cpu_small_cases_only",
            "capacity_reference_materialization": "forbidden",
            "full_state_gather": False,
            "norm_reduction": "fp64_scalar_all_gather_then_cpu_sum",
        }
        report = FlagOSStatevectorScaleRun(
            world_size=context.world_size,
            local_world_size=context.local_world_size,
            node_count=context.node_count,
            cases=tuple(cases),
            rank_placement=placement,
            environment=environment,
        )
        if context.rank == 0:
            _write_json(output, report.to_dict())
        torch.distributed.barrier()
        return 0 if report.accepted else 1
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            destroy_torch_distributed()


def _controller(
    *,
    output: Path,
    world_sizes: tuple[int, ...],
    timeout_seconds: float,
    capacity_wires: int,
) -> int:
    # Preserve the provider-required import order in the controller too.  The
    # workers repeat this independently in their fresh Python processes.
    _load_runtime()
    from flagquantum.runtime.distributed.scale_profile import build_scale_profile

    runs = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="flagquantum-flagos-scale-") as temp:
        temp_path = Path(temp)
        for world_size in world_sizes:
            run_output = temp_path / f"world-{world_size}.json"
            command = (
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={world_size}",
                str(Path(__file__).resolve()),
                "--worker",
                "--output",
                str(run_output),
                "--timeout-seconds",
                str(timeout_seconds),
                "--capacity-wires",
                str(capacity_wires),
            )
            _log("controller", "world_start", f"world_size={world_size}")
            try:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    check=False,
                    timeout=timeout_seconds * 8,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"world_size_{world_size}_timeout")
                continue
            if run_output.exists():
                runs.append(json.loads(run_output.read_text(encoding="utf-8")))
            if completed.returncode != 0:
                failures.append(f"world_size_{world_size}_exit_{completed.returncode}")
            _log(
                "controller",
                "world_complete",
                f"world_size={world_size} returncode={completed.returncode}",
            )
    profile = build_scale_profile(
        runs,
        environment={
            "source_revision": _source_revision(),
            "torch_fl_source_revision": os.environ.get(
                "TORCH_FL_SOURCE_REVISION", "unavailable"
            ),
            "controller_python": platform.python_version(),
            "requested_world_sizes": list(world_sizes),
            "capacity_wires": capacity_wires,
            "worker_failures": failures,
        },
    )
    payload = profile.to_dict()
    if failures:
        payload["status"] = "failed"
        payload["scale_ladder_accepted"] = False
        payload["statevector_forward_scale_profile_accepted"] = False
        payload["blockers"] = list(
            dict.fromkeys((*payload["blockers"], "worker_subprocess_failed"))
        )
    _write_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["scale_ladder_accepted"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-sizes", default="2,4,8")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--capacity-wires", type=int, default=24)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    world_sizes = tuple(int(item) for item in args.world_sizes.split(",") if item)
    if any(item not in {2, 4, 8} for item in world_sizes):
        parser.error("--world-sizes must contain only 2, 4, and 8")
    if args.capacity_wires < 8:
        parser.error("--capacity-wires must be at least 8")
    if args.worker:
        return _worker(
            output=args.output,
            timeout_seconds=args.timeout_seconds,
            capacity_wires=args.capacity_wires,
        )
    return _controller(
        output=args.output,
        world_sizes=world_sizes,
        timeout_seconds=args.timeout_seconds,
        capacity_wires=args.capacity_wires,
    )


if __name__ == "__main__":
    raise SystemExit(main())
