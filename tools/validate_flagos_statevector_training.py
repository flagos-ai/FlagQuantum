#!/usr/bin/env python
"""Validate the fail-closed FlagOS F3 sharded-training ladder."""

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
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _log(rank: int | str, phase: str, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[flagos-training rank={rank}] phase={phase}{suffix}", flush=True)


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError("FlagOS training validation requires Torch-FL") from exc
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _global_max(torch: Any, value: float, device: Any) -> float:
    metric = torch.tensor([value], dtype=torch.float64, device=device)
    torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.MAX)
    return float(metric.item())


def _rank_consistency(torch: Any, values: tuple[float, ...], context: Any) -> float:
    # This provider snapshot corrupts some real-valued diagnostic collectives,
    # while its same-precision complex all-gather is independently conformant.
    # Carry only the bounded parameter scalars as complex128 and compare on CPU.
    source = torch.tensor(values, dtype=torch.complex128)
    local = torch.empty(source.shape, dtype=torch.complex128, device=context.device)
    local.copy_(source)
    gathered = torch.empty(
        context.world_size * local.numel(),
        dtype=torch.complex128,
        device=context.device,
    )
    torch.distributed.all_gather_into_tensor(gathered, local)
    rows = gathered.detach().cpu().real.reshape(context.world_size, local.numel())
    return float(torch.max(torch.abs(rows - rows[0])).item())


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


def _circuit(
    fq: Any,
    torch: Any,
    *,
    n_wires: int,
    device: Any,
    dtype: Any,
    real_dtype: Any,
) -> tuple[Any, tuple[Any, ...]]:
    def parameter(value: float) -> Any:
        # Torch-FL's CUDA reference currently narrows torch.tensor(...,
        # device="flagos") to float32.  Allocation followed by an explicit copy
        # preserves the requested dtype and makes accidental FP64 narrowing an
        # observable validation failure instead of silently testing FP32 twice.
        target = torch.empty((), dtype=real_dtype, device=device)
        target.copy_(torch.tensor(value, dtype=real_dtype))
        return target.requires_grad_()

    theta = parameter(0.23)
    phi = parameter(-0.37)
    circuit = fq.Circuit(n_wires, dtype=dtype, device=device)
    # Match the independently covered ISSUE-042 differential: this includes a
    # repeated parameter and a two-wire gate spanning a shard boundary.
    circuit.h(0).ry(n_wires - 1, theta).rxx(0, n_wires - 1, phi)
    circuit.rz(1, theta).crx(0, n_wires - 1, phi)
    return circuit, (theta, phi)


def _cpu_reference(
    fq: Any, torch: Any, *, n_wires: int, optimizer: str | None, steps: int, lr: float
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    circuit, parameters = _circuit(
        fq,
        torch,
        n_wires=n_wires,
        device=torch.device("cpu"),
        dtype=torch.complex128,
        real_dtype=torch.float64,
    )
    optimizer_obj = None
    if optimizer is not None:
        cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.Adam
        optimizer_obj = cls(parameters, lr=lr)
    losses = []
    gradients = ()
    for _ in range(steps):
        if optimizer_obj is not None:
            optimizer_obj.zero_grad()
        state = circuit.state(refresh=True)
        probabilities = state.abs().square().reshape(-1)
        # The last logical wire is the least-significant basis bit.
        loss = probabilities[::2].sum() - probabilities[1::2].sum()
        loss.backward()
        losses.append(float(loss.detach()))
        gradients = tuple(float(parameter.grad.detach()) for parameter in parameters)
        if optimizer_obj is not None:
            optimizer_obj.step()
    return tuple(losses), gradients, tuple(float(item.detach()) for item in parameters)


def _run_case(
    torch: Any,
    fq: Any,
    *,
    name: str,
    dtype_name: str,
    n_wires: int,
    steps: int,
    lr: float,
    context: Any,
) -> Any:
    from flagquantum.runtime.backends.statevector.reverse import (
        StatevectorCheckpointPolicy,
        execute_torch_distributed_statevector_reverse,
    )
    from flagquantum.runtime.distributed.training_profile import FlagOSTrainingCase

    dtype = getattr(torch, dtype_name)
    real_dtype = torch.float32 if dtype_name == "complex64" else torch.float64
    tolerance = 6e-5 if dtype_name == "complex64" else 3e-10
    circuit, parameters = _circuit(
        fq,
        torch,
        n_wires=n_wires,
        device=context.device,
        dtype=dtype,
        real_dtype=real_dtype,
    )
    probe_losses, reference_gradients, _ = _cpu_reference(
        fq,
        torch,
        n_wires=n_wires,
        optimizer=None,
        steps=1,
        lr=lr,
    )
    reference_losses, _, reference_parameters = _cpu_reference(
        fq,
        torch,
        n_wires=n_wires,
        optimizer=None if name == "gradient_reference" else name.split("_", 1)[0],
        steps=1 if name == "gradient_reference" else steps,
        lr=lr,
    )
    probe = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_wire=n_wires - 1,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy="interval", interval=3),
        device=context.device,
    )
    probe.backward()
    probe_summary = probe.summary()
    actual_gradients = tuple(
        float(parameter.grad.detach().cpu()) for parameter in parameters
    )
    value_error = abs(float(probe.value.detach().cpu()) - probe_losses[0])
    gradient_error = max(
        abs(actual - expected)
        for actual, expected in zip(actual_gradients, reference_gradients)
    )
    rank_error = _rank_consistency(torch, actual_gradients, context)
    parameter_error = 0.0
    communication_count = int(probe_summary["backward_communication_count"])
    communication_bytes = int(probe_summary["backward_communication_bytes"])
    peak_memory = int(probe_summary["local_state_bytes"]) + int(
        probe_summary["peak_backward_scratch_bytes"]
    )
    if name != "gradient_reference":
        for parameter in parameters:
            parameter.grad = None
        optimizer = name.split("_", 1)[0]
        result = fq.train_distributed_statevector(
            circuit,
            steps=steps,
            observable_wire=n_wires - 1,
            optimizer=optimizer,
            lr=lr,
            rematerialization_interval=2,
        )
        summary = result.summary()
        loss_error = max(
            abs(actual - expected)
            for actual, expected in zip(result.losses, reference_losses)
        )
        value_error = max(value_error, loss_error)
        actual_parameters = tuple(float(item.detach().cpu()) for item in parameters)
        parameter_error = max(
            abs(actual - expected)
            for actual, expected in zip(actual_parameters, reference_parameters)
        )
        rank_error = max(
            rank_error, _rank_consistency(torch, actual_parameters, context)
        )
        communication_count += int(summary["communication_events"])
        communication_bytes += int(summary["communication_bytes"])
        peak_memory = max(peak_memory, int(summary["peak_memory_bytes"]))
        if summary["gradient_ownership_semantics"] != "reduced_across_ranks":
            raise AssertionError("training gradients were not reduced across ranks")
        if summary["optimizer_update_semantics"] != "sharded_across_ranks":
            raise AssertionError("optimizer owner routing was not sharded")
    passed = bool(
        probe_summary["parameter_gradient_ready"]
        and probe_summary["backward_distribution_semantics"] == "sharded_across_ranks"
        and probe_summary["gradient_distribution"] == "replicated_after_all_reduce"
        and not probe_summary["backward_uses_full_state_replay"]
        and not probe_summary["full_state_materialization"]
        and probe_summary["rank_ownership"]["full_state_owned"] is False
        and value_error <= tolerance
        and gradient_error <= tolerance
        and parameter_error <= tolerance
        and rank_error <= tolerance
        and communication_count > 0
        and communication_bytes > 0
    )
    _log(
        context.rank,
        "case_complete",
        f"name={name} dtype={dtype_name} value_error={value_error:.3e} "
        f"gradient_error={gradient_error:.3e} parameter_error={parameter_error:.3e} "
        f"rank_error={rank_error:.3e} actual_gradients={actual_gradients} "
        f"reference_gradients={reference_gradients}",
    )
    return FlagOSTrainingCase(
        name=name,
        dtype=dtype_name,
        passed=passed,
        n_wires=n_wires,
        steps=1 if name == "gradient_reference" else steps,
        local_amplitudes=int(probe.local_amplitudes),
        total_amplitudes=int(probe.total_amplitudes),
        value_max_abs_error=value_error,
        gradient_max_abs_error=gradient_error,
        parameter_max_abs_error=parameter_error,
        rank_consistency_error=rank_error,
        tolerance=tolerance,
        communication_count=communication_count,
        communication_bytes=communication_bytes,
        peak_memory_bytes=peak_memory,
        device_type=probe.value.device.type,
    )


def _worker(*, output: Path, timeout_seconds: float, steps: int) -> int:
    torch_fl, torch, fq = _load_runtime()
    from flagquantum.runtime.distributed.training_profile import (
        FLAGOS_TRAINING_CASES,
        FLAGOS_TRAINING_DTYPES,
        FlagOSTrainingRun,
    )

    context = fq.init_torch_distributed(
        backend="flagos", device="flagos", timeout_seconds=timeout_seconds
    )
    try:
        if context.world_size not in {2, 4, 8}:
            raise RuntimeError("F3 worker world size must be 2, 4, or 8")
        if context.node_count != 1 or context.local_world_size != context.world_size:
            raise RuntimeError("F3 is restricted to one complete node")
        # F3 measures training semantics and accuracy, not forward capacity.
        n_wires = max(4, int(math.log2(context.world_size)) + 2)
        cases = []
        for dtype_name in FLAGOS_TRAINING_DTYPES:
            for name in FLAGOS_TRAINING_CASES:
                _log(context.rank, "case_start", f"name={name} dtype={dtype_name}")
                cases.append(
                    _run_case(
                        torch,
                        fq,
                        name=name,
                        dtype_name=dtype_name,
                        n_wires=n_wires,
                        steps=steps,
                        lr=0.02,
                        context=context,
                    )
                )
        placement = _rank_placement(torch, context)
        device_name, _ = _physical_device_identity(torch, context.device.index)
        environment = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_fl": str(getattr(torch_fl, "__version__", "unknown")),
            "torch_cuda_runtime": str(torch.version.cuda),
            "hostname": socket.gethostname(),
            "device_name": device_name,
            "source_revision": _source_revision(),
            "torch_fl_source_revision": os.environ.get(
                "TORCH_FL_SOURCE_REVISION", "unavailable"
            ),
            "reference_materialization": "bounded_per_rank_cpu_complex128_only",
            "full_state_gather": False,
            "rank_consistency_reduction": (
                "complex128_scalar_all_gather_then_cpu_compare"
            ),
        }
        report = FlagOSTrainingRun(
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
            fq.destroy_torch_distributed()


def _controller(
    *, output: Path, world_sizes: tuple[int, ...], timeout_seconds: float, steps: int
) -> int:
    _load_runtime()
    from flagquantum.runtime.distributed.training_profile import build_training_profile

    runs, failures = [], []
    with tempfile.TemporaryDirectory(prefix="flagquantum-flagos-training-") as temp:
        root = Path(temp)
        for world_size in world_sizes:
            run_output = root / f"world-{world_size}.json"
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
                "--steps",
                str(steps),
            )
            _log("controller", "world_start", f"world_size={world_size}")
            try:
                completed = subprocess.run(
                    command, cwd=ROOT, check=False, timeout=timeout_seconds * 8
                )
            except subprocess.TimeoutExpired:
                failures.append(f"world_size_{world_size}_timeout")
                continue
            if run_output.exists():
                runs.append(json.loads(run_output.read_text()))
            if completed.returncode != 0:
                failures.append(f"world_size_{world_size}_exit_{completed.returncode}")
            _log(
                "controller",
                "world_complete",
                f"world_size={world_size} returncode={completed.returncode}",
            )
    profile = build_training_profile(
        runs,
        environment={
            "source_revision": _source_revision(),
            "torch_fl_source_revision": os.environ.get(
                "TORCH_FL_SOURCE_REVISION", "unavailable"
            ),
            "controller_python": platform.python_version(),
            "requested_world_sizes": list(world_sizes),
            "steps": steps,
            "worker_failures": failures,
        },
    )
    payload = profile.to_dict()
    if failures:
        payload["status"] = "failed"
        payload["training_ladder_accepted"] = False
        payload["sharded_backward_profile_accepted"] = False
        payload["sharded_optimizer_profile_accepted"] = False
        payload["blockers"] = list(
            dict.fromkeys((*payload["blockers"], "worker_subprocess_failed"))
        )
    _write_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["training_ladder_accepted"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-sizes", default="2,4,8")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    world_sizes = tuple(int(item) for item in args.world_sizes.split(",") if item)
    if any(item not in {2, 4, 8} for item in world_sizes):
        parser.error("--world-sizes must contain only 2, 4, and 8")
    if args.steps < 2:
        parser.error("--steps must be at least 2")
    if args.worker:
        return _worker(
            output=args.output, timeout_seconds=args.timeout_seconds, steps=args.steps
        )
    return _controller(
        output=args.output,
        world_sizes=world_sizes,
        timeout_seconds=args.timeout_seconds,
        steps=args.steps,
    )


if __name__ == "__main__":
    raise SystemExit(main())
