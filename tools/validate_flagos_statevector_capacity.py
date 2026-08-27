#!/usr/bin/env python
"""Measure matched single, replicated, and sharded FlagOS capacity evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.runtime.distributed.capacity_profile import (  # noqa: E402
    CAPACITY_RUN_SCHEMA,
    build_flagos_statevector_capacity_profile,
)

DEFAULT_MANIFEST = ROOT / "benchmarks/manifests/flagos_statevector_capacity_f5.json"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
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
        raise RuntimeError("F5 requires Torch-FL") from exc
    torch = importlib.import_module("torch")
    fq = importlib.import_module("flagquantum")
    if not hasattr(torch, "flagos"):
        raise RuntimeError("Torch-FL did not register torch.flagos")
    return torch_fl, torch, fq


def _manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "flagquantum_flagos_statevector_capacity_manifest_v1":
        raise ValueError("unsupported F5 manifest")
    if payload.get("release_eligible") is not False:
        raise ValueError("F5 development manifest cannot be release eligible")
    workload = payload.get("workload")
    acceptance = payload.get("acceptance")
    if not isinstance(workload, dict) or not isinstance(acceptance, dict):
        raise ValueError("F5 manifest is incomplete")
    if workload.get("dtype") not in {"complex64", "complex128"}:
        raise ValueError("F5 dtype is unsupported")
    if acceptance.get("completion_world_size") not in {2, 4, 8}:
        raise ValueError("F5 completion world size must be 2, 4, or 8")
    return payload


def _device_properties(torch: Any, index: int) -> tuple[str, int]:
    # On the locked CUDA boxing reference, native CUDA is the authoritative
    # physical identity.  Some provider snapshots expose placeholder FlagOS
    # properties (for example an unrelated vendor name and zero memory).
    if torch.version.cuda is not None and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(index)
        return str(properties.name), int(properties.total_memory)
    flagos_getter = getattr(torch.flagos, "get_device_properties", None)
    if flagos_getter is not None:
        try:
            properties = flagos_getter(index)
            return str(properties.name), int(properties.total_memory)
        except (AttributeError, RuntimeError, TypeError):
            pass
    return "unknown", 0


def _reset_peak(torch: Any, index: int) -> None:
    getter = getattr(torch.flagos, "reset_peak_memory_stats", None)
    if getter is not None:
        try:
            getter(index)
            return
        except (RuntimeError, TypeError):
            pass
    if torch.version.cuda is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(index)


def _peak_memory(torch: Any, index: int) -> int:
    getter = getattr(torch.flagos, "max_memory_allocated", None)
    if getter is not None:
        try:
            return int(getter(index))
        except (RuntimeError, TypeError):
            pass
    if torch.version.cuda is not None and torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated(index))
    return 0


def _empty_cache(torch: Any) -> None:
    getter = getattr(torch.flagos, "empty_cache", None)
    if getter is not None:
        try:
            getter()
            return
        except RuntimeError:
            pass
    if torch.version.cuda is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() or type(error).__name__ in {
        "OutOfMemoryError",
        "OutOfMemoryException",
    }


def _circuit(fq: Any, torch: Any, workload: Mapping[str, Any], device: Any) -> Any:
    n_wires = int(workload["n_wires"])
    dtype = getattr(torch, str(workload["dtype"]))
    return (
        fq.Circuit(n_wires, dtype=dtype, device=device)
        .h(0)
        .ry(1, theta=-0.173)
        .rx(n_wires - 1, theta=0.11)
        .cx(0, n_wires - 1)
        .rz(n_wires - 1, theta=-0.07)
        .cx(n_wires - 1, 1)
        .h(2)
        .cx(2, n_wires - 1)
    )


def _rank_path(prefix: Path, rank: int) -> Path:
    return prefix.parent / f"{prefix.name}.rank-{rank}.json"


def _base_rank_record(
    torch: Any, *, rank: int, local_rank: int, device: Any
) -> dict[str, Any]:
    name, total_memory = _device_properties(torch, int(device.index))
    return {
        "rank": rank,
        "local_rank": local_rank,
        "device": str(device),
        "device_type": device.type,
        "physical_device_name": name,
        "total_memory_bytes": total_memory,
        "peak_memory_bytes": 0,
        "status": "runtime_error",
        "oom_observed": False,
        "error": None,
        "local_amplitudes": 0,
        "local_state_bytes": 0,
        "communication_count": 0,
        "communication_bytes": 0,
        "norm_error": None,
        "validation_method": None,
        "tolerance": None,
        "elapsed_seconds": 0.0,
    }


def _run_local(
    torch: Any,
    fq: Any,
    workload: Mapping[str, Any],
    *,
    rank: int,
    local_rank: int,
    device: Any,
) -> dict[str, Any]:
    record = _base_rank_record(torch, rank=rank, local_rank=local_rank, device=device)
    _reset_peak(torch, int(device.index))
    started = time.perf_counter()
    try:
        state = _circuit(fq, torch, workload, device).state()
        torch.flagos.synchronize()
        record.update(
            status="unexpected_completion",
            local_amplitudes=int(state.numel()),
            local_state_bytes=int(state.numel() * state.element_size()),
        )
    except BaseException as error:
        if not _is_oom(error):
            raise
        record.update(
            status="expected_oom",
            oom_observed=True,
            error=f"{type(error).__name__}: {str(error).splitlines()[0]}",
        )
        _empty_cache(torch)
    record["elapsed_seconds"] = time.perf_counter() - started
    record["peak_memory_bytes"] = _peak_memory(torch, int(device.index))
    return record


def _run_sharded(
    torch: Any,
    fq: Any,
    workload: Mapping[str, Any],
    *,
    context: Any,
) -> dict[str, Any]:
    from flagquantum.runtime.backends.statevector import (
        execute_torch_distributed_statevector,
    )

    device = context.device
    record = _base_rank_record(
        torch, rank=context.rank, local_rank=context.local_rank, device=device
    )
    dtype_name = str(workload["dtype"])
    tolerance = 3e-5 if dtype_name == "complex64" else 2e-11
    _reset_peak(torch, int(device.index))
    circuit = _circuit(fq, torch, workload, "cpu")
    started = time.perf_counter()
    try:
        first = execute_torch_distributed_statevector(
            circuit, device=device, dtype=getattr(torch, dtype_name)
        )
        torch.flagos.synchronize()
        amplitudes = first.shard_state.amplitudes
        squared = amplitudes.real * amplitudes.real + amplitudes.imag * amplitudes.imag
        local_norm = torch.sum(squared, dtype=torch.float64).reshape(1)
        gathered = torch.empty(context.world_size, dtype=torch.float64, device=device)
        torch.distributed.all_gather_into_tensor(gathered, local_norm)
        norm_error = abs(float(gathered.detach().cpu().sum().item()) - 1.0)
        local_amplitudes = int(amplitudes.numel())
        local_state_bytes = int(amplitudes.numel() * amplitudes.element_size())
        communication_count = int(first.communication_count)
        communication_bytes = int(first.communication_bytes)
        record.update(
            status="passed",
            local_amplitudes=local_amplitudes,
            local_state_bytes=local_state_bytes,
            communication_count=communication_count,
            communication_bytes=communication_bytes,
            norm_error=norm_error,
            validation_method="single_full_width_forward_with_fp64_global_norm",
            tolerance=tolerance,
        )
    except BaseException as error:
        if not _is_oom(error):
            raise
        record.update(
            status="unexpected_oom",
            oom_observed=True,
            error=f"{type(error).__name__}: {str(error).splitlines()[0]}",
        )
        _empty_cache(torch)
    record["elapsed_seconds"] = time.perf_counter() - started
    record["peak_memory_bytes"] = _peak_memory(torch, int(device.index))
    return record


def _worker(
    *, mode: str, manifest_path: Path, output_prefix: Path, timeout_seconds: float
) -> int:
    _, torch, fq = _load_runtime()
    workload = _manifest(manifest_path)["workload"]
    context = None
    if mode == "single":
        rank = local_rank = 0
        device = torch.device("flagos", 0)
        record = _run_local(
            torch,
            fq,
            workload,
            rank=rank,
            local_rank=local_rank,
            device=device,
        )
    else:
        context = fq.init_torch_distributed(
            backend="flagos", device="flagos", timeout_seconds=timeout_seconds
        )
        rank = context.rank
        if mode == "replicated":
            record = _run_local(
                torch,
                fq,
                workload,
                rank=rank,
                local_rank=context.local_rank,
                device=context.device,
            )
        else:
            record = _run_sharded(torch, fq, workload, context=context)
    _write_json(_rank_path(output_prefix, rank), record)
    if context is not None:
        fq.destroy_torch_distributed()
    expected = "passed" if mode == "sharded" else "expected_oom"
    return 0 if record["status"] == expected else 2


def _run_payload(
    *,
    mode: str,
    workload: Mapping[str, Any],
    workload_sha256: str,
    ranks: list[dict[str, Any]],
    world_size: int,
    source_revision: str,
) -> dict[str, Any]:
    expected_oom = mode != "sharded"
    expected_status = "expected_oom" if expected_oom else "passed"
    status = (
        expected_status
        if all(rank["status"] == expected_status for rank in ranks)
        else "failed"
    )
    return {
        "schema": CAPACITY_RUN_SCHEMA,
        "mode": mode,
        "status": status,
        "expected_outcome": "capacity_failure" if expected_oom else "completion",
        "world_size": world_size,
        "local_world_size": world_size,
        "node_count": 1,
        "logical_device_type": "flagos",
        "outer_backend": "not_applicable" if mode == "single" else "flagos",
        "distribution_semantics": {
            "single": "single_device_fast_path",
            "replicated": "replicated_full_state_per_rank",
            "sharded": "sharded_across_ranks",
        }[mode],
        "full_state_materialization": mode != "sharded",
        "workload": dict(workload),
        "workload_sha256": workload_sha256,
        "ranks": ranks,
        "source_revision": source_revision,
        "torch_fl_source_revision": os.environ.get(
            "TORCH_FL_SOURCE_REVISION", "unavailable"
        ),
        "flagcx_route_verified": False,
        "host_staging_observed": None,
        "communication_claim_allowed": False,
        "scalability_claim_allowed": False,
        "production_support_claim_allowed": False,
        "release_gate_allowed": False,
    }


def _controller(*, manifest_path: Path, output: Path, timeout_seconds: float) -> int:
    _load_runtime()
    manifest = _manifest(manifest_path)
    workload = manifest["workload"]
    workload_sha256 = _canonical_sha256(workload)
    world_size = int(manifest["acceptance"]["completion_world_size"])
    source_revision = _source_revision()
    runs = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="flagquantum-flagos-capacity-") as temp:
        temp_path = Path(temp)
        for mode, ranks_expected in (
            ("single", 1),
            ("replicated", world_size),
            ("sharded", world_size),
        ):
            prefix = temp_path / mode
            command = [sys.executable]
            if mode != "single":
                command.extend(
                    (
                        "-m",
                        "torch.distributed.run",
                        "--standalone",
                        f"--nproc-per-node={world_size}",
                    )
                )
            command.extend(
                (
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--mode",
                    mode,
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(prefix),
                    "--timeout-seconds",
                    str(timeout_seconds),
                )
            )
            try:
                completed = subprocess.run(
                    command, cwd=ROOT, check=False, timeout=timeout_seconds * 4
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{mode}_timeout")
                continue
            rank_records = []
            for rank in range(ranks_expected):
                path = _rank_path(prefix, rank)
                if path.exists():
                    rank_records.append(json.loads(path.read_text(encoding="utf-8")))
            if completed.returncode != 0:
                failures.append(f"{mode}_exit_{completed.returncode}")
            runs.append(
                _run_payload(
                    mode=mode,
                    workload=workload,
                    workload_sha256=workload_sha256,
                    ranks=rank_records,
                    world_size=ranks_expected,
                    source_revision=source_revision,
                )
            )
    if failures:
        _write_json(
            output,
            {
                "schema": "flagquantum_flagos_statevector_capacity_f5_v1",
                "status": "failed",
                "runs": runs,
                "worker_failures": failures,
                "release_gate_allowed": False,
            },
        )
        return 2
    profile = build_flagos_statevector_capacity_profile(runs)
    payload = profile.to_dict()
    _write_json(output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if profile.accepted else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("single", "replicated", "sharded"))
    args = parser.parse_args(argv)
    if args.worker:
        if args.mode is None:
            parser.error("--worker requires --mode")
        return _worker(
            mode=args.mode,
            manifest_path=args.manifest,
            output_prefix=args.output,
            timeout_seconds=args.timeout_seconds,
        )
    if args.mode is not None:
        parser.error("--mode is internal and requires --worker")
    return _controller(
        manifest_path=args.manifest,
        output=args.output,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
