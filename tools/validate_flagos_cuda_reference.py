#!/usr/bin/env python
"""Validate FlagQuantum's ``flagos:0`` path on Torch-FL's CUDA backend.

Torch-FL must be imported before PyTorch for its CUDA distribution, so this
validator intentionally runs as a fresh process and controls import order.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENVIRONMENT_LOCK = ROOT / "ci" / "flagos_cuda_reference.lock.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(*args: str) -> str:
    try:
        return subprocess.run(
            ("git", *args),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _source_provenance() -> dict[str, Any]:
    status = _git_output("status", "--porcelain")
    revision = _git_output("rev-parse", "HEAD")
    if revision == "unavailable":
        revision = os.environ.get("FLAGQUANTUM_SOURCE_REVISION", "unavailable")
    if status == "unavailable":
        dirty_value = os.environ.get("FLAGQUANTUM_SOURCE_TREE_DIRTY")
        tree_dirty = None if dirty_value is None else dirty_value == "1"
    else:
        tree_dirty = bool(status)
    return {
        "revision": revision,
        "tree_dirty": tree_dirty,
    }


def _require_locked_environment(
    *, environment_lock: dict[str, Any], payload: dict[str, Any]
) -> None:
    torch_lock = environment_lock["torch"]
    torch_fl_lock = environment_lock["torch_fl"]
    expected = {
        "python": environment_lock["python"],
        "torch": torch_lock["package"],
        "torch_distribution": torch_lock["distribution"],
        "torch_cuda_runtime": torch_lock["cuda_runtime"],
        "torch_fl_package": torch_fl_lock["package"],
        "torch_fl_commit": torch_fl_lock["commit"],
        "torch_fl_build": torch_fl_lock["build"],
        "container_image": environment_lock["container_image"],
    }
    actual = {
        "python": platform.python_version(),
        "torch": payload["torch_version"],
        "torch_distribution": payload["torch_distribution"],
        "torch_cuda_runtime": payload["torch_cuda_runtime"],
        "torch_fl_package": payload["torch_fl_package"],
        "torch_fl_commit": os.environ.get("FLAGQUANTUM_TORCH_FL_COMMIT"),
        "torch_fl_build": _declared_torch_fl_build(),
        "container_image": os.environ.get("FLAGQUANTUM_FLAGOS_CONTAINER_IMAGE"),
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, got {actual[name]!r}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise RuntimeError("environment lock mismatch; " + "; ".join(mismatches))
    if not os.environ.get("FLAGQUANTUM_ACCELERATOR_MODEL"):
        raise RuntimeError(
            "environment lock mismatch; physical accelerator identity is missing"
        )
    expected_count = environment_lock["runtime_contract"]["visible_device_count"]
    if payload["device_count"] != expected_count:
        raise RuntimeError(
            "environment lock mismatch; visible_device_count: "
            f"expected {expected_count!r}, got {payload['device_count']!r}"
        )


def _declared_torch_fl_build() -> dict[str, Any] | None:
    encoded = os.environ.get("FLAGQUANTUM_TORCH_FL_BUILD")
    if encoded is None:
        return None
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _torch_distribution(torch_version: str) -> str:
    return "cuda_enabled" if "+cu" in torch_version else "cpu_control"


def _assert_close(torch: Any, actual: Any, expected: Any, *, atol: float) -> None:
    if not torch.allclose(actual.detach().cpu(), expected.detach().cpu(), atol=atol):
        delta = torch.max(torch.abs(actual.detach().cpu() - expected.detach().cpu()))
        raise AssertionError(
            f"FlagOS CUDA-reference mismatch; max error={delta.item()}"
        )


def _depth_circuit(fq: Any, *, depth: int, device: Any, dtype: Any) -> Any:
    circuit = fq.Circuit(3, device=device, dtype=dtype)
    circuit.h(0)
    for layer in range(depth):
        angle = (layer + 1) * 0.017
        circuit.rx(layer % 3, theta=angle)
        circuit.rz((layer + 1) % 3, theta=-0.7 * angle)
        circuit.cx(layer % 3, (layer + 1) % 3)
    return circuit


def _state_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    candidate = actual.detach().cpu().to(torch.complex128).reshape(-1)
    reference = expected.detach().cpu().to(torch.complex128).reshape(-1)
    overlap = torch.abs(torch.vdot(reference, candidate)).square()
    denominator = (
        torch.linalg.vector_norm(reference).square()
        * torch.linalg.vector_norm(candidate).square()
    )
    return {
        "max_abs_error": float(torch.max(torch.abs(candidate - reference)).item()),
        "norm_drift": abs(float(torch.linalg.vector_norm(candidate).item()) - 1.0),
        "state_infidelity": max(0.0, 1.0 - float((overlap / denominator).real.item())),
    }


def validate(
    *, device_name: str, atol: float, dtypes: tuple[str, ...], depths: tuple[int, ...]
) -> dict[str, Any]:
    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError(
            "Torch-FL is required. Install its CUDA backend in this environment."
        ) from exc

    # Import order is part of the Torch-FL CUDA backend contract.
    torch = importlib.import_module("torch")
    fq = importlib.import_module("flagquantum")
    fqb = importlib.import_module("flagquantum.backends")
    platforms = importlib.import_module("flagquantum.runtime.platforms")

    if not hasattr(torch, "flagos"):
        raise RuntimeError("Torch-FL imported without registering torch.flagos")
    if not torch.flagos.is_available():
        raise RuntimeError(
            "torch.flagos is registered but no CUDA-backed device is available"
        )

    device = fqb.resolve_device(device_name)
    if device.type != "flagos":
        raise AssertionError(f"expected flagos device, got {device}")

    platform = platforms.get_platform_runtime("flagos")
    discovered = platform.discover()
    if not discovered or discovered[0].device.type != "flagos":
        raise AssertionError("PlatformRuntime did not discover a flagos device")
    stream = platform.stream(device)
    event = platform.event(device)
    rng_state = platform.rng_state(device)
    platform.restore_rng_state(device, rng_state)

    validation_matrix = []
    first_preflight = None
    first_state = None
    for dtype_name in dtypes:
        dtype = getattr(torch, dtype_name)
        left = torch.tensor(
            [[[1.0 + 0.5j, -0.25j], [0.75, -1.0 + 0.25j]]],
            dtype=dtype,
            device=device,
        )
        right = torch.tensor(
            [[[0.5, 1.0j], [-0.5j, 0.25]]],
            dtype=dtype,
            device=device,
        )
        product = torch.bmm(left, right)
        _assert_close(
            torch,
            product,
            torch.bmm(left.cpu(), right.cpu()),
            atol=atol if dtype_name == "complex64" else 1e-11,
        )
        if product.device.type != "flagos":
            raise AssertionError("operator result escaped the logical flagos device")

        for depth in depths:
            reference = _depth_circuit(
                fq, depth=depth, device="cpu", dtype=torch.complex128
            )
            expected_state = reference.state()
            candidate = _depth_circuit(fq, depth=depth, device=device, dtype=dtype)
            actual_state, execution_plan = fqb.run_native(
                candidate,
                mode="statevector",
                device=device,
                return_plan=True,
            )
            if actual_state.device.type != "flagos":
                raise AssertionError(
                    "FlagQuantum statevector escaped the logical flagos device"
                )
            routing = dict(execution_plan.routing_plan or {})
            operator_preflight = routing.get("operator_preflight")
            numerical_validation = routing.get("numerical_validation")
            if not operator_preflight or not operator_preflight["supported"]:
                raise AssertionError(
                    "statevector_local_p0 preflight evidence is missing"
                )
            if not numerical_validation or not numerical_validation["passed"]:
                raise AssertionError("statevector numerical certification is missing")
            validation_matrix.append(
                {
                    "dtype": dtype_name,
                    "depth": depth,
                    "state_metrics": _state_metrics(
                        torch, actual_state, expected_state
                    ),
                    "numerical_validation": numerical_validation,
                    "accuracy_requirement_hash": routing["accuracy_requirement_hash"],
                    "precision_plan_hash": routing["precision_plan_hash"],
                }
            )
            first_preflight = first_preflight or operator_preflight
            first_state = first_state if first_state is not None else actual_state

    assert first_preflight is not None and first_state is not None

    event.record()
    platform.synchronize(device)
    event.synchronize()
    memory = platform.memory_snapshot(device)
    identity = platform.identity()
    return {
        "status": "passed",
        "validation_scope": "flagos_cuda_reference",
        "device": str(device),
        "device_name": discovered[0].name,
        "device_count": len(discovered),
        "torch_version": str(torch.__version__),
        "torch_fl_version": str(getattr(torch_fl, "__version__", "unknown")),
        "torch_fl_package": importlib.metadata.version("torch-fl"),
        "torch_distribution": _torch_distribution(str(torch.__version__)),
        "torch_cuda_runtime": str(torch.version.cuda),
        "platform_provider": identity.provider,
        "stream_type": type(stream).__name__,
        "event_type": type(event).__name__,
        "memory_allocated_bytes": memory.allocated_bytes,
        "memory_reserved_bytes": memory.reserved_bytes,
        "statevector_dtype": str(first_state.dtype),
        "operator_profile": first_preflight["profile"],
        "operator_profile_hash": first_preflight["profile_hash"],
        "operator_evidence_count": len(first_preflight["evidence_ids"]),
        "validated_dtypes": list(dtypes),
        "validated_depths": list(depths),
        "validation_matrix": validation_matrix,
        "atol": atol,
        "hardware_certification": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="flagos:0")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--dtypes", default="complex64,complex128")
    parser.add_argument("--depths", default="8,32,128")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--environment-lock", type=Path, default=DEFAULT_ENVIRONMENT_LOCK
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="fail unless runtime and provisioner identities match the lock",
    )
    args = parser.parse_args()
    dtypes = tuple(item.strip() for item in args.dtypes.split(",") if item.strip())
    depths = tuple(int(item) for item in args.depths.split(",") if item.strip())
    environment_lock = json.loads(args.environment_lock.read_text(encoding="utf-8"))
    payload = validate(
        device_name=args.device,
        atol=args.atol,
        dtypes=dtypes,
        depths=depths,
    )
    if args.strict_environment:
        _require_locked_environment(environment_lock=environment_lock, payload=payload)
    payload.update(
        {
            "schema": "flagquantum_flagos_cuda_reference_evidence_v2",
            "artifact_class": "development_reference",
            "distribution_semantics": "single_device_fast_path",
            "scalability_claim_allowed": False,
            "source": _source_provenance(),
            "environment_lock": {
                "path": (
                    str(args.environment_lock.relative_to(ROOT))
                    if args.environment_lock.is_relative_to(ROOT)
                    else str(args.environment_lock)
                ),
                "sha256": _sha256(args.environment_lock),
                "schema": environment_lock.get("schema"),
            },
            "environment": {
                "physical_accelerator": os.environ.get("FLAGQUANTUM_ACCELERATOR_MODEL"),
                "container_image": os.environ.get("FLAGQUANTUM_FLAGOS_CONTAINER_IMAGE"),
                "python": platform.python_version(),
                "torch_package": payload["torch_version"],
                "torch_distribution": payload["torch_distribution"],
                "torch_cuda_runtime": payload["torch_cuda_runtime"],
                "torch_fl_package": payload["torch_fl_package"],
                "torch_fl_commit": os.environ.get("FLAGQUANTUM_TORCH_FL_COMMIT"),
                "torch_fl_build": _declared_torch_fl_build(),
            },
            "host": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "claim_blockers": [
                "domestic_accelerator_hardware_not_exercised",
                "route_fallback_absence_not_certified",
                "production_performance_not_measured",
            ],
        }
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
