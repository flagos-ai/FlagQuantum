#!/usr/bin/env python
"""Reduce FlagOS statevector reverse errors to gates and tensor primitives."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_runtime() -> tuple[Any, Any, Any]:
    try:
        torch_fl = importlib.import_module("torch_fl")
    except ImportError as exc:
        raise RuntimeError("FlagOS reverse diagnosis requires Torch-FL") from exc
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


def _device_tensor(torch: Any, value: Any, *, dtype: Any, device: Any) -> Any:
    source = torch.tensor(value, dtype=dtype)
    target = torch.empty(source.shape, dtype=dtype, device=device)
    target.copy_(source)
    if target.dtype != dtype:
        raise RuntimeError(
            f"FlagOS dtype narrowing: requested={dtype} got={target.dtype}"
        )
    return target


def _parameter(torch: Any, value: float, *, dtype: Any, device: Any) -> Any:
    return _device_tensor(torch, value, dtype=dtype, device=device).requires_grad_()


def _max_error(torch: Any, actual: Any, expected: Any) -> float:
    actual_cpu = (
        actual.detach()
        .cpu()
        .to(torch.complex128 if actual.is_complex() else torch.float64)
    )
    expected_cpu = expected.detach().cpu().to(actual_cpu.dtype)
    return float(torch.max(torch.abs(actual_cpu - expected_cpu)).item())


def _primitive_cases(
    torch: Any, *, dtype_name: str, device: Any
) -> list[dict[str, Any]]:
    complex_dtype = getattr(torch, dtype_name)
    source_a = torch.tensor(
        [[0.25 + 0.5j, -0.75 + 0.125j], [0.33 - 0.2j, -0.4 - 0.6j]],
        dtype=torch.complex128,
    ).to(complex_dtype)
    source_b = torch.tensor(
        [[-0.1 + 0.7j, 0.2 - 0.3j], [0.8 + 0.05j, -0.9 + 0.4j]],
        dtype=torch.complex128,
    ).to(complex_dtype)
    a = _device_tensor(torch, source_a, dtype=complex_dtype, device=device)
    b = _device_tensor(torch, source_b, dtype=complex_dtype, device=device)
    operations: tuple[tuple[str, Callable[[Any, Any], Any]], ...] = (
        ("conj", lambda x, y: torch.conj(x)),
        ("matrix_hermitian", lambda x, y: x.mH),
        ("matrix_multiply", lambda x, y: x @ y),
        ("conjugate_inner_sum", lambda x, y: torch.sum(torch.conj(x) * y)),
        (
            "conjugate_inner_real",
            lambda x, y: torch.real(torch.sum(torch.conj(x) * y)),
        ),
        ("absolute_square", lambda x, y: x.abs().square()),
    )
    tolerance = 3e-5 if dtype_name == "complex64" else 3e-12
    cases = []
    for name, operation in operations:
        expected = operation(source_a, source_b)
        actual = operation(a, b)
        error = _max_error(torch, actual, expected)
        cases.append(
            {
                "kind": "primitive",
                "name": name,
                "dtype": dtype_name,
                "passed": error <= tolerance,
                "max_abs_error": error,
                "tolerance": tolerance,
                "actual_dtype": str(actual.dtype),
                "device_type": actual.device.type,
            }
        )
    return cases


def _build_circuit(
    fq: Any,
    torch: Any,
    *,
    name: str,
    complex_dtype: Any,
    real_dtype: Any,
    device: Any,
) -> tuple[Any, tuple[Any, ...], int]:
    theta = _parameter(torch, 0.23, dtype=real_dtype, device=device)
    phi = _parameter(torch, -0.37, dtype=real_dtype, device=device)
    if name == "ry":
        return (
            fq.Circuit(1, dtype=complex_dtype, device=device).ry(0, theta),
            (theta,),
            0,
        )
    if name == "rz_interference":
        circuit = (
            fq.Circuit(1, dtype=complex_dtype, device=device).h(0).rz(0, theta).h(0)
        )
        return circuit, (theta,), 0
    if name == "rxx":
        circuit = fq.Circuit(2, dtype=complex_dtype, device=device).h(0).rxx(0, 1, phi)
        return circuit, (phi,), 1
    if name == "crx":
        circuit = fq.Circuit(2, dtype=complex_dtype, device=device).h(0).crx(0, 1, phi)
        return circuit, (phi,), 1
    if name == "composite":
        circuit = fq.Circuit(4, dtype=complex_dtype, device=device)
        circuit.h(0).ry(3, theta).rxx(0, 3, phi).rz(1, theta).crx(0, 3, phi)
        return circuit, (theta, phi), 3
    raise ValueError(f"unknown reverse case {name!r}")


def _reverse_case(
    fq: Any, torch: Any, *, name: str, dtype_name: str, device: Any
) -> dict[str, Any]:
    from flagquantum.runtime.backends.statevector.reverse import (
        StatevectorCheckpointPolicy,
        execute_torch_distributed_statevector_reverse,
    )

    complex_dtype = getattr(torch, dtype_name)
    real_dtype = torch.float32 if dtype_name == "complex64" else torch.float64
    reference, reference_parameters, observable = _build_circuit(
        fq,
        torch,
        name=name,
        complex_dtype=torch.complex128,
        real_dtype=torch.float64,
        device=torch.device("cpu"),
    )
    expected_value = reference.expectation_z(observable).sum()
    expected_gradients = torch.autograd.grad(expected_value, reference_parameters)
    circuit, parameters, observable = _build_circuit(
        fq,
        torch,
        name=name,
        complex_dtype=complex_dtype,
        real_dtype=real_dtype,
        device=device,
    )
    result = execute_torch_distributed_statevector_reverse(
        circuit,
        observable_wire=observable,
        checkpoint_policy=StatevectorCheckpointPolicy(strategy="interval", interval=3),
        device=device,
    )
    result.backward()
    actual_gradients = tuple(parameter.grad for parameter in parameters)
    value_error = abs(
        float(result.value.detach().cpu()) - float(expected_value.detach())
    )
    gradient_errors = tuple(
        abs(float(actual.detach().cpu()) - float(expected.detach()))
        for actual, expected in zip(actual_gradients, expected_gradients)
    )
    tolerance = 6e-5 if dtype_name == "complex64" else 3e-10
    summary = result.summary()
    return {
        "kind": "reverse",
        "name": name,
        "dtype": dtype_name,
        "passed": value_error <= tolerance and max(gradient_errors) <= tolerance,
        "value_max_abs_error": value_error,
        "gradient_max_abs_error": max(gradient_errors),
        "gradient_errors": list(gradient_errors),
        "expected_gradients": [float(item.detach()) for item in expected_gradients],
        "actual_gradients": [float(item.detach().cpu()) for item in actual_gradients],
        "tolerance": tolerance,
        "device_type": result.value.device.type,
        "analytic_rotation_derivative_count": summary[
            "analytic_rotation_derivative_count"
        ],
        "backward_uses_full_state_replay": summary["backward_uses_full_state_replay"],
        "full_state_materialization": summary["full_state_materialization"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch_fl, torch, fq = _load_runtime()
    device = torch.device("flagos:0")
    torch.flagos.set_device(0)
    cases = []
    for dtype_name in ("complex64", "complex128"):
        cases.extend(_primitive_cases(torch, dtype_name=dtype_name, device=device))
        for name in ("ry", "rz_interference", "rxx", "crx", "composite"):
            cases.append(
                _reverse_case(
                    fq, torch, name=name, dtype_name=dtype_name, device=device
                )
            )
    primitive_failures = [
        item["name"]
        for item in cases
        if item["kind"] == "primitive" and not item["passed"]
    ]
    reverse_failures = [
        item["name"]
        for item in cases
        if item["kind"] == "reverse" and not item["passed"]
    ]
    payload = {
        "schema": "flagquantum_flagos_statevector_reverse_differential_v1",
        "status": (
            "passed" if not primitive_failures and not reverse_failures else "failed"
        ),
        "source_revision": _source_revision(),
        "torch_fl_source_revision": os.environ.get(
            "TORCH_FL_SOURCE_REVISION", "unavailable"
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_fl": str(getattr(torch_fl, "__version__", "unknown")),
            "torch_cuda_runtime": str(torch.version.cuda),
            "logical_device": "flagos:0",
        },
        "cases": cases,
        "primitive_failures": sorted(set(primitive_failures)),
        "reverse_failures": sorted(set(reverse_failures)),
        "flagcx_route_verified": False,
        "release_gate_allowed": False,
    }
    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
