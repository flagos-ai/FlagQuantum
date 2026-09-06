"""Executable PyTorch probes for packaged quantum operator profiles."""

from __future__ import annotations

from threading import Lock
from typing import Any

import torch

from .capabilities import (
    CapabilityEvidence,
    CapabilityPreflightReport,
    OperatorProfile,
    OperatorRequirement,
    load_operator_profile,
    preflight_operator_profile,
)

_PROBE_CACHE: dict[tuple[str, str, str, str], tuple[CapabilityEvidence, ...]] = {}
_PROBE_CACHE_LOCK = Lock()


def _dtype_name(dtype: str | torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _real_dtype_for(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in {torch.float32, torch.complex64} else torch.float64


def _make_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> torch.Tensor:
    count = 1
    for dimension in shape:
        count *= dimension
    values = (
        [complex((index + 1) / 7.0, (count - index) / 11.0) for index in range(count)]
        if dtype.is_complex
        else [(index + 1) / 7.0 for index in range(count)]
    )
    tensor = torch.tensor(values, dtype=dtype, device=device).reshape(shape)
    return tensor.requires_grad_(requires_grad)


def _execute_probe(
    operator: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    def make(shape: tuple[int, ...]) -> torch.Tensor:
        return _make_tensor(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    if operator == "aten::zeros":
        return torch.zeros((2, 4), dtype=dtype, device=device), ()
    if operator == "aten::ones_like":
        tensor = make((2, 4))
        return torch.ones_like(tensor), (tensor,)
    if operator == "aten::full_like":
        tensor = make((2, 4))
        return torch.full_like(tensor, 0.5), (tensor,)
    if operator == "aten::reshape":
        tensor = make((2, 4))
        return tensor.reshape(1, 2, 4), (tensor,)
    if operator == "aten::permute":
        tensor = make((2, 2, 2))
        return tensor.permute(0, 2, 1), (tensor,)
    if operator == "aten::transpose":
        tensor = make((2, 3))
        return tensor.transpose(0, 1), (tensor,)
    if operator == "aten::expand":
        tensor = make((1, 2, 2))
        return tensor.expand(3, -1, -1), (tensor,)
    if operator == "aten::bmm":
        left = make((2, 2, 3))
        right = make((2, 3, 2))
        return torch.bmm(left, right), (left, right)
    if operator == "aten::matmul":
        left = make((4, 4))
        right = make((4, 4))
        return torch.matmul(left, right), (left, right)
    if operator in {"aten::add", "aten::sub"}:
        left = make((2, 3))
        right = make((2, 3))
        output = left + right if operator == "aten::add" else left - right
        return output, (left, right)
    if operator == "aten::diagonal":
        tensor = make((2, 3, 3))
        return torch.diagonal(tensor, dim1=-2, dim2=-1), (tensor,)
    if operator == "aten::unsqueeze":
        tensor = make((2, 3))
        return tensor.unsqueeze(-1), (tensor,)
    if operator == "aten::mul":
        real_dtype = _real_dtype_for(dtype)
        left = torch.tensor(
            [[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        right = torch.tensor(
            [[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return left * right, (left, right)
    if operator == "aten::reciprocal":
        tensor = make((2, 3)).abs() + 0.5
        return torch.reciprocal(tensor), (tensor,)
    if operator == "aten::rsqrt":
        tensor = make((2, 3)).abs() + 0.5
        return torch.rsqrt(tensor), (tensor,)
    if operator == "aten::round":
        tensor = make((2, 3)) * 3.0 - 1.0
        return torch.round(tensor), (tensor,)
    if operator == "aten::remainder":
        tensor = make((2, 3)) * 7.0 - 3.0
        return torch.remainder(tensor, 4.0), (tensor,)
    if operator in {"aten::eq", "aten::gt"}:
        tensor = make((2, 3))
        output = tensor == 0.5 if operator == "aten::eq" else tensor > 0.5
        return output, (tensor,)
    if operator == "aten::where":
        left = make((2, 3))
        right = -left
        return torch.where(left > 0.5, left, right), (left, right)
    if operator == "aten::unbind":
        tensor = make((2, 3, 2))
        return torch.unbind(tensor, dim=1), (tensor,)
    if operator == "aten::flip":
        tensor = make((2, 3))
        return torch.flip(tensor, dims=(1,)), (tensor,)
    if operator == "aten::stack":
        left = make((2, 3))
        right = make((2, 3))
        return torch.stack((left, right), dim=1), (left, right)
    if operator == "aten::cat":
        left = make((2, 3))
        right = make((2, 3))
        return torch.cat((left, right), dim=1), (left, right)
    if operator == "aten::_to_copy":
        real_dtype = _real_dtype_for(dtype)
        tensor = torch.tensor(
            [[0.25, -0.5, 0.75]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return tensor.to(dtype=dtype), (tensor,)
    if operator == "aten::complex":
        real_dtype = _real_dtype_for(dtype)
        real = torch.tensor(
            [[0.25, -0.5, 0.75]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        imag = torch.tensor(
            [[-0.1, 0.2, -0.3]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return torch.complex(real, imag), (real, imag)
    if operator == "aten::cos":
        real_dtype = _real_dtype_for(dtype)
        tensor = torch.tensor(
            [[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return torch.cos(tensor), (tensor,)
    if operator == "aten::sin":
        real_dtype = _real_dtype_for(dtype)
        tensor = torch.tensor(
            [[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return torch.sin(tensor), (tensor,)
    if operator == "aten::exp":
        tensor = make((2, 3))
        return torch.exp(tensor), (tensor,)
    if operator == "aten::conj":
        tensor = make((2, 3))
        return torch.conj(tensor), (tensor,)
    if operator in {"aten::real", "aten::imag"}:
        real_dtype = _real_dtype_for(dtype)
        real = torch.tensor(
            [[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        imag = torch.tensor(
            [[-0.1, 0.2, -0.3], [0.4, -0.5, 0.6]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        tensor = torch.complex(real, imag)
        output = tensor.real if operator == "aten::real" else tensor.imag
        return output, (real, imag)
    if operator == "aten::neg":
        real_dtype = _real_dtype_for(dtype)
        tensor = torch.tensor(
            [[0.2, -0.3, 0.4], [0.5, -0.6, 0.7]],
            dtype=real_dtype,
            device=device,
            requires_grad=requires_grad,
        )
        return torch.neg(tensor), (tensor,)
    if operator == "aten::abs":
        tensor = make((2, 3))
        return torch.abs(tensor), (tensor,)
    if operator == "aten::sum":
        tensor = make((2, 3))
        return torch.sum(tensor, dim=1), (tensor,)
    if operator == "aten::isfinite":
        tensor = make((2, 3))
        return torch.isfinite(tensor), (tensor,)
    if operator == "aten::all":
        tensor = make((2, 3))
        return torch.all(torch.isfinite(tensor)), (tensor,)
    if operator == "aten::any":
        tensor = make((2, 3))
        return torch.any(tensor > 0), (tensor,)
    raise KeyError(f"no executable probe registered for {operator!r}")


def _output_tensors(output: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)) and all(
        isinstance(item, torch.Tensor) for item in output
    ):
        return tuple(output)
    raise TypeError(f"probe returned unsupported output type: {type(output).__name__}")


def _loss(output: Any) -> torch.Tensor:
    tensors = _output_tensors(output)
    return sum(torch.abs(tensor).square().sum() for tensor in tensors)


def _max_error(actual: Any, expected: Any) -> float:
    actual_tensors = _output_tensors(actual)
    expected_tensors = _output_tensors(expected)
    if len(actual_tensors) != len(expected_tensors):
        return float("inf")
    errors = []
    for left, right in zip(actual_tensors, expected_tensors):
        left_cpu = left.detach().cpu()
        right_cpu = right.detach().cpu()
        if left_cpu.dtype == torch.bool or right_cpu.dtype == torch.bool:
            errors.append(0.0 if torch.equal(left_cpu, right_cpu) else float("inf"))
        else:
            errors.append(float(torch.max(torch.abs(left_cpu - right_cpu)).item()))
    return max(errors)


def _probe_requirement(
    requirement: OperatorRequirement,
    *,
    dtype_name: str,
    device: torch.device,
    provider: str,
    profile_hash: str,
) -> CapabilityEvidence:
    dtype = getattr(torch, dtype_name)
    tolerance = 2e-5 if dtype_name in {"float32", "complex64"} else 1e-11
    forward = False
    backward = False
    details = ""
    try:
        actual, actual_leaves = _execute_probe(
            requirement.operator,
            dtype=dtype,
            device=device,
            requires_grad=requirement.backward,
        )
        expected, expected_leaves = _execute_probe(
            requirement.operator,
            dtype=dtype,
            device=torch.device("cpu"),
            requires_grad=requirement.backward,
        )
        actual_tensors = _output_tensors(actual)
        if any(tensor.device.type != device.type for tensor in actual_tensors):
            raise RuntimeError("result left the requested logical device")
        error = _max_error(actual, expected)
        if error > tolerance:
            raise RuntimeError(f"forward max error {error} exceeds {tolerance}")
        forward = True
        if requirement.backward:
            _loss(actual).backward()
            _loss(expected).backward()
            if any(leaf.grad is None for leaf in actual_leaves):
                raise RuntimeError("backward did not produce every required gradient")
            if any(
                leaf.grad is not None and leaf.grad.device.type != device.type
                for leaf in actual_leaves
            ):
                raise RuntimeError("gradient left the requested logical device")
            for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves):
                assert actual_leaf.grad is not None
                assert expected_leaf.grad is not None
                gradient_error = float(
                    torch.max(
                        torch.abs(
                            actual_leaf.grad.detach().cpu()
                            - expected_leaf.grad.detach().cpu()
                        )
                    ).item()
                )
                if gradient_error > tolerance:
                    raise RuntimeError(
                        f"gradient max error {gradient_error} exceeds {tolerance}"
                    )
            backward = True
        details = f"cpu_reference_max_error={error}; tolerance={tolerance}"
    except Exception as exc:
        details = f"{type(exc).__name__}: {exc}"
    return CapabilityEvidence(
        provider=provider,
        device_type=device.type,
        profile_hash=profile_hash,
        operator=requirement.operator,
        dtype=dtype_name,
        probe_source="runtime_probe",
        passed=forward and (backward or not requirement.backward),
        forward=forward,
        backward=backward,
        deterministic=None,
        details=details,
    )


def probe_operator_profile(
    profile: OperatorProfile,
    *,
    device: str | torch.device,
    dtype: str | torch.dtype,
    provider: str,
    refresh: bool = False,
) -> tuple[CapabilityEvidence, ...]:
    """Execute one dtype slice of a profile against a CPU reference."""

    resolved_device = torch.device(device)
    dtype_name = _dtype_name(dtype)
    if dtype_name not in {"float32", "float64", "complex64", "complex128"}:
        raise ValueError(f"unsupported statevector probe dtype: {dtype_name!r}")
    key = (profile.profile_hash, str(resolved_device), dtype_name, provider)
    with _PROBE_CACHE_LOCK:
        if not refresh and key in _PROBE_CACHE:
            return _PROBE_CACHE[key]
    evidence = tuple(
        _probe_requirement(
            requirement,
            dtype_name=dtype_name,
            device=resolved_device,
            provider=provider,
            profile_hash=profile.profile_hash,
        )
        for requirement in profile.requirements
        if dtype_name in requirement.dtypes
    )
    with _PROBE_CACHE_LOCK:
        _PROBE_CACHE[key] = evidence
    return evidence


def preflight_statevector_local_p0(
    *,
    device: str | torch.device,
    dtype: str | torch.dtype,
    provider: str,
    refresh: bool = False,
) -> CapabilityPreflightReport:
    """Probe and preflight the minimum local statevector workload."""

    profile = load_operator_profile("statevector_local_p0")
    dtype_name = _dtype_name(dtype)
    evidence = probe_operator_profile(
        profile,
        device=device,
        dtype=dtype_name,
        provider=provider,
        refresh=refresh,
    )
    return preflight_operator_profile(
        profile,
        evidence,
        device_type=torch.device(device).type,
        required_dtypes=(dtype_name,),
    )


def _preflight_split_real_imag_profile(
    profile_name: str,
    *,
    device: str | torch.device,
    provider: str,
    refresh: bool,
) -> CapabilityPreflightReport:
    profile = load_operator_profile(profile_name)
    evidence = probe_operator_profile(
        profile,
        device=device,
        dtype="float32",
        provider=provider,
        refresh=refresh,
    )
    return preflight_operator_profile(
        profile,
        evidence,
        device_type=torch.device(device).type,
        required_dtypes=("float32",),
    )


def preflight_split_real_imag_statevector_p0(
    *,
    device: str | torch.device,
    provider: str,
    refresh: bool = False,
) -> CapabilityPreflightReport:
    """Probe the pure-FP32 operator slice used by split statevector P0."""

    return _preflight_split_real_imag_profile(
        "split_real_imag_statevector_p0",
        device=device,
        provider=provider,
        refresh=refresh,
    )


def preflight_split_real_imag_statevector_p1(
    *,
    device: str | torch.device,
    provider: str,
    refresh: bool = False,
) -> CapabilityPreflightReport:
    """Probe P1 FP32 expectation and gradient-supporting operators."""

    return _preflight_split_real_imag_profile(
        "split_real_imag_statevector_p1",
        device=device,
        provider=provider,
        refresh=refresh,
    )


def preflight_split_real_imag_statevector_p2(
    *,
    device: str | torch.device,
    provider: str,
    refresh: bool = False,
) -> CapabilityPreflightReport:
    """Probe the FP32 surface used by selective Double-Single reductions."""

    return _preflight_split_real_imag_profile(
        "split_real_imag_statevector_p2_precision",
        device=device,
        provider=provider,
        refresh=refresh,
    )


def preflight_split_real_imag_statevector_p3(
    *,
    device: str | torch.device,
    provider: str,
    refresh: bool = False,
) -> CapabilityPreflightReport:
    """Probe the FP32 surface used by full Double-Single state evolution."""

    return _preflight_split_real_imag_profile(
        "split_real_imag_statevector_p3_double_single",
        device=device,
        provider=provider,
        refresh=refresh,
    )


def preflight_split_real_imag_statevector_p4(
    *,
    device: str | torch.device,
    provider: str,
    refresh: bool = False,
) -> CapabilityPreflightReport:
    """Probe full-state DS arithmetic plus device-side range reduction."""

    return _preflight_split_real_imag_profile(
        "split_real_imag_statevector_p4_device_double_single",
        device=device,
        provider=provider,
        refresh=refresh,
    )


__all__ = (
    "preflight_split_real_imag_statevector_p0",
    "preflight_split_real_imag_statevector_p1",
    "preflight_split_real_imag_statevector_p2",
    "preflight_split_real_imag_statevector_p3",
    "preflight_split_real_imag_statevector_p4",
    "preflight_statevector_local_p0",
    "probe_operator_profile",
)
