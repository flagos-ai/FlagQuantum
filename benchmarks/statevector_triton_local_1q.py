"""Measure generic Triton local-1q cold/warm/cache and VJP evidence."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from flagquantum.runtime.backends.statevector.triton import (
    _complex64_local_1q_kernel,
    _complex64_local_1q_vjp_adjoint_kernel,
    apply_complex64_local_1q,
    fused_complex64_local_1q_vjp_adjoint,
)


def _reference_apply(
    state: torch.Tensor, matrix: torch.Tensor, bit: int
) -> torch.Tensor:
    batch, size = state.shape
    paired = state.reshape(batch, size >> (bit + 1), 2, 1 << bit)
    return torch.einsum("ij,bhjw->bhiw", matrix, paired).reshape_as(state)


def _milliseconds(operation, repeats: int) -> list[float]:
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop))
    return values


def _cache_entries(kernel) -> int:
    total = 0
    for device_cache in kernel.device_caches.values():
        total += len(device_cache[0])
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(3107)
    device = torch.device("cuda")
    state = torch.randn(1, 1 << args.qubits, dtype=torch.complex64, device=device)
    adjoint = torch.randn_like(state)
    matrix = torch.randn(2, 2, dtype=torch.complex64, device=device)
    derivative = torch.randn(2, 2, dtype=torch.complex64, device=device)
    bit = min(7, args.qubits - 1)

    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    actual = apply_complex64_local_1q(state, matrix, bit_position=bit)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1e3
    forward_cache_after_cold = _cache_entries(_complex64_local_1q_kernel)

    triton_ms = _milliseconds(
        lambda: apply_complex64_local_1q(state, matrix, bit_position=bit), args.repeats
    )
    pytorch_ms = _milliseconds(
        lambda: _reference_apply(state, matrix, bit), args.repeats
    )
    expected = _reference_apply(state, matrix, bit)
    forward_error = float(torch.max(torch.abs(actual - expected)))

    # Matrix values and local bit positions are runtime data. Changing both must
    # reuse the same compiled specialization for the same shape/dtype family.
    updated_matrix = matrix * torch.exp(torch.tensor(0.17j, device=device))
    for updated_bit in (0, min(3, args.qubits - 1), args.qubits - 1):
        apply_complex64_local_1q(state, updated_matrix, bit_position=updated_bit)
    torch.cuda.synchronize()
    forward_cache_after_updates = _cache_entries(_complex64_local_1q_kernel)

    torch.cuda.synchronize()
    vjp_cold_start = time.perf_counter()
    next_adjoint, gradient = fused_complex64_local_1q_vjp_adjoint(
        state, adjoint, matrix, derivative, bit_position=bit
    )
    torch.cuda.synchronize()
    vjp_cold_ms = (time.perf_counter() - vjp_cold_start) * 1e3
    vjp_cache_after_cold = _cache_entries(_complex64_local_1q_vjp_adjoint_kernel)
    vjp_ms = _milliseconds(
        lambda: fused_complex64_local_1q_vjp_adjoint(
            state, adjoint, matrix, derivative, bit_position=bit
        ),
        args.repeats,
    )
    expected_adjoint = _reference_apply(adjoint, matrix.mH, bit)
    derivative_state = _reference_apply(state, derivative, bit)
    expected_gradient = torch.real(torch.sum(torch.conj(adjoint) * derivative_state))
    adjoint_error = float(torch.max(torch.abs(next_adjoint - expected_adjoint)))
    gradient_error = float(torch.abs(gradient - expected_gradient))

    updated_derivative = derivative * 0.91
    for updated_bit in (0, min(3, args.qubits - 1), args.qubits - 1):
        fused_complex64_local_1q_vjp_adjoint(
            state,
            adjoint,
            updated_matrix,
            updated_derivative,
            bit_position=updated_bit,
        )
    torch.cuda.synchronize()
    vjp_cache_after_updates = _cache_entries(_complex64_local_1q_vjp_adjoint_kernel)

    warm = statistics.median(triton_ms)
    baseline = statistics.median(pytorch_ms)
    compile_overhead = max(0.0, cold_ms - warm)
    saving = baseline - warm
    break_even = compile_overhead / saving if saving > 0 else None
    payload = {
        "device": torch.cuda.get_device_name(),
        "qubits": args.qubits,
        "state_gib": state.numel() * state.element_size() / 2**30,
        "forward": {
            "cold_wall_ms": cold_ms,
            "warm_median_ms": warm,
            "pytorch_median_ms": baseline,
            "speedup": baseline / warm,
            "break_even_calls": break_even,
            "max_abs_error": forward_error,
            "cache_entries_after_cold": forward_cache_after_cold,
            "cache_entries_after_parameter_and_bit_updates": forward_cache_after_updates,
        },
        "fused_vjp_adjoint": {
            "cold_wall_ms": vjp_cold_ms,
            "warm_median_ms": statistics.median(vjp_ms),
            "adjoint_max_abs_error": adjoint_error,
            "gradient_abs_error": gradient_error,
            "gradient_relative_error": gradient_error
            / max(float(torch.abs(expected_gradient)), 1e-30),
            "cache_entries_after_cold": vjp_cache_after_cold,
            "cache_entries_after_parameter_and_bit_updates": vjp_cache_after_updates,
            "materialized_derivative_state": False,
        },
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
