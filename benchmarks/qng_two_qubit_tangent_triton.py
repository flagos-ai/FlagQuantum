"""A800 benchmark for persistent RXX/RYY/RZZ parameter tangents."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from flagquantum.simulation.triton_kernels.two_qubit_pauli_tangent import (
    _reference,
    repeated_rxx_ryy_rzz_tangents,
)


def reference_tangents(state, angles):
    real = torch.autograd.functional.jacobian(
        lambda value: _reference(state, value).real, angles, vectorize=True
    )
    imag = torch.autograd.functional.jacobian(
        lambda value: _reference(state, value).imag, angles, vectorize=True
    )
    jacobian = torch.complex(real, imag)
    indices = torch.arange(int(state.shape[0]), device=state.device)
    return jacobian[indices, :, :, indices].movedim((-2, -1), (0, 1)).reshape(
        3 * angles.shape[1], *state.shape
    )


def metric(state, tangents):
    state = state.reshape(-1)
    jacobian = tangents[:, 0].reshape(tangents.shape[0], -1).T
    connection = torch.conj(state) @ jacobian
    value = torch.real(
        torch.conj(jacobian).T @ jacobian
        - torch.outer(torch.conj(connection), connection)
    )
    return 0.5 * (value + value.T)


def measure(function, warmup, repeats):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds": samples,
        "median_seconds": statistics.median(samples),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", type=int, default=2048)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    state = torch.randn(1, args.groups, 4, device="cuda", dtype=torch.complex64)
    state = state / torch.linalg.vector_norm(state)
    angles = torch.randn(1, args.depth, 3, device="cuda")
    cold_started = time.perf_counter()
    actual = repeated_rxx_ryy_rzz_tangents(state, angles)
    torch.cuda.synchronize()
    cold = time.perf_counter() - cold_started
    expected = reference_tangents(state, angles)
    actual_metric, expected_metric = metric(_reference(state, angles), actual), metric(_reference(state, angles), expected)
    gradient = torch.randn(3 * args.depth, device="cuda", generator=torch.Generator(device="cuda").manual_seed(args.seed + 1))
    identity = torch.eye(3 * args.depth, device="cuda")
    actual_direction = torch.linalg.solve(actual_metric + 1e-3 * identity, gradient)
    expected_direction = torch.linalg.solve(expected_metric + 1e-3 * identity, gradient)
    triton_result = measure(lambda: repeated_rxx_ryy_rzz_tangents(state, angles), args.warmup, args.repeats)
    reference_result = measure(lambda: reference_tangents(state, angles), args.warmup, args.repeats)
    payload = {
        "schema": "flagquantum.qng_two_qubit_tangent_triton.v1",
        "execution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "device": torch.cuda.get_device_name(0),
        "dtype": "complex64",
        "groups": args.groups,
        "depth": args.depth,
        "parameter_count": 3 * args.depth,
        "triton_cold_seconds": cold,
        "maximum_tangent_error": float(torch.max(torch.abs(actual - expected))),
        "relative_l2_tangent_error": float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected)),
        "relative_l2_metric_error": float(torch.linalg.vector_norm(actual_metric - expected_metric) / torch.linalg.vector_norm(expected_metric)),
        "qng_direction_cosine_similarity": float(torch.dot(actual_direction, expected_direction) / (torch.linalg.vector_norm(actual_direction) * torch.linalg.vector_norm(expected_direction))),
        "triton": triton_result,
        "vectorized_reverse": reference_result,
        "speedup_vectorized_reverse_over_triton": reference_result["median_seconds"] / triton_result["median_seconds"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
