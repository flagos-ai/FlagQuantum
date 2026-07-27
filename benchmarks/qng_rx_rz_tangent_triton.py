"""A800 benchmark for persistent Triton RX/RZ parameter tangents."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from flagquantum.simulation.triton_kernels import repeated_rx_rz_tangents


def reference(state, rx, rz):
    output = state
    for layer in range(int(rx.shape[1])):
        half_rx = rx[:, layer].reshape(-1, 1, 1) / 2
        half_rz = rz[:, layer].reshape(-1, 1, 1) / 2
        cosine, sine = torch.cos(half_rx), torch.sin(half_rx)
        zero, one = output[..., :1], output[..., 1:]
        output = torch.cat(
            (cosine * zero - 1j * sine * one, -1j * sine * zero + cosine * one),
            dim=-1,
        )
        output = output * torch.cat(
            (torch.exp(-1j * half_rz), torch.exp(1j * half_rz)), dim=-1
        )
    return output


def reverse_jacobian(state, rx, rz):
    tangents = []
    for family, parameter in (("rx", rx), ("rz", rz)):
        def selected(value):
            return reference(
                state,
                value if family == "rx" else rx,
                value if family == "rz" else rz,
            )

        real = torch.autograd.functional.jacobian(
            lambda value: selected(value).real, parameter, vectorize=True
        )
        imag = torch.autograd.functional.jacobian(
            lambda value: selected(value).imag, parameter, vectorize=True
        )
        jacobian = torch.complex(real, imag)
        indices = torch.arange(int(state.shape[0]), device=state.device)
        tangents.append(jacobian[indices, :, :, indices].movedim(-1, 0))
    # Convert family-major [all RX, all RZ] to layer-major [RX, RZ, ...].
    stacked = torch.stack(tangents).movedim(1, 0)
    return stacked.reshape(-1, *state.shape)


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


def quantum_metric(state, tangents):
    state = state.reshape(-1)
    jacobian = tangents[:, 0].reshape(tangents.shape[0], -1).transpose(0, 1)
    connection = torch.conj(state) @ jacobian
    metric = torch.real(
        torch.conj(jacobian).transpose(0, 1) @ jacobian
        - torch.outer(torch.conj(connection), connection)
    )
    return 0.5 * (metric + metric.transpose(0, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=260720)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    state = torch.randn(args.batch, args.pairs, 2, device="cuda", dtype=torch.complex64)
    state = state / torch.linalg.vector_norm(state.reshape(args.batch, -1), dim=1).reshape(-1, 1, 1)
    rx = torch.randn(args.batch, args.depth, device="cuda")
    rz = torch.randn(args.batch, args.depth, device="cuda")
    cold_started = time.perf_counter()
    actual = repeated_rx_rz_tangents(state, rx, rz)
    torch.cuda.synchronize()
    cold_seconds = time.perf_counter() - cold_started
    expected = reverse_jacobian(state, rx, rz)
    maximum_error = float(torch.max(torch.abs(actual - expected)))
    relative_l2 = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
    output_state = reference(state, rx, rz)
    actual_metric = quantum_metric(output_state, actual)
    expected_metric = quantum_metric(output_state, expected)
    metric_relative_l2 = float(
        torch.linalg.vector_norm(actual_metric - expected_metric)
        / torch.linalg.vector_norm(expected_metric)
    )
    direction_generator = torch.Generator(device="cuda").manual_seed(args.seed + 1)
    gradient = torch.randn(
        actual_metric.shape[0], generator=direction_generator, device="cuda"
    )
    identity = torch.eye(actual_metric.shape[0], device="cuda")
    actual_direction = torch.linalg.solve(actual_metric + 1e-3 * identity, gradient)
    expected_direction = torch.linalg.solve(expected_metric + 1e-3 * identity, gradient)
    direction_cosine = float(
        torch.dot(actual_direction, expected_direction)
        / (
            torch.linalg.vector_norm(actual_direction)
            * torch.linalg.vector_norm(expected_direction)
        )
    )
    triton_result = measure(
        lambda: repeated_rx_rz_tangents(state, rx, rz), args.warmup, args.repeats
    )
    reference_result = measure(
        lambda: reverse_jacobian(state, rx, rz), args.warmup, args.repeats
    )
    payload = {
        "schema": "flagquantum.qng_rx_rz_tangent_triton.v1",
        "execution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "device": torch.cuda.get_device_name(0),
        "dtype": "complex64",
        "batch": args.batch,
        "pairs": args.pairs,
        "depth": args.depth,
        "parameter_count_per_batch": 2 * args.depth,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "triton_cold_seconds": cold_seconds,
        "maximum_tangent_error": maximum_error,
        "relative_l2_tangent_error": relative_l2,
        "relative_l2_metric_error": metric_relative_l2,
        "qng_direction_cosine_similarity": direction_cosine,
        "triton": triton_result,
        "vectorized_reverse": reference_result,
        "speedup_vectorized_reverse_over_triton": (
            reference_result["median_seconds"] / triton_result["median_seconds"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
