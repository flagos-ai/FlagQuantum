"""Benchmark a persistent RX/RZ loop against vectorized PyTorch eager.

This is local single-device engineering evidence, not a scalability benchmark.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.simulation.triton_kernels import repeated_rx_rz  # noqa: E402


def _torch_loop(state, rx_angles, rz_angles):
    output = state
    for layer in range(int(rx_angles.shape[1])):
        rx = rx_angles[:, layer].reshape(-1, 1, 1) / 2
        rz = rz_angles[:, layer].reshape(-1, 1, 1) / 2
        cos_rx, sin_rx = torch.cos(rx), torch.sin(rx)
        zero, one = output[..., :1], output[..., 1:]
        output = torch.cat(
            (cos_rx * zero - 1j * sin_rx * one, -1j * sin_rx * zero + cos_rx * one),
            dim=-1,
        )
        output = output * torch.cat(
            (torch.exp(-1j * rz), torch.exp(1j * rz)), dim=-1
        )
    return output


def _measure(operation, warmup, iterations, repeats):
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    samples = []
    peaks = []
    peak_increments = []
    for _ in range(repeats):
        baseline = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / iterations)
        peak = int(torch.cuda.max_memory_allocated())
        peaks.append(peak)
        peak_increments.append(peak - baseline)
    return {
        "median_seconds": statistics.median(samples),
        "samples": samples,
        "peak_memory_bytes": max(peaks),
        "peak_increment_bytes": max(peak_increments),
    }


def _cold_measure(operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    return time.perf_counter() - started


def _jax_loop(state, rx_angles, rz_angles):
    import jax.numpy as jnp
    from jax import lax

    def layer(output, angles):
        rx, rz = angles
        rx = rx[:, None, None] / 2
        rz = rz[:, None, None] / 2
        cos_rx, sin_rx = jnp.cos(rx), jnp.sin(rx)
        zero, one = output[..., :1], output[..., 1:]
        output = jnp.concatenate(
            (cos_rx * zero - 1j * sin_rx * one, -1j * sin_rx * zero + cos_rx * one),
            axis=-1,
        )
        phases = jnp.concatenate((jnp.exp(-1j * rz), jnp.exp(1j * rz)), axis=-1)
        return output * phases, None

    return lax.scan(layer, state, (rx_angles.T, rz_angles.T))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=65536)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--mode", choices=("forward", "training"), default="forward")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(7)
    state = torch.randn(
        args.batch,
        args.pairs,
        2,
        device="cuda",
        dtype=torch.complex64,
        requires_grad=args.mode == "training",
    )
    rx = torch.randn(
        args.batch,
        args.depth,
        device="cuda",
        requires_grad=args.mode == "training",
    )
    rz = torch.randn(
        args.batch,
        args.depth,
        device="cuda",
        requires_grad=args.mode == "training",
    )
    actual = repeated_rx_rz(state, rx, rz)
    reference = _torch_loop(state, rx, rz)
    gradient = torch.randn_like(actual) if args.mode == "training" else None

    def triton_operation():
        output = repeated_rx_rz(state, rx, rz)
        if gradient is not None:
            torch.autograd.grad(output, (state, rx, rz), gradient)

    def torch_operation():
        output = _torch_loop(state, rx, rz)
        if gradient is not None:
            torch.autograd.grad(output, (state, rx, rz), gradient)

    triton_cold_seconds = _cold_measure(triton_operation)
    triton_result = _measure(
        triton_operation,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    torch_result = _measure(
        torch_operation,
        args.warmup,
        args.iterations,
        args.repeats,
    )
    compiled_result = None
    jax_result = None
    if args.mode == "forward":
        compiled_loop = torch.compile(_torch_loop, fullgraph=True, dynamic=False)
        compiled_result = _measure(
            lambda: compiled_loop(state, rx, rz),
            args.warmup,
            args.iterations,
            args.repeats,
        )
        import jax
        import jax.dlpack

        jax_state = jax.dlpack.from_dlpack(state)
        jax_rx = jax.dlpack.from_dlpack(rx)
        jax_rz = jax.dlpack.from_dlpack(rz)
        jax_loop = jax.jit(_jax_loop)
        jax_result = _measure(
            lambda: jax_loop(jax_state, jax_rx, jax_rz).block_until_ready(),
            args.warmup,
            args.iterations,
            args.repeats,
        )
    reference_norm = torch.sum(torch.abs(reference) ** 2, dim=(-2, -1))
    norm_delta = torch.abs(
        torch.sum(torch.abs(actual) ** 2, dim=(-2, -1)) - reference_norm
    )
    payload = {
        "benchmark": "single_qubit_loop",
        "benchmark_evidence_class": "local",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "distribution_semantics": "single_device_fast_path",
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "mode": args.mode,
        "batch": args.batch,
        "pairs": args.pairs,
        "depth": args.depth,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "triton_cold_seconds": triton_cold_seconds,
        "triton": triton_result,
        "torch_eager": torch_result,
        "torch_compile": compiled_result,
        "jax_scan": jax_result,
        "speedup": torch_result["median_seconds"] / triton_result["median_seconds"],
        "max_abs_error": float(torch.max(torch.abs(actual - reference)).item()),
        "norm_error": float(torch.max(norm_delta).item()),
        "relative_norm_error": float(torch.max(norm_delta / reference_norm).item()),
    }
    if gradient is not None:
        actual_gradients = torch.autograd.grad(actual, (state, rx, rz), gradient)
        reference_gradients = torch.autograd.grad(
            reference, (state, rx, rz), gradient
        )
        payload["gradient_max_abs_error"] = {
            name: float(torch.max(torch.abs(actual_gradient - reference_gradient)).item())
            for name, actual_gradient, reference_gradient in zip(
                ("state", "rx", "rz"), actual_gradients, reference_gradients
            )
        }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
