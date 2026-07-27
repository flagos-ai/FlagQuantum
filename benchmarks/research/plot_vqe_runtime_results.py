"""Regenerate VQE runtime, speedup, and memory SVGs from a result JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.vqe_triton_runtime import _write_chart  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--speedup-output", type=Path, required=True)
    parser.add_argument("--memory-output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    iterations = int(payload["iterations"])
    triton, eager = payload["triton"], payload["eager"]
    jax_result = payload.get("jax")
    speedup = float(payload["speedup_eager_over_triton"])
    _write_chart(
        iterations=iterations,
        series=(
            (
                "FlagQuantum IR + Triton fusion",
                triton["cumulative_seconds"],
                "#2468b4",
            ),
            ("FlagQuantum PyTorch eager", eager["cumulative_seconds"], "#d24b40"),
        )
        + (
            (("FlagQuantum JAX (jit)", jax_result["cumulative_seconds"], "#238b57"),)
            if jax_result is not None
            else ()
        ),
        title="FlagQuantum VQE: cumulative training runtime (post-warmup)",
        y_label="Total Runtime (seconds)",
        annotation=(
            f"eager/Triton: {speedup:.2f}x; "
            f"JAX/Triton: {payload['runtime_ratio_jax_over_triton']:.2f}x"
            if jax_result is not None
            else f"eager/Triton: {speedup:.2f}x; JAX not measured"
        ),
        path=args.runtime_output,
    )
    _write_chart(
        iterations=iterations,
        series=(
            (
                "PyTorch eager / IR+Triton",
                payload["cumulative_speedup_factor"],
                "#6b3fa0",
            ),
        )
        + (
            (
                (
                    "JAX jit / IR+Triton",
                    payload["cumulative_jax_over_triton_factor"],
                    "#238b57",
                ),
            )
            if jax_result is not None
            else ()
        ),
        title="FlagQuantum VQE: post-warmup runtime ratio vs IR+Triton",
        y_label="Runtime Ratio vs IR+Triton (>1: Triton faster)",
        annotation=(
            f"final eager: {speedup:.2f}x; "
            f"JAX: {payload['runtime_ratio_jax_over_triton']:.2f}x"
            if jax_result is not None
            else f"final eager: {speedup:.2f}x"
        ),
        path=args.speedup_output,
    )
    _write_chart(
        iterations=iterations,
        series=(
            (
                "FlagQuantum IR + Triton fusion",
                [value / 2**30 for value in triton["peak_memory_bytes"]],
                "#2468b4",
            ),
            (
                "FlagQuantum PyTorch eager",
                [value / 2**30 for value in eager["peak_memory_bytes"]],
                "#d24b40",
            ),
        ),
        title="FlagQuantum VQE: per-step peak GPU memory",
        y_label="Peak Allocated GPU Memory (GiB)",
        annotation="torch.cuda.max_memory_allocated",
        path=args.memory_output,
    )


if __name__ == "__main__":
    main()
