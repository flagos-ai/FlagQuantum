"""Probe the direct MPS-008 wrapper against its exact PyTorch semantic."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance
from flagquantum.kernels.triton.mps_sampling_collapse import (
    fused_mps_sampling_collapse,
)

SHAPE_MATRIX = (
    (32, 1, 1),
    (128, 8, 16),
    (512, 16, 32),
    (2048, 32, 64),
    (2048, 64, 64),
)


def _reference(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = int(site.shape[0])
    batches = torch.arange(batch, device=site.device)
    boundary = site[batches, 0, bits, :]
    norms = torch.linalg.vector_norm(boundary, dim=-1)
    if bool(torch.any(~torch.isfinite(norms))) or bool(torch.any(norms <= 1.0e-12)):
        raise RuntimeError("sampled MPS branch is not finite and positive")
    boundary = boundary / norms[:, None]
    collapsed = torch.zeros(batch, 1, 2, 1, dtype=site.dtype, device=site.device)
    collapsed[batches, 0, bits, 0] = 1
    propagated = torch.einsum("bl,blsr->bsr", boundary, next_site).unsqueeze(1)
    return collapsed, propagated


def _measure(
    operation: Callable[[], Any],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(group_size):
            operation()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / group_size)
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument(
        "--compiler-lane", choices=("stock_triton", "flagtree"), required=True
    )
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=261_006)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    expected_distribution = (
        "triton" if args.compiler_lane == "stock_triton" else "flagtree"
    )
    if distribution != expected_distribution or identity_status != "resolved":
        raise RuntimeError(
            f"lane {args.compiler_lane!r} requires {expected_distribution!r}, "
            f"found {distribution!r} with identity {identity_status!r}"
        )

    cases = []
    for index, (batch, right_dim, next_right_dim) in enumerate(SHAPE_MATRIX):
        generator = torch.Generator(device="cuda").manual_seed(args.seed + index)
        site = torch.randn(
            batch,
            1,
            2,
            right_dim,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
        next_site = torch.randn(
            batch,
            right_dim,
            2,
            next_right_dim,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
        bits = torch.arange(batch, device="cuda", dtype=torch.int64) % 2
        expected = _reference(site, next_site, bits)
        actual = fused_mps_sampling_collapse(site, next_site, bits)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=2e-5)
        direct = _measure(
            lambda site=site, next_site=next_site, bits=bits: (
                fused_mps_sampling_collapse(site, next_site, bits)
            ),
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        reference = _measure(
            lambda site=site, next_site=next_site, bits=bits: _reference(
                site, next_site, bits
            ),
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        difference = actual[1] - expected[1]
        cases.append(
            {
                "shape": {
                    "batch": batch,
                    "right_bond": right_dim,
                    "next_right_bond": next_right_dim,
                },
                "direct_kernel_wrapper": direct,
                "pytorch_reference": reference,
                "speedup_over_pytorch": float(
                    reference["median_seconds_per_invocation"]
                )
                / float(direct["median_seconds_per_invocation"]),
                "maximum_absolute_error": float(torch.max(torch.abs(difference))),
                "relative_l2_error": float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(expected[1]).clamp_min(
                        torch.finfo(torch.float32).eps
                    )
                ),
            }
        )

    properties = torch.cuda.get_device_properties(0)
    payload = {
        "schema": "flagquantum.kernel_probe.mps_sampling_collapse.v1",
        "source_revision": args.source_revision,
        "host_label": args.host_label,
        "compiler_lane": args.compiler_lane,
        "compiler": {
            "distribution": distribution,
            "version": version,
            "integration_path": integration_path,
            "identity_status": identity_status,
        },
        "device": {"name": properties.name, "total_memory": properties.total_memory},
        "warmup": args.warmup,
        "repeats": args.repeats,
        "group_size": args.group_size,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
