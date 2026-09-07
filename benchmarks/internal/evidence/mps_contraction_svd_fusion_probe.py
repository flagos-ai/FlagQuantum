#!/usr/bin/env python3
"""Compare separate and full-graph MPS contraction/SVD regions on CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flagquantum.simulation.mps.site_kernels import (  # noqa: E402
    _rxx_contraction_real,
)


def _parse_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item) for item in value.split(",") if item)
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("dims must be positive comma-separated values")
    return dims


def _time(call, inputs, iterations: int) -> tuple[float, int]:
    torch.cuda.reset_peak_memory_stats(inputs[0].device)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        call(*inputs)
        torch.cuda.synchronize(inputs[0].device)
        samples.append(time.perf_counter() - started)
    return statistics.median(samples), int(torch.cuda.max_memory_allocated())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=_parse_dims, default=(32, 64, 128))
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 2:
        raise SystemExit("iterations must be at least two")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records = []
    for dim in args.dims:
        generator = torch.Generator(device=device).manual_seed(3100 + dim)
        left = torch.randn(1, 1, dim, 2, dim, 2, device=device, generator=generator)
        right = torch.randn(1, 1, dim, 2, dim, 2, device=device, generator=generator)
        gate = torch.view_as_real(
            torch.eye(4, dtype=torch.complex64, device=device).reshape(1, 1, 4, 4)
        )

        def contraction(a, b, g):
            return _rxx_contraction_real(a, b, g)

        def fused(a, b, g):
            pair = torch.view_as_complex(_rxx_contraction_real(a, b, g).contiguous())
            return torch.linalg.svd(pair, full_matrices=False, driver="gesvd")

        compiled_contraction = torch.compile(contraction, fullgraph=True, dynamic=False)
        compiled_fused = torch.compile(fused, fullgraph=True, dynamic=False)

        def separate(a, b, g):
            pair = torch.view_as_complex(compiled_contraction(a, b, g).contiguous())
            return torch.linalg.svd(pair, full_matrices=False, driver="gesvd")

        inputs = (left, right, gate)
        separate_result = separate(*inputs)
        fused_result = compiled_fused(*inputs)
        torch.cuda.synchronize(device)
        reference_pair = torch.view_as_complex(
            _rxx_contraction_real(*inputs).contiguous()
        )
        separate_reconstruction = (
            separate_result[0]
            @ torch.diag_embed(separate_result[1]).to(separate_result[0].dtype)
            @ separate_result[2]
        )
        fused_reconstruction = (
            fused_result[0]
            @ torch.diag_embed(fused_result[1]).to(fused_result[0].dtype)
            @ fused_result[2]
        )
        separate_seconds, separate_peak = _time(separate, inputs, args.iterations)
        fused_seconds, fused_peak = _time(compiled_fused, inputs, args.iterations)
        records.append(
            {
                "active_dim": dim,
                "matrix_shape": list(reference_pair.shape[-2:]),
                "separate_median_seconds": separate_seconds,
                "fused_median_seconds": fused_seconds,
                "speedup_separate_over_fused": separate_seconds / fused_seconds,
                "separate_peak_allocated_bytes": separate_peak,
                "fused_peak_allocated_bytes": fused_peak,
                "separate_relative_reconstruction_error": float(
                    torch.linalg.vector_norm(separate_reconstruction - reference_pair)
                    / torch.linalg.vector_norm(reference_pair)
                ),
                "fused_relative_reconstruction_error": float(
                    torch.linalg.vector_norm(fused_reconstruction - reference_pair)
                    / torch.linalg.vector_norm(reference_pair)
                ),
            }
        )
    payload = {
        "schema": "flagquantum.mps_contraction_svd_fusion_probe.v1",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "status": "passed",
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
        "provenance": {
            "source_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "dirty_paths": tuple(
                line[3:]
                for line in subprocess.check_output(
                    ("git", "status", "--short"), cwd=ROOT, text=True
                ).splitlines()
            ),
        },
        "records": records,
        "promotion": {
            "allowed": all(
                record["speedup_separate_over_fused"] >= 1.05
                and record["fused_relative_reconstruction_error"] <= 2e-5
                for record in records
            ),
            "minimum_required_speedup": 1.05,
            "maximum_relative_reconstruction_error": 2e-5,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
