#!/usr/bin/env python3
"""Probe CUDA SVD drivers before selecting a FlagQuantum factorization backend."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device).manual_seed(108)
    matrix = torch.randn(
        (args.dimension, args.dimension),
        dtype=torch.complex64,
        device=device,
        generator=generator,
    ) * args.dimension**-0.5
    records = []
    for driver in ("gesvdj", "gesvd", "gesvda"):
        try:
            for _ in range(args.warmups):
                torch.linalg.svd(matrix, full_matrices=False, driver=driver)
                torch.cuda.synchronize(device)
            samples = []
            peak = 0
            result = None
            for _ in range(args.samples):
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = torch.linalg.svd(
                    matrix, full_matrices=False, driver=driver
                )
                torch.cuda.synchronize(device)
                samples.append(time.perf_counter() - started)
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
            assert result is not None
            u, s, vh = result
            residual = float(
                torch.linalg.vector_norm((u * s) @ vh - matrix)
                / torch.linalg.vector_norm(matrix)
            )
            records.append(
                {
                    "driver": driver,
                    "supported": True,
                    "samples_seconds": samples,
                    "mean_seconds": statistics.fmean(samples),
                    "peak_allocated_memory_bytes": peak,
                    "relative_reconstruction_residual": residual,
                }
            )
        except RuntimeError as error:
            records.append(
                {"driver": driver, "supported": False, "error": str(error)}
            )
    payload = {
        "schema": "flagquantum.issue108.svd_driver_probe.v1",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dimension": args.dimension,
        "dtype": "complex64",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
