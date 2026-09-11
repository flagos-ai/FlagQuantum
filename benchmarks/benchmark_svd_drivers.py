"""Reproducible CUDA SVD-driver calibration for MPS bond shapes."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


def _source_snapshot() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"source_commit": commit, "source_tree_dirty": dirty}


def measure(
    dimension: int,
    driver: str,
    *,
    warmups: int,
    samples: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(3000 + dimension)
    matrix = torch.randn(
        1,
        dimension,
        dimension,
        device="cuda",
        dtype=torch.complex64,
        generator=generator,
    ) / dimension**0.5
    for _ in range(warmups):
        torch.linalg.svd(matrix, full_matrices=False, driver=driver)
        torch.cuda.synchronize()
    timings = []
    peaks = []
    result = None
    for _ in range(samples):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        result = torch.linalg.svd(
            matrix,
            full_matrices=False,
            driver=driver,
        )
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        peaks.append(int(torch.cuda.max_memory_allocated()))
    assert result is not None
    u, singular, vh = result
    reconstruction = torch.matmul(u * singular.unsqueeze(-2), vh)
    residual = float(
        (torch.linalg.vector_norm(reconstruction - matrix)
        / torch.linalg.vector_norm(matrix)).item()
    )
    return {
        "dimension": dimension,
        "driver": driver,
        "supported": True,
        "samples_seconds": timings,
        "mean_seconds": statistics.fmean(timings),
        "p50_seconds": statistics.median(timings),
        "peak_allocated_memory_bytes": max(peaks),
        "relative_reconstruction_residual": residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bonds", type=int, nargs="+", default=(128, 256, 512))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if min((*args.bonds, args.samples), default=0) < 1 or args.warmups < 0:
        raise SystemExit("bonds/samples must be positive and warmups non-negative")
    records = []
    for bond in args.bonds:
        dimension = 2 * bond
        for driver in ("gesvd", "gesvdj", "gesvda"):
            try:
                records.append(
                    measure(
                        dimension,
                        driver,
                        warmups=args.warmups,
                        samples=args.samples,
                    )
                )
            except RuntimeError as error:
                records.append(
                    {
                        "dimension": dimension,
                        "driver": driver,
                        "supported": False,
                        "error": str(error),
                    }
                )
    payload = {
        "schema": "flagquantum.mps_svd_driver_calibration.v1",
        "dtype": "complex64",
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "bonds": args.bonds,
        "matrix_semantics": "one square two-site split matrix of dimension 2*bond",
        "warmups": args.warmups,
        "samples": args.samples,
        **_source_snapshot(),
        "records": records,
        "performance_claim_allowed": False,
        "scalability_claim_allowed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "record_count": len(records)}))


if __name__ == "__main__":
    main()
