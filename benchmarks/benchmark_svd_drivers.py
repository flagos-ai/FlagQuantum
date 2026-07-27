"""Reproducible CUDA SVD-driver microbenchmark for distributed-MPS tuning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def measure(shape: tuple[int, int, int], driver: str, repeats: int) -> dict[str, float | str | list[int]]:
    matrix = torch.randn(*shape, device="cuda", dtype=torch.complex64)
    torch.cuda.synchronize()
    started = time.perf_counter()
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False, driver=driver)
    torch.cuda.synchronize()
    warm = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(repeats):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False, driver=driver)
    torch.cuda.synchronize()
    steady = (time.perf_counter() - started) / repeats
    reconstruction = torch.matmul(u * s.unsqueeze(-2), vh)
    rel_residual = float(
        (torch.linalg.norm(reconstruction - matrix) / torch.linalg.norm(matrix)).item()
    )
    return {
        "shape": list(shape),
        "driver": driver,
        "warm_seconds": warm,
        "steady_seconds": steady,
        "relative_reconstruction_residual": rel_residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    shapes = ((2, 512, 512), (2, 1024, 1024), (4, 1024, 1024))
    records = []
    for shape in shapes:
        for driver in ("gesvd", "gesvdj", "gesvda"):
            try:
                records.append(measure(shape, driver, args.repeats))
            except RuntimeError as error:
                records.append({"shape": list(shape), "driver": driver, "error": str(error)})
    payload = {
        "schema": "flagquantum.svd_driver_microbenchmark.v1",
        "dtype": "complex64",
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
