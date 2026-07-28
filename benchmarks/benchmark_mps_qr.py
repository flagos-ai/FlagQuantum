"""Calibrate the no-truncation MPS QR path on CUDA."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch

from flagquantum.runtime.backends.mps.factorization import mps_qr_forward


def _measure(operation: Callable[[], Any], warmups: int, samples: int) -> dict[str, Any]:
    for _ in range(warmups):
        operation()
        torch.cuda.synchronize()
    timings = []
    peaks = []
    for _ in range(samples):
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        peaks.append(int(torch.cuda.max_memory_allocated()))
    return {
        "samples_seconds": timings,
        "mean_seconds": statistics.fmean(timings),
        "p50_seconds": statistics.median(timings),
        "peak_allocated_memory_bytes": max(peaks),
    }


def _source_snapshot() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"source_commit": commit, "source_tree_dirty": bool(status.strip())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bonds", type=int, nargs="+", default=(128, 256, 512))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    records = []
    for bond in args.bonds:
        generator = torch.Generator(device="cuda").manual_seed(4000 + bond)
        left = torch.randn(
            1,
            bond,
            2,
            bond,
            dtype=torch.complex64,
            device="cuda",
            generator=generator,
        ) / bond**0.5
        right = torch.randn(
            1,
            bond,
            2,
            bond,
            dtype=torch.complex64,
            device="cuda",
            generator=generator,
        ) / bond**0.5
        matrix = left.reshape(1, 2 * bond, bond)
        direct = _measure(
            lambda: torch.linalg.qr(matrix, mode="reduced"),
            args.warmups,
            args.samples,
        )
        full = _measure(
            lambda: mps_qr_forward(left, right),
            args.warmups,
            args.samples,
        )
        q, updated = mps_qr_forward(left, right)
        original_pair = torch.einsum("blsm,bmtr->blstr", left, right)
        updated_pair = torch.einsum("blsm,bmtr->blstr", q, updated)
        reconstruction = float(
            (
                torch.linalg.vector_norm(updated_pair - original_pair)
                / torch.linalg.vector_norm(original_pair)
            ).item()
        )
        q_matrix = q.reshape(1, 2 * bond, bond)
        identity = torch.eye(bond, dtype=q.dtype, device=q.device).unsqueeze(0)
        orthogonality = float(
            torch.linalg.vector_norm(
                q_matrix.mH @ q_matrix - identity
            ).item()
            / bond**0.5
        )
        records.append(
            {
                "bond": bond,
                "matrix_shape": [1, 2 * bond, bond],
                "direct_qr": direct,
                "full_mps_update": full,
                "relative_reconstruction_residual": reconstruction,
                "normalized_orthogonality_error": orthogonality,
            }
        )
    payload = {
        "schema": "flagquantum.mps_qr_calibration.v1",
        "dtype": "complex64",
        "device": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "bonds": args.bonds,
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
