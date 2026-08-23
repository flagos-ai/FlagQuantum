#!/usr/bin/env python3
"""Evaluate steady-state CUDA Graph coverage for local MPS contraction."""

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

from flagquantum.runtime.backends.mps.site_kernels import (  # noqa: E402
    _rxx_contraction_real,
)


def _parse_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item) for item in value.split(",") if item)
    if not dims or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("dims must be positive comma-separated values")
    return dims


def _median_batch_seconds(call, *, batches: int, replays: int) -> float:
    samples = []
    for _ in range(batches):
        started = time.perf_counter()
        for _ in range(replays):
            call()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / replays)
    return statistics.median(samples)


def _promotion(records: list[dict[str, object]], *, minimum_speedup: float) -> dict:
    contraction_passed = all(
        float(record["speedup_compiled_over_graph"]) >= minimum_speedup
        and float(record["relative_error"]) <= 2e-5
        for record in records
    )
    return {
        "allowed": False,
        "contraction_threshold_passed": contraction_passed,
        "full_factorization_capture_required": True,
        "full_factorization_capture_supported": False,
        "minimum_required_speedup": minimum_speedup,
        "maximum_relative_error": 2e-5,
        "blockers": ["cusolver_gesvd_invalidates_cuda_graph_capture"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=_parse_dims, default=(32, 64, 128))
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batches, args.replays) <= 0:
        raise SystemExit("batches and replays must be positive")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records = []
    last_pair = None
    for dim in args.dims:
        generator = torch.Generator(device=device).manual_seed(3200 + dim)
        static_inputs = (
            torch.randn(1, 1, dim, 2, dim, 2, device=device, generator=generator),
            torch.randn(1, 1, dim, 2, dim, 2, device=device, generator=generator),
            torch.view_as_real(
                torch.eye(4, dtype=torch.complex64, device=device).reshape(1, 1, 4, 4)
            ),
        )
        for _ in range(3):
            _rxx_contraction_real(*static_inputs)
        compiled_contraction = torch.compile(
            _rxx_contraction_real,
            fullgraph=True,
            dynamic=False,
        )
        for _ in range(3):
            compiled_output = compiled_contraction(*static_inputs)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = compiled_contraction(*static_inputs)
        graph.replay()
        torch.cuda.synchronize(device)
        relative_error = float(
            torch.linalg.vector_norm(graph_output - compiled_output)
            / torch.linalg.vector_norm(compiled_output)
        )
        compiled_seconds = _median_batch_seconds(
            lambda: compiled_contraction(*static_inputs),
            batches=args.batches,
            replays=args.replays,
        )
        graph_seconds = _median_batch_seconds(
            graph.replay,
            batches=args.batches,
            replays=args.replays,
        )
        records.append(
            {
                "active_dim": dim,
                "matrix_shape": [2 * dim, 2 * dim],
                "compiled_seconds_per_replay": compiled_seconds,
                "graph_seconds_per_replay": graph_seconds,
                "speedup_compiled_over_graph": compiled_seconds / graph_seconds,
                "relative_error": relative_error,
            }
        )
        last_pair = torch.view_as_complex(graph_output.contiguous())

    svd_capture = {"supported": False, "error_type": None, "error": None}
    assert last_pair is not None
    try:
        svd_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(svd_graph):
            torch.linalg.svd(last_pair, full_matrices=False, driver="gesvd")
        svd_graph.replay()
        torch.cuda.synchronize(device)
        svd_capture["supported"] = True
    except Exception as error:
        svd_capture["error_type"] = type(error).__name__
        svd_capture["error"] = str(error)

    payload = {
        "schema": "flagquantum.mps_cuda_graph_probe.v1",
        "artifact_class": "derived_development_report",
        "benchmark_evidence_class": "non_release_smoke",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "status": "passed",
        "environment": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
        },
        "workload": {
            "dtype": "complex64",
            "batch_size": 1,
            "batches": args.batches,
            "replays_per_batch": args.replays,
        },
        "records": records,
        "svd_capture": svd_capture,
        "promotion": _promotion(records, minimum_speedup=1.05),
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
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
