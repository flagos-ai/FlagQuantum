"""Probe the direct MPS-008 wrapper against its exact PyTorch semantic."""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance
from flagquantum.kernels.triton.mps_sampling_collapse import (
    fused_mps_sampling_collapse,
)

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.mps_sampling_collapse.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.mps_sampling_collapse.v1"
SEMANTIC_ID = "mps.sampling.collapse_wire.local"
IMPLEMENTATION_ID = "FQKI-TRITON-MPS-008-A"
RUNNER = "benchmarks/internal/evidence/mps_sampling_collapse_probe.py"
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
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


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed MPS-008 matrix."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MPS-008 benchmark evidence")

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
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "source_revision": args.source_revision,
        "command": shlex.join(sys.argv),
        "host_label": args.host_label,
        "reported_hostname": socket.gethostname(),
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler_lane": args.compiler_lane,
        "compiler": {
            "distribution": distribution,
            "version": version,
            "integration_path": integration_path,
            "identity_status": identity_status,
        },
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
        },
        "measurement": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "group_size": args.group_size,
            "synchronization": "once after warmup and once per timed group",
            "statistic": "median synchronized wall seconds per invocation",
        },
        "seed": args.seed,
        "cases": cases,
    }


def aggregate_runs(paths: list[Path]) -> dict[str, object]:
    """Validate and combine the canonical dual-host/compiler evidence matrix."""

    runs = [json.loads(path.read_text()) for path in paths]
    if len(runs) != len(HOSTS) * len(COMPILER_LANES):
        raise ValueError("MPS-008 evidence requires exactly four raw runs")
    if any(run.get("schema") != RUN_SCHEMA for run in runs):
        raise ValueError("MPS-008 raw-run schema mismatch")
    identities = {(run["host_label"], run["compiler_lane"]) for run in runs}
    expected_identities = {
        (host, compiler_lane) for host in HOSTS for compiler_lane in COMPILER_LANES
    }
    if identities != expected_identities:
        raise ValueError("MPS-008 evidence host/compiler matrix is incomplete")
    revisions = {run["source_revision"] for run in runs}
    measurements = {json.dumps(run["measurement"], sort_keys=True) for run in runs}
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError("MPS-008 evidence must use one source and measurement policy")
    expected_shapes = [
        {
            "batch": batch,
            "right_bond": right_dim,
            "next_right_bond": next_right_dim,
        }
        for batch, right_dim, next_right_dim in SHAPE_MATRIX
    ]
    if any([case["shape"] for case in run["cases"]] != expected_shapes for run in runs):
        raise ValueError("MPS-008 evidence shape matrix mismatch")

    cases = [case for run in runs for case in run["cases"]]
    speedups = [float(case["speedup_over_pytorch"]) for case in cases]
    maximum_absolute_error = max(
        float(case["maximum_absolute_error"]) for case in cases
    )
    maximum_relative_l2_error = max(float(case["relative_l2_error"]) for case in cases)
    all_cases_win = all(speedup > 1.0 for speedup in speedups)
    return {
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": revisions.pop(),
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "measurement": runs[0]["measurement"],
        "shape_matrix": expected_shapes,
        "runs": sorted(runs, key=lambda run: (run["host_label"], run["compiler_lane"])),
        "aggregate": {
            "case_count": len(cases),
            "minimum_speedup_over_pytorch": min(speedups),
            "maximum_speedup_over_pytorch": max(speedups),
            "maximum_absolute_error": maximum_absolute_error,
            "maximum_relative_l2_error": maximum_relative_l2_error,
            "all_cases_win": all_cases_win,
            "decision": (
                "eligible_for_dispatch_evaluation"
                if all_cases_win
                else "retain_experimental"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command_name", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--host-label", choices=HOSTS, required=True)
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261_006)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--input", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = (
        collect_run(args) if args.command_name == "run" else aggregate_runs(args.input)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
