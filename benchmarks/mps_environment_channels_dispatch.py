"""Collect and validate MPS-005 channel-transfer evidence on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.mps_environment_channels_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.mps_environment_channels_dispatch.v1"
SEMANTIC_ID = "mps.environment.transfer_channels"
IMPLEMENTATION_ID = "FQKI-TRITON-MPS-005-A"
RUNNER = "benchmarks/mps_environment_channels_dispatch.py"
SHAPE_MATRIX = (
    (5, 2, 4, 4),
    (8, 8, 8, 8),
    (8, 8, 16, 16),
    (16, 8, 16, 16),
    (32, 8, 8, 8),
)
COMPILER_LANES = ("stock_triton", "flagtree")
DISPATCH_VARIABLE = "FQ_TRITON_MPS_ENVIRONMENT"
RESULT_NAMES = (
    "direct_kernel_wrapper",
    "direct_pytorch_einsum",
    "public_catalog_dispatch",
    "public_pytorch_eager",
    "public_pytorch_compiled",
)


def _reference(channels: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    return torch.einsum("tbij,bipr,bjps->tbrs", channels, tensor.conj(), tensor)


def _inputs(
    channel_count: int,
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    channels = torch.randn(
        channel_count,
        batch,
        left_dim,
        left_dim,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    tensor = torch.randn(
        batch,
        left_dim,
        2,
        right_dim,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    channels = channels / max(1, left_dim) ** 0.5
    tensor_norm = torch.linalg.vector_norm(tensor, dim=(1, 2, 3), keepdim=True)
    return channels.contiguous(), (tensor / tensor_norm).contiguous()


@contextmanager
def _dispatch(enabled: bool) -> Iterator[None]:
    previous = os.environ.get(DISPATCH_VARIABLE)
    os.environ[DISPATCH_VARIABLE] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DISPATCH_VARIABLE, None)
        else:
            os.environ[DISPATCH_VARIABLE] = previous


def _measure(
    function: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(group_size):
            function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) / group_size)
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_delta_bytes": max(
            0, torch.cuda.max_memory_allocated() - allocated_before
        ),
    }


def _case(
    channel_count: int,
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.mps_environment import (
        fused_mps_environment_channels,
    )
    from flagquantum.simulation.mps.site_kernels import (
        environment_transfer_channels,
        reset_site_kernel_stats,
    )

    channels, tensor = _inputs(
        channel_count,
        batch,
        left_dim,
        right_dim,
        seed=seed,
    )
    reference = _reference(channels, tensor)
    direct = fused_mps_environment_channels(channels, tensor)
    with _dispatch(True):
        public = environment_transfer_channels(channels, tensor, compiled=True)
    with _dispatch(False):
        public_eager = environment_transfer_channels(channels, tensor, compiled=False)
        public_compiled = environment_transfer_channels(channels, tensor, compiled=True)
    for actual in (direct, public, public_eager, public_compiled):
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=1e-4)

    direct_kernel = _measure(
        lambda: fused_mps_environment_channels(channels, tensor),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    direct_reference = _measure(
        lambda: _reference(channels, tensor),
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(True):
        public_dispatch = _measure(
            lambda: environment_transfer_channels(channels, tensor, compiled=True),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(False):
        public_eager_result = _measure(
            lambda: environment_transfer_channels(channels, tensor, compiled=False),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    reset_site_kernel_stats(clear_cache=True)
    with _dispatch(False):
        public_compiled_result = _measure(
            lambda: environment_transfer_channels(channels, tensor, compiled=True),
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        )
    differences = (direct - reference, public - reference)
    denominator = torch.linalg.vector_norm(reference).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return {
        "shape": {
            "channels": channel_count,
            "batch": batch,
            "left_bond": left_dim,
            "physical_dimension": 2,
            "right_bond": right_dim,
        },
        "contraction_work": (
            channel_count * batch * left_dim * left_dim * right_dim * right_dim
        ),
        "dtype": "complex64",
        "layout": "contiguous_mps_environment_channels_and_site_tensor",
        "maximum_absolute_error": max(
            float(torch.max(torch.abs(difference))) for difference in differences
        ),
        "relative_l2_error": max(
            float(torch.linalg.vector_norm(difference) / denominator)
            for difference in differences
        ),
        "direct_kernel_wrapper": direct_kernel,
        "direct_pytorch_einsum": direct_reference,
        "public_catalog_dispatch": public_dispatch,
        "public_pytorch_eager": public_eager_result,
        "public_pytorch_compiled": public_compiled_result,
        "direct_kernel_speedup_over_pytorch": (
            float(direct_reference["median_seconds_per_invocation"])
            / float(direct_kernel["median_seconds_per_invocation"])
        ),
        "public_dispatch_speedup_over_eager": (
            float(public_eager_result["median_seconds_per_invocation"])
            / float(public_dispatch["median_seconds_per_invocation"])
        ),
        "public_dispatch_speedup_over_compiled": (
            float(public_compiled_result["median_seconds_per_invocation"])
            / float(public_dispatch["median_seconds_per_invocation"])
        ),
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed MPS-005 matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MPS-005 benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    expected_distribution = (
        "triton" if args.compiler_lane == "stock_triton" else "flagtree"
    )
    if identity_status != "resolved" or distribution != expected_distribution:
        raise RuntimeError(
            f"compiler lane {args.compiler_lane!r} requires distribution "
            f"{expected_distribution!r}, found {distribution!r} with status "
            f"{identity_status!r}"
        )

    properties = torch.cuda.get_device_properties(0)
    cases = [
        _case(
            channel_count,
            batch,
            left_dim,
            right_dim,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (channel_count, batch, left_dim, right_dim) in enumerate(
            SHAPE_MATRIX
        )
    ]
    return {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": args.source_revision,
        "runner": RUNNER,
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
            "identity_source": "python_package_metadata",
            "identity_status": identity_status,
        },
        "environment": {
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
            "cuda_runtime": torch.version.cuda,
            "pytorch": torch.__version__,
            "python": sys.version.split()[0],
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "group_size": args.group_size,
            "seed": args.seed,
        },
        "cases": cases,
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def validate_run(payload: Mapping[str, Any]) -> None:
    """Fail closed when one raw run does not meet the evidence contract."""

    expected = {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"run field {field!r} must equal {value!r}")
    for field in ("source_revision", "command", "host_label"):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise ValueError(f"run field {field!r} must be a non-empty string")
    lane = payload.get("compiler_lane")
    if lane not in COMPILER_LANES:
        raise ValueError(f"unsupported compiler lane {lane!r}")
    compiler = _mapping(payload.get("compiler"), "compiler")
    expected_distribution = "triton" if lane == "stock_triton" else "flagtree"
    if (
        compiler.get("distribution") != expected_distribution
        or compiler.get("identity_status") != "resolved"
        or not compiler.get("version")
    ):
        raise ValueError(f"unresolved compiler identity for lane {lane!r}")
    measurement = _mapping(payload.get("measurement"), "measurement")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("measurement repeats must be a positive integer")
    if not isinstance(measurement.get("warmup"), int) or measurement["warmup"] < 0:
        raise ValueError("measurement warmup must be a non-negative integer")
    if (
        not isinstance(measurement.get("group_size"), int)
        or measurement["group_size"] <= 0
    ):
        raise ValueError("measurement group_size must be a positive integer")

    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes: list[tuple[int, int, int, int]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shape = (
            shape.get("channels"),
            shape.get("batch"),
            shape.get("left_bond"),
            shape.get("right_bond"),
        )
        observed_shapes.append(observed_shape)
        expected_work = (
            observed_shape[0]
            * observed_shape[1]
            * observed_shape[2]
            * observed_shape[2]
            * observed_shape[3]
            * observed_shape[3]
        )
        if case.get("contraction_work") != expected_work:
            raise ValueError(f"cases[{index}] contraction_work is not reproducible")
        if case.get("maximum_absolute_error", float("inf")) > 1e-4:
            raise ValueError(f"cases[{index}] exceeds the absolute error tolerance")
        if case.get("relative_l2_error", float("inf")) > 2e-4:
            raise ValueError(f"cases[{index}] exceeds the relative error tolerance")
        for result_name in RESULT_NAMES:
            result = _mapping(case.get(result_name), f"cases[{index}].{result_name}")
            samples = _sequence(
                result.get("samples_seconds_per_invocation"),
                f"cases[{index}].{result_name}.samples_seconds_per_invocation",
            )
            if len(samples) != repeats or any(
                not isinstance(sample, (int, float)) or sample <= 0
                for sample in samples
            ):
                raise ValueError(
                    f"cases[{index}].{result_name} must contain "
                    f"{repeats} positive samples"
                )
            if result.get("median_seconds_per_invocation") != statistics.median(
                samples
            ):
                raise ValueError(
                    f"cases[{index}].{result_name} median is not reproducible"
                )
            for memory_field in ("peak_memory_bytes", "peak_memory_delta_bytes"):
                if (
                    not isinstance(result.get(memory_field), int)
                    or result[memory_field] < 0
                ):
                    raise ValueError(
                        f"cases[{index}].{result_name}.{memory_field} is invalid"
                    )
        ratios = {
            "direct_kernel_speedup_over_pytorch": (
                case["direct_pytorch_einsum"]["median_seconds_per_invocation"]
                / case["direct_kernel_wrapper"]["median_seconds_per_invocation"]
            ),
            "public_dispatch_speedup_over_eager": (
                case["public_pytorch_eager"]["median_seconds_per_invocation"]
                / case["public_catalog_dispatch"]["median_seconds_per_invocation"]
            ),
            "public_dispatch_speedup_over_compiled": (
                case["public_pytorch_compiled"]["median_seconds_per_invocation"]
                / case["public_catalog_dispatch"]["median_seconds_per_invocation"]
            ),
        }
        for field, value in ratios.items():
            if case.get(field) != value:
                raise ValueError(f"cases[{index}] {field} is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed MPS-005 shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]], *, required_hosts: Sequence[str]
) -> dict[str, object]:
    """Combine the two-host, two-compiler matrix into one catalog artifact."""

    if len(set(required_hosts)) != len(required_hosts) or not required_hosts:
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record the same source revision")
    observed = {
        (payload["host_label"], payload["compiler_lane"]) for payload in payloads
    }
    required = {
        (host, compiler_lane)
        for host in required_hosts
        for compiler_lane in COMPILER_LANES
    }
    if observed != required or len(payloads) != len(required):
        raise ValueError(
            "runs must contain each required host and compiler lane exactly once"
        )
    ordered = sorted(
        payloads, key=lambda item: (item["host_label"], item["compiler_lane"])
    )
    direct_wins = all(
        case["direct_kernel_speedup_over_pytorch"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    eager_wins = all(
        case["public_dispatch_speedup_over_eager"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    compiled_wins = all(
        case["public_dispatch_speedup_over_compiled"] > 1.0
        for run in ordered
        for case in run["cases"]
    )
    return {
        "benchmark": "mps_environment_channels_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": next(iter(revisions)),
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "evidence_scope": "development_hardware_evidence",
        "claim_evidence_type": "development_smoke",
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-device kernel benchmark is not distributed scalability evidence"
        ],
        "required_hosts": sorted(required_hosts),
        "required_compiler_lanes": list(COMPILER_LANES),
        "direct_kernel_win_on_all_cases": direct_wins,
        "public_dispatch_win_over_eager_on_all_cases": eager_wins,
        "public_dispatch_win_over_compiled_on_all_cases": compiled_wins,
        "dispatch_selection_decision": (
            "eligible_for_default" if eager_wins and compiled_wins else "retain_opt_in"
        ),
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "benchmark": "mps_environment_channels_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "evidence_scope": "development_hardware_evidence",
        "claim_evidence_type": "development_smoke",
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-device kernel benchmark is not distributed scalability evidence"
        ],
        "required_compiler_lanes": list(COMPILER_LANES),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"evidence field {field!r} must equal {value!r}")
    hosts = _sequence(payload.get("required_hosts"), "required_hosts")
    runs = _sequence(payload.get("runs"), "runs")
    rebuilt = merge_runs(runs, required_hosts=hosts)
    if dict(payload) != rebuilt:
        raise ValueError("aggregate is not the canonical merge of its raw runs")


def _read_payload(path: Path) -> Mapping[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _write_payload(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    run = subparsers.add_parser("run", help="collect one host/compiler run")
    run.add_argument("--host-label", default=socket.gethostname())
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261003)
    run.add_argument("--output", type=Path, required=True)

    merge = subparsers.add_parser("merge", help="merge raw runs")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--required-host", action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate", help="validate an aggregate")
    validate.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        if args.warmup < 0 or args.repeats <= 0 or args.group_size <= 0:
            raise ValueError(
                "warmup must be non-negative; repeats and group_size must be positive"
            )
        payload = collect_run(args)
        validate_run(payload)
        _write_payload(args.output, payload)
    elif args.action == "merge":
        payload = merge_runs(
            [_read_payload(path) for path in args.input],
            required_hosts=args.required_host,
        )
        validate_evidence(payload)
        _write_payload(args.output, payload)
    else:
        validate_evidence(_read_payload(args.input))
        payload = {"validated": str(args.input)}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
