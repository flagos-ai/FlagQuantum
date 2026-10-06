"""Collect and validate MPS-008 public dispatch evidence on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.mps_sampling_collapse_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.mps_sampling_collapse_dispatch.v1"
SEMANTIC_ID = "mps.sampling.collapse_wire.local"
IMPLEMENTATION_ID = "FQKI-TRITON-MPS-008-A"
RUNNER = "benchmarks/mps_sampling_collapse_dispatch.py"
HOSTS = ("jp-a800-171", "jp-a800-172")
COMPILER_LANES = ("stock_triton", "flagtree")
DISPATCH_VARIABLE = "FQ_TRITON_MPS_SAMPLING_COLLAPSE"
SHAPE_MATRIX = (
    (32, 1, 1),
    (128, 8, 16),
    (512, 16, 32),
    (2048, 32, 64),
    (2048, 64, 64),
)
RESULT_NAMES = (
    "direct_kernel_wrapper",
    "direct_pytorch_reference",
    "public_catalog_dispatch",
    "public_pytorch_reference",
)
MEASUREMENT_ORDERING = "balanced Latin square across all measured operations"


class TimingResult(TypedDict):
    samples_seconds_per_invocation: list[float]
    median_seconds_per_invocation: float
    peak_memory_bytes: int
    peak_memory_delta_bytes: int


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


def _counterbalanced_orders(
    names: Sequence[str], repeats: int
) -> tuple[tuple[str, ...], ...]:
    if not names or len(set(names)) != len(names):
        raise ValueError("measured operation names must be non-empty and unique")
    if repeats <= 0 or repeats % len(names) != 0:
        raise ValueError("repeats must be a positive multiple of operation count")
    count = len(names)
    if count % 2:
        raise ValueError("balanced ordering requires an even operation count")
    first_row = tuple(
        0 if index == 0 else (index + 1) // 2 if index % 2 else count - index // 2
        for index in range(count)
    )
    return tuple(
        tuple(names[(item + repeat) % count] for item in first_row)
        for repeat in range(repeats)
    )


def _measure_group(function: Callable[[], object], group_size: int) -> float:
    started = time.perf_counter()
    for _ in range(group_size):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / group_size


def _measure_operations(
    operations: Mapping[str, Callable[[], object]],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, TimingResult]:
    if tuple(operations) != RESULT_NAMES:
        raise ValueError("measurement operations must follow RESULT_NAMES")
    previous = os.environ.get(DISPATCH_VARIABLE)
    try:
        for _ in range(warmup):
            for name, function in operations.items():
                if name.startswith("public_"):
                    os.environ[DISPATCH_VARIABLE] = (
                        "1" if name == "public_catalog_dispatch" else "0"
                    )
                function()
        torch.cuda.synchronize()
        samples: dict[str, list[float]] = {name: [] for name in operations}
        peak_memory: dict[str, list[int]] = {name: [] for name in operations}
        peak_deltas: dict[str, list[int]] = {name: [] for name in operations}
        for order in _counterbalanced_orders(tuple(operations), repeats):
            for name in order:
                if name.startswith("public_"):
                    os.environ[DISPATCH_VARIABLE] = (
                        "1" if name == "public_catalog_dispatch" else "0"
                    )
                torch.cuda.reset_peak_memory_stats()
                allocated_before = torch.cuda.memory_allocated()
                samples[name].append(_measure_group(operations[name], group_size))
                peak = torch.cuda.max_memory_allocated()
                peak_memory[name].append(peak)
                peak_deltas[name].append(max(0, peak - allocated_before))
    finally:
        if previous is None:
            os.environ.pop(DISPATCH_VARIABLE, None)
        else:
            os.environ[DISPATCH_VARIABLE] = previous
    return {
        name: {
            "samples_seconds_per_invocation": samples[name],
            "median_seconds_per_invocation": statistics.median(samples[name]),
            "peak_memory_bytes": max(peak_memory[name]),
            "peak_memory_delta_bytes": max(peak_deltas[name]),
        }
        for name in operations
    }


def _input(
    batch: int,
    right_dim: int,
    next_right_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
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
    return site, next_site, bits


def _public_operation(
    site: torch.Tensor,
    next_site: torch.Tensor,
    bits: torch.Tensor,
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    from flagquantum.simulation.mps.sampling import _collapse_sampled_wire
    from flagquantum.simulation.mps.state import MPSState

    state = MPSState([site, next_site])

    def execute() -> tuple[torch.Tensor, torch.Tensor]:
        state.tensors[0] = site
        state.tensors[1] = next_site
        state.orthogonality_center = 0
        state._canonical_center_valid = True
        _collapse_sampled_wire(state, 0, bits)
        return state.tensors[0], state.tensors[1]

    return execute


def _case(
    batch: int,
    right_dim: int,
    next_right_dim: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.mps_sampling_collapse import (
        fused_mps_sampling_collapse,
    )
    from flagquantum.simulation.mps.site_kernels import (
        reset_site_kernel_stats,
        site_kernel_stats,
    )

    site, next_site, bits = _input(
        batch,
        right_dim,
        next_right_dim,
        seed=seed,
    )
    reference = _reference(site, next_site, bits)
    direct = fused_mps_sampling_collapse(site, next_site, bits)
    public_operation = _public_operation(site, next_site, bits)
    previous = os.environ.get(DISPATCH_VARIABLE)
    try:
        reset_site_kernel_stats(clear_cache=True)
        os.environ[DISPATCH_VARIABLE] = "1"
        public = public_operation()
        dispatch_stats = site_kernel_stats()
        if (
            dispatch_stats["triton_sampling_collapse_calls"] != 1
            or dispatch_stats["sampling_collapse_fallback_calls"] != 0
        ):
            raise RuntimeError("public MPS-008 dispatch did not use the catalog route")
        reset_site_kernel_stats(clear_cache=True)
        os.environ[DISPATCH_VARIABLE] = "0"
        public_reference = public_operation()
        reference_stats = site_kernel_stats()
        if (
            reference_stats["triton_sampling_collapse_calls"] != 0
            or reference_stats["sampling_collapse_fallback_calls"] != 1
        ):
            raise RuntimeError(
                "disabled MPS-008 dispatch did not use the reference route"
            )
    finally:
        if previous is None:
            os.environ.pop(DISPATCH_VARIABLE, None)
        else:
            os.environ[DISPATCH_VARIABLE] = previous
    for actual in (direct, public):
        torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], reference[1], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(public_reference[0], reference[0], rtol=0, atol=0)
    torch.testing.assert_close(public_reference[1], reference[1], rtol=1e-5, atol=1e-5)

    timings = _measure_operations(
        {
            "direct_kernel_wrapper": lambda: fused_mps_sampling_collapse(
                site, next_site, bits
            ),
            "direct_pytorch_reference": lambda: _reference(site, next_site, bits),
            "public_catalog_dispatch": public_operation,
            "public_pytorch_reference": public_operation,
        },
        warmup=warmup,
        repeats=repeats,
        group_size=group_size,
    )
    difference = public[1] - reference[1]
    return {
        "shape": {
            "batch": batch,
            "right_bond": right_dim,
            "next_right_bond": next_right_dim,
        },
        "dtype": "complex64",
        "layout": "contiguous_mps_sampling_step",
        "maximum_absolute_error": float(torch.max(torch.abs(difference))),
        "relative_l2_error": float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference[1]).clamp_min(
                torch.finfo(torch.float32).eps
            )
        ),
        **timings,
        "direct_kernel_speedup_over_pytorch": (
            timings["direct_pytorch_reference"]["median_seconds_per_invocation"]
            / timings["direct_kernel_wrapper"]["median_seconds_per_invocation"]
        ),
        "public_dispatch_speedup_over_pytorch": (
            timings["public_pytorch_reference"]["median_seconds_per_invocation"]
            / timings["public_catalog_dispatch"]["median_seconds_per_invocation"]
        ),
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one host/compiler lane of the fixed MPS-008 dispatch matrix."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MPS-008 dispatch evidence")
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
    cases = [
        _case(
            batch,
            right_dim,
            next_right_dim,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (batch, right_dim, next_right_dim) in enumerate(SHAPE_MATRIX)
    ]
    properties = torch.cuda.get_device_properties(0)
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
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": properties.name,
            "gpu_total_memory_bytes": properties.total_memory,
        },
        "measurement": {
            "clock": "time.perf_counter",
            "synchronization": "torch.cuda.synchronize after each invocation group",
            "ordering": MEASUREMENT_ORDERING,
            "order_cycle_length": len(RESULT_NAMES),
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
    if not isinstance(repeats, int) or repeats <= 0 or repeats % len(RESULT_NAMES) != 0:
        raise ValueError("measurement repeats must be a positive complete cycle")
    if measurement.get("ordering") != MEASUREMENT_ORDERING:
        raise ValueError("measurement ordering must be counterbalanced")

    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shapes.append(
            (shape.get("batch"), shape.get("right_bond"), shape.get("next_right_bond"))
        )
        if case.get("maximum_absolute_error", float("inf")) > 2e-5:
            raise ValueError(f"cases[{index}] exceeds the absolute error tolerance")
        if case.get("relative_l2_error", float("inf")) > 2e-5:
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
                raise ValueError(f"cases[{index}].{result_name} samples are invalid")
            if result.get("median_seconds_per_invocation") != statistics.median(
                samples
            ):
                raise ValueError(
                    f"cases[{index}].{result_name} median is not canonical"
                )
        expected_direct = (
            case["direct_pytorch_reference"]["median_seconds_per_invocation"]
            / case["direct_kernel_wrapper"]["median_seconds_per_invocation"]
        )
        expected_public = (
            case["public_pytorch_reference"]["median_seconds_per_invocation"]
            / case["public_catalog_dispatch"]["median_seconds_per_invocation"]
        )
        if case.get("direct_kernel_speedup_over_pytorch") != expected_direct:
            raise ValueError(f"cases[{index}] direct speedup is not reproducible")
        if case.get("public_dispatch_speedup_over_pytorch") != expected_public:
            raise ValueError(f"cases[{index}] public speedup is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed MPS-008 shape matrix")


def merge_runs(payloads: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Combine the two-host, two-compiler matrix into one catalog artifact."""

    for payload in payloads:
        validate_run(payload)
    expected = {(host, lane) for host in HOSTS for lane in COMPILER_LANES}
    observed = {
        (payload["host_label"], payload["compiler_lane"]) for payload in payloads
    }
    if observed != expected or len(payloads) != len(expected):
        raise ValueError("MPS-008 evidence host/compiler matrix is incomplete")
    revisions = {payload["source_revision"] for payload in payloads}
    measurements = {
        json.dumps(payload["measurement"], sort_keys=True) for payload in payloads
    }
    if len(revisions) != 1 or len(measurements) != 1:
        raise ValueError("MPS-008 evidence must use one source and measurement policy")
    ordered = sorted(
        payloads, key=lambda item: (item["host_label"], item["compiler_lane"])
    )
    direct_speedups = [
        float(case["direct_kernel_speedup_over_pytorch"])
        for run in ordered
        for case in run["cases"]
    ]
    public_speedups = [
        float(case["public_dispatch_speedup_over_pytorch"])
        for run in ordered
        for case in run["cases"]
    ]
    public_wins = all(speedup > 1.0 for speedup in public_speedups)
    return {
        "benchmark": "mps_sampling_collapse_dispatch",
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
        "required_hosts": list(HOSTS),
        "required_compiler_lanes": list(COMPILER_LANES),
        "minimum_direct_speedup_over_pytorch": min(direct_speedups),
        "maximum_direct_speedup_over_pytorch": max(direct_speedups),
        "minimum_public_speedup_over_pytorch": min(public_speedups),
        "maximum_public_speedup_over_pytorch": max(public_speedups),
        "public_dispatch_win_on_all_cases": public_wins,
        "dispatch_selection_decision": (
            "eligible_for_default" if public_wins else "retain_opt_in"
        ),
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    runs = _sequence(payload.get("runs"), "runs")
    rebuilt = merge_runs(runs)
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
    run = subparsers.add_parser("run")
    run.add_argument("--host-label", choices=HOSTS, required=True)
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=32)
    run.add_argument("--group-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=261_006)
    run.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "run":
        if args.warmup < 0 or args.repeats <= 0 or args.group_size <= 0:
            raise ValueError("warmup must be non-negative and timing counts positive")
        payload = collect_run(args)
        validate_run(payload)
        _write_payload(args.output, payload)
    elif args.action == "merge":
        payload = merge_runs([_read_payload(path) for path in args.input])
        validate_evidence(payload)
        _write_payload(args.output, payload)
    else:
        validate_evidence(_read_payload(args.input))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
