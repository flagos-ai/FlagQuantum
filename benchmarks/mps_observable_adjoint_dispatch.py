"""Collect and validate MPS-006 dispatch evidence on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from flagquantum.kernels.provenance import triton_compiler_provenance

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.mps_observable_adjoint.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.mps_observable_adjoint.v1"
SEMANTIC_ID = "mps.gradient.hermitian_observable_adjoint.local"
IMPLEMENTATION_ID = "FQKI-TRITON-MPS-006-A"
RUNNER = "benchmarks/mps_observable_adjoint_dispatch.py"
SHAPE_MATRIX = ((1, 4, 8), (1, 16, 32), (8, 16, 16), (8, 32, 32))
COMPILER_LANES = ("stock_triton", "flagtree")
DISPATCH_VARIABLE = "FQ_TRITON_MPS_OBSERVABLE_ADJOINT"


def _hermitian(value: torch.Tensor) -> torch.Tensor:
    return (value + value.mH).contiguous()


def _inputs(
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    complex_shape = (batch, left_dim, 2, right_dim)
    tensor = torch.randn(
        complex_shape,
        generator=generator,
        device="cuda",
        dtype=torch.complex64,
    )
    left = _hermitian(
        torch.randn(
            batch,
            left_dim,
            left_dim,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
    )
    right = _hermitian(
        torch.randn(
            batch,
            right_dim,
            right_dim,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
    )
    operator = _hermitian(
        torch.randn(
            2,
            2,
            generator=generator,
            device="cuda",
            dtype=torch.complex64,
        )
    )
    weights = torch.randn(
        batch,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )
    return tensor, left, right, operator, weights


def _execute(inputs: tuple[torch.Tensor, ...], *, dispatch: bool) -> torch.Tensor:
    from flagquantum.simulation.mps.observables import (
        mps_local_observable_adjoint,
    )

    previous = os.environ.get(DISPATCH_VARIABLE)
    os.environ[DISPATCH_VARIABLE] = "1" if dispatch else "0"
    try:
        return mps_local_observable_adjoint(*inputs, hermitian=True)
    finally:
        if previous is None:
            os.environ.pop(DISPATCH_VARIABLE, None)
        else:
            os.environ[DISPATCH_VARIABLE] = previous


def _measure(
    function: Callable[[], torch.Tensor], *, warmup: int, repeats: int
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    return {
        "samples_seconds": samples,
        "median_seconds": median,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "peak_memory_delta_bytes": max(
            0, torch.cuda.max_memory_allocated() - allocated_before
        ),
    }


def _case(
    batch: int,
    left_dim: int,
    right_dim: int,
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    inputs = _inputs(batch, left_dim, right_dim, seed=seed)
    actual = _execute(inputs, dispatch=True)
    reference = _execute(inputs, dispatch=False)
    difference = actual - reference
    maximum_error = float(torch.max(torch.abs(difference)))
    relative_l2 = float(
        torch.linalg.vector_norm(difference)
        / torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).eps)
    )
    torch.testing.assert_close(actual, reference, rtol=1e-3, atol=5e-4)

    dispatch_result = _measure(
        lambda: _execute(inputs, dispatch=True), warmup=warmup, repeats=repeats
    )
    reference_result = _measure(
        lambda: _execute(inputs, dispatch=False), warmup=warmup, repeats=repeats
    )
    return {
        "shape": {
            "batch": batch,
            "left_bond": left_dim,
            "physical_dimension": 2,
            "right_bond": right_dim,
        },
        "dtype": "complex64",
        "layout": "contiguous_mps_local_observable",
        "maximum_absolute_error": maximum_error,
        "relative_l2_error": relative_l2,
        "catalog_dispatch": dispatch_result,
        "pytorch_autograd": reference_result,
        "speedup_over_pytorch_autograd": (
            float(reference_result["median_seconds"])
            / float(dispatch_result["median_seconds"])
        ),
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the fixed MPS-006 matrix and return one raw run."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MPS-006 benchmark evidence")
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
            batch,
            left_dim,
            right_dim,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for index, (batch, left_dim, right_dim) in enumerate(SHAPE_MATRIX)
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
            "synchronization": "torch.cuda.synchronize after every invocation",
            "warmup": args.warmup,
            "repeats": args.repeats,
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

    cases = _sequence(payload.get("cases"), "cases")
    observed_shapes: list[tuple[int, int, int]] = []
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed_shapes.append(
            (shape.get("batch"), shape.get("left_bond"), shape.get("right_bond"))
        )
        if case.get("maximum_absolute_error", float("inf")) > 5e-4:
            raise ValueError(f"cases[{index}] exceeds the absolute error tolerance")
        if case.get("relative_l2_error", float("inf")) > 1e-3:
            raise ValueError(f"cases[{index}] exceeds the relative error tolerance")
        for result_name in ("catalog_dispatch", "pytorch_autograd"):
            result = _mapping(case.get(result_name), f"cases[{index}].{result_name}")
            samples = _sequence(
                result.get("samples_seconds"),
                f"cases[{index}].{result_name}.samples_seconds",
            )
            if len(samples) != repeats or any(
                not isinstance(sample, (int, float)) or sample <= 0
                for sample in samples
            ):
                raise ValueError(
                    f"cases[{index}].{result_name} must contain "
                    f"{repeats} positive samples"
                )
            if result.get("median_seconds") != statistics.median(samples):
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
        expected_speedup = (
            case["pytorch_autograd"]["median_seconds"]
            / case["catalog_dispatch"]["median_seconds"]
        )
        if case.get("speedup_over_pytorch_autograd") != expected_speedup:
            raise ValueError(f"cases[{index}] speedup is not reproducible")
    if tuple(observed_shapes) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed MPS-006 shape matrix")


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
    return {
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": next(iter(revisions)),
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "evidence_scope": "development_hardware_evidence",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "required_hosts": sorted(required_hosts),
        "required_compiler_lanes": list(COMPILER_LANES),
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    """Validate a checked-in aggregate without importing accelerator providers."""

    expected = {
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
        "evidence_scope": "development_hardware_evidence",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
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
    run.add_argument("--seed", type=int, default=260930)
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
        if args.warmup < 0 or args.repeats <= 0:
            raise ValueError("warmup must be non-negative and repeats must be positive")
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
