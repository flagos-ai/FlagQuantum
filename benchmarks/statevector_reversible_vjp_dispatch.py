"""Collect two-host, two-compiler dispatch evidence for GR-002."""

from __future__ import annotations

import argparse
import json
import re
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

RUN_SCHEMA = "flagquantum.kernel_benchmark_run.statevector_reversible_vjp_dispatch.v1"
EVIDENCE_SCHEMA = "flagquantum.kernel_benchmark.statevector_reversible_vjp_dispatch.v1"
SEMANTIC_ID = "gradient.vjp.reversible_1q.local"
IMPLEMENTATION_ID = "FQKI-TRITON-GR-002-A"
RUNNER = "benchmarks/statevector_reversible_vjp_dispatch.py"
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (1, 2, 0, True),
    (1, 1 << 10, 0, True),
    (1, 1 << 16, 8, True),
    (1, 1 << 20, 10, True),
    (1, 1 << 24, 12, True),
    (8, 1 << 20, 7, True),
)
RESULT_NAMES = ("catalog_dispatch", "pytorch_eager", "torch_compile")
PERFORMANCE_FLOOR = 1.0
STATE_ATOL = 3e-5
STATE_RTOL = 3e-5
GRADIENT_ATOL = 3e-3
GRADIENT_RTOL = 3e-5
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _measure_mutating(
    prepare: Callable[[], None],
    function: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    """Measure only one mutation after restoring inputs outside the timed region."""

    for _ in range(warmup):
        prepare()
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        prepare()
        torch.cuda.synchronize()
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
    }


def _measure(
    function: Callable[[], object], *, warmup: int, repeats: int
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - started)
    return {
        "samples_seconds_per_invocation": samples,
        "median_seconds_per_invocation": statistics.median(samples),
    }


def _apply_local(
    state: torch.Tensor, matrix: torch.Tensor, bit_position: int
) -> torch.Tensor:
    batch, size = state.shape
    high = size >> (bit_position + 1)
    width = 1 << bit_position
    paired = state.reshape(batch, high, 2, width)
    return torch.einsum("ij,bhjw->bhiw", matrix, paired).reshape_as(state)


def _pytorch_reference(
    before: torch.Tensor,
    adjoint: torch.Tensor,
    matrix: torch.Tensor,
    derivative: torch.Tensor,
    bit_position: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    next_adjoint = _apply_local(adjoint, matrix.mH, bit_position)
    derivative_state = _apply_local(before, derivative, bit_position)
    gradient = torch.real(torch.sum(torch.conj(adjoint) * derivative_state))
    return before, next_adjoint, gradient


def _median(result: Mapping[str, object]) -> float:
    value = result.get("median_seconds_per_invocation")
    if not isinstance(value, (int, float)):
        raise TypeError("benchmark median must be numeric")
    return float(value)


def _case(
    batch: int,
    amplitudes: int,
    bit_position: int,
    default_eligible: bool,
    *,
    seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.statevector_adjoint import (
        fused_complex64_local_1q_reversible_vjp,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    before = torch.randn(
        batch, amplitudes, device="cuda", dtype=torch.complex64, generator=generator
    )
    adjoint = torch.randn(
        batch, amplitudes, device="cuda", dtype=torch.complex64, generator=generator
    )
    angle = torch.tensor(0.37, device="cuda")
    cosine, sine = torch.cos(angle / 2), torch.sin(angle / 2)
    matrix = torch.stack(
        (
            torch.stack((cosine, -1j * sine)),
            torch.stack((-1j * sine, cosine)),
        )
    ).to(torch.complex64)
    derivative = torch.stack(
        (
            torch.stack((-0.5 * sine, -0.5j * cosine)),
            torch.stack((-0.5j * cosine, -0.5 * sine)),
        )
    ).to(torch.complex64)
    after = _apply_local(before, matrix, bit_position).contiguous()
    work_ket = torch.empty_like(after)
    work_adjoint = torch.empty_like(adjoint)
    compiled_reference = torch.compile(
        _pytorch_reference, fullgraph=True, dynamic=False
    )

    def prepare_catalog() -> None:
        work_ket.copy_(after)
        work_adjoint.copy_(adjoint)

    def catalog_dispatch() -> torch.Tensor:
        return fused_complex64_local_1q_reversible_vjp(
            work_ket,
            work_adjoint,
            matrix,
            derivative,
            bit_position=bit_position,
        )

    def pytorch_eager() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _pytorch_reference(before, adjoint, matrix, derivative, bit_position)

    def compiled_pytorch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return compiled_reference(before, adjoint, matrix, derivative, bit_position)

    prepare_catalog()
    actual_gradient = catalog_dispatch()
    expected_ket, expected_adjoint, expected_gradient = pytorch_eager()
    compiled_ket, compiled_adjoint, compiled_gradient = compiled_pytorch()
    torch.testing.assert_close(work_ket, expected_ket, atol=STATE_ATOL, rtol=STATE_RTOL)
    torch.testing.assert_close(
        work_adjoint, expected_adjoint, atol=STATE_ATOL, rtol=STATE_RTOL
    )
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=GRADIENT_ATOL, rtol=GRADIENT_RTOL
    )
    torch.testing.assert_close(
        compiled_ket, expected_ket, atol=STATE_ATOL, rtol=STATE_RTOL
    )
    torch.testing.assert_close(
        compiled_adjoint, expected_adjoint, atol=STATE_ATOL, rtol=STATE_RTOL
    )
    torch.testing.assert_close(
        compiled_gradient, expected_gradient, atol=GRADIENT_ATOL, rtol=GRADIENT_RTOL
    )
    ket_difference = work_ket - expected_ket
    adjoint_difference = work_adjoint - expected_adjoint
    gradient_difference = torch.abs(actual_gradient - expected_gradient)
    epsilon = torch.finfo(torch.float32).eps
    results = {
        "catalog_dispatch": _measure_mutating(
            prepare_catalog, catalog_dispatch, warmup=warmup, repeats=repeats
        ),
        "pytorch_eager": _measure(pytorch_eager, warmup=warmup, repeats=repeats),
        "torch_compile": _measure(compiled_pytorch, warmup=warmup, repeats=repeats),
    }
    catalog_median = _median(results["catalog_dispatch"])
    return {
        "shape": {
            "batch": batch,
            "amplitudes_per_batch": amplitudes,
            "bit_position": bit_position,
        },
        "dtype": "complex64",
        "layout": "contiguous_flat_statevector",
        "default_eligible": default_eligible,
        "restored_ket_reference_maximum_absolute_value": float(
            torch.max(torch.abs(expected_ket))
        ),
        "restored_ket_maximum_absolute_error": float(
            torch.max(torch.abs(ket_difference))
        ),
        "restored_ket_relative_l2_error": float(
            torch.linalg.vector_norm(ket_difference)
            / torch.linalg.vector_norm(expected_ket).clamp_min(epsilon)
        ),
        "next_adjoint_reference_maximum_absolute_value": float(
            torch.max(torch.abs(expected_adjoint))
        ),
        "next_adjoint_maximum_absolute_error": float(
            torch.max(torch.abs(adjoint_difference))
        ),
        "next_adjoint_relative_l2_error": float(
            torch.linalg.vector_norm(adjoint_difference)
            / torch.linalg.vector_norm(expected_adjoint).clamp_min(epsilon)
        ),
        "gradient_reference": float(expected_gradient),
        "gradient_absolute_error": float(gradient_difference),
        "gradient_relative_error": float(
            gradient_difference / torch.abs(expected_gradient).clamp_min(epsilon)
        ),
        **results,
        "speedup_over_pytorch_eager": _median(results["pytorch_eager"])
        / catalog_median,
        "speedup_over_torch_compile": _median(results["torch_compile"])
        / catalog_median,
    }


def collect_run(args: argparse.Namespace) -> dict[str, object]:
    """Execute one fixed compiler lane and return its raw measurements."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GR-002 benchmark evidence")
    distribution, version, integration_path, identity_status = (
        triton_compiler_provenance()
    )
    expected = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[args.compiler_lane]
    if (
        identity_status != "resolved"
        or distribution != expected[0]
        or integration_path != expected[1]
    ):
        raise RuntimeError(
            f"{args.compiler_lane} requires compiler identity {expected!r}; "
            f"found {(distribution, integration_path)!r}"
        )
    import triton

    properties = torch.cuda.get_device_properties(0)
    cases = [
        _case(
            batch,
            amplitudes,
            bit_position,
            default_eligible,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for index, (
            batch,
            amplitudes,
            bit_position,
            default_eligible,
        ) in enumerate(SHAPE_MATRIX)
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
        "compiler_lane": args.compiler_lane,
        "execution_semantics": "single_device_mutating_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "compiler": {
            "distribution": distribution,
            "version": version,
            "triton_api_version": triton.__version__,
            "integration_path": integration_path,
            "backend": triton.runtime.driver.active.get_current_target().backend,
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
            "synchronization": "input restore and synchronize before each timed invocation; synchronize after invocation",
            "input_restore_in_timed_region": False,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "group_size": 1,
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
    """Fail closed when one raw run violates the fixed evidence contract."""

    for field, value in {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_mutating_fast_path",
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
    }.items():
        if payload.get(field) != value:
            raise ValueError(f"run field {field!r} must equal {value!r}")
    revision = payload.get("source_revision")
    if not isinstance(revision, str) or _FULL_REVISION.fullmatch(revision) is None:
        raise ValueError("source_revision must be a full lowercase Git revision")
    lane = payload.get("compiler_lane")
    if lane not in COMPILER_LANES:
        raise ValueError("compiler_lane is not recognized")
    compiler = _mapping(payload.get("compiler"), "compiler")
    expected = {
        "stock_triton": ("triton", "direct"),
        "flagtree": ("flagtree", "flagtree"),
    }[lane]
    if (
        compiler.get("distribution") != expected[0]
        or compiler.get("integration_path") != expected[1]
        or compiler.get("backend") != "cuda"
        or compiler.get("identity_status") != "resolved"
    ):
        raise ValueError("run compiler identity does not match its lane")
    measurement = _mapping(payload.get("measurement"), "measurement")
    repeats = measurement.get("repeats")
    if not isinstance(repeats, int) or repeats <= 0:
        raise ValueError("measurement repeats must be positive")
    if measurement.get("group_size") != 1:
        raise ValueError("mutating evidence requires group_size=1")
    if measurement.get("input_restore_in_timed_region") is not False:
        raise ValueError("input restore must be outside the timed region")
    observed = []
    for index, case_value in enumerate(_sequence(payload.get("cases"), "cases")):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed.append(
            (
                shape.get("batch"),
                shape.get("amplitudes_per_batch"),
                shape.get("bit_position"),
                case.get("default_eligible"),
            )
        )
        checks = (
            (
                "restored_ket",
                case.get("restored_ket_reference_maximum_absolute_value"),
                case.get("restored_ket_maximum_absolute_error"),
                STATE_ATOL,
                STATE_RTOL,
            ),
            (
                "next_adjoint",
                case.get("next_adjoint_reference_maximum_absolute_value"),
                case.get("next_adjoint_maximum_absolute_error"),
                STATE_ATOL,
                STATE_RTOL,
            ),
            (
                "gradient",
                abs(case.get("gradient_reference"))
                if isinstance(case.get("gradient_reference"), (int, float))
                else None,
                case.get("gradient_absolute_error"),
                GRADIENT_ATOL,
                GRADIENT_RTOL,
            ),
        )
        for name, reference, error, atol, rtol in checks:
            if not isinstance(reference, (int, float)) or not isinstance(
                error, (int, float)
            ):
                raise ValueError(f"cases[{index}] {name} values must be numeric")
            if float(error) > atol + rtol * float(reference):
                raise ValueError(f"cases[{index}] exceeds the {name} tolerance")
        medians = {}
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
            median = statistics.median(samples)
            if result.get("median_seconds_per_invocation") != median:
                raise ValueError(f"cases[{index}].{result_name} median is invalid")
            medians[result_name] = median
        speedups = {
            "speedup_over_pytorch_eager": medians["pytorch_eager"]
            / medians["catalog_dispatch"],
            "speedup_over_torch_compile": medians["torch_compile"]
            / medians["catalog_dispatch"],
        }
        for name, speedup in speedups.items():
            if case.get(name) != speedup:
                raise ValueError(f"cases[{index}] {name} is not reproducible")
            if case.get("default_eligible") and speedup < PERFORMANCE_FLOOR:
                raise ValueError(f"cases[{index}] misses the {name} performance floor")
    if tuple(observed) != SHAPE_MATRIX:
        raise ValueError("run does not contain the fixed GR-002 shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]], *, required_hosts: Sequence[str]
) -> dict[str, object]:
    if not required_hosts or len(set(required_hosts)) != len(required_hosts):
        raise ValueError("required hosts must be unique and non-empty")
    for payload in payloads:
        validate_run(payload)
    revisions = {payload["source_revision"] for payload in payloads}
    if len(revisions) != 1:
        raise ValueError("all runs must record one source revision")
    observed = {(run["host_label"], run["compiler_lane"]) for run in payloads}
    expected = {(host, lane) for host in required_hosts for lane in COMPILER_LANES}
    if observed != expected or len(payloads) != len(expected):
        raise ValueError("runs must contain every host/compiler lane exactly once")
    ordered = sorted(
        payloads, key=lambda item: (item["host_label"], item["compiler_lane"])
    )
    eager_speedups = [
        float(case["speedup_over_pytorch_eager"])
        for run in ordered
        for case in run["cases"]
        if case["default_eligible"]
    ]
    compiled_speedups = [
        float(case["speedup_over_torch_compile"])
        for run in ordered
        for case in run["cases"]
        if case["default_eligible"]
    ]
    return {
        "benchmark": "statevector_reversible_vjp_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "source_revision": next(iter(revisions)),
        "runner": RUNNER,
        "execution_semantics": "single_device_mutating_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "evidence_scope": "development_hardware_evidence",
        "claim_evidence_type": "development_smoke",
        "benchmark_evidence_class": "local_non_release",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "scalability_blockers": [
            "single-device kernel evidence is not distributed scalability evidence"
        ],
        "required_hosts": sorted(required_hosts),
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "minimum_default_window_speedup_over_pytorch_eager": min(eager_speedups),
        "minimum_default_window_speedup_over_torch_compile": min(compiled_speedups),
        "default_window_passed": all(
            speedup >= PERFORMANCE_FLOOR
            for speedup in (*eager_speedups, *compiled_speedups)
        ),
        "dispatch_decision": "eligible_for_default",
        "runs": ordered,
    }


def validate_evidence(payload: Mapping[str, Any]) -> None:
    for field, value in {
        "benchmark": "statevector_reversible_vjp_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_mutating_fast_path",
        "distribution_semantics": "single_device_fast_path",
        "non_release_evidence": True,
        "release_gate_allowed": False,
        "scalability_claim_allowed": False,
        "required_compiler_lanes": list(COMPILER_LANES),
        "performance_floor": PERFORMANCE_FLOOR,
        "dispatch_decision": "eligible_for_default",
    }.items():
        if payload.get(field) != value:
            raise ValueError(f"evidence field {field!r} must equal {value!r}")
    required_hosts = _sequence(payload.get("required_hosts"), "required_hosts")
    runs = _sequence(payload.get("runs"), "runs")
    rebuilt = merge_runs(runs, required_hosts=required_hosts)
    for field in (
        "source_revision",
        "minimum_default_window_speedup_over_pytorch_eager",
        "minimum_default_window_speedup_over_torch_compile",
        "default_window_passed",
    ):
        if payload.get(field) != rebuilt[field]:
            raise ValueError(f"evidence field {field!r} is not reproducible")
    if not payload.get("scalability_blockers"):
        raise ValueError("local evidence must name its scalability blocker")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--host-label", required=True)
    run.add_argument("--compiler-lane", choices=COMPILER_LANES, required=True)
    run.add_argument("--source-revision", required=True)
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--repeats", type=int, default=30)
    run.add_argument("--seed", type=int, default=261004)
    run.add_argument("--output", type=Path, required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--required-host", action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        payload = collect_run(args)
        validate_run(payload)
        _write_json(args.output, payload)
    elif args.command == "merge":
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
        payload = merge_runs(payloads, required_hosts=args.required_host)
        validate_evidence(payload)
        _write_json(args.output, payload)
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        validate_evidence(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
