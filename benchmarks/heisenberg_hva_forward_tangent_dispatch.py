"""Collect two-host, two-compiler dispatch evidence for GR-006."""

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

RUN_SCHEMA = (
    "flagquantum.kernel_benchmark_run.heisenberg_hva_forward_tangent_dispatch.v1"
)
EVIDENCE_SCHEMA = (
    "flagquantum.kernel_benchmark.heisenberg_hva_forward_tangent_dispatch.v1"
)
SEMANTIC_ID = "gradient.forward_tangent.heisenberg_hva"
IMPLEMENTATION_ID = "FQKI-TRITON-GR-006-A"
RUNNER = "benchmarks/heisenberg_hva_forward_tangent_dispatch.py"
COMPILER_LANES = ("stock_triton", "flagtree")
SHAPE_MATRIX = (
    (2, 1, True),
    (4, 1, True),
    (4, 3, True),
    (6, 3, True),
    (8, 2, True),
    (10, 1, True),
)
RESULT_NAMES = ("catalog_dispatch", "pytorch_eager", "torch_compile")
PERFORMANCE_FLOOR = 1.0
STATE_ATOL = 2e-5
STATE_RTOL = 2e-5
TANGENT_ATOL = 2e-5
TANGENT_RTOL = 2e-5
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")


def _measure(
    function: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
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
    }


def _pauli_actions(
    n_wires: int, device: torch.device | str
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Build the fixed basis permutation and sign for one HVA layer."""

    dimension = 1 << n_wires
    basis = torch.arange(dimension, device=device)
    actions: list[tuple[torch.Tensor, torch.Tensor]] = []
    for family in range(3):
        for left in range(n_wires - 1):
            bit_left = n_wires - 1 - left
            bit_right = n_wires - 2 - left
            left_value = torch.bitwise_and(
                torch.bitwise_right_shift(basis, bit_left), 1
            )
            right_value = torch.bitwise_and(
                torch.bitwise_right_shift(basis, bit_right), 1
            )
            if family == 0:
                permutation = torch.bitwise_xor(
                    basis, (1 << bit_left) | (1 << bit_right)
                )
                sign = torch.ones(dimension, dtype=torch.float32, device=device)
            elif family == 1:
                permutation = torch.bitwise_xor(
                    basis, (1 << bit_left) | (1 << bit_right)
                )
                sign = torch.where(
                    left_value == right_value,
                    -torch.ones(dimension, dtype=torch.float32, device=device),
                    torch.ones(dimension, dtype=torch.float32, device=device),
                )
            else:
                permutation = basis
                sign = torch.where(
                    left_value == right_value,
                    torch.ones(dimension, dtype=torch.float32, device=device),
                    -torch.ones(dimension, dtype=torch.float32, device=device),
                )
            actions.append((permutation, sign))
    for wire in range(n_wires):
        bit = n_wires - 1 - wire
        value = torch.bitwise_and(torch.bitwise_right_shift(basis, bit), 1)
        sign = torch.where(
            value == 0,
            torch.ones(dimension, dtype=torch.float32, device=device),
            -torch.ones(dimension, dtype=torch.float32, device=device),
        )
        actions.append((basis, sign))
    return tuple(actions)


def _pytorch_reference(
    initial_state: torch.Tensor,
    parameters: torch.Tensor,
    actions: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    depth: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate the HVA state and all tangents with ordinary PyTorch."""

    count = int(parameters.numel())
    parameter_lanes = torch.arange(count, device=parameters.device).reshape(-1, 1)
    state = initial_state.reshape(-1)
    tangents = torch.zeros(
        count, state.numel(), dtype=state.dtype, device=state.device
    )
    parameter = 0
    for _ in range(depth):
        for permutation, sign in actions:
            transformed_state = state[permutation] * sign
            transformed_tangents = tangents[:, permutation] * sign.unsqueeze(0)
            half = 0.5 * parameters[parameter]
            sine, cosine = torch.sin(half), torch.cos(half)
            derivative = -0.5 * sine * state - 0.5j * cosine * transformed_state
            tangents = cosine * tangents - 1j * sine * transformed_tangents
            tangents = tangents + (
                parameter_lanes == parameter
            ) * derivative.unsqueeze(0)
            state = cosine * state - 1j * sine * transformed_state
            parameter += 1
    return state, tangents


def _median(result: Mapping[str, object]) -> float:
    value = result.get("median_seconds_per_invocation")
    if not isinstance(value, (int, float)):
        raise TypeError("benchmark median must be numeric")
    return float(value)


def _case(
    n_wires: int,
    depth: int,
    default_eligible: bool,
    *,
    seed: int,
    warmup: int,
    repeats: int,
    group_size: int,
) -> dict[str, object]:
    from flagquantum.kernels.triton.hva_forward_tangent import (
        heisenberg_hva_forward_tangents,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    dimension = 1 << n_wires
    parameter_count = (4 * n_wires - 3) * depth
    initial_state = torch.randn(
        dimension,
        device="cuda",
        dtype=torch.complex64,
        generator=generator,
    )
    initial_state = initial_state / torch.linalg.vector_norm(initial_state)
    parameters = torch.randn(
        parameter_count,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    actions = _pauli_actions(n_wires, initial_state.device)
    compiled_reference = torch.compile(
        _pytorch_reference, fullgraph=True, dynamic=False
    )

    def catalog_dispatch() -> tuple[torch.Tensor, torch.Tensor]:
        return heisenberg_hva_forward_tangents(
            initial_state, parameters, n_qubits=n_wires, depth=depth
        )

    def pytorch_eager() -> tuple[torch.Tensor, torch.Tensor]:
        return _pytorch_reference(initial_state, parameters, actions, depth)

    def compiled_pytorch() -> tuple[torch.Tensor, torch.Tensor]:
        return compiled_reference(initial_state, parameters, actions, depth)

    actual_state, actual_tangents = catalog_dispatch()
    expected_state, expected_tangents = pytorch_eager()
    compiled_state, compiled_tangents = compiled_pytorch()
    torch.testing.assert_close(
        actual_state, expected_state, atol=STATE_ATOL, rtol=STATE_RTOL
    )
    torch.testing.assert_close(
        actual_tangents,
        expected_tangents,
        atol=TANGENT_ATOL,
        rtol=TANGENT_RTOL,
    )
    torch.testing.assert_close(
        compiled_state, expected_state, atol=STATE_ATOL, rtol=STATE_RTOL
    )
    torch.testing.assert_close(
        compiled_tangents,
        expected_tangents,
        atol=TANGENT_ATOL,
        rtol=TANGENT_RTOL,
    )
    state_difference = actual_state - expected_state
    tangent_difference = actual_tangents - expected_tangents
    results = {
        "catalog_dispatch": _measure(
            catalog_dispatch,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "pytorch_eager": _measure(
            pytorch_eager,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
        "torch_compile": _measure(
            compiled_pytorch,
            warmup=warmup,
            repeats=repeats,
            group_size=group_size,
        ),
    }
    catalog_median = _median(results["catalog_dispatch"])
    return {
        "shape": {
            "n_wires": n_wires,
            "depth": depth,
            "parameter_count": parameter_count,
            "dimension": dimension,
        },
        "dtype": "complex64",
        "layout": "flat_statevector_and_parameter_major_tangents",
        "default_eligible": default_eligible,
        "state_reference_maximum_absolute_value": float(
            torch.max(torch.abs(expected_state))
        ),
        "state_maximum_absolute_error": float(torch.max(torch.abs(state_difference))),
        "state_relative_l2_error": float(
            torch.linalg.vector_norm(state_difference)
            / torch.linalg.vector_norm(expected_state).clamp_min(
                torch.finfo(torch.float32).eps
            )
        ),
        "tangent_reference_maximum_absolute_value": float(
            torch.max(torch.abs(expected_tangents))
        ),
        "tangent_maximum_absolute_error": float(
            torch.max(torch.abs(tangent_difference))
        ),
        "tangent_relative_l2_error": float(
            torch.linalg.vector_norm(tangent_difference)
            / torch.linalg.vector_norm(expected_tangents).clamp_min(
                torch.finfo(torch.float32).eps
            )
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
        raise RuntimeError("CUDA is required for GR-006 benchmark evidence")
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
            n_wires,
            depth,
            default_eligible,
            seed=args.seed + index,
            warmup=args.warmup,
            repeats=args.repeats,
            group_size=args.group_size,
        )
        for index, (n_wires, depth, default_eligible) in enumerate(SHAPE_MATRIX)
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
        "execution_semantics": "single_device_fast_path",
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
    """Fail closed when one raw run violates the fixed evidence contract."""

    for field, value in {
        "schema": RUN_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
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
    observed = []
    for index, case_value in enumerate(_sequence(payload.get("cases"), "cases")):
        case = _mapping(case_value, f"cases[{index}]")
        shape = _mapping(case.get("shape"), f"cases[{index}].shape")
        observed.append(
            (shape.get("n_wires"), shape.get("depth"), case.get("default_eligible"))
        )
        for prefix, absolute, relative in (
            (
                "state",
                case.get("state_maximum_absolute_error"),
                case.get("state_reference_maximum_absolute_value"),
            ),
            (
                "tangent",
                case.get("tangent_maximum_absolute_error"),
                case.get("tangent_reference_maximum_absolute_value"),
            ),
        ):
            if not isinstance(absolute, (int, float)) or not isinstance(
                relative, (int, float)
            ):
                raise ValueError(f"cases[{index}] {prefix} values must be numeric")
            atol = STATE_ATOL if prefix == "state" else TANGENT_ATOL
            rtol = STATE_RTOL if prefix == "state" else TANGENT_RTOL
            if float(absolute) > atol + rtol * float(relative):
                raise ValueError(f"cases[{index}] exceeds the {prefix} tolerance")
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
        raise ValueError("run does not contain the fixed GR-006 shape matrix")


def merge_runs(
    payloads: Sequence[Mapping[str, Any]], *, required_hosts: Sequence[str]
) -> dict[str, object]:
    """Combine the required host and compiler matrix into one artifact."""

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
        "benchmark": "heisenberg_hva_forward_tangent_dispatch",
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
    """Validate a checked-in aggregate without importing an accelerator provider."""

    for field, value in {
        "benchmark": "heisenberg_hva_forward_tangent_dispatch",
        "schema": EVIDENCE_SCHEMA,
        "semantic_id": SEMANTIC_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "runner": RUNNER,
        "execution_semantics": "single_device_fast_path",
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
    run.add_argument("--group-size", type=int, default=3)
    run.add_argument("--seed", type=int, default=261011)
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
