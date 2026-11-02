"""Measure ordinary Module overhead across generated CPU circuit families."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Case:
    width: int
    depth: int
    ring: bool


@dataclass(frozen=True)
class Measurement:
    route: str
    demand: str
    first_call_in_context_seconds: float
    seconds_per_call: list[float]
    median_seconds: float
    mad_seconds: float


def within_budget(public: float, kernel: float, ratio: float, allowance: float) -> bool:
    if any(not (0 <= value < float("inf")) for value in (public, kernel, allowance)):
        raise ValueError("timings and allowance must be finite and non-negative")
    if not 1 <= ratio < float("inf"):
        raise ValueError("budget ratio must be finite and at least one")
    return public <= ratio * kernel + allowance


def measure(case: Case, repeats: int, iterations: int) -> list[Measurement]:
    import torch

    import flagquantum as fq
    from flagquantum.runtime.executors.jax import compile_quantum_kernel

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    count = 2 * case.width * case.depth

    def build(parameters: Any, features: Any) -> fq.Circuit:
        circuit = fq.Circuit(case.width, dtype=torch.complex128)
        for layer in range(case.depth):
            for wire in range(case.width):
                index = 2 * (layer * case.width + wire)
                circuit.ry(wire, parameters[index] + features[wire] * (layer + 1))
                circuit.rz(wire, parameters[index + 1])
            for wire in range(case.width - 1):
                circuit.cx(wire, wire + 1)
            if case.ring and case.width > 2:
                circuit.cx(case.width - 1, 0)
        return circuit

    parameters = torch.linspace(-0.3, 0.4, count, dtype=torch.float64)
    features = torch.linspace(-0.2, 0.5, case.width, dtype=torch.float64)
    routes: dict[str, Callable[..., torch.Tensor]] = {}
    for name, backend in (("default_module", "pytorch"), ("jax_module", "jax")):
        policy = fq.RuntimePolicy(
            observable="z_sum", observable_qubits=tuple(range(case.width))
        )
        if backend == "jax":
            policy = fq.RuntimePolicy(
                execution_options=fq.ExecutionOptions(backend="jax"),
                observable="z_sum",
                observable_qubits=tuple(range(case.width)),
            )
        module = fq.Module(
            build,
            count,
            dtype=torch.float64,
            policy=policy,
        )

        def invoke(
            p: torch.Tensor, x: torch.Tensor, *, _module: fq.Module = module
        ) -> torch.Tensor:
            result = _module(x, parameters=p)
            if not isinstance(result, torch.Tensor):
                raise TypeError("Module must return a Torch tensor")
            return result.reshape(())

        routes[name] = invoke
    routes["jax_kernel_diagnostic"] = compile_quantum_kernel(
        build,
        parameters,
        n_qubits=case.width,
        accepts_inputs=True,
        observable="z_sum",
        observable_qubits=tuple(range(case.width)),
        compute_dtype="complex128",
    )
    measurements = []
    for demand in ("parameters", "inputs", "both", "no_grad", "inference"):
        parameters.requires_grad_(
            demand in {"parameters", "both", "no_grad", "inference"}
        )
        features.requires_grad_(demand in {"inputs", "both", "no_grad", "inference"})

        def evaluate(
            function: Callable[..., torch.Tensor], *, _demand: str = demand
        ) -> tuple[torch.Tensor, ...]:
            context = (
                torch.no_grad()
                if _demand == "no_grad"
                else torch.inference_mode()
                if _demand == "inference"
                else nullcontext()
            )
            with context:
                value = function(parameters, features).reshape(())
                selected = tuple(
                    tensor for tensor in (parameters, features) if tensor.requires_grad
                )
                gradients = (
                    torch.autograd.grad(value, selected) if value.requires_grad else ()
                )
                result = (
                    value.detach(),
                    *(gradient.detach() for gradient in gradients),
                )
                # Include CPU completion for both values and derivatives in timings.
                for tensor in result:
                    tensor.sum().item()
                return result

        first: dict[str, float] = {}
        samples: dict[str, list[float]] = {name: [] for name in routes}
        reference = None
        for name, function in routes.items():
            start = time.perf_counter()
            actual = evaluate(function)
            first[name] = time.perf_counter() - start
            if reference is None:
                reference = actual
            else:
                for result, expected in zip(actual, reference, strict=True):
                    torch.testing.assert_close(result, expected, atol=1e-10, rtol=1e-10)
            for _ in range(3):
                evaluate(function)
        names = list(routes)
        for repeat in range(repeats):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                start = time.perf_counter()
                for _ in range(iterations):
                    evaluate(routes[name])
                samples[name].append((time.perf_counter() - start) / iterations)
        for name in names:
            median = statistics.median(samples[name])
            measurements.append(
                Measurement(
                    name,
                    demand,
                    first[name],
                    samples[name],
                    median,
                    statistics.median(abs(value - median) for value in samples[name]),
                )
            )
    return measurements


def aggregate_process_trials(
    summaries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Retain every case while aggregating three fresh trials in each phase."""
    phases: dict[str, list[dict[tuple[Any, ...], float]]] = {}
    for phase in ("before", "after"):
        trials = summaries[phase]
        if len(trials) != 3:
            raise ValueError("each phase requires three complete process trials")
        phases[phase] = []
        for trial in trials:
            points: dict[tuple[Any, ...], float] = {}
            for case in trial["cases"]:
                if case["status"] != "completed":
                    raise ValueError(
                        "failed or timed-out cases cannot be aggregated as completed"
                    )
                shape = tuple(case["case"][key] for key in ("width", "depth", "ring"))
                for point in case["measurements"]:
                    samples = point["seconds_per_call"]
                    if len(samples) != trial["repeats"] or not all(
                        math.isfinite(x) and x > 0 for x in samples
                    ):
                        raise ValueError("invalid timing block samples")
                    median = statistics.median(samples)
                    if median != point["median_seconds"]:
                        raise ValueError(
                            "recorded median differs from retained block samples"
                        )
                    key = (*shape, point["demand"], point["route"])
                    if key in points:
                        raise ValueError("duplicate case/demand/route")
                    points[key] = median
                expected = {
                    (*shape, demand, route)
                    for demand in (
                        "parameters",
                        "inputs",
                        "both",
                        "no_grad",
                        "inference",
                    )
                    for route in (
                        "default_module",
                        "jax_module",
                        "jax_kernel_diagnostic",
                    )
                }
                if {key for key in points if key[:3] == shape} != expected:
                    raise ValueError("case is missing a route or derivative demand")
            phases[phase].append(points)
    all_trials = [trial for phase in phases.values() for trial in phase]
    keys = all_trials[0].keys()
    if not keys or any(trial.keys() != keys for trial in all_trials):
        raise ValueError("all trials must retain identical case/demand/route keys")
    points = []
    medians = {}
    for key in sorted(keys):
        record = dict(
            zip(("width", "depth", "ring", "demand", "route"), key, strict=True)
        )
        for phase in ("before", "after"):
            values = [trial[key] for trial in phases[phase]]
            median = statistics.median(values)
            record[phase] = {
                "process_medians_seconds": values,
                "median_seconds": median,
                "mad_seconds": statistics.median(abs(x - median) for x in values),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
            medians[phase, key] = median
        record["before_over_after"] = (
            record["before"]["median_seconds"] / record["after"]["median_seconds"]
        )
        points.append(record)
    contexts = []
    for demand in ("parameters", "inputs", "both", "no_grad", "inference"):
        record = {"demand": demand}
        for route in ("default_module", "jax_module", "jax_kernel_diagnostic"):
            ratios = [
                p["before_over_after"]
                for p in points
                if p["demand"] == demand and p["route"] == route
            ]
            record[route] = {
                "median_ratio": statistics.median(ratios),
                "min_ratio": min(ratios),
                "max_ratio": max(ratios),
            }
        contexts.append(record)
    budgets = {}
    for phase in ("before", "after"):
        public_passes = default_passes = total = 0
        for key in keys:
            if key[-1] != "default_module":
                continue
            total += 1
            default = medians[phase, key]
            public = medians[phase, (*key[:-1], "jax_module")]
            kernel = medians[phase, (*key[:-1], "jax_kernel_diagnostic")]
            public_passes += int(public <= 1.15 * kernel + 5e-6)
            default_passes += int(default <= 1.25 * min(default, public) + 5e-6)
        budgets[phase] = {
            "public_overhead_passes": public_passes,
            "default_accessibility_passes": default_passes,
            "total": total,
        }
    return {
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "closure_status": "not_ready_to_close_918",
        "numerical_checks": sum(len(trial) for trial in all_trials),
        "fresh_shape_processes": sum(len(trial) // 15 for trial in all_trials),
        "aggregation": "median of three process medians, then ratio; no confidence interval with n=3",
        "contexts": contexts,
        "proposed_development_budgets": budgets,
        "points": points,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--enforce", action="store_true", help="fail on proposed performance budgets"
    )
    parser.add_argument("--worker", nargs=3, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.repeats, args.iterations, args.timeout) < 1:
        parser.error("repeats, iterations and timeout must be positive")
    if args.worker:
        width, depth, ring = args.worker
        result = measure(Case(width, depth, bool(ring)), args.repeats, args.iterations)
        args.output.write_text(
            json.dumps([asdict(row) for row in result], indent=2) + "\n"
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    cases = (
        [Case(2, 1, False)]
        if args.smoke
        else [
            Case(width, depth, ring)
            for width, depth, ring in itertools.product(
                (2, 4, 6), (1, 3), (False, True)
            )
            if width > 2 or not ring
        ]
    )
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    results: list[dict[str, Any]] = []
    failed = False
    for index, case in enumerate(cases):
        print(f"case {index + 1}/{len(cases)}: {case}", flush=True)
        output = args.output / f"case-{index:02d}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.module_execution_matrix",
            "--output",
            str(output),
            "--worker",
            str(case.width),
            str(case.depth),
            str(int(case.ring)),
            "--repeats",
            str(args.repeats),
            "--iterations",
            str(args.iterations),
        ]
        status = "completed"
        with (args.output / f"case-{index:02d}.log").open("w") as log:
            try:
                child = subprocess.run(
                    command,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                    check=False,
                )
                if child.returncode:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        record: dict[str, Any] = {"case": asdict(case), "status": status}
        if status == "completed":
            rows = json.loads(output.read_text())
            budgets = []
            for demand in ("parameters", "inputs", "both", "no_grad", "inference"):
                timings = {
                    row["route"]: row["median_seconds"]
                    for row in rows
                    if row["demand"] == demand
                }
                budgets.append(
                    {
                        "demand": demand,
                        "public_overhead_passed": within_budget(
                            timings["jax_module"],
                            timings["jax_kernel_diagnostic"],
                            1.15,
                            5e-6,
                        ),
                        "default_route_passed": within_budget(
                            timings["default_module"],
                            min(timings["default_module"], timings["jax_module"]),
                            1.25,
                            5e-6,
                        ),
                    }
                )
            record.update(measurements=rows, proposed_budgets=budgets)
            if args.enforce:
                failed |= any(
                    not row["public_overhead_passed"] or not row["default_route_passed"]
                    for row in budgets
                )
        else:
            failed = True
        results.append(record)
    root = Path(__file__).resolve().parents[1]

    def git(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, timeout=15
        ).strip()

    report = {
        "schema": "flagquantum.module_execution_matrix.development.v1",
        "revision": git("rev-parse", "HEAD"),
        "source_dirty": bool(git("status", "--porcelain")),
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {
            name: version(name) for name in ("torch", "jax", "jaxlib", "flagquantum")
        },
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "device": "cpu",
        "precision": "complex128",
        "batching": "one_example",
        "repeats": args.repeats,
        "iterations": args.iterations,
        "case_timeout_seconds": args.timeout,
        "budget_status": "proposed_development_targets_not_release_thresholds",
        "limitations": [
            "one fresh process per shape, within-process timing blocks",
            "context-first calls reuse earlier context caches; not cold-process startup",
            "kernel route is diagnostic; no external-framework ranking",
            "Torch/BLAS configured to one thread; XLA thread count is not constrained",
        ],
        "cases": results,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
