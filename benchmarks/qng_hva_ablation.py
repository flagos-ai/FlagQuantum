"""Frozen QNG algorithm/fusion experiment behind the existing HVA entry point."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmarks.qng_hva_geometry import (
    Geometry,
    UnsupportedRouteError,
    Workload,
    independent_reference,
)
from benchmarks.training_comparison import ROOT, digest, source_identity
from flagquantum.benchmarking.contract import runtime_metadata, write_json_atomic

SCHEMA = "flagquantum.qng_hva_ablation.v1"
METHODS = ("reverse_jacobian", "unfused_forward", "fused_forward", "pennylane_metric")


def protocol_path(device: str) -> Path:
    return ROOT / f"benchmarks/manifests/qng_hva_ablation_{device}_v1.json"


def identity(protocol: Path) -> dict[str, Any]:
    source = source_identity(protocol)
    for name in (
        "qng_hva_ablation.py",
        "qng_hva_geometry.py",
        "qng_hva_time_to_accuracy.py",
    ):
        path = ROOT / "benchmarks" / name
        source["producer_sha256"][str(path.relative_to(ROOT))] = digest(path)
    return source


def synchronize(workload: Workload) -> None:
    if workload.device == "cuda":
        torch.cuda.synchronize()


def verify(
    geometry: Geometry, parameters: torch.Tensor, damping: float, tolerance: float
) -> dict[str, Any]:
    """Check state tangents, blocked metric, energy gradient and actual update."""
    workload = geometry.workload
    state, tangent, metric, gradient = independent_reference(
        workload, parameters.detach().cpu().double().numpy()
    )
    observed = geometry.metric(parameters)
    hamiltonian = torch.as_tensor(
        workload.hamiltonian(),
        dtype=workload.state(parameters).dtype,
        device=parameters.device,
    )
    value = workload.state(parameters)
    loss = torch.vdot(value, hamiltonian @ value).real
    actual_gradient = torch.autograd.grad(loss, parameters)[0]
    reference_direction = np.linalg.solve(
        metric + damping * np.eye(workload.count), gradient
    )
    direction = torch.linalg.solve(
        observed
        + damping
        * torch.eye(workload.count, device=parameters.device, dtype=parameters.dtype),
        actual_gradient,
    )
    errors = {
        "metric_max_absolute_error": float(
            np.max(np.abs(observed.detach().cpu().numpy() - metric))
        ),
        "gradient_max_absolute_error": float(
            np.max(np.abs(actual_gradient.detach().cpu().numpy() - gradient))
        ),
        "direction_relative_l2_error": float(
            np.linalg.norm(direction.detach().cpu().numpy() - reference_direction)
            / max(np.linalg.norm(reference_direction), 1e-12)
        ),
    }
    if geometry.framework_metric is None:
        actual_state, actual_tangent = geometry.tangent(parameters)
        errors["state_max_absolute_error"] = float(
            np.max(np.abs(actual_state.detach().cpu().numpy() - state))
        )
        errors["tangent_max_absolute_error"] = float(
            np.max(np.abs(actual_tangent.detach().cpu().numpy() - tangent))
        )
    passed = all(np.isfinite(value) and value <= tolerance for value in errors.values())
    return {
        "passed": bool(passed),
        "tolerance": tolerance,
        "reference": "independent_numpy_dense_centered_difference_1e-5",
        "tangents_exposed": geometry.framework_metric is None,
        **errors,
    }


def run_case(
    manifest: dict[str, Any],
    workload: Workload,
    method: str,
    seed: int,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    """Matched optimizer trajectory; oracle and timing scopes remain explicit."""
    wall_started = time.perf_counter()
    if workload.device == "cuda" and not torch.cuda.is_available():
        raise UnsupportedRouteError(
            "CUDA is unavailable; CPU substitution is forbidden"
        )
    dtype = torch.float64 if workload.precision == "complex128" else torch.float32
    generator = torch.Generator().manual_seed(seed)
    initial = (
        manifest["initial_std"]
        * torch.randn(workload.count, generator=generator, dtype=torch.float64)
    ).to(device=workload.device, dtype=dtype)
    parameters = initial.clone().requires_grad_()
    setup_started = time.perf_counter()
    geometry = Geometry(method, workload)
    hamiltonian = torch.as_tensor(
        workload.hamiltonian(),
        device=workload.device,
        dtype=torch.complex128 if dtype == torch.float64 else torch.complex64,
    )
    exact = float(np.linalg.eigvalsh(workload.hamiltonian())[0])
    eye = torch.eye(workload.count, dtype=dtype, device=workload.device)
    damping, lr = manifest["damping"], manifest["learning_rate"]

    def energy() -> torch.Tensor:
        state = workload.state(parameters)
        return torch.vdot(state, hamiltonian @ state).real

    phases: dict[str, float] = {}

    def direction() -> torch.Tensor:
        synchronize(workload)
        phase_started = time.perf_counter()
        gradient = torch.autograd.grad(energy(), parameters)[0]
        synchronize(workload)
        phases["objective_gradient_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        metric = geometry.metric(parameters)
        synchronize(workload)
        phases["metric_seconds"] = time.perf_counter() - phase_started
        phase_started = time.perf_counter()
        update = torch.linalg.solve(metric + damping * eye, gradient)
        synchronize(workload)
        phases["solve_seconds"] = time.perf_counter() - phase_started
        return update

    synchronize(workload)
    setup_seconds = time.perf_counter() - setup_started
    cold_started = time.perf_counter()
    direction()
    synchronize(workload)
    first_call_seconds = time.perf_counter() - cold_started
    oracle_started = time.perf_counter()
    parity = verify(geometry, parameters, damping, manifest["parity_tolerance"])
    oracle_seconds = time.perf_counter() - oracle_started
    if not parity["passed"]:
        raise ValueError(f"independent QNG geometry/update parity failed: {parity}")
    warmup_started = time.perf_counter()
    for _ in range(manifest["warmup_calls"]):
        direction()
    synchronize(workload)
    warmup_seconds = time.perf_counter() - warmup_started
    initial_started = time.perf_counter()
    with torch.no_grad():
        initial_energy = float(energy())
    synchronize(workload)
    initial_assessment_seconds = time.perf_counter() - initial_started
    startup = (
        setup_seconds + first_call_seconds + warmup_seconds + initial_assessment_seconds
    )
    target = manifest["target_relative_energy_error"]
    target_training = (
        0.0 if abs(initial_energy - exact) / abs(exact) <= target else None
    )
    if workload.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    trace: list[dict[str, Any]] = []
    started = time.perf_counter()
    for step in range(1, (3 if smoke else manifest["steps"]) + 1):
        step_started = time.perf_counter()
        update = direction()
        with torch.no_grad():
            parameters.add_(-lr * update)
        synchronize(workload)
        update_seconds = time.perf_counter() - step_started
        validation_started = time.perf_counter()
        with torch.no_grad():
            value = float(energy())
        synchronize(workload)
        validation_seconds = time.perf_counter() - validation_started
        elapsed = time.perf_counter() - started
        error = abs(value - exact) / abs(exact)
        if not np.isfinite(value) or not bool(torch.isfinite(parameters).all()):
            raise ValueError("non-finite optimizer trajectory")
        if target_training is None and error <= target:
            target_training = elapsed
        trace.append(
            {
                **phases,
                "step": step,
                "energy": value,
                "relative_energy_error": error,
                "parameters": parameters.detach().cpu().tolist(),
                "direction": update.detach().cpu().tolist(),
                "update_seconds": update_seconds,
                "validation_seconds": validation_seconds,
                "elapsed_seconds": elapsed,
            }
        )
        print(
            json.dumps({"method": method, "seed": seed, "step": step, "error": error}),
            flush=True,
        )
    training_seconds = time.perf_counter() - started
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "status": "completed",
        "workload": asdict(workload),
        "parameter_count": workload.count,
        "method": method,
        "seed": seed,
        "parity": parity,
        "objective_gradient": "FlagQuantum_Torch_reverse_identical_for_all_geometry_routes",
        "metric_backend": (
            "pennylane_default.qubit_adjoint_metric_tensor_torch"
            if method == "pennylane_metric"
            else "FlagQuantum_HVA"
        ),
        "interface": "torch",
        "dtype": workload.precision,
        "shots": None,
        "threads": torch.get_num_threads(),
        "jit": "Triton_first_call_included" if method == "fused_forward" else "off",
        "synchronization": (
            "cuda_synchronize_each_timed_boundary"
            if workload.device == "cuda"
            else "synchronous_cpu"
        ),
        "exact_ground_energy": exact,
        "initial_parameters": initial.cpu().tolist(),
        "initial_energy": initial_energy,
        "trace": trace,
        "completed_steps": len(trace),
        "final_relative_energy_error": trace[-1]["relative_energy_error"],
        "time_to_target_training_seconds": target_training,
        "time_to_target_total_seconds": (
            None if target_training is None else startup + target_training
        ),
        "backend_setup_seconds": setup_seconds,
        "first_call_seconds": first_call_seconds,
        "warmup_seconds": warmup_seconds,
        "initial_assessment_seconds": initial_assessment_seconds,
        "oracle_seconds": oracle_seconds,
        "training_seconds": training_seconds,
        "total_measured_seconds": startup + training_seconds,
        "worker_wall_seconds": time.perf_counter() - wall_started,
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if workload.device == "cuda" else None
        ),
    }


def campaign(output: Path, *, device: str, smoke: bool) -> int:
    output.mkdir(parents=True, exist_ok=False)
    protocol = protocol_path(device)
    manifest = json.loads(protocol.read_text())
    source = identity(protocol)
    metadata = runtime_metadata(
        runner="qng_hva_time_to_accuracy",
        schema=SCHEMA,
        source=source,
        protocol=manifest,
        smoke=smoke,
        scalability_claim_allowed=False,
        execution_semantics="single_device_fast_path",
        benchmark_evidence_class="comparison_non_release",
    )
    write_json_atomic(output / "freeze.json", metadata)
    cases = []
    for shape in manifest["workloads"][:1] if smoke else manifest["workloads"]:
        for method in METHODS:
            for seed in manifest["seeds"][:1] if smoke else manifest["seeds"]:
                name = f"q{shape['qubits']}-d{shape['depth']}-{method}-{seed}"
                result_path = output / f"{name}.json"
                command = [
                    sys.executable,
                    "-m",
                    "benchmarks.qng_hva_time_to_accuracy",
                    "--ablation",
                    "--worker",
                    "--device",
                    device,
                    "--method",
                    method,
                    "--qubits",
                    str(shape["qubits"]),
                    "--depth",
                    str(shape["depth"]),
                    "--seed",
                    str(seed),
                    "--output",
                    str(result_path),
                ]
                if smoke:
                    command.append("--smoke")
                environment = dict(
                    os.environ,
                    OMP_NUM_THREADS="1",
                    OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1",
                )
                started = time.perf_counter()
                print(f"Starting {name}", flush=True)
                with (output / f"{name}.log").open("w") as log:
                    try:
                        completed = subprocess.run(
                            command,
                            cwd=ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=manifest["case_timeout_seconds"],
                            check=False,
                        )
                        if result_path.exists():
                            result = json.loads(result_path.read_text())
                            if (
                                result["source"] != source
                                or result["protocol"] != manifest
                            ):
                                raise ValueError(
                                    "worker changed frozen source/protocol"
                                )
                            status = result["status"]
                            if completed.returncode and status == "completed":
                                status = "failed"
                        else:
                            status = "failed"
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                cases.append(
                    {
                        "case": name,
                        "status": status,
                        "result": result_path.name,
                        "process_wall_seconds": time.perf_counter() - started,
                        "qubits": shape["qubits"],
                        "depth": shape["depth"],
                        "method": method,
                        "seed": seed,
                    }
                )
                write_json_atomic(output / "summary.json", {**metadata, "cases": cases})
    (output / "REPORT.md").write_text(render_report(output))
    return int(any(c["status"] in ("failed", "timeout") for c in cases))


def render_report(directory: Path) -> str:
    freeze = json.loads((directory / "freeze.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if (
        summary["source"] != freeze["source"]
        or summary["protocol"] != freeze["protocol"]
    ):
        raise ValueError("summary source/protocol does not match freeze")
    lines = [
        "# Matched QNG algorithm and fusion ablation",
        "",
        f"Producer: `{freeze['source']['code_version']}`. Protocol: `{freeze['protocol']['protocol_id']}`. Smoke: `{freeze['smoke']}`.",
        "",
        "The objective gradient, damping, layer-block projection and update rule are identical. Only metric construction changes. PennyLane supplies its framework metric, with the same FlagQuantum objective gradient; this isolates geometry and is not an end-to-end PennyLane optimizer ranking.",
        "",
        "First-call/JIT and warmup are separate and included in total cost and total time to target. Oracle validation and common imports are outside measured total; fresh-process wall includes them and interpreter startup. Peak RSS includes the entire worker. Fixed updates continue after reaching the target. No quantum advantage, state-of-the-art or scalability claim is made.",
        "",
        "| Wires | Depth | Parameters | Method | Seed | Status | Final error | Training s | First call s | Total s | Total target s | RSS MiB |",
        "| ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    results: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for case in summary["cases"]:
        if case["status"] != "completed":
            path = directory / case["result"]
            reason = (
                json.loads(path.read_text()).get("error", "")
                if path.exists()
                else "See retained log; no completed result."
            )
            lines.append(
                f"| {case['qubits']} | {case['depth']} | — | {case['method']} | {case['seed']} | {case['status']}: {reason} | — | — | — | — | — | — |"
            )
            continue
        result = json.loads((directory / case["result"]).read_text())
        if (
            result["source"] != freeze["source"]
            or result["protocol"] != freeze["protocol"]
            or not result["parity"]["passed"]
            or result["status"] != "completed"
        ):
            raise ValueError(f"unvalidated or mismatched result: {case['case']}")
        results[(case["qubits"], case["depth"], case["seed"], case["method"])] = result
        target = result["time_to_target_total_seconds"]
        text = "not reached" if target is None else f"{target:.6f}"
        lines.append(
            f"| {case['qubits']} | {case['depth']} | {result['parameter_count']} | [{case['method']}]({case['result']}) | {case['seed']} | completed | {result['final_relative_energy_error']:.6g} | {result['training_seconds']:.6f} | {result['first_call_seconds']:.6f} | {result['total_measured_seconds']:.6f} | {text} | {result['process_peak_rss_bytes']/2**20:.1f} |"
        )
    lines += [
        "",
        "## Separate savings",
        "",
        "Ratios use paired median metric construction phase samples from the complete optimizer trajectories. Reverse/unfused isolates derivative algorithm; unfused/fused isolates fusion. A ratio below one is a loss. No ratio is emitted for missing, unsupported or failed pairs.",
        "",
        "| Wires | Depth | Seed | Reverse / unfused metric time | Unfused / fused metric time |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for qubits, depth, seed in sorted({key[:3] for key in results}):

        def ratio(
            left: str,
            right: str,
            qubits: int = qubits,
            depth: int = depth,
            seed: int = seed,
        ) -> str:
            a, b = results.get((qubits, depth, seed, left)), results.get(
                (qubits, depth, seed, right)
            )
            if a is None or b is None:
                return "unavailable"
            return f"{statistics.median(p['metric_seconds'] for p in a['trace'])/statistics.median(p['metric_seconds'] for p in b['trace']):.4f}"

        lines.append(
            f"| {qubits} | {depth} | {seed} | {ratio('reverse_jacobian', 'unfused_forward')} | {ratio('unfused_forward', 'fused_forward')} |"
        )
    lines += [
        "",
        "All seeds, every parameter vector and update direction, per-step timings, parity errors, process wall and memory remain in the JSON. Unsupported fused CPU cases cannot establish fusion savings; a completed CUDA campaign is required for that conclusion.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=METHODS, default="reverse_jacobian")
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    if not args.worker:
        return campaign(args.output, device=args.device, smoke=args.smoke)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = protocol_path(args.device)
    manifest = json.loads(protocol.read_text())
    shape = {"qubits": args.qubits, "depth": args.depth}
    if shape not in manifest["workloads"] or args.seed not in manifest["seeds"]:
        parser.error("worker case is outside the frozen manifest")
    metadata = runtime_metadata(
        runner="qng_hva_time_to_accuracy",
        schema=SCHEMA,
        source=identity(protocol),
        protocol=manifest,
        smoke=args.smoke,
        scalability_claim_allowed=False,
        execution_semantics="single_device_fast_path",
        benchmark_evidence_class="comparison_non_release",
    )
    try:
        result = run_case(
            manifest,
            Workload(args.qubits, args.depth, manifest["precision"], args.device),
            args.method,
            args.seed,
            smoke=args.smoke,
        )
    except UnsupportedRouteError as exc:
        result = {"status": "unsupported", "error": str(exc)}
    except Exception as exc:  # Isolated CLI boundary retains the actual failure.
        traceback.print_exc()
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json_atomic(args.output, {**metadata, **result})
    return int(result["status"] == "failed")
