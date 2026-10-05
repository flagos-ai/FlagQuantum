#!/usr/bin/env python3
"""Measure steady-state independent-simulation throughput on one CPU socket."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import selectors
import shutil
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch

from .contract import runtime_metadata, write_json_atomic
from .simulator_compare import ABSOLUTE_TOLERANCE, SEED
from .simulator_workload_corpus import (
    ENGINE_NAMES,
    WORKLOAD_NAMES,
    EngineName,
    WorkloadName,
    _engine_callable,
    _engine_versions,
    build_workload,
)

SCHEMA = "flagquantum.socket_local_throughput.v1"
RUNNER = "socket_local_throughput"
_STABILITY_THRESHOLD = 0.20


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse Linux taskset notation into an ordered, duplicate-free CPU tuple."""

    cpus: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("CPU list contains an empty item")
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid CPU range: {item}")
            first, last = (int(part) for part in parts)
            if first < 0 or last < first:
                raise ValueError(f"invalid CPU range: {item}")
            cpus.extend(range(first, last + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError("CPU identifiers must be non-negative")
            cpus.append(cpu)
    if not cpus or len(set(cpus)) != len(cpus):
        raise ValueError("CPU list must be non-empty and contain no duplicates")
    return tuple(cpus)


def split_cpu_groups(cpus: Sequence[int], workers: int) -> tuple[tuple[int, ...], ...]:
    """Split one physical-socket CPU set into equal contiguous worker groups."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if len(cpus) % workers:
        raise ValueError("worker count must divide the socket CPU count")
    width = len(cpus) // workers
    if width < 1:
        raise ValueError("each worker must own at least one CPU")
    return tuple(
        tuple(int(cpu) for cpu in cpus[index : index + width])
        for index in range(0, len(cpus), width)
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between zero and one")
    ordered = sorted(float(value) for value in values)
    position = percentile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def latency_summary(samples: Sequence[float]) -> dict[str, Any]:
    """Return reproducible latency statistics for one throughput cell."""

    values = tuple(float(value) for value in samples)
    if not values or any(value <= 0 for value in values):
        raise ValueError("latency samples must be positive and non-empty")
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    return {
        "samples_seconds": values,
        "sample_count": len(values),
        "p50_seconds": median,
        "p95_seconds": _percentile(values, 0.95),
        "mean_seconds": statistics.fmean(values),
        "min_seconds": min(values),
        "max_seconds": max(values),
        "relative_median_absolute_deviation": mad / median,
    }


def _configure_worker(cpu_group: Sequence[int], threads: int) -> tuple[int, ...]:
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("socket-local affinity requires Linux sched_setaffinity")
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["OMP_PLACES"] = "cores"
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return tuple(cpu_group)


def _worker_subprocess_main(payload_text: str) -> int:
    """Execute one taskset-pinned worker using a line-oriented start barrier."""

    payload = json.loads(payload_text)
    worker = int(payload["worker"])
    cpu_group = tuple(int(cpu) for cpu in payload["cpu_group"])
    threads = int(payload["threads"])
    engine = cast(EngineName, payload["engine"])
    workload = cast(WorkloadName, payload["workload"])
    n_qubits = int(payload["n_wires"])
    warmup = int(payload["warmup"])
    tasks = int(payload["tasks"])
    seed = int(payload["seed"])

    assigned_cpus = _configure_worker(cpu_group, threads)
    circuit = build_workload(workload, n_qubits=n_qubits, seed=seed)
    execute = _engine_callable(engine, circuit, seed=seed, threads=threads)
    output: torch.Tensor | None = None
    for _ in range(warmup):
        output = execute()
    gc.collect()
    print(
        json.dumps(
            {
                "kind": "ready",
                "worker": worker,
                "pid": os.getpid(),
                "taskset_cpus": assigned_cpus,
                "observed_main_thread_cpus": sorted(os.sched_getaffinity(0)),
            }
        ),
        flush=True,
    )
    if sys.stdin.readline().strip() != "start":
        raise RuntimeError("worker did not receive the synchronized start token")

    started = time.perf_counter()
    latencies: list[float] = []
    for _ in range(tasks):
        call_started = time.perf_counter()
        output = execute()
        latencies.append(time.perf_counter() - call_started)
    finished = time.perf_counter()
    if output is None:
        output = execute()

    if engine == "flagquantum_native":
        max_error = 0.0
    else:
        reference_circuit = build_workload(workload, n_qubits=n_qubits, seed=seed)
        reference = _engine_callable(
            "flagquantum_native",
            reference_circuit,
            seed=seed,
            threads=threads,
        )()
        reference = reference.detach().cpu().to(torch.complex128).reshape(-1)
        measured = output.detach().cpu().to(torch.complex128).reshape(-1)
        max_error = float(torch.max(torch.abs(measured - reference)).item())

    print(
        json.dumps(
            {
                "kind": "result",
                "worker": worker,
                "started": started,
                "finished": finished,
                "latencies": latencies,
                "max_abs_error": max_error,
            }
        ),
        flush=True,
    )
    return 0


def _worker_environment(threads: int) -> dict[str, str]:
    environment = os.environ.copy()
    value = str(threads)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = value
    environment["OMP_PROC_BIND"] = "close"
    environment["OMP_PLACES"] = "cores"
    return environment


def _start_worker(*, payload: Mapping[str, Any], taskset: str) -> subprocess.Popen[str]:
    cpu_list = ",".join(str(cpu) for cpu in payload["cpu_group"])
    return subprocess.Popen(
        (
            taskset,
            "-c",
            cpu_list,
            sys.executable,
            "-m",
            "flagquantum.benchmarking.socket_local_throughput",
            "--worker-json",
            json.dumps(payload, separators=(",", ":")),
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=_worker_environment(int(payload["threads"])),
    )


def _collect_messages(
    processes: Sequence[subprocess.Popen[str]],
    *,
    expected_kind: str,
    timeout: float,
) -> dict[int, Mapping[str, Any]]:
    selector = selectors.DefaultSelector()
    try:
        for index, process in enumerate(processes):
            if process.stdout is None:
                raise RuntimeError("worker stdout pipe is unavailable")
            selector.register(process.stdout, selectors.EVENT_READ, index)
        deadline = time.monotonic() + timeout
        messages: dict[int, Mapping[str, Any]] = {}
        while len(messages) < len(processes):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"socket-local worker {expected_kind} timed out")
            events = selector.select(remaining)
            if not events:
                raise TimeoutError(f"socket-local worker {expected_kind} timed out")
            for key, _ in events:
                index = int(key.data)
                process = processes[index]
                line = cast(Any, key.fileobj).readline()
                if not line:
                    stderr = process.stderr.read() if process.stderr is not None else ""
                    raise RuntimeError(
                        f"worker {index} exited before {expected_kind}: {stderr.strip()}"
                    )
                message = json.loads(line)
                if not isinstance(message, Mapping):
                    raise RuntimeError(
                        "socket-local worker returned an invalid message"
                    )
                if message.get("kind") != expected_kind:
                    raise RuntimeError(
                        f"worker {index} returned {message.get('kind')!r}; "
                        f"expected {expected_kind!r}"
                    )
                messages[index] = message
                selector.unregister(key.fileobj)
        return messages
    finally:
        selector.close()


def run_configuration(
    *,
    workload: WorkloadName,
    n_qubits: int,
    engine: EngineName,
    cpus: Sequence[int],
    workers: int,
    total_tasks: int,
    warmup: int,
    seed: int = SEED,
    timeout: float = 1800.0,
) -> dict[str, Any]:
    """Run one equal-core worker configuration in fresh pinned processes."""

    cpu_groups = split_cpu_groups(cpus, workers)
    threads = len(cpu_groups[0])
    if total_tasks < workers or total_tasks % workers:
        raise ValueError("total_tasks must be a positive multiple of workers")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if engine not in ENGINE_NAMES:
        raise ValueError(f"unsupported engine: {engine}")
    if workload not in WORKLOAD_NAMES:
        raise ValueError(f"unsupported workload: {workload}")

    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("socket-local throughput requires the Linux taskset tool")
    tasks_per_worker = total_tasks // workers
    processes = [
        _start_worker(
            taskset=taskset,
            payload={
                "worker": index,
                "cpu_group": cpu_group,
                "threads": threads,
                "engine": engine,
                "workload": workload,
                "n_wires": n_qubits,
                "warmup": warmup,
                "tasks": tasks_per_worker,
                "seed": seed,
            },
        )
        for index, cpu_group in enumerate(cpu_groups)
    ]
    try:
        ready = _collect_messages(processes, expected_kind="ready", timeout=timeout)
        for index, group in enumerate(cpu_groups):
            assigned = tuple(ready[index]["taskset_cpus"])
            if assigned != tuple(group):
                raise RuntimeError(
                    f"worker {index} affinity {assigned} does not match its group "
                    f"{tuple(group)}"
                )

        for process in processes:
            if process.stdin is None:
                raise RuntimeError("worker stdin pipe is unavailable")
            process.stdin.write("start\n")
            process.stdin.flush()
        results = _collect_messages(processes, expected_kind="result", timeout=timeout)
        for process in processes:
            process.wait(timeout=30)
        bad = [process.pid for process in processes if process.returncode != 0]
        if bad:
            raise RuntimeError(f"socket-local worker process failed: {bad}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    latencies = [
        float(value)
        for result in results.values()
        for value in cast(Sequence[float], result["latencies"])
    ]
    window = max(float(result["finished"]) for result in results.values()) - min(
        float(result["started"]) for result in results.values()
    )
    errors = [float(result["max_abs_error"]) for result in results.values()]
    summary = latency_summary(latencies)
    return {
        "configuration": {
            "workers": workers,
            "threads_per_worker": threads,
            "total_physical_cores": len(cpus),
            "cpu_groups": cpu_groups,
        },
        "total_tasks": total_tasks,
        "tasks_per_worker": tasks_per_worker,
        "steady_state_wall_seconds": window,
        "tasks_per_second": total_tasks / window,
        "latency": summary,
        "correctness": {
            "passed": max(errors) <= ABSOLUTE_TOLERANCE,
            "absolute_tolerance": ABSOLUTE_TOLERANCE,
            "maximum_worker_max_abs_error": max(errors),
        },
        "stable": summary["relative_median_absolute_deviation"] <= _STABILITY_THRESHOLD,
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _validate_matrix(
    cpus: Sequence[int], workers: Sequence[int], total_tasks: int
) -> None:
    if platform.system() != "Linux":
        raise RuntimeError("socket-local throughput measurement requires Linux")
    if not workers or len(set(workers)) != len(workers):
        raise ValueError("worker counts must be unique and non-empty")
    for count in workers:
        split_cpu_groups(cpus, count)
        if total_tasks < count or total_tasks % count:
            raise ValueError("total_tasks must be divisible by every worker count")
    available = os.sched_getaffinity(0)
    unavailable = sorted(set(cpus) - available)
    if unavailable:
        raise ValueError(f"requested CPUs are outside process affinity: {unavailable}")


def _derive_comparison(case: dict[str, Any]) -> None:
    native = case["engines"]["flagquantum_native"]["tasks_per_second"]
    case["comparison"] = {
        "engine_over_flagquantum_throughput": {
            engine: result["tasks_per_second"] / native
            for engine, result in case["engines"].items()
            if engine != "flagquantum_native"
        },
        "ratio_semantics": "values above one mean the external engine is faster",
    }


def run_benchmark(
    *,
    workloads: Sequence[WorkloadName],
    n_qubits: int,
    engines: Sequence[EngineName],
    cpus: Sequence[int],
    worker_counts: Sequence[int],
    total_tasks: int,
    warmup: int,
    seed: int = SEED,
    socket_id: int | None = None,
) -> dict[str, Any]:
    """Measure a matched worker/thread matrix while holding total cores fixed."""

    if not workloads or len(set(workloads)) != len(workloads):
        raise ValueError("workloads must be unique and non-empty")
    if not engines or len(set(engines)) != len(engines):
        raise ValueError("engines must be unique and non-empty")
    unknown_workloads = sorted(set(workloads) - set(WORKLOAD_NAMES))
    unknown_engines = sorted(set(engines) - set(ENGINE_NAMES))
    if unknown_workloads or unknown_engines:
        raise ValueError(
            "unsupported workloads or engines: "
            + ", ".join((*unknown_workloads, *unknown_engines))
        )
    _validate_matrix(cpus, worker_counts, total_tasks)

    cases: list[dict[str, Any]] = []
    for workload in workloads:
        for workers in worker_counts:
            engine_order = tuple(engines[workers % len(engines) :]) + tuple(
                engines[: workers % len(engines)]
            )
            results = {
                engine: run_configuration(
                    workload=workload,
                    n_qubits=n_qubits,
                    engine=engine,
                    cpus=cpus,
                    workers=workers,
                    total_tasks=total_tasks,
                    warmup=warmup,
                    seed=seed,
                )
                for engine in engine_order
            }
            case = {
                "workload": workload,
                "n_wires": n_qubits,
                "configuration": {
                    "workers": workers,
                    "threads_per_worker": len(cpus) // workers,
                    "total_physical_cores": len(cpus),
                },
                "engines": {engine: results[engine] for engine in engines},
            }
            _derive_comparison(case)
            cases.append(case)

    passed = all(
        result["correctness"]["passed"]
        for case in cases
        for result in case["engines"].values()
    )
    stable = all(
        result["stable"] for case in cases for result in case["engines"].values()
    )
    return runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        benchmark="socket_local_independent_simulation_throughput",
        artifact_class="measured_comparison_run",
        benchmark_evidence_class="comparison_non_release",
        claim_evidence_type="development_hardware_evidence",
        distribution_semantics="single_socket_replicated_process_throughput",
        scalability_claim_allowed=False,
        scalability_blockers=(
            "independent_workloads_are_replicated_not_one_sharded_statevector",
            "single_socket_result_does_not_measure_capacity_expansion",
        ),
        release_gate_allowed=False,
        non_release_evidence=True,
        passed=passed and stable,
        correctness_passed=passed,
        all_measurements_stable=stable,
        environment={
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "socket_id": socket_id,
            "physical_cpu_list": tuple(cpus),
            "torch": torch.__version__,
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        methodology={
            "measurement_scope": "steady_state_user_facing_run_call",
            "process_startup_included": False,
            "circuit_construction_included": False,
            "conversion_included": True,
            "result_retrieval_included": True,
            "synchronized_start": True,
            "fresh_processes_per_cell": True,
            "exclusive_cpu_groups": True,
            "worker_launch": "taskset_and_thread_environment_before_python_import",
            "smt_used": False,
            "total_tasks_per_cell": total_tasks,
            "warmup_per_worker": warmup,
            "latency_percentiles": (50, 95),
            "maximum_relative_median_absolute_deviation": _STABILITY_THRESHOLD,
            "engine_order": "rotated_per_worker_configuration",
        },
        workloads=tuple(workloads),
        n_wires=n_qubits,
        engines=tuple(engines),
        worker_counts=tuple(worker_counts),
        engine_versions={engine: _engine_versions(engine) for engine in engines},
        cases=cases,
    )


def render_markdown(payload: Mapping[str, Any], *, artifact_name: str) -> str:
    """Render absolute throughput and tail-latency data from one artifact."""

    engines = tuple(payload["engines"])
    baseline: dict[tuple[str, str], float] = {}
    for case in payload["cases"]:
        if case["configuration"]["workers"] == 1:
            for engine, result in case["engines"].items():
                baseline[(case["workload"], engine)] = result["tasks_per_second"]

    lines = [
        "# Socket-local independent-simulation throughput",
        "",
        f"Generated from [`{artifact_name}`]({artifact_name}). Process startup and circuit ",
        "construction are excluded; every timed task is one user-facing exact-statevector run.",
        "All configurations use the same physical CPU set with disjoint worker affinity.",
        "",
        "| Workload | Workers x threads | Engine | Tasks/s | p50 | p95 | vs 1xsocket |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        workers = case["configuration"]["workers"]
        threads = case["configuration"]["threads_per_worker"]
        for engine in engines:
            result = case["engines"][engine]
            throughput = result["tasks_per_second"]
            speedup = throughput / baseline[(case["workload"], engine)]
            lines.append(
                f"| {case['workload']} | {workers}x{threads} | {engine} | "
                f"{throughput:.3f} | {result['latency']['p50_seconds'] * 1000:.3f} ms | "
                f"{result['latency']['p95_seconds'] * 1000:.3f} ms | {speedup:.3f}x |"
            )
    lines.extend(
        (
            "",
            "Ratios describe replicated independent work on one socket, not single-circuit",
            "latency scaling, multi-socket scaling, distributed statevectors, or cold start.",
            "",
        )
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workloads", nargs="+", choices=WORKLOAD_NAMES, required=True)
    parser.add_argument("--n-wires", type=int, default=22)
    parser.add_argument("--engines", nargs="+", choices=ENGINE_NAMES, required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--workers", nargs="+", type=int, required=True)
    parser.add_argument("--total-tasks", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--socket-id", type=int)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-json":
        return _worker_subprocess_main(sys.argv[2])
    args = _parser().parse_args()
    payload = run_benchmark(
        workloads=cast(Sequence[WorkloadName], args.workloads),
        n_qubits=args.n_wires,
        engines=cast(Sequence[EngineName], args.engines),
        cpus=parse_cpu_list(args.cpu_list),
        worker_counts=args.workers,
        total_tasks=args.total_tasks,
        warmup=args.warmup,
        seed=args.seed,
        socket_id=args.socket_id,
    )
    write_json_atomic(args.json_output, payload)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(payload, artifact_name=args.json_output.name),
            encoding="utf-8",
        )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
