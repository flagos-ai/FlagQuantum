#!/usr/bin/env python3
"""Measure isolated peak RSS for cross-framework CPU statevector batch tasks."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import torch

from .batched_statevector_corpus import (
    _ENGINE_LABELS,
    ENGINE_NAMES,
    EngineName,
    _engine_callable,
    build_parameter_batch,
    run_case,
)
from .contract import runtime_metadata, write_json_atomic
from .simulator_compare import SEED
from .simulator_workload_corpus import (
    WORKLOAD_NAMES,
    WorkloadName,
    _configure_threads,
)

SCHEMA = "flagquantum.batched_statevector_memory.v1"
RUNNER = "batched_statevector_memory"


def _peak_rss_bytes() -> int:
    """Normalize ``ru_maxrss`` to bytes on the supported Unix hosts."""

    resource = import_module("resource")
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _worker_payload(
    *,
    workload: WorkloadName,
    n_qubits: int,
    batch_size: int,
    engine: EngineName,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    """Execute one cold task in a fresh process and report its high-water RSS."""

    _configure_threads(threads)
    batched, scalar = build_parameter_batch(
        workload, n_qubits=n_qubits, batch_size=batch_size, seed=seed
    )
    function = _engine_callable(engine, batched, scalar, seed=seed, threads=threads)
    gc.collect()
    before = _peak_rss_bytes()
    started = time.perf_counter()
    output = function()
    elapsed = time.perf_counter() - started
    after = _peak_rss_bytes()
    state = output.detach().cpu().to(torch.complex128)
    norms = torch.linalg.vector_norm(state, dim=-1)
    return {
        "engine": engine,
        "peak_rss_bytes": after,
        "pre_execution_peak_rss_bytes": before,
        "execution_peak_rss_growth_bytes": max(0, after - before),
        "cold_execution_seconds": elapsed,
        "output_shape": tuple(int(value) for value in state.shape),
        "maximum_norm_error": float(torch.max(torch.abs(norms - 1)).item()),
        "measurement": "fresh_process_ru_maxrss",
    }


def _probe_engine(
    *,
    workload: WorkloadName,
    n_qubits: int,
    batch_size: int,
    engine: EngineName,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    """Run one engine in an isolated interpreter so RSS peaks do not overlap."""

    with tempfile.TemporaryDirectory(prefix="flagquantum-rss-") as directory:
        output = Path(directory) / "worker.json"
        command = (
            sys.executable,
            "-m",
            "flagquantum.benchmarking.batched_statevector_memory",
            "--worker-output",
            str(output),
            "--workloads",
            workload,
            "--n-wires",
            str(n_qubits),
            "--batch-sizes",
            str(batch_size),
            "--engines",
            engine,
            "--threads",
            str(threads),
            "--seed",
            str(seed),
        )
        environment = os.environ.copy()
        environment.update(
            {
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
            }
        )
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        if completed.returncode != 0 or not output.is_file():
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"isolated RSS probe failed for {engine}: {detail or 'no payload'}"
            )
        return cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))


def _summarize_memory_probes(
    probes: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    """Return robust medians while retaining every fresh-process observation."""

    if not probes:
        raise ValueError("at least one isolated memory probe is required")
    engine = probes[0]["engine"]
    output_shape = probes[0]["output_shape"]
    if any(
        probe["engine"] != engine or probe["output_shape"] != output_shape
        for probe in probes
    ):
        raise RuntimeError("isolated memory probes disagree on engine or output shape")

    def integer_median(name: str) -> int:
        return int(statistics.median(int(probe[name]) for probe in probes))

    return {
        "engine": engine,
        "peak_rss_bytes": integer_median("peak_rss_bytes"),
        "pre_execution_peak_rss_bytes": integer_median("pre_execution_peak_rss_bytes"),
        "execution_peak_rss_growth_bytes": integer_median(
            "execution_peak_rss_growth_bytes"
        ),
        "cold_execution_seconds": statistics.median(
            float(probe["cold_execution_seconds"]) for probe in probes
        ),
        "output_shape": output_shape,
        "maximum_norm_error": max(
            float(probe["maximum_norm_error"]) for probe in probes
        ),
        "measurement": "fresh_process_ru_maxrss",
        "aggregation": "median",
        "sample_count": len(probes),
        "samples": probes,
    }


def run_benchmark(
    *,
    workloads: tuple[WorkloadName, ...],
    n_qubits: tuple[int, ...],
    batch_sizes: tuple[int, ...],
    engines: tuple[EngineName, ...],
    threads: int,
    warmup: int,
    iterations: int,
    seed: int = SEED,
    memory_probes: int = 3,
) -> dict[str, Any]:
    """Measure timings/correctness together and RSS in isolated subprocesses."""

    if threads < 1:
        raise ValueError("threads must be positive")
    if memory_probes < 1:
        raise ValueError("memory_probes must be positive")
    if not workloads or not n_qubits or not batch_sizes or not engines:
        raise ValueError(
            "workloads, widths, batch sizes, and engines must not be empty"
        )
    if any(width < 1 for width in n_qubits) or any(size < 1 for size in batch_sizes):
        raise ValueError("widths and batch sizes must be positive")

    thread_environment = _configure_threads(threads)
    memory_results: dict[tuple[WorkloadName, int, int, EngineName], dict[str, Any]] = {}
    # Linux preserves a process's historical ru_maxrss across exec. Collect every
    # isolated probe before this parent executes a large statevector, otherwise a
    # later worker inherits the parent's earlier high-water mark and all engines
    # appear to have the same peak.
    for workload in workloads:
        for width in n_qubits:
            for batch_size in batch_sizes:
                for engine in engines:
                    memory_results[(workload, width, batch_size, engine)] = (
                        _summarize_memory_probes(
                            tuple(
                                _probe_engine(
                                    workload=workload,
                                    n_qubits=width,
                                    batch_size=batch_size,
                                    engine=engine,
                                    threads=threads,
                                    seed=seed,
                                )
                                for _ in range(memory_probes)
                            )
                        )
                    )

    cases: list[dict[str, Any]] = []
    for workload in workloads:
        for width in n_qubits:
            for batch_size in batch_sizes:
                case = run_case(
                    workload=workload,
                    n_qubits=width,
                    batch_size=batch_size,
                    engines=engines,
                    threads=threads,
                    warmup=warmup,
                    iterations=iterations,
                    seed=seed,
                )
                for engine in engines:
                    case["engines"][engine]["isolated_memory"] = memory_results[
                        (workload, width, batch_size, engine)
                    ]
                cases.append(case)

    passed = all(case["correctness"]["passed"] for case in cases)
    stable = all(case["stability"]["passed"] for case in cases)
    return runtime_metadata(
        runner=RUNNER,
        schema=SCHEMA,
        benchmark="cross_framework_batched_statevector_peak_rss",
        artifact_class="measured_comparison_run",
        benchmark_evidence_class="comparison_non_release",
        claim_evidence_type="unknown",
        distribution_semantics="median_of_fresh_processes_per_engine_case",
        scalability_claim_allowed=False,
        release_gate_allowed=False,
        non_release_evidence=True,
        scalability_blockers=("single_host_peak_rss_comparison",),
        passed=passed,
        correctness_passed=passed,
        all_measurements_stable=stable,
        platform=platform.platform(),
        environment={
            "device": "cpu",
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
            "torch": torch.__version__,
            "torch_threads": threads,
            "thread_environment": thread_environment,
        },
        methodology={
            "warmup": warmup,
            "iterations": iterations,
            "timing_scope": "complete_N_statevector_user_task",
            "circuit_construction_in_timing": False,
            "one_time_engine_initialization_in_timing": False,
            "per_call_conversion_and_result_retrieval_in_timing": True,
            "pennylane_native_batch_preprocessing_in_timing": False,
            "pennylane_native_batch_result_materialization_in_timing": True,
            "pennylane_native_batching": "broadcast_expand_preprocessed_once",
            "memory_scope": "fresh_process_high_water_resident_set",
            "memory_api": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "memory_probe_warmup": 0,
            "memory_probe_iterations": memory_probes,
            "process_baseline_included": True,
            "circuit_construction_in_peak_rss": True,
            "exact_statevector": True,
            "external_bridge_batching": "repeated_single_item_bridge",
        },
        workloads=workloads,
        n_wires=n_qubits,
        batch_sizes=batch_sizes,
        engines=engines,
        cases=tuple(cases),
    )


def render_markdown(payload: dict[str, Any], *, artifact_name: str) -> str:
    """Render exact time and isolated peak RSS in one prominent table."""

    lines = [
        "# Cross-framework CPU batched statevector time and peak RSS",
        "",
        f"Generated from [`{artifact_name}`]({artifact_name}). Timings are warm",
        "same-process medians. Each peak RSS sample comes from one fresh process",
        "running one complete cold task, so one engine cannot",
        "contaminate another's high-water mark. Each displayed value is the median",
        "of those probes; raw observations remain in JSON. The process baseline and",
        "circuit construction are included.",
    ]
    if "pennylane_lightning_native_batch" in payload["engines"]:
        lines.extend(
            (
                "PennyLane Lightning native batch uses public broadcast expansion with",
                "one-time device preprocessing outside warm timing. The separately named",
                "bridge engine executes one public FlagQuantum bridge call per batch item.",
            )
        )
    lines.extend(
        (
            "",
            "| Workload | Qubits | Batch | Engine | Median time (ms) | vs FQ batch | Peak RSS (MiB) | Execution RSS growth (MiB) |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        )
    )
    for case in payload["cases"]:
        ratios = case["comparison"]["engine_over_flagquantum_batch_median"]
        for engine in payload["engines"]:
            result = case["engines"][engine]
            ratio = 1.0 if engine == "flagquantum_native_batch" else ratios[engine]
            memory = result["isolated_memory"]
            lines.append(
                "| {workload} | {width} | {batch} | {engine} | {time:.3f} | "
                "{ratio:.2f}x | {peak:.1f} | {growth:.1f} |".format(
                    workload=case["workload"]["name"],
                    width=case["workload"]["n_wires"],
                    batch=case["workload"]["batch_size"],
                    engine=_ENGINE_LABELS[cast(EngineName, engine)],
                    time=result["batch_total"]["median_seconds"] * 1000,
                    ratio=ratio,
                    peak=memory["peak_rss_bytes"] / 2**20,
                    growth=memory["execution_peak_rss_growth_bytes"] / 2**20,
                )
            )
    lines.extend(
        (
            "",
            "Ratios above one mean FlagQuantum native batch was faster. Peak RSS",
            "includes framework/interpreter baseline and is local comparison evidence,",
            "not a universal framework ranking or a release/scalability claim.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOAD_NAMES, default=WORKLOAD_NAMES[:5]
    )
    parser.add_argument("--n-wires", type=int, nargs="+", default=(18,))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(32,))
    parser.add_argument(
        "--engines", nargs="+", choices=ENGINE_NAMES, default=ENGINE_NAMES
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--memory-probes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    workloads = tuple(cast(WorkloadName, item) for item in args.workloads)
    engines = tuple(cast(EngineName, item) for item in args.engines)
    if args.worker_output is not None:
        if (
            len(workloads) != 1
            or len(args.n_wires) != 1
            or len(args.batch_sizes) != 1
            or len(engines) != 1
        ):
            parser.error(
                "worker mode requires exactly one workload, width, batch, and engine"
            )
        payload = _worker_payload(
            workload=workloads[0],
            n_qubits=args.n_wires[0],
            batch_size=args.batch_sizes[0],
            engine=engines[0],
            threads=args.threads,
            seed=args.seed,
        )
        args.worker_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        return 0

    payload = run_benchmark(
        workloads=workloads,
        n_qubits=tuple(args.n_wires),
        batch_sizes=tuple(args.batch_sizes),
        engines=engines,
        threads=args.threads,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        memory_probes=args.memory_probes,
    )
    if args.json_output is not None:
        write_json_atomic(args.json_output, payload)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(
                payload,
                artifact_name=(
                    args.json_output.name
                    if args.json_output is not None
                    else "batched-statevector-memory.json"
                ),
            ),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ("RUNNER", "SCHEMA", "main", "render_markdown", "run_benchmark")
