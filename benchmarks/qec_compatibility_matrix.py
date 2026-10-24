"""Observe one bounded QEC case per fresh CPU process; no performance gate."""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import torch

from flagquantum.benchmarking.contract import runtime_metadata, write_json_atomic
from flagquantum.qec import (
    DetectorErrorModel,
    Pauli,
    PhenomenologicalNoise,
    PyMatchingDecoder,
    RotatedSurfaceCode,
    build_memory_circuit,
    compile_memory_sampler,
)
from flagquantum.qec.adapters.stim import (
    compile_stim_sampler,
    memory_to_stim_text,
    stim_circuit_to_dem_text,
)


class _Observation(TypedDict):
    warmup: bool
    completed_shots: int
    logical_failures: int
    batches: int
    phases_seconds: dict[str, float]
    pipeline_seconds: float
    shots_per_second: float
    visible_batch_tensor_bytes_max: int


def _observe(
    *, distance: int, basis: str, entrance: str, shots: int, batch_size: int
) -> _Observation:
    started = time.perf_counter()
    phases: dict[str, float] = {}
    code = RotatedSurfaceCode(distance)
    product = Pauli(x_qubits=code.data_qubits) if basis == "x" else None
    memory = build_memory_circuit(code, rounds=3, product=product)
    phases["memory_construction"] = time.perf_counter() - started
    noise = PhenomenologicalNoise(
        data_flip=0.005, phase_flip=0.007, measurement_flip=0.009
    )
    tick = time.perf_counter()
    source = memory_to_stim_text(memory, noise=noise)
    phases["native_export"] = time.perf_counter() - tick
    tick = time.perf_counter()
    model = DetectorErrorModel.from_stim_text(stim_circuit_to_dem_text(source))
    phases["strict_dem_construction"] = time.perf_counter() - tick
    tick = time.perf_counter()
    decoder = PyMatchingDecoder.from_detector_error_model(
        model.merge_duplicate_mechanisms()
    )
    phases["decoder_setup"] = time.perf_counter() - tick
    tick = time.perf_counter()
    sampler = (
        compile_memory_sampler(memory, noise=noise, seed=53)
        if entrance == "native"
        else compile_stim_sampler(source, seed=53)
    )
    phases["sampler_compile"] = time.perf_counter() - tick
    phases.update(sampling=0.0, decoding=0.0, count_reduction=0.0)
    completed = failures = batches = visible = 0
    with sampler as active_sampler:
        while completed < shots:
            requested = min(batch_size, shots - completed)
            tick = time.perf_counter()
            sample = active_sampler.sample(shots=requested)
            phases["sampling"] += time.perf_counter() - tick
            tick = time.perf_counter()
            predicted = decoder.decode_batch(sample.detectors)
            phases["decoding"] += time.perf_counter() - tick
            tick = time.perf_counter()
            if (
                sample.shots != requested
                or sample.detectors.shape != (requested, model.num_detectors)
                or sample.observables.shape != (requested, model.num_observables)
                or predicted.shape != sample.observables.shape
            ):
                raise ValueError("incomplete detector/logical batch")
            failures += int((predicted != sample.observables).any(dim=1).sum())
            completed += requested
            batches += 1
            visible = max(
                visible,
                sum(
                    t.numel() * t.element_size()
                    for t in (sample.detectors, sample.observables, predicted)
                ),
            )
            phases["count_reduction"] += time.perf_counter() - tick
            del sample, predicted
    wall = time.perf_counter() - started
    return {
        "warmup": False,
        "completed_shots": completed,
        "logical_failures": failures,
        "batches": batches,
        "phases_seconds": phases,
        "pipeline_seconds": wall,
        "shots_per_second": completed / wall,
        "visible_batch_tensor_bytes_max": visible,
    }


def _run_case(
    *,
    distance: int,
    basis: str,
    entrance: str,
    shots: int,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    for name, value, choices in (
        ("distance", distance, (3, 5, 7, 9, 13, 17)),
        ("shots", shots, range(1, 4098)),
        ("batch_size", batch_size, range(1, 129)),
        ("warmup", warmup, range(4)),
        ("iterations", iterations, range(1, 4)),
    ):
        if type(value) is not int or value not in choices:
            raise ValueError(f"{name} is outside the declared observation budget")
    if basis not in ("x", "z") or entrance not in ("native", "exported_stim"):
        raise ValueError("unsupported basis or experiment entrance")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("one local CPU process is required")
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=10
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True, timeout=10
    ).strip()
    if dirty:
        raise RuntimeError("commit all source changes before collecting evidence")
    rows = []
    for index in range(warmup + iterations):
        row = _observe(
            distance=distance,
            basis=basis,
            entrance=entrance,
            shots=shots,
            batch_size=batch_size,
        )
        row["warmup"] = index < warmup
        rows.append(row)
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = int(rss * (1 if sys.platform == "darwin" else 1024))
    except ImportError:
        rss_bytes = None
    return runtime_metadata(
        runner="qec_compatibility_matrix",
        schema="flagquantum.qec_compatibility_matrix.local_observations.v1",
        source_revision=revision,
        tracked_tree_clean=True,
        device="cpu",
        distribution_semantics="single_process",
        local_world_size=1,
        node_count=1,
        rank_ownership={"0": "whole_workload"},
        communication_bytes=0,
        non_release_evidence=True,
        scalability_claim_allowed=False,
        release_gate_allowed=False,
        blockers=["local CPU observations; no accelerator or capacity certification"],
        versions={name: version(name) for name in ("torch", "stim", "pymatching")},
        machine=platform.machine(),
        torch_threads=torch.get_num_threads(),
        logical_cpu_count=os.cpu_count(),
        thread_environment={
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        workload=dict(
            distance=distance,
            basis=basis,
            entrance=entrance,
            rounds=3,
            shots=shots,
            batch_size=batch_size,
            seed=53,
            noise=dict(data_flip=0.005, phase_flip=0.007, measurement_flip=0.009),
        ),
        methodology=dict(
            warmup=warmup,
            iterations=iterations,
            seed_restarted_each_repetition=True,
            independent_statistical_trials=False,
            dem_law="strict conversion of original exported source",
            pipeline_includes_close=True,
            pipeline_excludes="initial imports, metadata, report IO",
            source_origin="native memory export for both entrances",
        ),
        process_max_rss_bytes=rss_bytes,
        pipeline_median_seconds=statistics.median(
            r["pipeline_seconds"] for r in rows[warmup:]
        ),
        observations=rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--basis", choices=("x", "z"), required=True)
    parser.add_argument(
        "--entrance", choices=("native", "exported_stim"), required=True
    )
    parser.add_argument("--shots", type=int, default=1025)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("output")
    write_json_atomic(output, _run_case(**args))


if __name__ == "__main__":
    main()
