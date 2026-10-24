"""Measure one surface-code circuit-location DEM in a fresh CPU process.

Run once per distance. Python allocation peaks exclude prior imports, layout
creation and the initial lowering used to count instructions. Process RSS is a
whole-process high-water mark, not the numerical primitive's exclusive memory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import torch

from flagquantum.noise import NoiseModel, bit_flip_channel
from flagquantum.qec import (
    DetectorErrorModel,
    RotatedSurfaceCode,
    build_memory_circuit,
)
from flagquantum.qec.dem.construction import _gate_bound_mechanisms
from flagquantum.qec.execution.signatures import _lowered_program


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True
    ).strip()
    if dirty:
        raise RuntimeError("commit tracked changes before recording benchmark evidence")
    code = RotatedSurfaceCode(distance=args.distance)
    memory = build_memory_circuit(code, rounds=args.rounds)
    noise = NoiseModel().add("cx", bit_flip_channel(0.01))
    program = _lowered_program(memory)
    faults = len(_gate_bound_mechanisms(noise, program))
    parities = len(memory.detectors) + len(memory.observables)
    rss_unit = 1 if sys.platform == "darwin" else 1024
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
    tracemalloc.start()
    start = time.perf_counter()
    model = DetectorErrorModel.from_memory_circuit(memory, noise=noise)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    record = {
        "source_revision": revision,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "world_size": 1,
        "local_world_size": 1,
        "node_count": 1,
        "rank_ownership": {"0": "whole_model"},
        "communication_bytes": 0,
        "distribution_semantics": "single_process",
        "scalability_claim_allowed": False,
        "release_gate_allowed": False,
        "non_release_evidence": True,
        "benchmark_evidence_class": "local_non_release",
        "scalability_blockers": [
            "local_cpu_construction_not_release_scalability_evidence"
        ],
        "blockers": ["local_cpu_construction_not_release_scalability_evidence"],
        "distance": args.distance,
        "rounds": args.rounds,
        "physical_qubits": program.n_wires,
        "instructions": len(program.instructions),
        "parities": parities,
        "fault_locations": faults,
        "reverse_instruction_visits": len(program.instructions) * parities,
        "fault_response_queries": faults * parities,
        "model_errors_after_merge": model.num_errors,
        "model_construction_seconds_with_tracemalloc": elapsed,
        "peak_traced_python_bytes": peak,
        "process_max_rss_before_bytes": rss_before,
        "process_max_rss_after_bytes": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        * rss_unit,
    }
    text = json.dumps(record, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
