"""Fixed development controls for PMH reuse; no SDK ranking or promotion."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass
from functools import partial
from importlib.metadata import version
from pathlib import Path

from flagquantum.compiler import CouplingMap, compile
from flagquantum.compiler.optimization._linear_cnot import (
    _linear_matrix,
    _pmh_synthesize,
    resynthesize_cnot_blocks,
)
from flagquantum.core.ir import CircuitIR, Instruction

_UPSTREAM = "0c26a5a158c4d62b2b303a8c2f3c6eb2b14653b4"


@dataclass
class Metrics:
    operations: int
    two_body_count: int
    two_body_depth: int


@dataclass
class Observation:
    width: int
    family: str
    seed: int
    topology: str
    source_hash: str
    baseline: Metrics
    candidate: Metrics
    seconds: dict[str, list[float]]
    traced_peak_bytes: dict[str, int]


def _source(width: int, family: str, seed: int) -> CircuitIR:
    rng = random.Random(seed)
    pairs: list[tuple[int, int]] = []
    if family == "serial_mix":
        for _ in range(8 * width):
            first, second = rng.sample(range(width), 2)
            pairs.append((first, second))
    elif family == "parallel_layers":
        for _ in range(10):
            order = list(range(width))
            rng.shuffle(order)
            pairs.extend((order[i], order[i + 1]) for i in range(0, width, 2))
    else:
        assert family == "parity_relay"
        order = list(range(width))
        rng.shuffle(order)
        for index in range(0, width - 2, 3):
            first, middle, last = order[index : index + 3]
            pairs.extend(((first, middle), (middle, last)) * 2)
    return CircuitIR(width, tuple(Instruction("cx", pair) for pair in pairs))


def _identity(program: CircuitIR, *, coupling_map: CouplingMap | None) -> CircuitIR:
    return program


def _compile(source: CircuitIR, topology: str, candidate: bool) -> CircuitIR:
    coupling = CouplingMap.line(source.n_wires) if topology == "line" else None
    transform = resynthesize_cnot_blocks if candidate else _identity
    return compile(
        source,
        coupling_map=coupling,
        optimization_level=0,
        passes={"development.linear": partial(transform, coupling_map=coupling)},
    )


def _metrics(program: CircuitIR) -> Metrics:
    levels = [0] * program.n_wires
    count = 0
    for item in program.instructions:
        if len(item.wires) == 2:
            count += 1
            layer = max(levels[q] for q in item.wires) + 1
            for q in item.wires:
                levels[q] = layer
    return Metrics(len(program.instructions), count, max(levels, default=0))


def _upstream_conformance() -> int:
    import numpy as np
    from qiskit.synthesis.linear import synth_cnot_count_full_pmh

    if version("qiskit") != "2.6.2":
        raise ValueError("Upstream conformance requires the inspected Qiskit 2.6.2")
    count = 0
    for width in (2, 3, 4, 8, 16, 32):
        for seed in range(20):
            rng = random.Random(seed)
            operations = []
            for _ in range(4 * width):
                control, target = rng.sample(range(width), 2)
                operations.append((control, target))
            matrix = _linear_matrix(operations, width)
            circuit = synth_cnot_count_full_pmh(
                np.array(
                    [
                        [(row >> column) & 1 for column in range(width)]
                        for row in matrix
                    ],
                    dtype=bool,
                )
            )
            expected = tuple(
                (
                    circuit.find_bit(item.qubits[0]).index,
                    circuit.find_bit(item.qubits[1]).index,
                )
                for item in circuit.data
            )
            assert _pmh_synthesize(matrix) == expected, (width, seed)
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    upstream_equal = _upstream_conformance()
    rows: list[Observation] = []
    for width in (4, 8, 16, 32):
        for family in ("serial_mix", "parallel_layers", "parity_relay"):
            for seed in range(3):
                source = _source(width, family, seed)
                source_hash = source.content_hash
                for topology in ("all_to_all", "line"):
                    times: dict[str, list[float]] = {"baseline": [], "candidate": []}
                    metrics: dict[str, Metrics] = {}
                    for block in range(8):
                        labels = (
                            ("baseline", "candidate")
                            if block % 2 == 0
                            else ("candidate", "baseline")
                        )
                        for label in labels:
                            gc.collect()
                            started = time.perf_counter()
                            result = _compile(source, topology, label == "candidate")
                            elapsed = time.perf_counter() - started
                            if block:
                                times[label].append(elapsed)
                            value = _metrics(result)
                            assert label not in metrics or metrics[label] == value
                            metrics[label] = value
                            assert source.content_hash == source_hash
                    peaks: dict[str, int] = {}
                    for label in times:
                        gc.collect()
                        tracemalloc.start()
                        try:
                            result = _compile(source, topology, label == "candidate")
                            peaks[label] = tracemalloc.get_traced_memory()[1]
                        finally:
                            tracemalloc.stop()
                        assert _metrics(result) == metrics[label]
                    rows.append(
                        Observation(
                            width,
                            family,
                            seed,
                            topology,
                            source_hash,
                            metrics["baseline"],
                            metrics["candidate"],
                            times,
                            peaks,
                        )
                    )
                    checkpoint = args.output.with_suffix(".partial")
                    checkpoint.write_text(
                        json.dumps(
                            {"complete": False, "rows": [asdict(row) for row in rows]},
                            indent=2,
                        )
                        + "\n"
                    )
                    checkpoint.replace(args.output)
                    print(
                        f"completed {width} {family} {seed} {topology}",
                        file=sys.stderr,
                        flush=True,
                    )
    time_ratios = [
        statistics.median(row.seconds["candidate"])
        / statistics.median(row.seconds["baseline"])
        for row in rows
    ]
    peak_ratios = [
        row.traced_peak_bytes["candidate"] / row.traced_peak_bytes["baseline"]
        for row in rows
    ]
    regressions = [
        asdict(row)
        for row in rows
        if row.candidate.two_body_count > row.baseline.two_body_count
        or row.candidate.two_body_depth > row.baseline.two_body_depth
    ]
    report = {
        "complete": True,
        "implementation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=15
        ).strip(),
        "source_commit": _UPSTREAM,
        "evidence_class": "development",
        "distribution_semantics": "single_device_fast_path",
        "scalability_claim_allowed": False,
        "scope": "Explicit private pass versus an identity-pass control, optimization level zero, fresh topology per call. Not default-compiler or cross-SDK performance evidence. Inputs are inspected development constructions, not held-out scientific workloads.",
        "python": sys.version,
        "platform": platform.platform(),
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[1]
                / "flagquantum/compiler/optimization/_linear_cnot.py"
            ).read_bytes()
        ).hexdigest(),
        "upstream_equal_gate_sequences": upstream_equal,
        "rows": [asdict(row) for row in rows],
        "quality_regressions": regressions,
        "median_case_time_ratio": statistics.median(time_ratios),
        "worst_time_ratio": max(time_ratios),
        "worst_traced_peak_ratio": max(peak_ratios),
        "default_admission_passed": not regressions
        and max(time_ratios) <= 1.10
        and max(peak_ratios) <= 1.05,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "quality_regressions": len(regressions),
                "default_admission_passed": report["default_admission_passed"],
            }
        ),
        flush=True,
    )
    if not report["default_admission_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
