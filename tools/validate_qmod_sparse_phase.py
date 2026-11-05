"""Explicit full-size sparse-phase validation with independent classical costs.

Run each case in a fresh process to retain per-case peak RSS and timings.
The original files and layer counts are preserved. This tool is an opt-in
large-state validation and is separate from the routine integration suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth

WIDTH = 24
SIZE = 1 << WIDTH
CHUNK = 1 << 16
FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
BLOBS = {
    "resiliency_planning": "0034d5aba2801be900a83a9171b36c1c0a5c77e8",
    "resiliency_planning_AMD": "0034d5aba2801be900a83a9171b36c1c0a5c77e8",
    "network_traffic_optimization": "9534cb1d36f6b7c77f207cdff43ca4d67030d689",
}


def resiliency_cost(bits, real=Fraction):
    linear = real("0.025") * sum(bits[index] for index in (*range(6), *range(12, 18)))
    linear += real("0.05") * (bits[6] + bits[18])
    crossing = real("0.25") * sum(
        bits[a] * bits[b] for a, b in ((7, 19), (8, 20), (9, 21), (11, 23))
    )
    crossing += bits[1] * bits[14] + bits[2] * bits[13]
    constraints = (
        1 - 2 * bits[7] + bits[0] + bits[1],
        -2 * bits[19] + bits[12] + bits[13],
        -2 * bits[8] + bits[3] + bits[5],
        1 - 2 * bits[20] + bits[15] + bits[17],
        -2 * bits[9] + bits[0] + bits[2] + bits[3] + bits[4],
        -2 * bits[21] + bits[12] + bits[14] + bits[15] + bits[16],
        1 - 2 * bits[10] + bits[1] + bits[2] + bits[6],
        1 - 2 * bits[22] + bits[13] + bits[14] + bits[18],
        -2 * bits[11] + bits[4] + bits[5] + bits[6],
        -2 * bits[23] + bits[16] + bits[17] + bits[18],
    )
    slack = sum(1 - bits[index] for index in (10, 7, 22, 20))
    return (
        linear
        + crossing
        + real("0.125") * (sum(value**2 for value in constraints) + slack)
    )


def network_cost(bits, real=Fraction):
    weights = (
        "0.133333333333333",
        "0.2",
        "0.133333333333333",
        "0.066666666666667",
        "0.066666666666667",
        "0.133333333333333",
        "0.133333333333333",
        "0.066666666666667",
    )
    linear = sum(real(weights[index % 8]) * bit for index, bit in enumerate(bits))
    flow = []
    for group, offset in enumerate((0, 8, 16)):
        data = bits[offset : offset + 8]
        supply = [0, 0, 0]
        supply[group] = 1
        flow.extend(
            (
                supply[0] - data[0] - data[1],
                data[0] - data[2] - data[3] + supply[1],
                data[1] - data[4] - data[5] + supply[2],
                data[3] + data[5] - data[6],
                data[2] + data[4] - data[7] - (1 if group == 0 else 0),
                data[6] + data[7] - (1 if group != 0 else 0),
            )
        )
    capacity = 0
    for index in range(8):
        first, second, third = bits[index], bits[index + 8], bits[index + 16]
        capacity += (
            first * second + first * third + second * third - 2 * first * second * third
        )
    return linear + real("0.2") * (sum(value**2 for value in flow) + capacity)


def independent_terms(cost, degree, width=WIDTH):
    """Recover low-degree coefficients by exact classical finite differences."""
    coefficients = {0: cost([0] * width)}
    for size in range(1, degree + 1):
        for subset in combinations(range(width), size):
            mask = sum(1 << bit for bit in subset)
            value = cost([(mask >> bit) & 1 for bit in range(width)])
            proper = (mask - 1) & mask
            while True:
                value -= coefficients.get(proper, 0)
                if proper == 0:
                    break
                proper = (proper - 1) & mask
            if value:
                coefficients[mask] = value
    return {mask: float(value) for mask, value in coefficients.items() if value}


def reference_circuit(params, terms, width=WIDTH):
    circuit = fq.Circuit(width, dtype=torch.complex128)
    for wire in range(width):
        circuit.h(wire)
    for gamma, beta in zip(params[::2], params[1::2], strict=True):
        for mask, coefficient in terms.items():
            angle = gamma * coefficient
            wires = tuple(width - 1 - bit for bit in range(width) if mask & (1 << bit))
            if not wires:
                circuit.x(0).phase(0, theta=angle).x(0).phase(0, theta=angle)
            elif len(wires) == 1:
                circuit.phase(wires[0], theta=angle)
            elif len(wires) == 2:
                circuit.cphase(*wires, theta=angle)
            else:
                diagonal = torch.ones(1 << len(wires), dtype=torch.complex128)
                diagonal[-1] = torch.exp(
                    torch.tensor(1j * angle, dtype=torch.complex128)
                )
                circuit.any(*wires, matrix=torch.diag(diagonal))
        for wire in range(width):
            circuit.rx(wire, theta=beta)
    return circuit


def maximum_error(actual, expected):
    error = 0.0
    if actual.numel() != expected.numel():
        raise AssertionError("state sizes differ")
    for start in range(0, actual.numel(), CHUNK):
        error = max(
            error,
            float(
                (actual[start : start + CHUNK] - expected[start : start + CHUNK])
                .abs()
                .max()
            ),
        )
    if error > 1e-12:
        raise AssertionError(f"full-state maximum absolute error {error}")
    return error


def phase_cost_check(source, count, cost):
    params = [0.0] * count
    params[0] = 0.19
    program = synth.synthesize(
        source, classical_inputs={"params": params}, memory_limit_bytes=512 * 1024**2
    )
    state = fq.run(program).statevector().reshape(-1)
    error = 0.0
    for start in range(0, SIZE, CHUNK):
        values = torch.arange(start, start + CHUNK, dtype=torch.int64)
        bits = [((values >> bit) & 1).to(torch.float64) for bit in range(WIDTH)]
        expected = torch.exp(1j * params[0] * cost(bits, float)) / (SIZE**0.5)
        error = max(error, float((state[start : start + CHUNK] - expected).abs().max()))
    if error > 1e-12:
        raise AssertionError(f"classical full-domain cost phase error {error}")
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(BLOBS), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--check-phase", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or not (-0.1 <= args.scale <= 0.1) or not args.scale:
        parser.error("use positive threads and a nonzero scale within ±0.1")
    torch.set_num_threads(args.threads)
    source = FIXTURES / f"{args.case}.qmod"
    raw = source.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != BLOBS[args.case]:
        raise ValueError("original source blob mismatch")
    cost = (
        network_cost if args.case == "network_traffic_optimization" else resiliency_cost
    )
    degree, count = (3, 10) if cost is network_cost else (2, 40)
    params = [
        args.scale * (index + 1) * (-1 if index % 2 else 1) for index in range(count)
    ]
    times = {}
    started = time.perf_counter()
    checkpoint = started
    program = synth.synthesize(
        source, classical_inputs={"params": params}, memory_limit_bytes=512 * 1024**2
    )
    times["synthesis_seconds"] = time.perf_counter() - checkpoint
    print(f"{args.case} scale={args.scale} synthesized", flush=True)
    checkpoint = time.perf_counter()
    actual = fq.run(program).statevector().reshape(-1)
    times["execution_seconds"] = time.perf_counter() - checkpoint
    print(f"{args.case} scale={args.scale} executed", flush=True)
    checkpoint = time.perf_counter()
    terms = independent_terms(cost, degree)
    reference = reference_circuit(params, terms)
    times["reference_construction_seconds"] = time.perf_counter() - checkpoint
    checkpoint = time.perf_counter()
    expected = fq.run(reference).statevector().reshape(-1)
    times["reference_execution_seconds"] = time.perf_counter() - checkpoint
    checkpoint = time.perf_counter()
    error = maximum_error(actual, expected)
    norm = float(actual.abs().square().sum())
    if abs(norm - 1) > 1e-9:
        raise AssertionError(f"state norm {norm}")
    times["comparison_seconds"] = time.perf_counter() - checkpoint
    del actual, expected, reference
    print(f"{args.case} scale={args.scale} full state verified", flush=True)
    phase_error = None
    if args.check_phase:
        checkpoint = time.perf_counter()
        phase_error = phase_cost_check(source, count, cost)
        times["classical_phase_check_seconds"] = time.perf_counter() - checkpoint
    resources = program.report["resources"]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "case": args.case,
        "source_blob": blob,
        "parameters": params,
        "threads": args.threads,
        "state_dtype": "complex128",
        "state_bytes": SIZE * 16,
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": program.report["workspace_qubits"],
        "oracle_terms": len(terms),
        "oracle_degree": degree,
        "max_abs_state_error": error,
        "state_norm": norm,
        "max_abs_classical_phase_error": phase_error,
        "timings": times,
        "total_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
        "validation": "passed",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
