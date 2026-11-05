"""Validate unchanged Knapsack originals against direct dense-diagonal Circuits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
BLOBS = {
    "qaoa": "7cc0e92318f042f1d16b31345f8b3a0f6139dac6",
    "qaoa_knapsack": "3452eef85ccab147d98c7927314acfd70384a768",
    "algo_design_QCE_tutorial_part_II": "5f536d451355bf821353c1772d85e4d508f69be5",
}


def reference_circuit(parameters: Sequence[float]) -> fq.Circuit:
    """Use all 32 classical costs directly, without the parser or polynomial compiler."""
    costs = torch.tensor(
        [
            -(3 * a + 5 * b) if 2 * a + 3 * b <= 12 else 0
            for b in range(4)
            for a in range(8)
        ],
        dtype=torch.float64,
    )
    circuit = fq.Circuit(5, dtype=torch.complex128)
    for wire in range(5):
        circuit.h(wire)
    for gamma, beta in zip(parameters[:3], parameters[3:], strict=True):
        circuit.any(*range(5), matrix=torch.diag(torch.exp(1j * gamma * costs)))
        for wire in range(5):
            circuit.rx(wire, theta=beta)
    return circuit


@dataclass(frozen=True)
class Validation:
    case: str
    source_blob: str
    parameters: list[float]
    qubits: int
    operations: int
    depth: int
    workspace: int
    named_registers: dict[str, tuple[int, ...]]
    max_abs_state_error: float
    max_abs_probability_error: float
    state_norm: float
    output_qubits_cover_state: bool
    total_seconds: float
    peak_rss_bytes: int
    validation: str = "passed"


def validate(case: str, scale: float) -> Validation:
    if not math.isfinite(scale) or not 0 < abs(scale) <= 0.1:
        raise ValueError("use a nonzero finite scale within +/-0.1")
    source = FIXTURES / f"{case}.qmod"
    raw = source.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if blob != BLOBS[case]:
        raise ValueError("original source blob mismatch")
    parameters = [scale * (i + 1) * (-1 if i % 2 else 1) for i in range(6)]
    started = time.perf_counter()
    program = synth.synthesize(
        source,
        classical_inputs={"params": parameters},
        max_qubits=5,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
    result = fq.run(program)
    actual = result.statevector().reshape(-1)
    expected = fq.run(reference_circuit(parameters)).statevector().reshape(-1)
    error = float((actual - expected).abs().max())
    distribution = program.probabilities(result, registers=("v.a", "v.b"))
    probability_error = max(
        abs(distribution.get((a, b), 0.0) - float(expected[a + 8 * b].abs().square()))
        for a in range(8)
        for b in range(4)
    )
    norm = float(actual.abs().square().sum())
    ports = [wire for wires in program.registers.values() for wire in wires]
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("expected actual Compiler resource estimates")
    if error > 1e-12 or probability_error > 1e-12 or abs(norm - 1) > 1e-12:
        raise AssertionError(
            f"state/probability/norm mismatch: {error}, {probability_error}, {norm}"
        )
    if (
        resources.n_qubits != 5
        or program.report["workspace_qubits"] != 0
        or sorted(ports) != list(range(5))
    ):
        raise AssertionError(
            "five named output bits must cover the state without workspace"
        )
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return Validation(
        case,
        blob,
        parameters,
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
        program.report["workspace_qubits"],
        dict(program.registers),
        error,
        probability_error,
        norm,
        True,
        time.perf_counter() - started,
        peak if sys.platform == "darwin" else peak * 1024,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(BLOBS), required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    report = asdict(validate(args.case, args.scale))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
