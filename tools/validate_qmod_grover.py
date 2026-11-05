"""Independent dense reflections and analytic vectors for pinned Grover originals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
DTYPE = torch.complex128


def marked_states(case: str) -> tuple[int, tuple[int, ...]]:
    """Enumerate the published SAT clauses or graph, without the Qmod parser."""
    if case == "3_sat_grover":
        # Signed literals: +k denotes bit k-1, -k its Boolean negation.
        clauses = (
            (2, 3, 4),
            (-1, 2, 3),
            (-1, 2, -3),
            (-1, -2, 3),
            (1, -2, -3),
            (1, -2, 3),
            (-1, -2, -4),
            (-1, -2, 4),
            (-2, -3, -4),
            (2, -3, 4),
            (1, -3, 4),
            (1, -2, -4),
            (-1, -2, -3),
        )
        marked = []
        for word in range(16):
            if all(
                any(
                    bool(word & (1 << (abs(literal) - 1))) == (literal > 0)
                    for literal in clause
                )
                for clause in clauses
            ):
                marked.append(word)
        return 4, tuple(marked)
    if case != "grover_max_cut":
        raise ValueError("unknown original Grover case")
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
    # The original marks <= CUT_SIZE, deliberately not a maximization rewrite.
    return 5, tuple(
        word
        for word in range(32)
        if sum(((word >> a) ^ (word >> b)) & 1 for a, b in edges) <= 4
    )


def reference_circuit(case: str, iterations: int, workspace: int = 0) -> fq.Circuit:
    """Use dense diagonal marking and 2|s><s|-I, independent of gate lowering."""
    width, marked = marked_states(case)
    size = 1 << width
    diagonal = torch.ones(size, dtype=DTYPE)
    diagonal[list(marked)] = -1
    diffuser = torch.full((size, size), 2 / size, dtype=DTYPE) - torch.eye(
        size, dtype=DTYPE
    )
    circuit = fq.Circuit(width + workspace, dtype=DTYPE)
    # Physical Circuit positions run MSB first; source q[i] is numeric bit i.
    target = tuple(range(width))
    for position in target:
        circuit.h(position)
    for _ in range(iterations):
        circuit.any(*target, matrix=torch.diag(diagonal))
        circuit.any(*target, matrix=diffuser)
    return circuit


def analytic_state(case: str, iterations: int) -> torch.Tensor:
    """Closed two-dimensional amplification formula, including absolute signs."""
    width, marked = marked_states(case)
    size, good = 1 << width, len(marked)
    theta = math.asin(math.sqrt(good / size))
    positive = math.sin((2 * iterations + 1) * theta) / math.sqrt(good)
    negative = math.cos((2 * iterations + 1) * theta) / math.sqrt(size - good)
    return torch.tensor(
        [positive if word in marked else negative for word in range(size)], dtype=DTYPE
    )


@dataclass(frozen=True)
class Validation:
    case: str
    source_blob: str
    iterations: int
    marked_states: tuple[int, ...]
    qubits: int
    operations: int
    depth: int
    workspace: int
    max_abs_state_error: float
    analytic_max_abs_error: float
    workspace_nonzero_probability: float
    state_norm: float
    validation: str = "passed"


def validate(case: str, iterations: int) -> Validation:
    if iterations not in (0, 1, 2):
        raise ValueError("original acceptance uses r=0/1/2")
    manifest = json.loads((FIXTURES / "grover_sources.json").read_text())
    entry = next(
        item for item in manifest["fixtures"] if item["file"] == case + ".qmod"
    )
    path = FIXTURES / entry["file"]
    raw = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if (
        blob != entry["git_blob_sha1"]
        or hashlib.sha256(raw).hexdigest() != entry["sha256"]
    ):
        raise ValueError("original source digest mismatch")
    if (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        != manifest["license_sha256"]
    ):
        raise ValueError("original license digest mismatch")
    program = synth.synthesize(
        path, classical_inputs={"r": iterations}, memory_limit_bytes=64 * 1024**2
    )
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("resource report missing")
    width, marked = marked_states(case)
    workspace = resources.n_qubits - width
    register = "x" if case == "3_sat_grover" else "v"
    if tuple(program.registers[register]) != tuple(range(width)):
        raise AssertionError("source numeric bit order changed")
    actual = fq.run(program).statevector().reshape(-1)
    expected = (
        fq.run(reference_circuit(case, iterations, workspace)).statevector().reshape(-1)
    )
    closed = torch.zeros_like(actual)
    closed[:: 1 << workspace] = analytic_state(case, iterations)
    error = float((actual - expected).abs().max())
    analytic_error = float((actual - closed).abs().max())
    scratch = actual.reshape(1 << width, 1 << workspace)
    dirty = float(scratch[:, 1:].abs().square().sum())
    norm = float(actual.abs().square().sum())
    if max(error, analytic_error) > 1e-12 or dirty > 1e-24 or abs(norm - 1) > 1e-10:
        raise AssertionError(
            f"full-state errors {error}/{analytic_error}, dirty {dirty}, norm {norm}"
        )
    return Validation(
        case,
        blob,
        iterations,
        marked,
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
        workspace,
        error,
        analytic_error,
        dirty,
        norm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", choices=("3_sat_grover", "grover_max_cut"), required=True
    )
    parser.add_argument("--iterations", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = asdict(validate(args.case, args.iterations))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
