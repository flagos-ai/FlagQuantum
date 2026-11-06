"""Independent full-state Circuits for three unchanged predicate-oracle originals."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.services.synthesis import _synthesize

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
DTYPE = torch.complex128


def marked_states(case: str) -> tuple[int, tuple[int, ...]]:
    """Use mathematical differences, fixed record layout and the original >=4."""
    if case == "grover_operator":
        return 4, tuple(
            x + 4 * y
            for y in range(4)
            for x in range(4)
            if x + y < 4 and (x * y) % 4 == 2
        )
    if case == "oracles_workshop":
        return 4, tuple(x + 4 * y for y in range(4) for x in range(4) if x - y == 2)
    if case != "grover":
        raise ValueError("unknown predicate-oracle original")
    edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4))
    return 5, tuple(
        word
        for word in range(32)
        if sum(((word >> a) ^ (word >> b)) & 1 for a, b in edges) >= 4
    )


def reference_circuit(case: str, iterations: int, workspace: int) -> fq.Circuit:
    width, marked = marked_states(case)
    size = 1 << width
    diagonal = torch.ones(size, dtype=DTYPE)
    diagonal[list(marked)] = -1
    reflection = torch.full((size, size), 2 / size, dtype=DTYPE) - torch.eye(
        size, dtype=DTYPE
    )
    circuit = fq.Circuit(width + workspace, dtype=DTYPE)
    for bit in range(width):
        circuit.h(bit)
    for _ in range(iterations):
        circuit.any(*range(width), matrix=torch.diag(diagonal))
        circuit.any(*range(width), matrix=reflection)
    return circuit


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
    workspace_nonzero_probability: float
    state_norm: float
    validation: str = "passed"


def validate(case: str, iterations: int = 2) -> Validation:
    if iterations not in (0, 1, 2) or (case != "grover" and iterations != 2):
        raise ValueError("fixed originals use two rounds; grover uses r=0/1/2")
    manifest = json.loads((FIXTURES / "predicate_oracle_sources.json").read_text())
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
    program = _synthesize(
        path,
        classical_inputs={"r": iterations} if case == "grover" else {},
        memory_limit_bytes=64 * 1024**2,
    )
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("missing resource report")
    width, marked = marked_states(case)
    workspace = resources.n_qubits - width
    expected_ports = (
        {"nodes": tuple(range(5))}
        if case == "grover"
        else {"vars.x": (2, 3), "vars.y": (0, 1)}
    )
    if dict(program.registers) != expected_ports:
        raise AssertionError("record layout or source numeric bit order changed")
    actual = fq.run(program).statevector().reshape(-1)
    expected = (
        fq.run(reference_circuit(case, iterations, workspace)).statevector().reshape(-1)
    )
    error = float((actual - expected).abs().max())
    dirty = float(
        actual.reshape(1 << width, 1 << workspace)[:, 1:].abs().square().sum()
    )
    norm = float(actual.abs().square().sum())
    if error > 1e-12 or dirty > 1e-24 or abs(norm - 1) > 1e-10:
        raise AssertionError(
            f"absolute full-state error {error}; dirty {dirty}; norm {norm}"
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
        dirty,
        norm,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("grover_operator", "oracles_workshop", "grover"),
        required=True,
    )
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = asdict(validate(args.case, args.iterations))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
