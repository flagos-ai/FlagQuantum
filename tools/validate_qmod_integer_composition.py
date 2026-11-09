"""Independent full-state Circuits for unchanged integer-composition originals."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.services.synthesis import _synthesize
from tools.validate_qmod_large_pauli import real_literal

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
CREDIT_DATA = (
    ((0.17, -0.43, 0.29), (-0.31, 0.57, 0.11)),
    ((-0.71, 0.23, 0.44), (0.32, -0.19, -0.51)),
)
DTYPE = torch.complex128


def function_body(source: str, name: str) -> str:
    match = re.search(r"\bqfunc\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", source)
    if match is None:
        raise ValueError(f"missing original helper {name}")
    depth = 1
    for end in range(match.end(), len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[match.end() : end]
    raise ValueError("unterminated original helper")


def fermi_reference(source: str, iterations: int) -> fq.Circuit:
    circuit = fq.Circuit(8, dtype=DTYPE)
    for bit in range(4):
        circuit.x(7 - bit)
    for index in range(10):
        body = function_body(source, f"G_expanded___{index}")
        matrix_start = body.index("[")
        depth = 0
        for end in range(matrix_start, len(body)):
            if body[end] == "[":
                depth += 1
            elif body[end] == "]":
                depth -= 1
                if depth == 0:
                    break
        literal = ast.parse(body[matrix_start : end + 1], mode="eval").body
        if not isinstance(literal, ast.List) or len(literal.elts) != 4:
            raise ValueError("expected original 4x4 matrix")
        rows = []
        for row in literal.elts:
            if not isinstance(row, ast.List) or len(row.elts) != 4:
                raise ValueError("expected original matrix row")
            rows.append([real_literal(ast.unparse(entry)) for entry in row.elts])
        view = function_body(source, f"given_rotation_expanded___{index}")
        bits = [int(bit) for bit in re.findall(r"qba\[(\d+)\]", view)]
        if len(bits) != 2:
            raise ValueError("expected original ordered pair")
        # Matrix basis follows numeric significance: array position zero is LSB.
        circuit.any(7 - bits[1], 7 - bits[0], matrix=torch.tensor(rows, dtype=DTYPE))
    x = torch.tensor([[0, 1], [1, 0]], dtype=DTYPE)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=DTYPE)
    exchange = (torch.kron(x, x) + torch.kron(y, y)) / 2
    rotations = []
    for index in range(3):
        body = function_body(source, f"K_expanded___{index}")
        time_arg = re.search(r"\},\s*([^,]+),\s*1,\s*1,\s*qba\s*\)", body)
        if time_arg is None:
            raise ValueError("expected original first-order exchange")
        rotations.append(torch.matrix_exp(-1j * real_literal(time_arg[1]) * exchange))
    for _ in range(iterations):
        for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
            circuit.any(7 - second, 7 - first, matrix=rotations[0])
        circuit.cphase(7, 3, theta=0.6).cphase(5, 1, theta=0.6)
        for first, second in ((1, 2), (5, 6)):
            circuit.any(7 - second, 7 - first, matrix=rotations[1])
        circuit.cphase(5, 1, theta=0.6).cphase(4, 0, theta=0.6)
        for first, second in ((1, 2), (5, 6)):
            circuit.any(7 - second, 7 - first, matrix=rotations[2])
    return circuit


def credit_reference(data1: tuple[float, ...], data2: tuple[float, ...]) -> fq.Circuit:
    circuit = fq.Circuit(3, dtype=DTYPE)
    pairs = ((0, 1), (0, 2), (1, 2))
    for _ in range(2):
        for bit in range(3):
            circuit.h(2 - bit)
        for bit, value in enumerate(data1):
            circuit.rz(2 - bit, theta=2 * value)
        for first, second in pairs:
            circuit.rzz(
                2 - first,
                2 - second,
                theta=2 * (data1[first] - math.pi) * (data1[second] - math.pi),
            )
    # The adjoint reverses each E H layer, not only its rotation signs.
    for _ in range(2):
        for first, second in reversed(pairs):
            circuit.rzz(
                2 - first,
                2 - second,
                theta=-2 * (data2[first] - math.pi) * (data2[second] - math.pi),
            )
        for bit in reversed(range(3)):
            circuit.rz(2 - bit, theta=-2 * data2[bit])
        for bit in reversed(range(3)):
            circuit.h(2 - bit)
    return circuit


def bb84_reference(source: str) -> fq.Circuit:
    match = re.search(
        r"bb84\s*\(\s*(\[[^]]*\])\s*,\s*(\[[^]]*\])\s*,\s*(\[[^]]*\])", source
    )
    if match is None:
        raise ValueError("missing original BB84 data")
    circuit = fq.Circuit(8, dtype=DTYPE)
    for group, gate in zip(match.groups(), ("x", "h", "h"), strict=True):
        data = ast.literal_eval(group)
        if (
            not isinstance(data, list)
            or len(data) != 8
            or any(type(bit) is not int or bit not in (0, 1) for bit in data)
        ):
            raise ValueError("invalid original BB84 bits")
        for bit, value in enumerate(data):
            if value:
                circuit.gate(gate, 7 - bit)
    return circuit


@dataclass(frozen=True)
class Validation:
    case: str
    source_blob: str
    iterations: int | None
    data_case: int | None
    classical_inputs: dict[str, int | tuple[float, ...]]
    qubits: int
    operations: int
    depth: int
    workspace: int
    max_abs_state_error: float
    state_norm: float
    elapsed_seconds: float
    validation: str = "passed"


def validate(case: str, *, iterations: int = 1, data_case: int = 0) -> Validation:
    started = time.perf_counter()
    manifest = json.loads((FIXTURES / "integer_composition_sources.json").read_text())
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
    source = raw.decode()
    bindings: dict[str, int | tuple[float, ...]] = {}
    if case == "fermi_hubbard_1D":
        if iterations not in (1, 2):
            raise ValueError("original acceptance uses iterations 1 and 2")
        bindings = {"num_iter": iterations}
        reference = fermi_reference(source, iterations)
    elif case == "credit_card_fraud":
        if data_case not in (0, 1):
            raise ValueError("choose one of two fixed nonzero data pairs")
        data1, data2 = CREDIT_DATA[data_case]
        bindings = {"data1": data1, "data2": data2}
        reference = credit_reference(data1, data2)
    else:
        reference = bb84_reference(source)
    program = _synthesize(
        path,
        classical_inputs=bindings,
        max_workspace=0,
        memory_limit_bytes=64 * 1024**2,
    )
    actual = fq.run(program).statevector().reshape(-1)
    expected = fq.run(reference).statevector().reshape(-1)
    error = float((actual - expected).abs().max())
    norm = float(actual.abs().square().sum())
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("resource report missing")
    if error > 1e-12 or abs(norm - 1) > 1e-10:
        raise AssertionError(f"full-state error {error}, norm {norm}")
    width = 3 if case == "credit_card_fraud" else 8
    ports = [bit for group in program.registers.values() for bit in group]
    if (
        resources.n_qubits != width
        or program.report["workspace_qubits"] != 0
        or sorted(ports) != list(range(width))
    ):
        raise AssertionError("original output width, ownership or workspace changed")
    return Validation(
        case,
        blob,
        iterations if case == "fermi_hubbard_1D" else None,
        data_case if case == "credit_card_fraud" else None,
        bindings,
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
        0,
        error,
        norm,
        time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("fermi_hubbard_1D", "credit_card_fraud", "bb84_implementation"),
        required=True,
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--data-case", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = asdict(
        validate(args.case, iterations=args.iterations, data_case=args.data_case)
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
