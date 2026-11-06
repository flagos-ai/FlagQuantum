"""Validate original QAOA programs against an independent arithmetic Circuit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.services.synthesis import _synthesize

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"


class ArithmeticOracle:
    """Interpret classical source arithmetic without the Qmod parser or lowering."""

    def __init__(self, source: str) -> None:
        shape = re.search(r"x:\s*qbit\[(\d+)\]\[(\d+)\]", source)
        count = re.search(r"params:\s*real\[(\d+)\]", source)
        if shape is None or count is None:
            raise ValueError(
                "oracle requires one rectangular bit field and real parameters"
            )
        self.columns, rows = map(int, shape.groups())
        self.width = self.columns * rows
        self.count = int(count[1])
        start = re.search(r"\bphase\s*\(", source)
        if start is None:
            raise ValueError("oracle requires a phase cost")
        opening, nesting = start.end(), 0
        for end in range(opening, len(source)):
            character = source[end]
            if character in "([":
                nesting += 1
            elif character in ")]":
                nesting -= 1
            elif character == "," and nesting == 0:
                break
        self.tree = ast.parse(source[opening:end], mode="eval").body
        self.degree = min(self.width, self.degree_of(self.tree))

    def degree_of(self, node: ast.AST) -> int:
        if isinstance(node, ast.Constant):
            return 0
        if isinstance(node, ast.Subscript):
            return 1
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return self.degree_of(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = self.degree_of(node.left), self.degree_of(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return max(left, right)
            if isinstance(node.op, ast.Mult):
                return left + right
            if isinstance(node.op, ast.Div) and right == 0:
                return left
            if (
                isinstance(node.op, ast.Pow)
                and isinstance(node.right, ast.Constant)
                and type(node.right.value) is int
            ):
                return left * node.right.value
        raise ValueError("unsupported independent arithmetic degree")

    def cost(self, bits: Sequence[int]) -> Fraction:
        def evaluate(node: ast.AST) -> Fraction:
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return Fraction(str(node.value))
            if isinstance(node, ast.Subscript) and isinstance(
                node.value, ast.Subscript
            ):
                field, row, column = node.value.value, node.value.slice, node.slice
                if not (
                    isinstance(field, ast.Attribute)
                    and field.attr == "x"
                    and isinstance(field.value, ast.Name)
                    and field.value.id == "v"
                    and isinstance(row, ast.Constant)
                    and type(row.value) is int
                    and isinstance(column, ast.Constant)
                    and type(column.value) is int
                    and 0 <= column.value < self.columns
                    and 0 <= row.value * self.columns + column.value < self.width
                ):
                    raise ValueError("invalid independent bit address")
                return Fraction(bits[row.value * self.columns + column.value])
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -evaluate(node.operand)
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left), evaluate(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    return left / right
                if (
                    isinstance(node.op, ast.Pow)
                    and right.denominator == 1
                    and 0 <= right <= 8
                ):
                    return left ** int(right)
            raise ValueError("unsupported independent arithmetic")

        return evaluate(self.tree)

    def terms(self) -> dict[int, Fraction]:
        coefficients = {0: self.cost([0] * self.width)}
        for size in range(1, self.degree + 1):
            for subset in combinations(range(self.width), size):
                mask = sum(1 << bit for bit in subset)
                value = self.cost([(mask >> bit) & 1 for bit in range(self.width)])
                proper = (mask - 1) & mask
                while True:
                    value -= coefficients.get(proper, Fraction(0))
                    if proper == 0:
                        break
                    proper = (proper - 1) & mask
                if value:
                    coefficients[mask] = value
        return {mask: value for mask, value in coefficients.items() if value}


def reference_circuit(oracle: ArithmeticOracle, params: Sequence[float]) -> fq.Circuit:
    circuit = fq.Circuit(oracle.width, dtype=torch.complex128)
    for wire in range(oracle.width):
        circuit.h(wire)
    # Convert exact classical monomials to parity characters using b=(1-Z)/2.
    # This reference uses neither Compiler polynomials nor its controlled gates.
    characters: dict[int, Fraction] = {}
    for mask, value in oracle.terms().items():
        coefficient = value / (1 << mask.bit_count())
        subset = mask
        while True:
            characters[subset] = characters.get(subset, Fraction(0)) + (
                -coefficient if subset.bit_count() % 2 else coefficient
            )
            if subset == 0:
                break
            subset = (subset - 1) & mask
    half = len(params) // 2
    for gamma, beta in zip(params[:half], params[half:], strict=True):
        for mask, coefficient in characters.items():
            if not coefficient:
                continue
            angle = gamma * float(coefficient)
            wires = tuple(
                oracle.width - 1 - bit for bit in range(oracle.width) if mask >> bit & 1
            )
            if not wires:
                circuit.x(0).phase(0, theta=angle).x(0).phase(0, theta=angle)
            else:
                for control in wires[:-1]:
                    circuit.cnot(control, wires[-1])
                circuit.rz(wires[-1], theta=-2 * angle)
                for control in reversed(wires[:-1]):
                    circuit.cnot(control, wires[-1])
        for wire in range(oracle.width):
            circuit.rx(wire, theta=beta)
    return circuit


@dataclass(frozen=True)
class Validation:
    case: str
    source_blob: str
    parameters: tuple[float, ...]
    qubits: int
    operations: int
    depth: int
    workspace: int
    max_abs_state_error: float
    max_workspace_probability: float
    state_norm: float
    elapsed_seconds: float
    validation: str = "passed"


def validate(case: str, scale: float) -> Validation:
    started = time.perf_counter()
    manifest = json.loads((FIXTURES / "phase_factor_sources.json").read_text())
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
    oracle = ArithmeticOracle(raw.decode())
    params = tuple(scale * (i + 1) * (-1 if i % 2 else 1) for i in range(oracle.count))
    program = _synthesize(
        path, classical_inputs={"params": params}, memory_limit_bytes=512 * 1024**2
    )
    actual = fq.run(program).statevector().reshape(-1)
    reference = fq.run(reference_circuit(oracle, params)).statevector().reshape(-1)
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("resource report missing")
    workspace = resources.n_qubits - oracle.width
    expected = torch.zeros_like(actual)
    expected[:: 1 << workspace] = reference
    error = float((actual - expected).abs().max())
    leak = float(actual.reshape(-1, 1 << workspace)[:, 1:].abs().square().sum())
    norm = float(actual.abs().square().sum())
    ports = [wire for wires in program.registers.values() for wire in wires]
    if (
        sorted(ports) != list(range(oracle.width))
        or program.report["workspace_qubits"] != workspace
    ):
        raise AssertionError("named outputs and hidden workspace disagree")
    if error > 1e-12 or leak > 1e-24 or abs(norm - 1) > 1e-10:
        raise AssertionError(
            f"full-state error={error}, workspace leak={leak}, norm={norm}"
        )
    return Validation(
        case,
        blob,
        params,
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
        workspace,
        error,
        leak,
        norm,
        time.perf_counter() - started,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("facility_location", "traveling_salesman_problem"),
        required=True,
    )
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = asdict(validate(args.case, args.scale))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
