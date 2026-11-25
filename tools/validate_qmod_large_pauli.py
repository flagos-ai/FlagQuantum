"""Verify unchanged large Pauli originals with independent sampling and Circuit."""

from __future__ import annotations

import argparse
import ast
import bisect
import hashlib
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.ecosystem._qmod.source import load_qmod

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
SEED = 0


@dataclass(frozen=True)
class Term:
    support: tuple[tuple[int, str], ...]
    coefficient: float


def real_literal(source: str) -> float:
    def evaluate(node: ast.AST) -> complex:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float, complex))
            and not isinstance(node.value, bool)
        ):
            return complex(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = evaluate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            left, right = evaluate(node.left), evaluate(node.right)
            return left + right if isinstance(node.op, ast.Add) else left - right
        raise ValueError("reference coefficients must be closed numeric literals")

    value = evaluate(ast.parse(source.strip(), mode="eval").body)
    if value.imag or not math.isfinite(value.real):
        raise ValueError("reference requires finite real Pauli coefficients")
    return value.real


def original_terms(source: str) -> tuple[Term, ...]:
    """Parse source data independently, without importing the production parser."""
    terms = []
    for match in re.finditer(r"\bSparsePauliTerm\s*\{", source):
        begin = match.end()
        depth = 1
        for end in range(begin, len(source)):
            if source[end] == "{":
                depth += 1
            elif source[end] == "}":
                depth -= 1
                if depth == 0:
                    break
        body = source[begin:end]
        support = []
        seen = set()
        for indexed in re.findall(r"\bIndexedPauli\s*\{([^}]*)\}", body, re.S):
            axis = re.search(r"\bpauli\s*=\s*Pauli::([IXYZ])\b", indexed)
            position = re.search(r"\bindex\s*=\s*(\d+)\b", indexed)
            if axis is None or position is None:
                raise ValueError("invalid independent Pauli record")
            index = int(position[1])
            if not 0 <= index < 12 or index in seen:
                raise ValueError("invalid independent Pauli index")
            seen.add(index)
            if axis[1] != "I":
                support.append((index, axis[1]))
        coefficient = re.search(r"\bcoefficient\s*=\s*([^\n}]+)", body)
        if coefficient is None:
            raise ValueError("missing independent coefficient")
        terms.append(Term(tuple(support), real_literal(coefficient[1])))
    if len(terms) != 551:
        raise ValueError("the unchanged originals must retain 551 terms")
    return tuple(terms)


def reference_schedule(
    terms: tuple[Term, ...], *, stochastic: bool
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not stochastic:
        return tuple(range(len(terms))), tuple(term.coefficient for term in terms)
    cumulative = []
    total = 0.0
    for term in terms:
        total += abs(term.coefficient)
        cumulative.append(total)
    rng = random.Random(SEED)
    # Invert the CDF using individual uniform draws, independently of choices().
    indices = tuple(
        bisect.bisect_right(cumulative, rng.random() * total, 0, len(terms) - 1)
        for _ in range(2000)
    )
    rotation_angle = math.fsum(abs(term.coefficient) for term in terms) / 2000
    angles = tuple(
        math.copysign(rotation_angle, terms[index].coefficient) for index in indices
    )
    return indices, angles


def reference_circuit(
    terms: tuple[Term, ...], indices: tuple[int, ...], angles: tuple[float, ...]
) -> fq.Circuit:
    circuit = fq.Circuit(12, dtype=torch.complex128)
    for index, angle in zip(indices, angles, strict=True):
        support = sorted(terms[index].support)
        if not support:
            circuit.x(0).phase(0, theta=-angle).x(0).phase(0, theta=-angle)
            continue
        # Use S-dagger/H for Y and the opposite parity anchor from production.
        # No production Pauli mapper, phase expansion or native parser is called.
        for bit, axis in support:
            qubit = 11 - bit
            if axis == "Y":
                circuit.gate("sdg", qubit)
            if axis != "Z":
                circuit.h(qubit)
        anchor = 11 - support[-1][0]
        for bit, _ in support[:-1]:
            circuit.cx(11 - bit, anchor)
        circuit.rz(anchor, theta=2 * angle)
        for bit, _ in reversed(support[:-1]):
            circuit.cx(11 - bit, anchor)
        for bit, axis in reversed(support):
            qubit = 11 - bit
            if axis != "Z":
                circuit.h(qubit)
            if axis == "Y":
                circuit.s(qubit)
    return circuit


@dataclass(frozen=True)
class Validation:
    case: str
    source_blob: str
    term_count: int
    source_pauli_factors: int
    seed: int | None
    sample_count: int | None
    sampled_indices: tuple[int, ...]
    sampled_indices_sha256: str
    rotation_payloads_sha256: str
    all_owned_rotations_match_reference_sequence: bool
    qubits: int
    operations: int
    depth: int
    workspace: int
    max_abs_state_error: float
    state_norm: float
    synthesis_seconds: float
    execution_seconds: float
    reference_seconds: float
    validation: str = "passed"


def validate(case: str) -> Validation:
    manifest = json.loads((FIXTURES / "large_pauli_sources.json").read_text())
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
        raise ValueError("original Pauli source digest mismatch")
    if (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        != manifest["license_sha256"]
    ):
        raise ValueError("original license digest mismatch")
    source = raw.decode()
    terms = original_terms(source)
    stochastic = case == "hamiltonian_evolution"
    indices, angles = reference_schedule(terms, stochastic=stochastic)
    # Compare every emitted physical rotation with the independent sampled
    # sequence. Duplicate source terms need not have distinct physical payloads.
    imported = load_qmod(path)
    rotations = [
        op
        for op in imported.program.body.blocks[0].operations
        if op.name == "qreg.pauli_rotation"
    ]
    if len(rotations) != len(indices):
        raise AssertionError("Pauli factors were lost or added")
    payloads = []
    for op, index, angle in zip(rotations, indices, angles, strict=True):
        if (
            op.attributes["support"] != terms[index].support
            or op.attributes["angle"] != angle
        ):
            raise AssertionError(
                "production rotation differs from the reference sample sequence"
            )
        payloads.append((terms[index].support, angle))
    started = time.perf_counter()
    program = synth.synthesize(
        path, max_qubits=12, max_workspace=0, memory_limit_bytes=64 * 1024**2
    )
    synthesis_seconds = time.perf_counter() - started
    print(f"{case}: synthesized {len(rotations)} compact factors", flush=True)
    started = time.perf_counter()
    actual = fq.run(program).statevector().reshape(-1)
    execution_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reference = reference_circuit(terms, indices, angles)
    expected = fq.run(reference).statevector().reshape(-1)
    reference_seconds = time.perf_counter() - started
    error = float((actual - expected).abs().max())
    norm = float(actual.abs().square().sum())
    if error > 1e-12 or abs(norm - 1) > 1e-10:
        raise AssertionError(f"full-state error={error}, norm={norm}")
    resources = program.report["resources"]
    if not isinstance(resources, ResourceEstimate):
        raise TypeError("resource report missing")
    if (
        resources.n_qubits != 12
        or program.report["workspace_qubits"] != 0
        or dict(program.registers)
        != {("qbv" if stochastic else "state"): tuple(range(12))}
    ):
        raise AssertionError("original register width, bit order or workspace changed")
    return Validation(
        case,
        blob,
        len(terms),
        len(rotations),
        SEED if stochastic else None,
        2000 if stochastic else None,
        indices,
        hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest(),
        hashlib.sha256(
            json.dumps(payloads, separators=(",", ":")).encode()
        ).hexdigest(),
        True,
        resources.n_qubits,
        resources.n_operations,
        resources.depth,
        0,
        error,
        norm,
        synthesis_seconds,
        execution_seconds,
        reference_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("example_exponentiation", "hamiltonian_evolution"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = asdict(validate(args.case))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "sampled_indices"}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
