"""Compare unchanged ordinary QPE originals with independent matrix Circuits."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import torch

import flagquantum as fq
from flagquantum import synth
from flagquantum.compiler.resource_estimation import ResourceEstimate

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
CASES = ("quantum_counting_qpe", "qpe_for_unitary_matrix", "qpe_for_matrix")
DTYPE = torch.complex128


class Validation(TypedDict):
    case: str
    source_blob: str
    qubits: int
    operations: int
    depth: int
    workspace: int
    phase_bits: int
    signed_phase: bool
    max_abs_state_error: float
    max_phase_probability_error: float
    workspace_nonzero_probability: float
    validation: str


def literal_array(source: str, anchor: str) -> torch.Tensor:
    """Read reference numeric literals without the production Qmod parser."""
    start = source.index("[", source.index(anchor) + len(anchor))
    depth = 0
    for stop in range(start, len(source)):
        depth += (source[stop] == "[") - (source[stop] == "]")
        if depth == 0:
            return torch.tensor(
                ast.literal_eval(re.sub(r"//[^\n]*", "", source[start : stop + 1])),
                dtype=DTYPE,
            )
    raise ValueError("unterminated reference array")


def reference(case: str, source: str) -> tuple[fq.Circuit, int, int]:
    bits = {
        "quantum_counting_qpe": 5,
        "qpe_for_unitary_matrix": 8,
        "qpe_for_matrix": 7,
    }[case]
    data = {
        "quantum_counting_qpe": 4,
        "qpe_for_unitary_matrix": 3,
        "qpe_for_matrix": 2,
    }[case]
    dimension = 1 << data
    circuit = fq.Circuit(bits + data, dtype=DTYPE)
    wires = tuple(range(bits, bits + data))
    if case == "quantum_counting_qpe":
        h = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
        preparation = torch.ones((1, 1), dtype=DTYPE)
        for _ in range(data):
            preparation = torch.kron(preparation, h)
        marked = torch.tensor(
            [-1 if (word & 3) + (word >> 2) <= 2 else 1 for word in range(dimension)],
            dtype=DTYPE,
        )
        zero = torch.eye(dimension, dtype=DTYPE)
        zero[0, 0] = -1
        unitary = -preparation @ zero @ preparation.mH @ torch.diag(marked)
        circuit.any(*wires, matrix=preparation)
    else:
        anchor = (
            "unitary("
            if case == "qpe_for_unitary_matrix"
            else "unitary_for_HM: real[][] ="
        )
        unitary = literal_array(source, anchor)
        if case == "qpe_for_matrix":
            initial = literal_array(source, "initial_vector: real[] =")
            assert abs(initial.norm().item() - 1) < 1e-12
            delta = torch.eye(dimension, dtype=DTYPE)[:, 0] - initial
            preparation = (
                torch.eye(dimension, dtype=DTYPE)
                - 2 * torch.outer(delta, delta.conj()) / delta.norm().square()
            )
            circuit.any(*wires, matrix=preparation)
    identity = torch.eye(dimension, dtype=DTYPE)
    torch.testing.assert_close(unitary.mH @ unitary, identity, atol=1e-12, rtol=0)
    for bit in range(bits):
        circuit.h(bit)
    for numeric_bit in range(bits):
        power = torch.linalg.matrix_power(unitary, 1 << numeric_bit)
        controlled = torch.zeros((2 * dimension, 2 * dimension), dtype=DTYPE)
        controlled[:dimension, :dimension] = identity
        controlled[dimension:, dimension:] = power
        circuit.any(bits - 1 - numeric_bit, *wires, matrix=controlled)
    axis = torch.arange(1 << bits, dtype=torch.float64)
    fourier = torch.exp(
        -2j * math.pi * torch.outer(axis, axis) / (1 << bits)
    ) / math.sqrt(1 << bits)
    circuit.any(*range(bits), matrix=fourier)
    return circuit, bits, data


def validate(case: str) -> Validation:
    manifest = json.loads((FIXTURES / "static_qpe_sources.json").read_text())
    row = next(r for r in manifest["fixtures"] if r["file"] == case + ".qmod")
    path = FIXTURES / row["file"]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    started = time.perf_counter()
    p = synth.synthesize(path, memory_limit_bytes=64 * 1024**2)
    print(case, "synthesis_seconds", time.perf_counter() - started, flush=True)
    resources = p.report["resources"]
    workspace = p.report["workspace_qubits"]
    assert isinstance(resources, ResourceEstimate) and isinstance(workspace, int)
    expected_circuit, bits, data = reference(case, raw.decode())
    expected = fq.run(expected_circuit).statevector().reshape(-1)
    print(case, "reference_ready_seconds", time.perf_counter() - started, flush=True)
    result = fq.run(p)
    print(case, "execution_ready_seconds", time.perf_counter() - started, flush=True)
    actual = result.statevector().reshape(expected.numel(), -1)
    error = float((actual[:, 0] - expected).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    name = (
        "phase_result"
        if case == "qpe_for_matrix"
        else "phase_var" if case == "quantum_counting_qpe" else "phase"
    )
    fractions = p.report["qmod_fractional_bits"]
    signed_names = p.report.get("qmod_signed_registers", ())
    assert isinstance(fractions, Mapping) and isinstance(signed_names, tuple)
    assert fractions[name] == bits
    signed = name in signed_names
    assert signed == (case == "qpe_for_matrix")
    probabilities = dict(p.probabilities(result, registers=(name,)))
    expected_probabilities = (
        expected.reshape(1 << bits, 1 << data).abs().square().sum(dim=1)
    )
    probability_error = max(
        abs(probabilities.get((word,), 0) - float(value))
        for word, value in enumerate(expected_probabilities)
    )
    # Raw keys are unchanged. Decode two's-complement fixed-point values explicitly.
    decoded = {
        (
            (word - (1 << bits) if signed and word & (1 << (bits - 1)) else word)
            / (1 << bits)
        ): value
        for (word,), value in probabilities.items()
    }
    assert all(
        (-0.5 <= value < 0.5) if signed else (0 <= value < 1) for value in decoded
    )
    print(case, "errors", error, probability_error, dirty, flush=True)
    assert max(error, probability_error) < 1e-12 and dirty < 1e-24, (
        error,
        probability_error,
        dirty,
    )
    return {
        "case": case,
        "source_blob": row["git_blob_sha1"],
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": workspace,
        "phase_bits": bits,
        "signed_phase": signed,
        "max_abs_state_error": error,
        "max_phase_probability_error": probability_error,
        "workspace_nonzero_probability": dirty,
        "validation": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(args.case), indent=2) + "\n")


if __name__ == "__main__":
    main()
