"""Original IQAE/QMCI vectors against independent Grover matrices and analytic amplitudes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import TypedDict

import torch

import flagquantum as fq
from flagquantum.compiler.resource_estimation import ResourceEstimate
from flagquantum.services.synthesis import _synthesize

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/classiq_qmod"
CASES = ("quantum_counting", "quantum_counting_iqae", "qmci_pi_estimation")
DTYPE = torch.complex128


class Validation(TypedDict):
    case: str
    repetitions: int
    source_blob: str
    qubits: int
    operations: int
    depth: int
    workspace: int
    marked_states: int
    max_abs_state_error: float
    analytic_max_abs_error: float
    workspace_nonzero_probability: float
    indicator_probability: float
    analytic_indicator_probability: float
    dropped_problem_is_semantic_data: bool
    validation: str


def marked(case: str, word: int) -> bool:
    if case == "qmci_pi_estimation":
        return (word & 1) ** 2 + (word >> 1) ** 2 < 4
    return (word & 3) + (word >> 2) <= 2


def reference(case: str, repetitions: int) -> fq.Circuit:
    width = 2 if case == "qmci_pi_estimation" else 4
    size = 1 << width
    permutation = torch.zeros((2 * size, 2 * size), dtype=DTYPE)
    for flag in (0, 1):
        for word in range(size):
            permutation[
                (flag ^ marked(case, word)) * size + word, flag * size + word
            ] = 1
    h = torch.tensor([[1, 1], [1, -1]], dtype=DTYPE) / math.sqrt(2)
    preparation = torch.eye(2, dtype=DTYPE)
    for _ in range(width):
        preparation = torch.kron(preparation, h)
    preparation = permutation @ preparation
    sign = torch.diag(torch.tensor([1] * size + [-1] * size, dtype=DTYPE))
    zero = torch.eye(2 * size, dtype=DTYPE)
    zero[0, 0] = -1
    grover = -preparation @ zero @ preparation.conj().T @ sign
    c = fq.Circuit(width + 1, dtype=DTYPE).any(*range(width + 1), matrix=preparation)
    for _ in range(repetitions):
        c.any(*range(width + 1), matrix=grover)
    return c


def analytic(case: str, repetitions: int) -> torch.Tensor:
    size = 4 if case == "qmci_pi_estimation" else 16
    good = sum(marked(case, word) for word in range(size))
    theta = math.asin(math.sqrt(good / size))
    state = torch.zeros(2 * size, dtype=DTYPE)
    for word in range(size):
        flag = int(marked(case, word))
        state[flag * size + word] = (
            math.sin((2 * repetitions + 1) * theta) / math.sqrt(good)
            if flag
            else math.cos((2 * repetitions + 1) * theta) / math.sqrt(size - good)
        )
    return state


def validate(case: str, repetitions: int) -> Validation:
    manifest = json.loads((FIXTURES / "iteration_sources.json").read_text())
    row = next(r for r in manifest["fixtures"] if r["file"] == case + ".qmod")
    source = FIXTURES / row["file"]
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["sha256"]
    assert (
        hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        == row["git_blob_sha1"]
    )
    assert (
        hashlib.sha256((FIXTURES / manifest["license_file"]).read_bytes()).hexdigest()
        == manifest["license_sha256"]
    )
    p = _synthesize(
        source, classical_inputs={"k": repetitions}, memory_limit_bytes=64 * 1024**2
    )
    resources = p.report["resources"]
    workspace = p.report["workspace_qubits"]
    if not isinstance(resources, ResourceEstimate) or not isinstance(workspace, int):
        raise TypeError("missing actual resource report")
    result = fq.run(p)
    expected = fq.run(reference(case, repetitions)).statevector().reshape(-1)
    actual = result.statevector().reshape(expected.numel(), -1)
    closed = analytic(case, repetitions)
    error = float((actual[:, 0] - expected).abs().max())
    analytic_error = float((actual[:, 0] - closed).abs().max())
    dirty = float(actual[:, 1:].abs().square().sum())
    probability = dict(p.probabilities(result, registers=("indicator",))).get((1,), 0)
    expected_probability = float(closed.reshape(2, -1)[1].abs().square().sum())
    assert max(error, analytic_error, abs(probability - expected_probability)) < 1e-12
    assert dirty < 1e-24
    return {
        "case": case,
        "repetitions": repetitions,
        "source_blob": row["git_blob_sha1"],
        "qubits": resources.n_qubits,
        "operations": resources.n_operations,
        "depth": resources.depth,
        "workspace": workspace,
        "marked_states": 4 if case == "qmci_pi_estimation" else 6,
        "max_abs_state_error": error,
        "analytic_max_abs_error": analytic_error,
        "workspace_nonzero_probability": dirty,
        "indicator_probability": probability,
        "analytic_indicator_probability": expected_probability,
        "dropped_problem_is_semantic_data": True,
        "validation": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--repetitions", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(validate(args.case, args.repetitions), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
